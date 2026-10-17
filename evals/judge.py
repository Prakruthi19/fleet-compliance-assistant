"""
LLM-judged answer quality with DeepEval. Runs in its own venv because
DeepEval's dependency pins clash with the app's:

    venv-eval/Scripts/python.exe evals/judge.py evals/runs/v1.jsonl [--judge groq|azure]

Metrics (each 0-1):
  faithfulness       — are the answer's claims supported by the retrieved contexts? (hallucination)
  answer_relevancy   — does the answer address the question?
  contextual_recall  — do the retrieved contexts contain what the reference answer needs?
  citations_valid    — (no LLM) every [n] in the answer points to a real source, and there is at least one

The default judge is the local Ollama model with thinking disabled. A 4B
judge is noisy; use --judge groq (gpt-oss-120b) or azure for scores you'd report.
"""
import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

load_dotenv()


class LocalOllama(DeepEvalBaseLLM):
    """DeepEval's built-in Ollama wrapper can't disable qwen's thinking mode, which makes judging ~10x slower."""

    def __init__(self, model: str, url: str):
        self.model, self.url = model, url

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": 8192}}
        if schema:
            body["format"] = schema.model_json_schema()
        resp = httpx.post(f"{self.url}/api/chat", json=body, timeout=900)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return schema.model_validate_json(content) if schema else content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"ollama:{self.model}"


class OpenAICompatible(DeepEvalBaseLLM):
    """Any OpenAI-compatible host (Groq by default). JSON mode + schema in the
    prompt, because strict json_schema support varies by model."""

    def __init__(self, model: str, base_url: str, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=8)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {}
        if schema:
            messages.insert(0, {"role": "system", "content": "Reply with only a JSON object matching this schema:\n"
                                + json.dumps(schema.model_json_schema())})
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, **kwargs)
        content = resp.choices[0].message.content
        return schema.model_validate_json(content) if schema else content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"openai-compatible:{self.model}"


def judge_model(kind: str):
    if kind == "groq":
        # Default judge is a different model family than the qwen generator,
        # so the judge isn't grading its own answers.
        return OpenAICompatible(os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"),
                                os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                                os.environ["LLM_API_KEY"])
    if kind == "azure":
        from deepeval.models import AzureOpenAIModel
        return AzureOpenAIModel(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
        )
    return LocalOllama(os.environ.get("LLM_MODEL", "qwen3.5:4b"),
                       os.environ.get("OLLAMA_URL", "http://localhost:11434"))


def citations_valid(answer: str, n_sources: int) -> float:
    refs = [int(x) for group in re.findall(r"\[([\d,\s]+)\]", answer) for x in re.findall(r"\d+", group)]
    return float(bool(refs) and all(1 <= r <= n_sources for r in refs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_file", type=Path)
    ap.add_argument("--judge", choices=["local", "azure", "groq"], default="local")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--same-as", type=Path,
                    help="a previous *-judged-*.jsonl: judge only the ids it scored, for a like-for-like comparison")
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy,contextual_recall",
                    help="comma-separated subset (the judge's daily token quota is the bottleneck)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.run_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.same_as:
        prev = [json.loads(l) for l in args.same_as.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep = {r["id"] for r in prev if r.get("faithfulness") is not None}
        rows = [r for r in rows if r["id"] in keep]
    rows = rows[: args.limit] if args.limit else rows
    model = judge_model(args.judge)
    all_metrics = {
        "faithfulness": lambda: FaithfulnessMetric(model=model, async_mode=False, include_reason=False),
        "answer_relevancy": lambda: AnswerRelevancyMetric(model=model, async_mode=False, include_reason=False),
        "contextual_recall": lambda: ContextualRecallMetric(model=model, async_mode=False, include_reason=False),
    }
    metrics = {k: all_metrics[k] for k in args.metrics.split(",")}

    out_path = args.run_file.with_name(f"{args.run_file.stem}-judged-{args.judge}{args.tag}.jsonl")
    scored = []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            case = LLMTestCase(input=row["question"], actual_output=row["answer"],
                               expected_output=row["reference"], retrieval_context=row["contexts"])
            scores = {"citations_valid": citations_valid(row["answer"], len(row["contexts"]))}
            for name, make in metrics.items():
                try:
                    m = make()
                    m.measure(case)
                    scores[name] = m.score
                except Exception as e:  # small judges sometimes emit invalid JSON
                    print(f"  {name} failed: {e!r}", file=sys.stderr)
                    scores[name] = None
            scored.append(scores)
            f.write(json.dumps({"id": row["id"], **scores}) + "\n")
            f.flush()
            print(f"{i}/{len(rows)} {scores}", flush=True)

    summary = {"run_file": str(args.run_file), "judge": model.get_model_name(), "n": len(rows)}
    for name in ["citations_valid", *metrics]:
        vals = [s[name] for s in scored if s[name] is not None]
        summary[name] = round(statistics.mean(vals), 3) if vals else None
        summary[f"{name}_failed"] = len(scored) - len(vals)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
