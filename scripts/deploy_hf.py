"""
Deploy the app to a Hugging Face Docker Space.

    venv/Scripts/python.exe scripts/deploy_hf.py [--space Prakruthi/federal-rules-guide]

Reads from .env: HF_TOKEN, QDRANT_CLOUD_URL, QDRANT_CLOUD_KEY, LLM_PROVIDER,
LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_REASONING_EFFORT. Secrets are set on
the Space (never uploaded as files). The Space is built from the committed
code (git archive HEAD) plus data/state.db, which is gitignored.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Federal Rules Guide
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Plain-English answers to U.S. federal regulations
---

Plain-English answers to U.S. federal regulations (eCFR), with the exact rule
cited, plus a compliance checklist per business type.

Source: https://github.com/Prakruthi19/fleet-compliance-assistant
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="Prakruthi/federal-rules-guide")
    ap.add_argument("--ask-limit", default="20", help="questions per visitor per hour")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    secrets = {
        "QDRANT_URL": os.environ["QDRANT_CLOUD_URL"],
        "QDRANT_API_KEY": os.environ["QDRANT_CLOUD_KEY"],
        "LLM_API_KEY": os.environ["LLM_API_KEY"],
    }
    variables = {
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "openai"),
        "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        "LLM_MODEL": os.environ["LLM_MODEL"],
        "LLM_REASONING_EFFORT": os.environ.get("LLM_REASONING_EFFORT", ""),
        "ASK_LIMIT_PER_HOUR": args.ask_limit,
        "REGRAG_DATA_DIR": "/app/data",
    }
    for k, v in secrets.items():
        api.add_space_secret(args.space, k, v)
    for k, v in variables.items():
        if v:
            api.add_space_variable(args.space, k, v)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        stage.mkdir()
        archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(stage)], input=archive, check=True)
        shutil.copy(ROOT / "data" / "state.db", stage / "data" / "state.db")
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(folder_path=str(stage), repo_id=args.space, repo_type="space",
                          commit_message="Deploy from git HEAD",
                          ignore_patterns=["evals/**", "src/**", "files/**", "*.ipynb"])
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
