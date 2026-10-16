export type Source = {
  n: number
  citation: string
  heading: string
  breadcrumb: string
  url: string
  text: string
  chunk_id: string
}

export type Timings = Partial<Record<
  'route_s' | 'search_s' | 'rerank_s' | 'generate_s' | 'first_token_s' | 'total_s', number
>>

export type PipelineEvent =
  | { type: 'status'; stage: string }
  | { type: 'scope'; titles: number[]; parts: string[]; method: string }
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; text: string }
  | { type: 'done'; timings: Timings }
  | { type: 'cache'; hit: boolean }

export type TitleInfo = { number: number; name: string; semantic: boolean }
export type Health = { ok: boolean; llm: string; corpus_version: string; cache: boolean }

/** POST /api/ask and invoke onEvent for each server-sent event as it arrives. */
export async function ask(question: string, onEvent: (e: PipelineEvent) => void, signal?: AbortSignal) {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`)

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += value
    let sep
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)))
      }
    }
  }
}

export async function getTitles(): Promise<TitleInfo[]> {
  const res = await fetch('/api/titles')
  return res.ok ? res.json() : []
}

export async function getHealth(): Promise<Health | null> {
  try {
    const res = await fetch('/health')
    return res.ok ? res.json() : null
  } catch {
    return null
  }
}

export async function sendFeedback(body: {
  question: string; answer: string; rating: -1 | 1; comment: string; sources: string[]
}) {
  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
