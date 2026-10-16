import { useEffect, useRef, useState } from 'react'
import {
  ask, getHealth, getTitles, sendFeedback,
  type Health, type PipelineEvent, type Source, type Timings, type TitleInfo,
} from './api'

type Stage = 'route' | 'search' | 'generate' | 'done' | 'error'

type Turn = {
  id: number
  question: string
  stage: Stage
  scope?: { titles: number[]; parts: string[]; method: string }
  sources: Source[]
  answer: string
  timings?: Timings
  cached?: boolean
  error?: string
  feedback?: -1 | 1
}

const EXAMPLES = [
  'How often must a trucking company review each driver’s driving record?',
  'What records must an employer keep about overtime hours worked?',
  'What must be on the label of a dietary supplement?',
  'When is a facility required to report a release of a hazardous substance?',
]

const STAGES: { key: Stage; label: string }[] = [
  { key: 'route', label: 'Route' },
  { key: 'search', label: 'Search & rerank' },
  { key: 'generate', label: 'Write answer' },
  { key: 'done', label: 'Done' },
]

function stageIndex(s: Stage) {
  return STAGES.findIndex(x => x.key === s)
}

/** Render answer text, turning [1] / [2, 3] into clickable citation chips. */
function AnswerText({ text, sources, onCite }: {
  text: string; sources: Source[]; onCite: (s: Source) => void
}) {
  const parts = text.split(/(\[[\d,\s]+\])/g)
  return (
    <div className="answer">
      {parts.map((p, i) => {
        const m = p.match(/^\[([\d,\s]+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        return m[1].split(',').map(n => n.trim()).filter(Boolean).map(n => {
          const src = sources[Number(n) - 1]
          return src
            ? <button key={`${i}-${n}`} className="cite" title={src.citation} onClick={() => onCite(src)}>{n}</button>
            : <span key={`${i}-${n}`} className="cite invalid" title="No such source">{n}</span>
        })
      })}
    </div>
  )
}

function Trace({ turn }: { turn: Turn }) {
  const current = stageIndex(turn.stage)
  return (
    <ol className="trace" aria-label="Pipeline progress">
      {STAGES.map((s, i) => {
        const state = turn.stage === 'error' ? (i === 0 ? 'error' : 'todo')
          : i < current || turn.stage === 'done' ? 'done' : i === current ? 'active' : 'todo'
        let detail = ''
        if (s.key === 'route' && turn.scope) {
          detail = turn.scope.titles.length
            ? `Title ${turn.scope.titles.join(', ')}${turn.scope.parts.length ? ` · Part ${turn.scope.parts.join(', ')}` : ''}`
            : 'All titles'
        }
        if (s.key === 'search' && turn.sources.length) detail = `${turn.sources.length} sources`
        if (s.key === 'done' && turn.timings?.total_s != null) detail = `${turn.timings.total_s.toFixed(1)} s`
        return (
          <li key={s.key} className={state}>
            <span className="dot" aria-hidden />
            <span className="label">{s.label}</span>
            {detail && <span className="detail">{detail}</span>}
          </li>
        )
      })}
    </ol>
  )
}

function TimingBar({ t, cached }: { t: Timings; cached?: boolean }) {
  if (cached) return <div className="timings"><span className="badge">Cached answer</span></div>
  const items: [string, number | undefined][] = [
    ['route', t.route_s], ['search', t.search_s], ['rerank', t.rerank_s],
    ['first token', t.first_token_s], ['total', t.total_s],
  ]
  return (
    <div className="timings">
      {items.filter(([, v]) => v != null).map(([k, v]) => (
        <span key={k}><b>{v!.toFixed(1)}s</b> {k}</span>
      ))}
    </div>
  )
}

function Feedback({ turn, onRate }: { turn: Turn; onRate: (r: -1 | 1, comment: string) => void }) {
  const [comment, setComment] = useState('')
  const [open, setOpen] = useState(false)
  if (turn.feedback) return <div className="feedback muted">Thanks for the feedback.</div>
  return (
    <div className="feedback">
      <span className="muted">Was this helpful?</span>
      <button className="ghost" onClick={() => onRate(1, '')} aria-label="Helpful">👍</button>
      <button className="ghost" onClick={() => setOpen(true)} aria-label="Not helpful">👎</button>
      {open && (
        <form onSubmit={e => { e.preventDefault(); onRate(-1, comment) }} className="feedback-form">
          <input value={comment} onChange={e => setComment(e.target.value)}
                 placeholder="What was wrong? (e.g. wrong citation, missing rule)" autoFocus />
          <button type="submit">Send</button>
        </form>
      )}
    </div>
  )
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [panel, setPanel] = useState<Source | null>(null)
  const [titles, setTitles] = useState<TitleInfo[]>([])
  const [health, setHealth] = useState<Health | null | undefined>(undefined)
  const bottom = useRef<HTMLDivElement>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    getHealth().then(setHealth)
    getTitles().then(setTitles).catch(() => setTitles([]))
  }, [])

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns])

  const update = (id: number, fn: (t: Turn) => Turn) =>
    setTurns(ts => ts.map(t => (t.id === id ? fn(t) : t)))

  async function submit(q: string) {
    q = q.trim()
    if (!q || busy) return
    const id = Date.now()
    setTurns(ts => [...ts, { id, question: q, stage: 'route', sources: [], answer: '' }])
    setQuestion('')
    setBusy(true)
    abort.current = new AbortController()
    try {
      await ask(q, (e: PipelineEvent) => {
        update(id, t => {
          switch (e.type) {
            case 'scope': return { ...t, scope: e, stage: 'search' }
            case 'sources': return { ...t, sources: e.sources, stage: 'generate' }
            case 'token': return { ...t, answer: t.answer + e.text, stage: 'generate' }
            case 'done': return { ...t, timings: e.timings, stage: 'done' }
            case 'cache': return { ...t, cached: true, stage: 'done' }
            default: return t
          }
        })
      }, abort.current.signal)
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        update(id, t => ({ ...t, stage: 'error', error: (err as Error).message }))
      }
    } finally {
      setBusy(false)
    }
  }

  function rate(turn: Turn, rating: -1 | 1, comment: string) {
    sendFeedback({ question: turn.question, answer: turn.answer, rating, comment,
                   sources: turn.sources.map(s => s.citation) })
    update(turn.id, t => ({ ...t, feedback: rating }))
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>RegRAG</h1>
        <p className="muted">Answers from the Code of Federal Regulations, with citations.</p>
        <h2>Indexed titles</h2>
        <ul className="titles">
          {titles.map(t => (
            <li key={t.number}>
              <span className="tnum">{t.number}</span>
              <span className="tname">{t.name}</span>
              <span className={`tier ${t.semantic ? 'sem' : 'kw'}`}
                    title={t.semantic ? 'Semantic + keyword search' : 'Keyword search only (semantic embedding pending)'}>
                {t.semantic ? 'hybrid' : 'keyword'}
              </span>
            </li>
          ))}
          {!titles.length && <li className="muted">No titles loaded.</li>}
        </ul>
        <div className="status muted">
          {health === undefined ? 'Connecting…'
            : health ? <>Model <code>{health.llm}</code> · corpus <code>{health.corpus_version}</code>{health.cache ? ' · cache on' : ''}</>
            : <span className="err">API offline — start it with <code>uvicorn regrag.api:app</code></span>}
        </div>
      </aside>

      <main className="main">
        <div className="thread">
          {!turns.length && (
            <div className="empty">
              <h2>Ask about federal regulations</h2>
              <p className="muted">Try one of these:</p>
              <div className="examples">
                {EXAMPLES.map(e => <button key={e} className="example" onClick={() => submit(e)}>{e}</button>)}
              </div>
            </div>
          )}
          {turns.map(turn => (
            <article key={turn.id} className="turn">
              <div className="question">{turn.question}</div>
              <Trace turn={turn} />
              {turn.error && <div className="err">Something went wrong: {turn.error}</div>}
              {turn.answer && <AnswerText text={turn.answer} sources={turn.sources} onCite={setPanel} />}
              {turn.stage !== 'done' && turn.stage !== 'error' && !turn.answer && (
                <div className="muted pulse">{turn.stage === 'generate' ? 'Writing the answer…' : 'Working…'}</div>
              )}
              {turn.sources.length > 0 && (
                <details className="sources" open={turn.stage === 'done'}>
                  <summary>Sources ({turn.sources.length})</summary>
                  <ol>
                    {turn.sources.map(s => (
                      <li key={s.chunk_id}>
                        <button className="link" onClick={() => setPanel(s)}>
                          <b>{s.citation}</b> {s.heading.replace(/^§\s*[\d.]+\s*/, '')}
                        </button>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
              {turn.stage === 'done' && (
                <>
                  {(turn.timings || turn.cached) && <TimingBar t={turn.timings ?? {}} cached={turn.cached} />}
                  <Feedback turn={turn} onRate={(r, c) => rate(turn, r, c)} />
                </>
              )}
            </article>
          ))}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={e => { e.preventDefault(); submit(question) }}>
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(question) } }}
            placeholder="Ask a regulatory question…"
            rows={2}
            aria-label="Question"
          />
          {busy
            ? <button type="button" onClick={() => abort.current?.abort()}>Stop</button>
            : <button type="submit" disabled={!question.trim()}>Ask</button>}
        </form>
      </main>

      {panel && (
        <aside className="panel" aria-label="Source text">
          <div className="panel-head">
            <div>
              <div className="panel-cite">{panel.citation}</div>
              <div className="muted small">{panel.breadcrumb}</div>
            </div>
            <button className="ghost" onClick={() => setPanel(null)} aria-label="Close">✕</button>
          </div>
          <div className="panel-body">
            {panel.text.split('\n').map((p, i) => <p key={i}>{p}</p>)}
          </div>
          <a href={panel.url} target="_blank" rel="noreferrer">Open on ecfr.gov ↗</a>
        </aside>
      )}
    </div>
  )
}
