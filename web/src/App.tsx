import { useEffect, useState } from 'react'
import {
  Apple, ArrowLeft, Ban, Check, CircleCheck, ClipboardList, Clock, ExternalLink, Factory, FileText,
  Lightbulb, ListChecks, MessageCircleQuestion, Repeat, RotateCcw, Scale, Send, ThumbsDown, ThumbsUp,
  TriangleAlert, Truck, User, Users, type LucideIcon,
} from 'lucide-react'
import {
  ask, getObligations, sendFeedback,
  type ObligationCard, type PipelineEvent, type Profile, type Source, type Timings,
} from './api'

/*
 * Layout: one screen, no page scroll on desktop. Three columns —
 *   left rail   : choices (business type + questions, or examples + history)
 *   center      : the main content (checklist grid, or the current answer)
 *   right panel : details for whatever is selected (card details, rule text)
 * Each column scrolls on its own only if it has to.
 */

type Stage = 'route' | 'search' | 'generate' | 'done' | 'error'
type PanelSource = Omit<Source, 'n'>

type Turn = {
  id: number
  question: string
  stage: Stage
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
  'When must a facility report a release of a hazardous substance?',
]

const PROFILE_ICONS: Record<string, LucideIcon> = { trucking: Truck, employer: Users, food: Apple, facility: Factory }
const profileIcon = (id: string) => PROFILE_ICONS[id] ?? ClipboardList

const LABEL_ICONS: Record<string, LucideIcon> = {
  who: User, what: CircleCheck, deadline: Clock, 'how often': Repeat, records: FileText,
  exceptions: Ban, penalty: TriangleAlert, note: Lightbulb,
}

const STAGES: { key: Stage; label: string }[] = [
  { key: 'route', label: 'Reading your question' },
  { key: 'search', label: 'Finding the rules' },
  { key: 'generate', label: 'Writing the answer' },
]

/* Per-viewer conveniences only; the page works without storage. */
function readStore<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}
function writeStore(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* storage unavailable */ }
}

const sectionOf = (citation: string) => citation.split(' ').pop()

/* ---------------- Answer rendering ---------------- */

/** Inline text: **bold** spans and [1] / [2, 3] citation chips (shown as the section number). */
function Inline({ text, sources, onCite }: {
  text: string; sources: PanelSource[]; onCite: (s: PanelSource) => void
}) {
  return (
    <>
      {/* "fact [2]." -> "fact. [2]" so the chip doesn't strand the full stop */}
      {text.replace(/\s*((?:\[[\d,\s]+\]\s*)+)([.,;])/g, '$2 $1').split(/(\[[\d,\s]+\]|\*\*[^*]+\*\*)/g).map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**') && p.length > 4) return <b key={i}>{p.slice(2, -2)}</b>
        const m = p.match(/^\[([\d,\s]+)\]$/)
        if (!m) return <span key={i}>{p}</span>
        return m[1].split(',').map(n => n.trim()).filter(Boolean).map(n => {
          const src = sources[Number(n) - 1]
          return src
            ? <button key={`${i}-${n}`} className="cite" title={src.heading} onClick={() => onCite(src)}>
                §&nbsp;{sectionOf(src.citation)}
              </button>
            : null
        })
      })}
    </>
  )
}

/**
 * The answer prompt asks for "**Answer:** ..." plus "- **Label:** fact [n]"
 * bullets; render that as a summary box and icon rows. Anything else falls
 * back to plain paragraphs, so partial (streaming) text renders too.
 */
function FormattedAnswer({ text, sources, onCite }: {
  text: string; sources: PanelSource[]; onCite: (s: PanelSource) => void
}) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean)
  const points: React.ReactNode[] = []
  const other: React.ReactNode[] = []
  let summary: React.ReactNode = null
  lines.forEach((line, i) => {
    const s = line.match(/^\*\*(?:Answer|Short answer):\*\*\s*(.*)$/i)
    if (s) {
      summary = <p key={i} className="tldr"><Inline text={s[1]} sources={sources} onCite={onCite} /></p>
      return
    }
    const bullet = line.match(/^[-*•]\s+(.*)$/)
    if (bullet) {
      const labeled = bullet[1].match(/^\*\*([^*:]+):\*\*\s*(.*)$/)
      const label = labeled?.[1].trim()
      const Icon = (label && LABEL_ICONS[label.toLowerCase()]) || CircleCheck
      points.push(
        <div key={i} className="point">
          <span className={`picon l-${(label ?? 'what').toLowerCase().replace(' ', '-')}`} aria-hidden><Icon size={18} /></span>
          <div>
            {label && <div className="plabel">{label}</div>}
            <Inline text={labeled ? labeled[2] : bullet[1]} sources={sources} onCite={onCite} />
          </div>
        </div>,
      )
      return
    }
    other.push(<p key={i}><Inline text={line} sources={sources} onCite={onCite} /></p>)
  })
  return (
    <div className="answer">
      {summary}
      {other}
      {points.length > 0 && <div className="points">{points}</div>}
    </div>
  )
}

function Progress({ stage }: { stage: Stage }) {
  const current = STAGES.findIndex(s => s.key === stage)
  return (
    <ol className="progress" aria-label="Progress">
      {STAGES.map((s, i) => (
        <li key={s.key} className={i < current ? 'done' : i === current ? 'active' : ''}>
          <span className="dot" aria-hidden />{s.label}
        </li>
      ))}
    </ol>
  )
}

function Feedback({ turn, onRate }: { turn: Turn; onRate: (r: -1 | 1, comment: string) => void }) {
  const [comment, setComment] = useState('')
  const [open, setOpen] = useState(false)
  if (turn.feedback) return <span className="muted small">Thanks for the feedback.</span>
  return (
    <div className="feedback">
      <span className="muted small">Helpful?</span>
      <button className="ghost" onClick={() => onRate(1, '')} aria-label="Helpful"><ThumbsUp size={16} /></button>
      <button className="ghost" onClick={() => setOpen(true)} aria-label="Not helpful"><ThumbsDown size={16} /></button>
      {open && (
        <form onSubmit={e => { e.preventDefault(); onRate(-1, comment) }} className="feedback-form">
          <input value={comment} onChange={e => setComment(e.target.value)}
                 placeholder="What was wrong?" autoFocus />
          <button type="submit"><Send size={15} /> Send</button>
        </form>
      )}
    </div>
  )
}

/* ---------------- Right panel ---------------- */

function RuleText({ source, onBack }: { source: PanelSource; onBack?: () => void }) {
  return (
    <div className="rule">
      {onBack && <button className="ghost back" onClick={onBack}><ArrowLeft size={16} /> Back</button>}
      <div className="rule-cite">{source.citation}</div>
      <div className="muted small">{source.heading}</div>
      <div className="rule-body">{source.text.split('\n').map((p, i) => <p key={i}>{p}</p>)}</div>
      <a href={source.url} target="_blank" rel="noreferrer">Read the full rule on ecfr.gov <ExternalLink size={14} /></a>
    </div>
  )
}

function CardDetail({ card, done, onToggle, onCite, onAsk }: {
  card: ObligationCard; done: boolean; onToggle: () => void
  onCite: (s: PanelSource) => void; onAsk: (q: string) => void
}) {
  return (
    <div className="detail">
      <h2>{card.name}</h2>
      <p className="tldr">{card.summary}</p>
      <div className="facts">
        {card.applies_to && <span className="pill who"><User size={13} /> {card.applies_to}</span>}
        {card.how_often && <span className="pill when"><Clock size={13} /> {card.how_often}</span>}
      </div>
      {card.requirements.length > 0 && (
        <>
          <h4>What you must do</h4>
          <ul className="reqs">
            {card.requirements.map((r, i) => {
              const src = card.sources[r.source - 1]
              return (
                <li key={i}>
                  {r.text}
                  {src && <button className="cite" onClick={() => onCite(src)}>§&nbsp;{sectionOf(src.citation)}</button>}
                </li>
              )
            })}
          </ul>
        </>
      )}
      {card.records.length > 0 && (
        <>
          <h4>Records to keep</h4>
          <div className="records">{card.records.map(r => <span key={r} className="pill rec"><FileText size={13} /> {r}</span>)}</div>
        </>
      )}
      <div className="detail-actions">
        <button className={done ? 'secondary' : ''} onClick={onToggle}>{done ? <><RotateCcw size={15} /> Mark as not done</> : <><Check size={15} /> Mark as done</>}</button>
        <button className="secondary" onClick={() => onAsk(`${card.name}: `)}><MessageCircleQuestion size={15} /> Ask a follow-up</button>
      </div>
    </div>
  )
}

/* ---------------- Checklist tab ---------------- */

function ChecklistTab({ profiles, onAsk }: { profiles: Profile[] | null; onAsk: (q: string) => void }) {
  const [profileId, setProfileId] = useState<string | null>(() => readStore('profile', null))
  const [answers, setAnswers] = useState<Record<string, boolean>>(() => readStore('answers', {}))
  const [done, setDone] = useState<Record<string, boolean>>(() => readStore('done', {}))
  const [selected, setSelected] = useState<string | null>(null)
  const [rule, setRule] = useState<PanelSource | null>(null)

  useEffect(() => writeStore('profile', profileId), [profileId])
  useEffect(() => writeStore('answers', answers), [answers])
  useEffect(() => writeStore('done', done), [done])

  if (profiles === null) return <div className="center muted pulse">Loading…</div>
  if (!profiles.length) return <div className="center notice">The checklist isn’t available yet. Try “Ask a question”.</div>

  const profile = profiles.find(p => p.id === profileId)
  const applies = (c: ObligationCard) => !c.when || answers[c.when]
  const active = profile ? profile.cards.filter(applies) : []
  const skipped = profile ? profile.cards.filter(c => !applies(c)) : []
  const key = (c: ObligationCard) => `${profile!.id}/${c.id}`
  const doneCount = active.filter(c => done[key(c)]).length
  const card = active.find(c => c.id === selected) ?? active[0]
  const toggle = (c: ObligationCard) => setDone(d => ({ ...d, [key(c)]: !d[key(c)] }))

  function choose(id: string) {
    setProfileId(id)
    setSelected(null)
    setRule(null)
  }

  return (
    <div className="cols">
      <aside className="rail">
        <h3>Your business</h3>
        <div className="profile-list">
          {profiles.map(p => {
            const Icon = profileIcon(p.id)
            return (
            <button key={p.id} className={`profile p-${p.id} ${p.id === profileId ? 'on' : ''}`} onClick={() => choose(p.id)}>
              <span className="icon" aria-hidden><Icon size={22} /></span>
              <span><span className="pname">{p.name}</span><span className="pblurb">{p.blurb}</span></span>
            </button>
            )
          })}
        </div>
        {profile && profile.questions.length > 0 && (
          <>
            <h3>About you</h3>
            <div className="questions">
              {profile.questions.map(q => (
                <label key={q.id} className="toggle">
                  <input type="checkbox" checked={!!answers[q.id]}
                         onChange={e => setAnswers(a => ({ ...a, [q.id]: e.target.checked }))} />
                  {q.label}
                </label>
              ))}
            </div>
          </>
        )}
      </aside>

      <section className="center-col">
        {!profile ? (
          <div className="welcome">
            <div className="big-icon" aria-hidden><ClipboardList size={44} /></div>
            <h2>Which federal rules apply to my business?</h2>
            <p className="muted">Pick your business type on the left. You’ll get a checklist of the main
              federal rules, each explained in plain English with the exact rule linked.</p>
            <div className="welcome-picks">
              {profiles.map(p => (
                <button key={p.id} className="secondary" onClick={() => choose(p.id)}>
                  {(() => { const Icon = profileIcon(p.id); return <Icon size={16} /> })()} {p.name}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="center-head">
              <h2 className="with-icon">{(() => { const Icon = profileIcon(profile.id); return <span className={`hicon p-${profile.id}`} aria-hidden><Icon size={20} /></span> })()} {profile.name} checklist</h2>
              <div className="progress-wrap">
                <div className="progressbar" role="progressbar" aria-valuemin={0}
                     aria-valuemax={active.length} aria-valuenow={doneCount}>
                  <div style={{ width: `${active.length ? (100 * doneCount) / active.length : 0}%` }} />
                </div>
                <span className="small muted">{doneCount}/{active.length} done</span>
              </div>
            </div>
            <div className="tiles">
              {active.map(c => (
                <div key={c.id} role="button" tabIndex={0}
                     className={`tile ${c.id === card?.id ? 'on' : ''} ${done[key(c)] ? 'is-done' : ''}`}
                     onClick={() => { setSelected(c.id); setRule(null) }}
                     onKeyDown={e => { if (e.key === 'Enter') { setSelected(c.id); setRule(null) } }}>
                  <label className="check" onClick={e => e.stopPropagation()}>
                    <input type="checkbox" checked={!!done[key(c)]} onChange={() => toggle(c)} />
                    <span className="box" aria-hidden />
                    <span className="sr-only">Mark {c.name} as done</span>
                  </label>
                  <div className="tile-body">
                    <div className="tile-name">{c.name}</div>
                    <div className="tile-sum">{c.summary}</div>
                    {c.how_often && <span className="pill when"><Clock size={13} /> {c.how_often}</span>}
                  </div>
                </div>
              ))}
            </div>
            {skipped.length > 0 && (
              <p className="muted small skipped">Hidden based on your answers: {skipped.map(c => c.name).join(' · ')}</p>
            )}
          </>
        )}
      </section>

      <aside className="side">
        {rule ? <RuleText source={rule} onBack={() => setRule(null)} />
          : card ? <CardDetail card={card} done={!!done[key(card)]} onToggle={() => toggle(card)}
                               onCite={setRule} onAsk={onAsk} />
          : <div className="side-empty muted">Details for the rule you pick show up here.</div>}
      </aside>
    </div>
  )
}

/* ---------------- Ask tab ---------------- */

/** Citation numbers used in an answer, in order of first use. */
function citedNumbers(answer: string): number[] {
  const nums: number[] = []
  for (const g of answer.matchAll(/\[([\d,\s]+)\]/g)) {
    for (const n of g[1].split(',')) {
      const v = Number(n.trim())
      if (v && !nums.includes(v)) nums.push(v)
    }
  }
  return nums
}

function SourceButton({ s, onOpen }: { s: PanelSource; onOpen: (s: PanelSource) => void }) {
  return (
    <button className="source" onClick={() => onOpen(s)}>
      <span className="rule-cite">§ {sectionOf(s.citation)}</span>
      <span className="small">{s.heading.replace(/^§\s*[\d.]+\s*/, '')}</span>
    </button>
  )
}

function AskTab({ question, setQuestion }: { question: string; setQuestion: (q: string) => void }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [rule, setRule] = useState<PanelSource | null>(null)
  const [abort, setAbort] = useState<AbortController | null>(null)

  const update = (id: number, fn: (t: Turn) => Turn) =>
    setTurns(ts => ts.map(t => (t.id === id ? fn(t) : t)))
  const turn = turns.find(t => t.id === currentId) ?? null

  async function submit(q: string) {
    q = q.trim()
    if (!q || busy) return
    const id = Date.now()
    setTurns(ts => [{ id, question: q, stage: 'route', sources: [], answer: '' }, ...ts])
    setCurrentId(id)
    setRule(null)
    setQuestion('')
    setBusy(true)
    const ctrl = new AbortController()
    setAbort(ctrl)
    try {
      await ask(q, (e: PipelineEvent) => {
        update(id, t => {
          switch (e.type) {
            case 'scope': return { ...t, stage: 'search' }
            case 'sources': return { ...t, sources: e.sources, stage: 'generate' }
            case 'token': return { ...t, answer: t.answer + e.text, stage: 'generate' }
            case 'done': return { ...t, timings: e.timings, stage: 'done' }
            case 'cache': return { ...t, cached: true, stage: 'done' }
            default: return t
          }
        })
      }, ctrl.signal)
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        update(id, t => ({ ...t, stage: 'error', error: (err as Error).message }))
      }
    } finally {
      setBusy(false)
    }
  }

  function rate(t: Turn, rating: -1 | 1, comment: string) {
    sendFeedback({ question: t.question, answer: t.answer, rating, comment, sources: t.sources.map(s => s.citation) })
    update(t.id, x => ({ ...x, feedback: rating }))
  }

  return (
    <div className="cols ask-cols">
      <aside className="rail">
        <h3>Try asking</h3>
        <div className="example-list">
          {EXAMPLES.map(e => <button key={e} className="example" onClick={() => submit(e)} disabled={busy}>{e}</button>)}
        </div>
        {turns.length > 0 && (
          <>
            <h3>Your questions</h3>
            <div className="history">
              {turns.map(t => (
                <button key={t.id} className={`hist ${t.id === currentId ? 'on' : ''}`}
                        onClick={() => { setCurrentId(t.id); setRule(null) }}>
                  {t.question}
                </button>
              ))}
            </div>
          </>
        )}
      </aside>

      <section className="center-col">
        <form className="composer" onSubmit={e => { e.preventDefault(); submit(question) }}>
          <textarea
            autoFocus
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(question) } }}
            placeholder="Type your question in plain words…"
            rows={2}
            aria-label="Question"
          />
          {busy
            ? <button type="button" onClick={() => abort?.abort()}>Stop</button>
            : <button type="submit" disabled={!question.trim()}>Ask</button>}
        </form>

        {!turn ? (
          <div className="welcome">
            <div className="big-icon" aria-hidden><MessageCircleQuestion size={44} /></div>
            <h2>Ask anything about federal rules</h2>
            <p className="muted">You’ll get a short answer, the key points, and the exact rule it comes from.</p>
          </div>
        ) : (
          <article className="turn">
            <div className="question">{turn.question}</div>
            {turn.stage !== 'done' && turn.stage !== 'error' && <Progress stage={turn.stage} />}
            {turn.error && <div className="notice err">Something went wrong: {turn.error}</div>}
            {turn.answer && <FormattedAnswer text={turn.answer} sources={turn.sources} onCite={setRule} />}
            {turn.stage === 'done' && (
              <div className="turn-foot">
                <span className="muted small">
                  {turn.cached ? 'Answered instantly (saved answer)'
                    : turn.timings?.total_s != null ? `Answered in ${turn.timings.total_s.toFixed(1)} s` : ''}
                </span>
                <Feedback turn={turn} onRate={(r, c) => rate(turn, r, c)} />
              </div>
            )}
          </article>
        )}
      </section>

      <aside className="side">
        {rule ? <RuleText source={rule} onBack={() => setRule(null)} />
          : turn && turn.sources.length > 0 ? (
            <div className="detail">
              {(() => {
                const cited = citedNumbers(turn.answer).map(n => turn.sources[n - 1]).filter(Boolean)
                const rest = turn.sources.filter(s => !cited.includes(s))
                return (
                  <>
                    <h3>Rules this answer is based on</h3>
                    <div className="source-list">
                      {cited.map(s => <SourceButton key={s.chunk_id} s={s} onOpen={setRule} />)}
                      {!cited.length && <span className="muted small">Waiting for the answer…</span>}
                    </div>
                    {rest.length > 0 && turn.stage === 'done' && (
                      <details className="also">
                        <summary className="muted small">Also checked ({rest.length})</summary>
                        <div className="source-list">
                          {rest.map(s => <SourceButton key={s.chunk_id} s={s} onOpen={setRule} />)}
                        </div>
                      </details>
                    )}
                  </>
                )
              })()}
            </div>
          )
          : <div className="side-empty muted">The rules behind each answer show up here. Click any § chip to read the rule.</div>}
      </aside>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState<'checklist' | 'ask'>(() => readStore('tab', 'checklist'))
  const [profiles, setProfiles] = useState<Profile[] | null>(null)
  const [question, setQuestion] = useState('')

  useEffect(() => { getObligations().then(setProfiles).catch(() => setProfiles([])) }, [])
  useEffect(() => writeStore('tab', tab), [tab])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden><Scale size={26} /></span>
          <div>
            <h1>Federal Rules Guide</h1>
            <p>Plain-English answers to U.S. federal regulations, with the exact rule cited.</p>
          </div>
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'checklist'} className={tab === 'checklist' ? 'on' : ''}
                  onClick={() => setTab('checklist')}><ListChecks size={17} /> My checklist</button>
          <button role="tab" aria-selected={tab === 'ask'} className={tab === 'ask' ? 'on' : ''}
                  onClick={() => setTab('ask')}><MessageCircleQuestion size={17} /> Ask a question</button>
        </nav>
      </header>

      <main className="main">
        {/* Both stay mounted so switching tabs keeps answers and selections. */}
        <div className="tabpanel" hidden={tab !== 'checklist'}>
          <ChecklistTab profiles={profiles} onAsk={q => { setQuestion(q); setTab('ask') }} />
        </div>
        <div className="tabpanel" hidden={tab !== 'ask'}>
          <AskTab question={question} setQuestion={setQuestion} />
        </div>
      </main>

      <footer className="foot">
        Covers transportation, labor, food &amp; drug and environmental rules from the{' '}
        <a href="https://www.ecfr.gov" target="_blank" rel="noreferrer">eCFR</a>. General information, not legal advice.
      </footer>
    </div>
  )
}
