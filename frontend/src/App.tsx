import { FormEvent, useEffect, useId, useState } from "react";
import skyline from "./assets/genesis-skyline.webp";
import {
  askDocumentQuestion,
  askResearchQuestion,
  createResearch,
  endDocumentSession,
  getDocumentSession,
  getResearch,
  uploadPdfs,
  type ChatAnswer,
  type ChatHistory,
  type DocumentSession,
  type ResearchMode,
  type ResearchJob,
  type AIUseCase,
  type DatasetReference,
  type SourceCitation,
} from "./api";

const researchModels = [
  { id: "gemini-3.1-flash-lite", label: "Gemini 3.1 Flash-Lite" },
  { id: "gemini-3.5-flash-lite", label: "Gemini 3.5 Flash-Lite" },
] as const;

type GenerationModel = (typeof researchModels)[number];

function ModelPicker({ model, onChange, disabled }: {
  model: GenerationModel;
  onChange: (model: GenerationModel) => void;
  disabled: boolean;
}) {
  const group = useId();
  return (
    <fieldset className="model-picker" disabled={disabled}>
      <legend>Generation model</legend>
      {researchModels.map((option) => (
        <label key={option.id} className="model-option">
          <input type="radio" name={group} value={option.id} checked={model.id === option.id}
            onChange={() => onChange(option)} />
          {option.label}
        </label>
      ))}
    </fieldset>
  );
}

// ─── Demo data mirrors the full backend ResearchReportData shape ──────────────
const demoReport = {
  executive_summary:
    "GENESIS turns verified public sources into a concise AI strategy brief. Start a live run once provider quotas are available.",
  company_overview:
    "Demo mode is intentionally local and does not call model or scraping providers.",
  financial_performance:
    "Demo: revenue and market-position metrics would appear here, sourced from public filings and news.",
  products_services:
    "Demo: core product lineup and services overview would be populated from scraped company pages.",
  competitive_landscape:
    "Demo: competitor analysis comparing product positioning, pricing, and market share.",
  industry_analysis:
    "Demo: industry trends, regulatory factors, and macro-economic tailwinds/headwinds.",
  key_challenges:
    "Provider quota and source availability are surfaced as explicit workflow status — not hidden behind a successful-looking report.",
  implementation_roadmap:
    "Validate sources → retrieve evidence → prioritize use cases → operationalize with measured guardrails.",
  ai_use_cases: [
    {
      title: "Research intelligence copilot",
      problem_statement: "Research is fragmented across sources.",
      ai_solution: "Citation-backed RAG workspace.",
      expected_benefits: ["40% faster research cycles", "Traceable citations"],
      complexity: "Medium",
      roi_timeline: "3–6 months",
      required_tech: ["LangGraph", "ChromaDB", "OpenAI"],
    },
    {
      title: "Customer signal triage",
      problem_statement: "Teams miss emerging issues.",
      ai_solution: "Classify and route feedback with human review.",
      expected_benefits: ["50% reduction in missed signals"],
      complexity: "Medium",
      roi_timeline: "3–6 months",
      required_tech: ["LLM classifier", "Slack integration"],
    },
  ],
  datasets_found: [
    {
      ref: "unverified://demo-dataset",
      title: "Industry benchmark dataset (demo)",
      size: "45 MB",
      votes: 128,
      keyword_matched: "industry",
    },
  ],
  citations: [
    {
      citation_id: "[SRC-1]",
      title: "Demo source document",
      source_type: "web",
      url_or_path: "https://example.com",
      snippet: "This is a demo citation snippet.",
    },
  ],
};

// Map backend current_step values to human-readable progress labels
const STEP_LABELS: Record<string, string> = {
  researcher_completed: "Web research complete",
  researcher_failed: "Web research failed — continuing",
  knowledge_completed: "Knowledge graph built",
  knowledge_skipped: "Knowledge graph skipped (no documents)",
  strategist_completed: "AI use cases generated",
  strategist_fallback: "AI use cases (fallback)",
  dataset_completed: "Datasets discovered",
  dataset_failed: "Dataset search failed — continuing",
  writer_completed: "Report assembled",
  writer_fallback: "Report assembled (fallback)",
};

// ─── Skeleton loader shown while job is running ───────────────────────────────
function Skeleton() {
  return (
    <div className="skeleton" aria-hidden="true">
      <div className="skeleton-block short" />
      <div className="skeleton-block" />
      <div className="skeleton-block" />
      <div className="skeleton-block tall" />
      <div className="skeleton-block short" />
      <div className="skeleton-block" />
      <div className="skeleton-block xshort" />
    </div>
  );
}

// ─── Individual section blocks in the report ──────────────────────────────────
function ReportSection({ title, content }: { title: string; content: string }) {
  return (
    <div>
      <h3>{title}</h3>
      <p>{content}</p>
    </div>
  );
}

function UseCaseCard({ item, index }: { item: AIUseCase; index: number }) {
  return (
    <div className="usecase">
      <span>0{index + 1}</span>
      <div>
        <h4>{item.title}</h4>
        <p>{item.ai_solution}</p>
        {item.expected_benefits && (
          <ul className="usecase-benefits">
            {item.expected_benefits.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        )}
      </div>
      <small>
        {item.complexity} · {item.roi_timeline}
      </small>
    </div>
  );
}

function DatasetRow({ ds }: { ds: DatasetReference }) {
  const isReal = !ds.ref.startsWith("unverified://");
  return (
    <div className="dataset-row">
      <span>
        {isReal ? (
          <a href={`https://www.kaggle.com/datasets/${ds.ref}`} target="_blank" rel="noreferrer">
            {ds.title}
          </a>
        ) : (
          <span className="unverified">{ds.title}</span>
        )}
      </span>
      <small>
        {ds.keyword_matched && <>{ds.keyword_matched} · </>}
        {ds.size} · ★ {ds.votes}
      </small>
    </div>
  );
}

function CitationRow({ c }: { c: SourceCitation }) {
  const isUrl = /^https?:\/\//i.test(c.url_or_path);
  return (
    <div className="citation-row">
      <span className="source-tag">{c.citation_id}</span>
      <span className="source-title">
        {isUrl ? (
          <a href={c.url_or_path} target="_blank" rel="noreferrer">
            {c.title}
          </a>
        ) : (
          c.title
        )}
      </span>
      <small>{c.page ? `PDF page ${c.page}` : c.source_type}</small>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function ChatPanel({ askQuestion, title, placeholder }: {
  askQuestion: (question: string, history: ChatHistory) => Promise<ChatAnswer>;
  title: string;
  placeholder: string;
}) {
  const headingId = useId();
  const questionId = useId();
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<(ChatAnswer & { question: string })[]>([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || sending) return;
    setSending(true);
    setError("");
    try {
      const history = turns.slice(-3).map((turn) => ({
        question: turn.question,
        answer: turn.answer.slice(0, 1200),
      }));
      const reply = await askQuestion(text, history);
      setTurns((previous) => [...previous, { question: text, ...reply }]);
      setQuestion("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not answer this question.");
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="research-chat" aria-labelledby={headingId}>
      <div className="chat-heading">
        <h2 id={headingId}>{title}</h2>
        {turns.length > 0 && (
          <button type="button" className="secondary chat-clear" disabled={sending} onClick={() => {
            setTurns([]);
            setError("");
          }}>Clear chat</button>
        )}
      </div>
      {turns.length > 0 && (
        <div className="chat-transcript" role="log" aria-label={title}>
          {turns.map((turn, index) => (
            <div className="chat-turn" key={index}>
              <p className="chat-question">{turn.question}</p>
              <p className="chat-answer">{turn.answer}</p>
              {turn.citations.map((citation) => <CitationRow key={citation.citation_id} c={citation} />)}
            </div>
          ))}
        </div>
      )}
      <form onSubmit={ask} className="chat-composer">
        <label htmlFor={questionId}>Question</label>
        <div>
          <input id={questionId} placeholder={placeholder} value={question}
            onChange={(event) => setQuestion(event.target.value)} required maxLength={1000} readOnly={sending} />
          <button type="submit" disabled={sending || !question.trim()}>{sending ? "Searching…" : "Ask"}</button>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
    </section>
  );
}

function DocumentWorkspace({ hidden }: { hidden: boolean }) {
  const [model, setModel] = useState<GenerationModel>(researchModels[0]);
  const [files, setFiles] = useState<File[]>([]);
  const [session, setSession] = useState<DocumentSession | null>(null);
  const [sessionToken, setSessionToken] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [pollPaused, setPollPaused] = useState(false);
  const reqId = session?.req_id;
  const indexing = session?.status === "queued" || session?.status === "running";

  useEffect(() => {
    if (!reqId || !indexing || pollPaused) return;
    let active = true;
    let failures = 0;
    let timer: number;
    const poll = async () => {
      try {
        const next = await getDocumentSession({ req_id: reqId, session_token: sessionToken });
        if (!active) return;
        failures = 0;
        setError("");
        setSession((previous) => previous?.req_id === reqId ? next : previous);
        if (next.status !== "queued" && next.status !== "running") return;
      } catch (reason) {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Could not check indexing status.");
        if (++failures >= 3) {
          setPollPaused(true);
          return;
        }
      }
      timer = window.setTimeout(poll, 1200);
    };
    timer = window.setTimeout(poll, 1200);
    return () => { active = false; window.clearTimeout(timer); };
  }, [indexing, pollPaused, reqId, sessionToken]);

  async function prepare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending || session || !files.length) return;
    const form = event.currentTarget;
    setPending(true);
    setError("");
    try {
      const { session_token, ...started } = await uploadPdfs(files, model.id);
      setSessionToken(session_token);
      setSession(started);
      setFiles([]);
      form.reset();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not upload these PDFs.");
    } finally {
      setPending(false);
    }
  }

  async function endSession() {
    if (!reqId || pending) return;
    setPending(true);
    setError("");
    try {
      await endDocumentSession({ req_id: reqId, session_token: sessionToken });
      setSession(null);
      setSessionToken("");
      setPollPaused(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not end the PDF session.");
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="workspace" aria-label="PDF workspace" hidden={hidden}>
      <form className="panel" onSubmit={prepare}>
        <h2>Chat with PDFs</h2>
        <label>
          PDF files
          <input type="file" accept=".pdf,application/pdf" multiple required disabled={pending || Boolean(session)}
            onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
        </label>
        <p className="upload-note">Choose up to five PDFs with selectable text. Text is sent to Google to index the PDFs and answer questions. Sessions expire after one hour.</p>
        <ModelPicker model={model} onChange={setModel} disabled={pending || Boolean(session)} />
        {session ? (
          <button className="secondary" type="button" disabled={pending} onClick={endSession}>End PDF session</button>
        ) : (
          <button type="submit" disabled={pending || !files.length || files.length > 5}>
            {pending ? "Uploading…" : "Prepare PDFs for chat"}
          </button>
        )}
        {files.length > 5 && <p className="error" role="alert">Choose no more than five PDFs.</p>}
        {error && <p className="error" role="alert">{error}</p>}
      </form>
      <section className="panel results">
        {session ? (
          <>
            <h2>Session documents</h2>
            <div aria-live="polite">
              <p>{pollPaused ? "Indexing status is unavailable." : indexing ? "Indexing PDF pages…" : session.error || "PDF indexing finished."}</p>
              <ul className="document-list">
                {session.files.map((file, index) => (
                  <li key={index}>
                    <strong>{file.name}</strong>
                    {!indexing && <small>{file.indexed_pages} pages indexed</small>}
                    {file.warning && <p className="warning">{file.warning}</p>}
                  </li>
                ))}
              </ul>
            </div>
            {pollPaused && (
              <button className="secondary" type="button" disabled={pending} onClick={() => setPollPaused(false)}>
                Check indexing status
              </button>
            )}
            {(session.status === "completed" || session.status === "partial") && (
              <ChatPanel key={session.req_id} title="Ask about your PDFs" placeholder="What does the document say about…?"
                askQuestion={(question, history) => askDocumentQuestion(
                  { req_id: session.req_id, session_token: sessionToken }, question, history,
                )} />
            )}
          </>
        ) : (
          <div className="empty">
            <div className="empty-mark" aria-hidden="true"><span /></div>
            <h2>Ready for your documents</h2>
            <p>Upload PDFs to ask questions and see which pages support the answers.</p>
          </div>
        )}
      </section>
    </section>
  );
}

function App() {
  const [workspace, setWorkspace] = useState<"research" | "documents">("research");
  const [company, setCompany] = useState("Tesla");
  const [industry, setIndustry] = useState("Electric Vehicles");
  const [model, setModel] = useState<GenerationModel>(researchModels[0]);
  const [activeModel, setActiveModel] = useState("");
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [demo, setDemo] = useState(true);
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const jobId = job?.id;
  const isRunning = job?.status === "queued" || job?.status === "running";

  // Poll every 2s while job is running or queued
  useEffect(() => {
    if (!jobId || !isRunning) return;
    let active = true;
    let failures = 0;
    let timer: number;
    const poll = async () => {
      try {
        const nextJob = await getResearch(jobId);
        if (!active) return;
        failures = 0;
        setError("");
        setJob(nextJob);
        if (nextJob.status === "queued" || nextJob.status === "running") {
          timer = window.setTimeout(poll, 2000);
        }
      } catch (reason) {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "Polling failed");
        if (++failures < 3) {
          timer = window.setTimeout(poll, 2000);
        } else {
          setJob((previous) => previous?.id === jobId ? { ...previous, status: "failed" } : previous);
        }
      }
    };
    timer = window.setTimeout(poll, 2000);
    return () => { active = false; window.clearTimeout(timer); };
  }, [jobId, isRunning]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (starting) return;
    const submitter = (event.nativeEvent as SubmitEvent).submitter;
    const mode: ResearchMode = submitter instanceof HTMLButtonElement && submitter.value === "chat" ? "chat" : "report";
    setError("");
    setJob(null);
    setActiveModel("");
    if (demo) {
      setJob({
        id: "demo",
        status: "completed",
        company_name: company,
        industry,
        mode: "report",
        chat_ready: false,
        errors: [],
        report: demoReport,
      });
      return;
    }
    setStarting(true);
    try {
      const startedJob = await createResearch({
        company_name: company,
        industry,
        provider: "gemini",
        model_name: model.id,
        retrieval_mode: "graph_rag",
        vector_db_choice: "chroma",
        mode,
      });
      setActiveModel(model.label);
      setJob(startedJob);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start research");
    } finally {
      setStarting(false);
    }
  }

  const report = job?.report;

  return (
    <main>
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">G</span>
          <span>GENESIS</span>
        </div>
        <a className="workspace-link" href="#research">Research workspace <span aria-hidden="true">↓</span></a>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <h1 id="hero-title"><span className="hero-line">Evidence-led</span><br />AI strategy<span className="hero-accent">.</span></h1>
          <p className="lede">Turn company and market signals into a research brief, then ask questions about the sources.</p>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <img src={skyline} alt="" fetchPriority="high" width="1400" height="788" />
        </div>
      </section>

      <div className="workspace-switch" id="research" role="group" aria-label="Workspace">
        <button type="button" className="secondary" aria-pressed={workspace === "research"}
          onClick={() => setWorkspace("research")}>Company research</button>
        <button type="button" className="secondary" aria-pressed={workspace === "documents"}
          onClick={() => setWorkspace("documents")}>Chat with PDFs</button>
      </div>

      <DocumentWorkspace hidden={workspace !== "documents"} />

      <section className="workspace" hidden={workspace !== "research"} aria-label="Company research workspace">
        {/* ── Left: Input Form ── */}
        <form onSubmit={submit} className="panel">
          <h2>Start research</h2>
          <label>
            Company
            <input value={company} onChange={(e) => setCompany(e.target.value)} required minLength={2} />
          </label>
          <label>
            Industry
            <input value={industry} onChange={(e) => setIndustry(e.target.value)} required minLength={2} />
          </label>
          <label className="toggle">
            <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} />
            Demo mode <small>No provider calls</small>
          </label>
          {!demo && <ModelPicker model={model} onChange={setModel} disabled={starting} />}
          <button type="submit" disabled={starting}>{demo ? "Open demo brief" : "Run live research"}</button>
          {!demo && <button type="submit" value="chat" className="secondary prepare-chat" disabled={starting}>Prepare sources for chat</button>}
          {error && <p className="error">{error}</p>}
        </form>

        {/* ── Right: Results Panel ── */}
        <section className="panel results">
          {/* Empty state */}
          {!job && (
            <div className="empty">
              <div className="empty-mark" aria-hidden="true"><span /></div>
              <h2>Ready for analysis</h2>
              <p>Open a demo brief or start a live run to see findings and sources here.</p>
            </div>
          )}

          {job && (
            <>
              {/* Status badge + live step label */}
              <div className="status-row" aria-live="polite">
                <div className={`status ${job.status}`}>
                  <span className={`dot ${job.status}`} />
                  {job.status === "partial" ? "Completed with retrieval warnings" : job.status}
                </div>
                {isRunning && (
                  <span className="status-detail">
                    {job.mode === "chat" ? "Preparing source documents…" : "Research is running…"}
                  </span>
                )}
              </div>

              <p className="eyebrow">
                {job.company_name} / {job.industry}
              </p>
              {activeModel && <p className="job-model">Selected model: {activeModel}</p>}

              {/* Per-agent warnings */}
              {job.errors.map((item) => (
                <p className="warning" key={item}>
                  {item}
                </p>
              ))}
              {job.error && <p className="error">{job.error}</p>}

              {/* Skeleton while running */}
              {isRunning && <Skeleton />}

              {job.chat_ready && (
                <ChatPanel key={job.id} title="Ask about these sources"
                  placeholder="What challenges did the sources identify?"
                  askQuestion={(question, history) => askResearchQuestion(job.id, question, history)} />
              )}

              {/* Full report */}
              {report && (
                <article>
                  <h2>Executive brief</h2>
                  <p>{report.executive_summary}</p>

                  {/* Four-up narrative grid */}
                  <div className="grid">
                    <ReportSection title="Company overview" content={report.company_overview} />
                    <ReportSection title="Financial performance" content={report.financial_performance} />
                    <ReportSection title="Products & services" content={report.products_services} />
                    <ReportSection title="Competitive landscape" content={report.competitive_landscape} />
                  </div>

                  <div className="grid">
                    <ReportSection title="Industry analysis" content={report.industry_analysis} />
                    <ReportSection title="Key challenges" content={report.key_challenges} />
                  </div>

                  <ReportSection title="Implementation roadmap" content={report.implementation_roadmap} />

                  {/* AI Use Cases */}
                  <h3>Prioritized AI use cases</h3>
                  {report.ai_use_cases.map((item, index) => (
                    <UseCaseCard key={item.title} item={item} index={index} />
                  ))}

                  {/* Datasets */}
                  {report.datasets_found?.length > 0 && (
                    <>
                      <h3 className="report-subheading">Supporting datasets</h3>
                      {report.datasets_found.map((ds) => (
                        <DatasetRow key={ds.ref + ds.keyword_matched} ds={ds} />
                      ))}
                    </>
                  )}

                  {/* Citations */}
                  {report.citations?.length > 0 && (
                    <>
                      <h3 className="report-subheading">Sources</h3>
                      {report.citations.map((c) => (
                        <CitationRow key={c.citation_id} c={c} />
                      ))}
                    </>
                  )}
                </article>
              )}
            </>
          )}
        </section>
      </section>
    </main>
  );
}

export default App;
