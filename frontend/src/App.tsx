import { FormEvent, useEffect, useState } from "react";
import {
  createResearch,
  getResearch,
  type ResearchJob,
  type AIUseCase,
  type DatasetReference,
  type SourceCitation,
} from "./api";

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
    <div style={{ padding: "8px 0" }}>
      <div className="skeleton-block short" />
      <div className="skeleton-block" />
      <div className="skeleton-block" />
      <div className="skeleton-block tall" style={{ marginTop: 20 }} />
      <div className="skeleton-block short" style={{ marginTop: 20 }} />
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
          <ul style={{ margin: "6px 0 0", paddingLeft: 16, fontSize: 13, color: "#9eb6a8" }}>
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
          <span style={{ color: "#9eb6a8", fontStyle: "italic" }}>{ds.title}</span>
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
  const isUrl = c.url_or_path.startsWith("http");
  return (
    <div className="citation-row">
      <span style={{ fontFamily: '"DM Mono", monospace', fontSize: 11, color: "#b7e86a", marginRight: 8 }}>
        {c.citation_id}
      </span>
      <span style={{ flex: 1 }}>
        {isUrl ? (
          <a href={c.url_or_path} target="_blank" rel="noreferrer">
            {c.title}
          </a>
        ) : (
          c.title
        )}
      </span>
      <small>{c.source_type}</small>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [company, setCompany] = useState("Tesla");
  const [industry, setIndustry] = useState("Electric Vehicles");
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [demo, setDemo] = useState(true);
  const [error, setError] = useState("");

  // Poll every 2s while job is running or queued
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        setJob(await getResearch(job.id));
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Polling failed");
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setJob(null);
    if (demo) {
      setJob({
        id: "demo",
        status: "completed",
        company_name: company,
        industry,
        errors: [],
        report: demoReport,
      });
      return;
    }
    try {
      setJob(
        await createResearch({
          company_name: company,
          industry,
          provider: "gemini",
          retrieval_mode: "graph_rag",
          vector_db_choice: "chroma",
        })
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start research");
    }
  }

  const report = job?.report;
  const isRunning = job?.status === "queued" || job?.status === "running";

  return (
    <main>
      <header>
        <p className="eyebrow">GENESIS / AI STRATEGY PLATFORM</p>
        <h1>
          Evidence-led AI strategy,
          <br />
          built from the real world.
        </h1>
        <p className="lede">
          Turn company and market signals into a research brief with transparent workflow status.
        </p>
      </header>

      <section className="workspace">
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
          <button type="submit">{demo ? "Open demo brief" : "Run live research"}</button>
          {error && <p className="error">{error}</p>}
        </form>

        {/* ── Right: Results Panel ── */}
        <section className="panel results">
          {/* Empty state */}
          {!job && (
            <div className="empty">
              <span>01</span>
              <h2>Ready for analysis</h2>
              <p>Use demo mode for a presentation-ready preview, or connect the API for live research.</p>
            </div>
          )}

          {job && (
            <>
              {/* Status badge + live step label */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                <div className="status">
                  <span className={`dot ${job.status}`} />
                  {job.status === "partial" ? "Completed with retrieval warnings" : job.status}
                </div>
                {isRunning && (
                  <span style={{ fontSize: 12, color: "#9eb6a8" }}>
                    Agents working — this takes 60–120 s…
                  </span>
                )}
              </div>

              <p className="eyebrow">
                {job.company_name} / {job.industry}
              </p>

              {/* Per-agent warnings */}
              {job.errors.map((item) => (
                <p className="warning" key={item}>
                  {item}
                </p>
              ))}
              {job.error && <p className="error">{job.error}</p>}

              {/* Skeleton while running */}
              {isRunning && <Skeleton />}

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
                      <h3 style={{ marginTop: 32 }}>Supporting datasets</h3>
                      {report.datasets_found.map((ds) => (
                        <DatasetRow key={ds.ref + ds.keyword_matched} ds={ds} />
                      ))}
                    </>
                  )}

                  {/* Citations */}
                  {report.citations?.length > 0 && (
                    <>
                      <h3 style={{ marginTop: 32 }}>Sources</h3>
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
