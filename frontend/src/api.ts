export type JobStatus = "queued" | "running" | "completed" | "partial" | "failed";
export type ResearchMode = "report" | "chat";

export type ResearchJob = {
  id: string;
  status: JobStatus;
  company_name: string;
  industry: string;
  mode: ResearchMode;
  chat_ready: boolean;
  error?: string | null;
  errors: string[];
  report?: Report | null;
};

export type AIUseCase = {
  title: string;
  problem_statement: string;
  ai_solution: string;
  expected_benefits?: string[];
  complexity: string;
  roi_timeline: string;
  required_tech?: string[];
};

export type DatasetReference = {
  ref: string;
  title: string;
  size?: string;
  votes?: number;
  keyword_matched?: string;
};

export type SourceCitation = {
  citation_id: string;
  title: string;
  source_type: string;
  url_or_path: string;
  snippet?: string;
  document_id?: string | null;
  page?: number | null;
};

/** Mirrors backend ResearchReportData (ai/state.py). All fields included. */
export type Report = {
  executive_summary: string;
  company_overview: string;
  financial_performance: string;
  products_services: string;
  competitive_landscape: string;
  industry_analysis: string;
  key_challenges: string;
  implementation_roadmap: string;
  ai_use_cases: AIUseCase[];
  datasets_found: DatasetReference[];
  citations: SourceCitation[];
};

type ResearchInput = {
  company_name: string;
  industry: string;
  provider: string;
  model_name: string;
  retrieval_mode: string;
  vector_db_choice: string;
  mode: ResearchMode;
};

export type ChatAnswer = { answer: string; citations: SourceCitation[] };
export type ChatHistory = { question: string; answer: string }[];

export type DocumentSession = {
  req_id: string;
  status: JobStatus;
  model_name: string;
  files: { name: string; indexed_pages: number; warning: string | null }[];
  error: string | null;
};

type DocumentAccess = { req_id: string; session_token: string };

async function requireSuccess(response: Response, fallback: string) {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : fallback);
  }
}

function documentRequest(access: DocumentAccess, path = "", init?: RequestInit) {
  const headers = new Headers(init?.headers);
  headers.set("Authorization", `Bearer ${access.session_token}`);
  return fetch(`/api/documents/${encodeURIComponent(access.req_id)}${path}`, { ...init, headers });
}

export async function uploadPdfs(files: File[], model: string): Promise<DocumentSession & DocumentAccess> {
  const body = new FormData();
  body.set("model_name", model);
  for (const file of files) body.append("files", file);
  const response = await fetch("/api/documents", { method: "POST", body });
  await requireSuccess(response, "Could not start PDF chat. Is the API running?");
  return response.json();
}

export async function getDocumentSession(access: DocumentAccess): Promise<DocumentSession> {
  const response = await documentRequest(access);
  await requireSuccess(response, "Could not retrieve the PDF session.");
  return response.json();
}

export async function endDocumentSession(access: DocumentAccess): Promise<void> {
  const response = await documentRequest(access, "", { method: "DELETE" });
  if (response.status !== 404) await requireSuccess(response, "Could not end the PDF session.");
}

export async function askDocumentQuestion(
  access: DocumentAccess, question: string, history: ChatHistory,
): Promise<ChatAnswer> {
  const response = await documentRequest(access, "/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  await requireSuccess(response, "Could not answer this question.");
  return response.json();
}

export async function createResearch(input: ResearchInput): Promise<ResearchJob> {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new Error("Could not start research. Is the API running?");
  return response.json();
}

export async function getResearch(id: string): Promise<ResearchJob> {
  const response = await fetch(`/api/research/${id}`);
  if (!response.ok) throw new Error("Could not retrieve the research job.");
  return response.json();
}

export async function askResearchQuestion(
  jobId: string,
  question: string,
  history: ChatHistory,
): Promise<ChatAnswer> {
  const response = await fetch(`/api/research/${encodeURIComponent(jobId)}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  await requireSuccess(response, "Could not answer this question.");
  return response.json();
}
