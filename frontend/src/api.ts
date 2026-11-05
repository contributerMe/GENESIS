export type JobStatus = "queued" | "running" | "completed" | "partial" | "failed";

export type ResearchJob = {
  id: string;
  status: JobStatus;
  company_name: string;
  industry: string;
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
  retrieval_mode: string;
  vector_db_choice: string;
};

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
