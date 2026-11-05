from copy import deepcopy
from typing import List, Dict, Optional, Any, TypedDict
from pydantic import BaseModel, Field, field_validator


class CompanyInputs(BaseModel):
    """Validated input parameters for market research workflow."""
    company_name: str = Field(..., min_length=2, description="Name of the company to analyze")
    industry: str = Field(..., min_length=2, description="Industry or market sector")
    provider: str = Field("openai", description="LLM Provider (gemini, groq, openai, openrouter, ollama)")
    model_name: Optional[str] = Field(None, description="Specific model override")
    # This is request-scoped only. Never persist, log, or return it to a client.
    api_key: Optional[str] = Field(None, repr=False, description="User-supplied BYO API key")
    vector_db_choice: str = Field("chroma", description="Vector database choice (chroma or pinecone)")
    retrieval_mode: str = Field("graph_rag", description="Retrieval paradigm (graph_rag or hybrid_rerank)")
    uploaded_files: Optional[List[str]] = Field(None, description="Optional paths to user uploaded documents (.pdf, .txt, .md, .docx)")

    model_config = {"protected_namespaces": ()}

    @field_validator("company_name", "industry")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"gemini", "groq", "openai", "openrouter", "ollama"}:
            raise ValueError("provider must be gemini, groq, openai, openrouter, or ollama")
        return value

    @field_validator("vector_db_choice")
    @classmethod
    def validate_vector_store(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"chroma", "pinecone"}:
            raise ValueError("vector_db_choice must be chroma or pinecone")
        return value

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"graph_rag", "hybrid_rerank"}:
            raise ValueError("retrieval_mode must be graph_rag or hybrid_rerank")
        return value

class SourceCitation(BaseModel):
    """Source reference tag for data lineage and citation symlinks."""
    citation_id: str = Field(..., description="Unique tag identifier, e.g., [SRC-1]")
    title: str = Field(..., description="Document or page title")
    source_type: str = Field(..., description="web_scrape, user_upload, or pdf")
    url_or_path: str = Field(..., description="HTTP URL or absolute local file path / symlink")
    snippet: str = Field("", description="Relevant text snippet used for inference")

class ScrapedDocument(BaseModel):
    """Clean structured document model for scraped web/PDF content or user uploads."""
    title: str = Field(..., description="Page or document title")
    url: str = Field(..., description="Source URL or file path")
    content: str = Field(..., description="Clean text content")
    category: str = Field(..., description="Content category (overview, financial, competitors, news, challenges, user_doc)")
    source_type: str = Field("web", description="web, pdf, or user_upload")
    file_path: Optional[str] = Field(None, description="Absolute local file path if uploaded or saved locally")
    citation_id: Optional[str] = Field(None, description="Assigned citation tag identifier e.g. [SRC-1]")

class KnowledgeEntity(BaseModel):
    """GraphRAG Entity Node."""
    name: str = Field(..., description="Entity name (e.g., Tesla, Model Y, Lithium)")
    type: str = Field(..., description="Entity category (Company, Competitor, Product, Financial Metric, Technology, Challenge)")
    description: str = Field(..., description="Brief description or context")

class KnowledgeRelation(BaseModel):
    """GraphRAG Relationship Edge."""
    source_entity: str = Field(..., description="Subject entity name")
    target_entity: str = Field(..., description="Object entity name")
    relationship_type: str = Field(..., description="Predicate (COMPETES_WITH, REQUIRES_TECH, FACING_CHALLENGE, SUPPLIES)")
    context: str = Field("", description="Supporting context snippet")

class GraphRAGData(BaseModel):
    """GraphRAG Knowledge Graph Representation."""
    entities: List[KnowledgeEntity] = Field(default_factory=list)
    relations: List[KnowledgeRelation] = Field(default_factory=list)

class AIUseCase(BaseModel):
    """Structured Pydantic JSON schema for a single AI Use Case."""
    title: str = Field(..., description="Short descriptive title of the AI solution")
    problem_statement: str = Field(..., description="Specific operational or business challenge addressed")
    ai_solution: str = Field(..., description="Technical AI/ML/GenAI solution description")
    expected_benefits: List[str] = Field(..., description="Quantifiable strategic and operational benefits")
    complexity: str = Field("Medium", description="Implementation complexity (Low, Medium, High)")
    roi_timeline: str = Field(..., description="Estimated timeline to achieve positive ROI (e.g. 6-12 months)")
    required_tech: List[str] = Field(..., description="Key technologies, models, or frameworks required")

class AIUseCasesResponse(BaseModel):
    """Strict list container for 5 target AI use cases."""
    use_cases: List[AIUseCase] = Field(..., description="List of 5 prioritized AI use cases")

class DatasetReference(BaseModel):
    """Structured model for Kaggle / Open Dataset match."""
    ref: str = Field(..., description="Dataset reference ID or URL")
    title: str = Field(..., description="Dataset title")
    size: str = Field("Unknown", description="File size")
    votes: int = Field(0, description="Community votes or rating")
    keyword_matched: str = Field("", description="Keyword that triggered the match")

class ResearchReportData(BaseModel):
    """Structured C-Level Executive Market Research Report."""
    executive_summary: str = Field(..., description="Concise C-level summary of findings")
    company_overview: str = Field(..., description="Background, history, business model")
    financial_performance: str = Field(..., description="Revenue, market position, growth metrics")
    products_services: str = Field(..., description="Core product lineup and services")
    competitive_landscape: str = Field(..., description="Competitor analysis and market share")
    industry_analysis: str = Field(..., description="Industry trends and regulatory factors")
    key_challenges: str = Field(..., description="Critical pain points and technology gaps")
    ai_use_cases: List[AIUseCase] = Field(default_factory=list, description="Top strategic AI recommendations")
    implementation_roadmap: str = Field(..., description="Prioritized roadmap and milestones")
    datasets_found: List[DatasetReference] = Field(default_factory=list, description="Supporting datasets found")
    citations: List[SourceCitation] = Field(default_factory=list, description="Supporting source citations with URLs and file symlinks")

# ==========================================
# LangGraph Global State Schema
# ==========================================

class ResearchState(TypedDict, total=False):
    """Global Graph State object passed between LangGraph Agent Nodes."""
    inputs: Dict[str, Any]
    scraped_documents: List[Dict[str, Any]]
    uploaded_documents: List[Dict[str, Any]]
    graph_rag_data: Optional[Dict[str, Any]]
    retrieved_context: List[str]
    research_findings: Dict[str, str]
    ai_use_cases: List[Dict[str, Any]]
    datasets: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    errors: List[str]
    current_step: str


def redact_state_for_persistence(state: ResearchState) -> ResearchState:
    """Return a copy safe to write to disk, logs, or a job database.

    API keys and local upload paths are credentials/identifiers, not research
    results. The backend must use this function before persisting a job state.
    """
    redacted = deepcopy(state)
    inputs = redacted.get("inputs")
    if inputs:
        inputs.pop("api_key", None)
        inputs.pop("uploaded_files", None)
    for document in redacted.get("uploaded_documents", []):
        document.pop("file_path", None)
        if str(document.get("url", "")).startswith("file://"):
            document["url"] = ""
    for citation in redacted.get("citations", []):
        if str(citation.get("url_or_path", "")).startswith("file://"):
            citation["url_or_path"] = ""
    return redacted
