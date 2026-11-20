from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, model_validator

from ai.state import CompanyInputs, ResearchReportData, SourceCitation


ResearchMode = Literal["report", "chat"]
ChatQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class ResearchRequest(CompanyInputs):
    api_key: None = None
    uploaded_files: None = None
    mode: ResearchMode = "report"

    @model_validator(mode="after")
    def validate_chat_store(self) -> Self:
        if self.mode == "chat" and self.vector_db_choice != "chroma":
            raise ValueError("Chat preparation requires the Chroma vector store.")
        return self


class ResearchJobResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "partial", "failed"]
    company_name: str
    industry: str
    mode: ResearchMode = "report"
    chat_ready: bool = False
    report: ResearchReportData | None = None
    errors: list[str] = Field(default_factory=list)
    error: str | None = None


class ChatTurn(BaseModel):
    question: ChatQuestion
    answer: str = Field(min_length=1, max_length=1200)


class ChatRequest(BaseModel):
    question: ChatQuestion
    history: list[ChatTurn] = Field(default_factory=list, max_length=3)


class ChatResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]


DocumentModel = Literal["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


class DocumentFile(BaseModel):
    name: str
    indexed_pages: int = 0
    warning: str | None = None


class DocumentSessionResponse(BaseModel):
    req_id: UUID
    status: Literal["queued", "running", "completed", "partial", "failed"] = "queued"
    model_name: DocumentModel
    files: list[DocumentFile]
    error: str | None = None


class CreatedDocumentSession(DocumentSessionResponse):
    session_token: str


class DocumentCitation(SourceCitation):
    document_id: UUID
    page: int = Field(ge=1)


class DocumentChatResponse(ChatResponse):
    citations: list[DocumentCitation]
