import asyncio
import logging
import secrets
from dataclasses import dataclass
from io import BytesIO
from uuid import UUID, uuid4
from weakref import finalize

import pdfplumber
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.embeddings.vector_factory import VectorStoreFactory
from ai.llm import EmbeddingFactory
from ai.settings import get_settings
from backend.schemas import DocumentFile, DocumentModel, DocumentSessionResponse

logger = logging.getLogger(__name__)

MAX_PDFS = 5
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_PAGES_PER_PDF = 100
_MAX_SESSIONS = 16
_SESSION_SECONDS = 60 * 60
_INDEX_TIMEOUT_SECONDS = 180


@dataclass
class IndexedDocuments:
    index: VectorStore


@dataclass
class DocumentSession:
    response: DocumentSessionResponse
    token: str
    expiry: asyncio.TimerHandle
    indexed: IndexedDocuments | None = None


class DocumentSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[UUID, DocumentSession] = {}

    def create(self, model: DocumentModel, filenames: list[str]) -> DocumentSession | None:
        if len(self._sessions) >= _MAX_SESSIONS:
            return None
        req_id = uuid4()
        expiry = asyncio.get_running_loop().call_later(_SESSION_SECONDS, self.remove, req_id)
        response = DocumentSessionResponse(
            req_id=req_id, model_name=model, files=[DocumentFile(name=name) for name in filenames],
        )
        session = DocumentSession(response, secrets.token_urlsafe(32), expiry)
        self._sessions[req_id] = session
        return session

    def get(self, req_id: UUID, authorization: str) -> DocumentSession | None:
        session = self._sessions.get(req_id)
        scheme, _, token = authorization.partition(" ")
        if session and scheme.lower() == "bearer" and secrets.compare_digest(token.encode(), session.token.encode()):
            return session
        return None

    def remove(self, req_id: UUID) -> None:
        session = self._sessions.pop(req_id, None)
        if session is not None:
            session.expiry.cancel()
            session.indexed = None

    def clear(self) -> None:
        for req_id in list(self._sessions):
            self.remove(req_id)

    async def index_pdfs(self, req_id: UUID, uploads: list[tuple[str, bytes]]) -> None:
        session = self._sessions.get(req_id)
        if session is None:
            return
        response = session.response
        response.status = "running"
        try:
            indexed, files = await asyncio.wait_for(
                asyncio.to_thread(_index_pdfs, req_id, uploads), timeout=_INDEX_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("PDF indexing failed for request %s (%s)", req_id, type(exc).__name__)
            response.status = "failed"
            response.error = "PDF indexing timed out." if isinstance(exc, TimeoutError) else (
                "Could not index these PDFs. The embedding provider may be unavailable or over its quota."
            )
            return
        if self._sessions.get(req_id) is not session:
            return
        response.files = files
        session.indexed = indexed
        if indexed is None:
            response.status = "failed"
            response.error = "None of the selected PDFs contained searchable text."
        else:
            response.status = "partial" if any(file.warning for file in files) else "completed"


def _read_pdf(
    data: bytes, file: DocumentFile, document_id: str, first_source: int,
) -> list[Document]:
    if b"%PDF-" not in data[:1024]:
        file.warning = "This file is not a valid PDF."
        return []
    settings = get_settings()
    remaining = settings.max_document_chars
    pages: list[Document] = []
    try:
        with pdfplumber.open(BytesIO(data), unicode_norm="NFC") as pdf:
            file_has_more_pages = len(pdf.pages) > MAX_PAGES_PER_PDF
            for page in pdf.pages[:MAX_PAGES_PER_PDF]:
                try:
                    content = (page.extract_text() or "").strip()
                    if not content:
                        continue
                    pages.append(Document(page_content=content[:remaining], metadata={
                        "citation_id": f"[SRC-{first_source + len(pages)}]",
                        "document_id": document_id,
                        "title": file.name,
                        "source_type": "pdf",
                        "page": page.page_number,
                    }))
                    truncated_page = len(content) > remaining
                    remaining = max(0, remaining - len(content))
                    if remaining == 0:
                        if truncated_page or page.page_number < len(pdf.pages):
                            file.warning = "Only part of this PDF was indexed because it reached the text limit."
                        break
                finally:
                    page.close()
            if file_has_more_pages and file.warning is None:
                file.warning = f"Only the first {MAX_PAGES_PER_PDF} pages were checked for text."
    except Exception:
        file.warning = "Could not read this PDF. Check whether it is damaged or password protected."
        return []
    file.indexed_pages = len(pages)
    if not pages:
        file.warning = "No selectable text was found. Scanned PDFs need OCR before upload."
    return pages


def _index_pdfs(
    req_id: UUID, uploads: list[tuple[str, bytes]],
) -> tuple[IndexedDocuments | None, list[DocumentFile]]:
    pages: list[Document] = []
    files = []
    for name, data in uploads:
        file = DocumentFile(name=name)
        pages.extend(_read_pdf(data, file, str(uuid4()), len(pages) + 1))
        files.append(file)
    if not pages:
        return None, files
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    embeddings = EmbeddingFactory.get_embeddings(provider="gemini")
    index = VectorStoreFactory.create_job_index(chunks, embeddings, str(req_id), namespace="documents")
    indexed = IndexedDocuments(index)
    cleanup = finalize(indexed, index.delete_collection)
    cleanup.atexit = False
    return indexed, files


document_sessions = DocumentSessionStore()
