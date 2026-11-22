import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, TypeVar
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from ai.rag.chat_engine import DynamicRAGChatEngine
from ai.settings import get_settings
from backend.documents import MAX_PDFS, MAX_UPLOAD_BYTES, DocumentSession, document_sessions
from backend.jobs import job_store
from backend.schemas import (
    ChatRequest, ChatResponse, CreatedDocumentSession, DocumentChatResponse, DocumentModel, DocumentSessionResponse,
    ResearchJobResponse, ResearchRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        document_sessions.clear()


app = FastAPI(title="GENESIS API", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
_document_model = TypeAdapter(DocumentModel)
ChatReply = TypeVar("ChatReply", bound=ChatResponse)
app.add_middleware(
    CORSMiddleware,
    # Read from CORS_ORIGINS env var — defaults to localhost:5173 for dev.
    # Set CORS_ORIGINS=https://your-domain.com in staging/production .env.
    allow_origins=_settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "genesis-api"}


@app.post("/api/research", response_model=ResearchJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(request: ResearchRequest) -> ResearchJobResponse:
    """Queue a research job. Provider keys remain server-side in environment variables."""
    job = job_store.create(request)
    asyncio.create_task(job_store.execute(job.id, request))
    return job


@app.get("/api/research/{job_id}", response_model=ResearchJobResponse)
async def get_research(job_id: str) -> ResearchJobResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research job not found")
    return job


@app.post("/api/research/{job_id}/chat", response_model=ChatResponse)
async def answer_research_question(job_id: str, request: ChatRequest) -> ChatResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Research job not found. Start a new session.")
    chat = job_store.get_chat(job_id)
    if chat is None:
        detail = "Sources are still being prepared." if job.status in {"queued", "running"} else "This job has no searchable sources."
        raise HTTPException(status_code=409, detail=detail)

    engine, retriever = chat
    return await _answer_chat(job_id, request, partial(engine.answer_from_retriever, retriever=retriever), ChatResponse)


class UploadTooLarge(MultiPartException):
    pass


@app.post("/api/documents", response_model=CreatedDocumentSession, status_code=status.HTTP_202_ACCEPTED)
async def start_document_session(request: Request, background_tasks: BackgroundTasks) -> CreatedDocumentSession:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(status_code=415, detail="Send the PDFs as a multipart upload.")

    async def bounded_body():
        received = 0
        try:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES + 64 * 1024:
                    raise UploadTooLarge("Upload exceeds the 40 MB total limit.")
                yield chunk
        except ClientDisconnect:
            raise MultiPartException("The upload was interrupted.") from None

    try:
        form = await MultiPartParser(
            request.headers, bounded_body(), max_files=MAX_PDFS, max_fields=1, max_part_size=1024,
        ).parse()
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=exc.message) from None
    except MultiPartException:
        raise HTTPException(status_code=400, detail=f"Upload between 1 and {MAX_PDFS} PDFs.") from None

    uploads = []
    total_size = 0
    try:
        try:
            model = _document_model.validate_python(form.get("model_name"))
        except ValidationError:
            raise HTTPException(status_code=422, detail="Choose one of the supported Gemini chat models.") from None
        files = form.getlist("files")
        if not files or len(files) + 1 != len(form.multi_items()):
            raise HTTPException(status_code=400, detail=f"Upload between 1 and {MAX_PDFS} PDFs.")
        for file in files:
            if not isinstance(file, UploadFile):
                raise HTTPException(status_code=400, detail="Send PDFs in the files field.")
            filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
            if not filename.lower().endswith(".pdf") or len(filename) > 255:
                raise HTTPException(status_code=415, detail="Only PDF files with valid filenames are supported.")
            data = await file.read(_settings.max_upload_size_bytes + 1)
            total_size += len(data)
            if len(data) > _settings.max_upload_size_bytes or total_size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"The limit is {_settings.max_upload_size_mb} MB per PDF and 40 MB total.",
                )
            uploads.append((filename, data))
    finally:
        await form.close()

    session = document_sessions.create(model, [name for name, _ in uploads])
    if session is None:
        raise HTTPException(status_code=429, detail="PDF sessions are at capacity. End an existing session or try again later.")
    background_tasks.add_task(document_sessions.index_pdfs, session.response.req_id, uploads)
    return CreatedDocumentSession(**session.response.model_dump(), session_token=session.token)


def _get_document_session(req_id: UUID, authorization: str) -> DocumentSession:
    session = document_sessions.get(req_id, authorization)
    if session is None:
        raise HTTPException(status_code=404, detail="PDF session not found or expired. Start a new session.")
    return session


@app.get("/api/documents/{req_id}", response_model=DocumentSessionResponse)
async def get_document_session(req_id: UUID, authorization: str = Header(default="")) -> DocumentSessionResponse:
    return _get_document_session(req_id, authorization).response


@app.delete("/api/documents/{req_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_document_session(req_id: UUID, authorization: str = Header(default="")) -> Response:
    _get_document_session(req_id, authorization)
    document_sessions.remove(req_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/documents/{req_id}/chat", response_model=DocumentChatResponse)
async def answer_document_question(
    req_id: UUID, request: ChatRequest, authorization: str = Header(default=""),
) -> DocumentChatResponse:
    session = _get_document_session(req_id, authorization)
    indexed = session.indexed
    if indexed is None:
        detail = (
            session.response.error or "This session has no searchable PDFs."
            if session.response.status == "failed" else "Your PDFs are still being prepared."
        )
        raise HTTPException(status_code=409, detail=detail)

    def search(query: str, limit: int):
        return indexed.index.similarity_search(query, k=limit)

    engine = DynamicRAGChatEngine(provider="gemini", model_name=session.response.model_name)
    return await _answer_chat(
        str(req_id), request, partial(engine.answer_from_search, search=search, private=True), DocumentChatResponse,
    )


async def _answer_chat(
    request_id: str, request: ChatRequest, answer: Callable[..., dict[str, Any]], response_type: type[ChatReply],
) -> ChatReply:
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                answer, request.question,
                history=[turn.model_dump() for turn in request.history],
            ),
            timeout=60,
        )
        reply = response_type.model_validate(result)
        for citation in reply.citations:
            if not citation.url_or_path.startswith(("https://", "http://")):
                citation.url_or_path = ""
        return reply
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Chat timed out. Try again.") from None
    except Exception as exc:
        if getattr(exc, "status_code", getattr(exc, "code", None)) == 429:
            raise HTTPException(status_code=429, detail="The provider's chat quota was reached. Try again later.") from None
        logging.getLogger(__name__).warning("Chat failed for request %s (%s)", request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="Chat is temporarily unavailable. Try again.") from None
