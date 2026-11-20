import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ai.settings import get_settings
from backend.jobs import job_store
from backend.schemas import ChatRequest, ChatResponse, ResearchJobResponse, ResearchRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="GENESIS API", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    # Read from CORS_ORIGINS env var — defaults to localhost:5173 for dev.
    # Set CORS_ORIGINS=https://your-domain.com in staging/production .env.
    allow_origins=_settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                engine.answer_from_retriever, request.question, retriever,
                history=[turn.model_dump() for turn in request.history],
            ),
            timeout=60,
        )
        reply = ChatResponse.model_validate(result)
        for citation in reply.citations:
            if not citation.url_or_path.startswith(("https://", "http://")):
                citation.url_or_path = ""
        return reply
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Chat timed out. Try again.") from None
    except Exception as exc:
        if getattr(exc, "status_code", getattr(exc, "code", None)) == 429:
            raise HTTPException(status_code=429, detail="The provider's chat quota was reached. Try again later.") from None
        logging.getLogger(__name__).warning("Chat failed for job %s (%s)", job_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="Chat is temporarily unavailable. Try again.") from None
