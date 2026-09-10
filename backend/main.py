import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ai.settings import get_settings
from backend.jobs import job_store
from backend.schemas import ResearchJobResponse, ResearchRequest

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
