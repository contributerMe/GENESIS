"""Small in-memory job manager for the demo service.

Replace this with a durable queue (for example Redis/Celery) before running
multiple API instances or accepting long-lived production jobs.
"""
import asyncio
import logging
from collections import OrderedDict
from typing import Optional
from uuid import uuid4

from backend.schemas import ResearchJobResponse, ResearchRequest
from ai.graph import get_research_graph
from ai.state import ResearchState, redact_state_for_persistence

logger = logging.getLogger(__name__)

# Maximum number of jobs held in memory at once.
# Oldest job is evicted when this limit is exceeded.
_MAX_JOBS = 200

# Hard timeout (seconds) for one research workflow run.
# Prevents a hung scraper or LLM call from leaking a thread forever.
_JOB_TIMEOUT_SECONDS = 300


class ResearchJobStore:
    def __init__(self) -> None:
        # OrderedDict preserves insertion order so we can evict the oldest entry.
        self._jobs: OrderedDict[str, ResearchJobResponse] = OrderedDict()

    def create(self, request: ResearchRequest) -> ResearchJobResponse:
        job = ResearchJobResponse(
            id=str(uuid4()), status="queued", company_name=request.company_name, industry=request.industry
        )
        self._jobs[job.id] = job
        # Evict oldest job if we've exceeded the memory cap
        if len(self._jobs) > _MAX_JOBS:
            evicted_id, _ = self._jobs.popitem(last=False)
            logger.info("Job store at capacity (%d). Evicted oldest job: %s", _MAX_JOBS, evicted_id)
        return job

    def get(self, job_id: str) -> Optional[ResearchJobResponse]:
        return self._jobs.get(job_id)

    async def execute(self, job_id: str, request: ResearchRequest) -> None:
        job = self._jobs[job_id]
        job.status = "running"
        try:
            # asyncio.to_thread runs the synchronous LangGraph workflow in a
            # threadpool without blocking the event loop.
            # asyncio.wait_for enforces a hard ceiling — if the workflow hangs
            # (e.g. a scraper stalls), the job is marked failed rather than
            # leaking a thread indefinitely.
            final_state: ResearchState = await asyncio.wait_for(
                asyncio.to_thread(self._run_workflow, request),
                timeout=_JOB_TIMEOUT_SECONDS,
            )
            safe_state = redact_state_for_persistence(final_state)
            job.report = safe_state.get("final_report")
            job.errors = safe_state.get("errors", [])
            job.status = "partial" if job.errors else "completed"
        except asyncio.TimeoutError:
            logger.error("Research job %s timed out after %ds", job_id, _JOB_TIMEOUT_SECONDS)
            job.status = "failed"
            job.error = f"Research timed out after {_JOB_TIMEOUT_SECONDS}s. The workflow took too long to complete."
        except Exception:
            logger.exception("Research job %s failed", job_id)
            job.status = "failed"
            job.error = "The research workflow could not be completed. Check server logs using this job ID."

    @staticmethod
    def _run_workflow(request: ResearchRequest) -> ResearchState:
        initial_state: ResearchState = {
            "inputs": request.model_dump(),
            "scraped_documents": [],
            "uploaded_documents": [],
            "research_findings": {},
            "ai_use_cases": [],
            "datasets": [],
            "citations": [],
            "errors": [],
        }
        return get_research_graph().invoke(initial_state)


job_store = ResearchJobStore()
