import json
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from ai.state import AIUseCasesResponse, ResearchReportData
from backend.main import app


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.index_builds = 0

    def embed_documents(self, texts):
        self.index_builds += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def source(name):
    return {"title": name, "url": f"https://example.com/{name}", "content": f"Evidence from {name}.",
            "source_type": "web"}


class ResearchChatTests(unittest.TestCase):
    def start(self, client, mode="chat"):
        response = client.post("/api/research", json={
            "company_name": "Example Company", "industry": "Manufacturing", "provider": "gemini",
            "retrieval_mode": "hybrid_rerank", "mode": mode,
        })
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        for _ in range(200):
            job = client.get(f"/api/research/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                return job
            time.sleep(0.02)
        self.fail("Research job did not finish")

    def test_chat_only_keeps_same_company_jobs_isolated_and_reuses_embeddings(self):
        embeddings = CountingEmbeddings()
        with (
            patch("ai.agents.researcher.ModernWebScraper.search_and_scrape", side_effect=[[source("first")], [source("second")]]),
            patch("ai.llm.EmbeddingFactory.get_embeddings", return_value=embeddings),
            patch("ai.llm.LiteLLMRouter.completion", return_value="Evidence [SRC-1] [SRC-99].") as completion,
            TestClient(app) as client,
        ):
            first, second = self.start(client), self.start(client)
            completion.assert_not_called()
            for job, name in [(first, "first"), (second, "second"), (first, "first")]:
                self.assertTrue(job["chat_ready"])
                self.assertIsNone(job["report"])
                response = client.post(f"/api/research/{job['id']}/chat", json={
                    "question": "What did the source say?",
                    "history": [{"question": "What was found?", "answer": "Earlier response"}],
                })
                self.assertEqual(response.status_code, 200)
                reply = response.json()
                self.assertEqual([c["url_or_path"] for c in reply["citations"]], [f"https://example.com/{name}"])
                self.assertNotIn("[SRC-99]", reply["answer"])
                self.assertIn("What was found?", json.loads(completion.call_args.kwargs["messages"][1]["content"])["conversation_history"][0]["question"])
            self.assertEqual(embeddings.index_builds, 2)

    def test_full_report_and_chat_share_the_source_index(self):
        embeddings = CountingEmbeddings()
        report = ResearchReportData(**{
            name: "Fixture report" for name, field in ResearchReportData.model_fields.items() if field.is_required()
        })
        with (
            patch("ai.agents.researcher.ModernWebScraper.search_and_scrape", return_value=[source("report")]),
            patch("ai.agents.dataset.KaggleDatasetTool.search_datasets", return_value=[]),
            patch("ai.llm.EmbeddingFactory.get_embeddings", return_value=embeddings),
            patch("ai.llm.LiteLLMRouter.completion", side_effect=[AIUseCasesResponse(use_cases=[]), report, "Evidence [SRC-1]."]),
            TestClient(app) as client,
        ):
            job = self.start(client, mode="report")
            self.assertEqual(job["status"], "completed")
            self.assertTrue(job["chat_ready"])
            response = client.post(f"/api/research/{job['id']}/chat", json={"question": "What did you learn?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["citations"][0]["url_or_path"], job["report"]["citations"][0]["url_or_path"])
            self.assertEqual(embeddings.index_builds, 1)

    def test_chat_reports_missing_jobs_and_unavailable_sources(self):
        with (
            patch("ai.agents.researcher.ModernWebScraper.search_and_scrape", return_value=[]),
            TestClient(app) as client,
        ):
            job = self.start(client)
            self.assertEqual(job["status"], "failed")
            self.assertFalse(job["chat_ready"])
            self.assertEqual(client.post(f"/api/research/{job['id']}/chat", json={"question": "Why?"}).status_code, 409)
            self.assertEqual(client.post("/api/research/missing/chat", json={"question": "Why?"}).status_code, 404)
            self.assertEqual(client.post(f"/api/research/{job['id']}/chat", json={"question": "  "}).status_code, 422)
