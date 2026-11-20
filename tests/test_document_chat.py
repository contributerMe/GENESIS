import json
import re
import unittest
from unittest.mock import patch
from uuid import UUID

import chromadb
from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from ai.llm import LiteLLMRouter
from ai.settings import Settings
from backend.main import app


def make_pdf(*pages: str) -> bytes:
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        objects.append((
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + i * 2} 0 R >>"
        ).encode())
        stream = f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(pdf)


class TestEmbeddings(Embeddings):
    def __init__(self):
        self.builds = 0

    def embed_documents(self, texts):
        self.builds += 1
        return [[1.0, 0.0] if "Launch" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class DocumentChatTests(unittest.TestCase):
    @staticmethod
    def upload(client, *files):
        return client.post("/api/documents", data={"model_name": "gemini-3.5-flash-lite"}, files=[
            ("files", (name, body, "application/pdf")) for name, body in files
        ])

    @staticmethod
    def session(created):
        return f"/api/documents/{created['req_id']}", {"Authorization": f"Bearer {created['session_token']}"}

    def test_pdf_chat_indexes_once_and_isolates_sessions_and_pages(self):
        embeddings = TestEmbeddings()

        def cite_launch(messages, **kwargs):
            self.assertTrue(kwargs["private"])
            self.assertEqual(kwargs["provider"], "gemini")
            self.assertEqual(kwargs["model_name"], "gemini-3.5-flash-lite")
            passages = json.loads(messages[1]["content"])["context_documents"]
            launch = next(passage for passage in passages if "Launch" in passage)
            self.assertIn("PDF page 2", launch)
            tag = re.search(r"\[SRC-\d+\]", launch)[0]
            return f"{launch.splitlines()[1]} {tag}"

        with (
            TestClient(app) as client,
            patch("backend.documents.EmbeddingFactory.get_embeddings", return_value=embeddings) as factory,
            patch("ai.rag.chat_engine.LiteLLMRouter.completion", side_effect=cite_launch) as completion,
        ):
            a = self.upload(client, ("Notes.pdf", make_pdf("Introduction", "Launch April 14")), ("Blank.pdf", make_pdf("")))
            b = self.upload(client, ("Notes.pdf", make_pdf("Introduction", "Launch June 23")))
            self.assertEqual((a.status_code, b.status_code), (202, 202))
            first = a.json()
            second = b.json()
            self.assertEqual(UUID(first["req_id"]).version, 4)
            self.assertNotEqual(first["req_id"], second["req_id"])
            path_a, auth_a = self.session(first)
            path_b, auth_b = self.session(second)
            self.assertEqual(client.get(path_a).status_code, 404)
            self.assertEqual(client.get(path_a, headers=auth_b).status_code, 404)
            session = client.get(path_a, headers=auth_a).json()
            self.assertEqual(session["status"], "partial")
            self.assertEqual(session["files"][0]["indexed_pages"], 2)
            self.assertIn("OCR", session["files"][1]["warning"])
            self.assertNotIn("session_token", session)
            question = {"question": "When is the launch?"}
            result_a = client.post(path_a + "/chat", headers=auth_a, json=question).json()
            result_b = client.post(path_b + "/chat", headers=auth_b, json=question).json()
            self.assertIn("April 14", result_a["answer"])
            self.assertIn("June 23", result_b["answer"])
            citation = result_a["citations"][0]
            self.assertEqual((citation["title"], citation["page"], citation["url_or_path"]), ("Notes.pdf", 2, ""))
            self.assertNotEqual(citation["document_id"], result_b["citations"][0]["document_id"])
            self.assertEqual(len(result_a["citations"]), 1)
            completion.side_effect = None
            completion.return_value = "Unsupported claim [SRC-999]."
            unknown = client.post(path_a + "/chat", headers=auth_a, json={"question": "Who is the publisher?"}).json()
            self.assertIn("couldn't find enough evidence", unknown["answer"])
            self.assertEqual(unknown["citations"], [])
            self.assertEqual(embeddings.builds, 2)
            factory.assert_called_with(provider="gemini")
            self.assertEqual(client.delete(path_a, headers=auth_b).status_code, 404)
            self.assertEqual(client.delete(path_a, headers=auth_a).status_code, 204)
            self.assertEqual(client.post(path_a + "/chat", headers=auth_a, json=question).status_code, 404)
            with self.assertRaises(NotFoundError):
                chromadb.EphemeralClient().get_collection(f"documents_{first['req_id']}")

    def test_bad_uploads_and_limits(self):
        with TestClient(app) as client, patch("backend.documents.EmbeddingFactory.get_embeddings") as factory:
            invalid = self.upload(client, ("Fake.pdf", b"Not a PDF"))
            path, auth = self.session(invalid.json())
            failure = client.get(path, headers=auth).json()
            self.assertEqual(failure["status"], "failed")
            self.assertIn("not a valid PDF", failure["files"][0]["warning"])
            self.assertEqual(client.post(path + "/chat", headers=auth, json={"question": "Hello?"}).status_code, 409)
            self.assertEqual(self.upload(client, ("Notes.txt", b"hello")).status_code, 415)
            self.assertEqual(self.upload(client, *((f"{i}.pdf", b"") for i in range(6))).status_code, 400)
            with patch("backend.main.MAX_UPLOAD_BYTES", 4):
                self.assertEqual(self.upload(client, ("Large.pdf", b"x" * 5)).status_code, 413)
                self.assertEqual(self.upload(client, ("Large.pdf", b"x" * 70000)).status_code, 413)
            factory.assert_not_called()

    def test_pdf_generation_disables_tracing_and_provider_fallback(self):
        settings = Settings(_env_file=None, llm_fallback_models="openai/gpt-4o-mini")
        with (
            patch("ai.llm.get_settings", return_value=settings),
            patch.object(LiteLLMRouter, "get_api_key", return_value="test-key"),
            patch.object(LiteLLMRouter, "_execute_with_retries", side_effect=RuntimeError("unavailable")) as send,
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                LiteLLMRouter.completion(provider="gemini", prompt="Synthetic PDF excerpt", private=True)
            send.assert_called_once()
            self.assertTrue(send.call_args.args[0]["no-log"])
