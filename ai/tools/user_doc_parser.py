"""
User Document Ingestion Engine
Parses user-uploaded documents (.pdf, .txt, .md, .docx) into structured ScrapedDocument objects
with citation tagging and file path / symlink metadata for Dynamic RAG.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from ai.state import ScrapedDocument
from ai.tools.pdf_parser import PDFParser
from ai.settings import get_settings

logger = logging.getLogger(__name__)


class UserDocumentParser:
    """
    Parser for user-supplied custom documents.
    Extracts raw text and metadata from PDF, TXT, Markdown, and Word documents.
    """

    @staticmethod
    def parse_file(filepath: str) -> Optional[ScrapedDocument]:
        """
        Parse a single file path and return a structured ScrapedDocument object.
        """
        if not os.path.exists(filepath):
            logger.warning(f"User uploaded file not found: {filepath}")
            return None

        settings = get_settings()
        if not os.path.isfile(filepath):
            logger.warning("Upload path is not a regular file: %s", filepath)
            return None
        if os.path.getsize(filepath) > settings.max_upload_size_bytes:
            logger.warning("Upload exceeds the %s MB size limit: %s", settings.max_upload_size_mb, filepath)
            return None

        abs_path = os.path.abspath(filepath)
        filename = os.path.basename(filepath)
        ext = filename.split(".")[-1].lower() if "." in filename else ""

        content = ""
        source_type = "user_upload"

        try:
            if ext == "pdf":
                content = PDFParser.extract_text(abs_path)
                source_type = "pdf"
            elif ext in ["txt", "md", "markdown"]:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif ext == "docx":
                try:
                    import docx
                    doc = docx.Document(abs_path)
                    content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                except ImportError:
                    logger.warning("python-docx not installed. Install python-docx for .docx parsing.")
                    return None
            else:
                logger.warning("Rejected unsupported upload type: %s", ext or "no extension")
                return None

            if not content or not content.strip():
                logger.warning(f"File {filename} contained no readable text.")
                return None

            return ScrapedDocument(
                title=f"[User Upload] {filename}",
                url=f"file://{abs_path}",
                content=content.strip()[:settings.max_document_chars],
                category="user_doc",
                source_type=source_type,
                file_path=abs_path
            )

        except Exception as e:
            logger.error(f"Error parsing user document '{filepath}': {e}")
            return None

    @classmethod
    def parse_multiple(cls, filepaths: List[str]) -> List[ScrapedDocument]:
        """
        Parse multiple user uploaded file paths into a list of ScrapedDocument objects.
        """
        docs: List[ScrapedDocument] = []
        for fp in filepaths:
            doc = cls.parse_file(fp)
            if doc:
                docs.append(doc)
        logger.info(f"UserDocumentParser ingested {len(docs)} user-uploaded document(s).")
        return docs
