"""
PDF Metadata & Text Extraction Engine
Downloads and extracts clean text content from PDF reports and industry whitepapers.
"""

import os
import logging
import ipaddress
import socket
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
import pdfplumber

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Validates PDF size/metadata and parses page text.
    """

    @staticmethod
    def extract_text(filepath: str) -> str:
        """
        Extract text from local PDF file using pdfplumber.
        """
        try:
            text_pages = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages[:30]:  # Limit to first 30 pages
                    extracted = page.extract_text()
                    if extracted:
                        text_pages.append(extracted.strip())

            return "\n\n".join(text_pages)
        except Exception as e:
            logger.error(f"Failed to extract PDF text from '{filepath}': {e}")
            return ""

    @staticmethod
    def download_and_parse(url: str, output_dir: str = "data/pdfs") -> Optional[Dict[str, Any]]:
        """
        Download PDF URL, validate metadata, and return structured document dict.
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
                raise ValueError("Only absolute HTTP(S) URLs are allowed")
            # Block obvious SSRF targets before fetching. Production should also
            # enforce this at the network boundary because DNS can change.
            for address in socket.getaddrinfo(parsed_url.hostname, None, type=socket.SOCK_STREAM):
                ip = ipaddress.ip_address(address[4][0])
                if not ip.is_global:
                    raise ValueError("URLs resolving to private or reserved addresses are not allowed")
            os.makedirs(output_dir, exist_ok=True)
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            
            filepath = os.path.join(output_dir, filename)

            # Download
            response = requests.get(url, timeout=(5, 15), stream=True, allow_redirects=False)
            response.raise_for_status()
            if "pdf" not in response.headers.get("content-type", "").lower():
                raise ValueError("URL did not return a PDF content type")
            max_download_bytes = 25 * 1024 * 1024
            if int(response.headers.get("content-length", 0)) > max_download_bytes:
                raise ValueError("PDF exceeds the 25 MB download limit")

            with open(filepath, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    downloaded += len(chunk)
                    if downloaded > max_download_bytes:
                        raise ValueError("PDF exceeds the 25 MB download limit")
                    f.write(chunk)

            text = PDFParser.extract_text(filepath)
            if not text:
                return None

            return {
                "title": filename.replace(".pdf", "").replace("_", " "),
                "url": url,
                "content": text[:8000],
                "category": "pdf_report",
                "source_type": "pdf"
            }

        except Exception as e:
            logger.error(f"Error downloading PDF from '{url}': {e}")
            return None
