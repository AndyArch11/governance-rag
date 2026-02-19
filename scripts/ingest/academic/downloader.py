"""Download utilities for academic references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional

import requests


@dataclass
class DownloadResult:
    success: bool
    path: Optional[str] = None
    error: Optional[str] = None


def _is_redirect_page(html_content: str) -> bool:
    """Detect HTML redirect pages that shouldn't be ingested.

    Identifies pages that are placeholders for redirects and don't contain
    actual content. Common patterns:
    - '<title>Redirecting...' or 'Redirect'
    - Meta refresh tags
    - JavaScript redirects
    - Very short content (< 100 chars) that's just redirect markup
    """
    if not html_content:
        return False

    lower = html_content.lower()

    # Check for redirect in title
    if "<title>" in lower:
        title_match = re.search(r"<title>([^<]+)</title>", lower)
        if title_match:
            title = title_match.group(1).strip()
            if any(
                word in title
                for word in ["redirect", "redirecting", "please wait", "moving", "found"]
            ):
                return True

    # Check for meta refresh
    if 'meta http-equiv="refresh"' in lower or 'meta http-equiv="refresh"' in lower:
        return True

    # Check for JavaScript redirects with minimal content
    if "window.location" in lower or "location.href" in lower:
        # If mostly JavaScript redirect code with little actual content, skip it
        text_len = len(html_content)
        if text_len < 500 and "window.location" in lower:
            return True

    # Check for common redirect patterns (frameset, empty body, etc.)
    if "<frameset" in lower or "<frame" in lower:
        return True

    # If page is almost empty with redirect indicators
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html_content, re.IGNORECASE | re.DOTALL)
    if body_match:
        body_content = body_match.group(1).strip()
        if len(body_content) < 50 and ("redirect" in lower or "please" in body_content.lower()):
            return True

    return False


def download_reference_pdf(url: str, dest_dir: str, max_size_mb: int = 50) -> DownloadResult:
    """Download a PDF reference.

    Streams content, enforces size limit, and stores with SHA256 filename.
    """
    if not url:
        return DownloadResult(success=False, error="missing_url")

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=30) as response:
            if response.status_code != 200:
                return DownloadResult(success=False, error=f"http_{response.status_code}")

            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return DownloadResult(success=False, error="not_pdf")

            max_bytes = max_size_mb * 1024 * 1024
            hasher = sha256()
            bytes_written = 0

            tmp_file = dest_path / "download.tmp"
            with tmp_file.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        tmp_file.unlink(missing_ok=True)
                        return DownloadResult(success=False, error="max_size_exceeded")
                    hasher.update(chunk)
                    handle.write(chunk)

            file_id = hasher.hexdigest()
            final_path = dest_path / f"{file_id}.pdf"
            if final_path.exists():
                tmp_file.unlink(missing_ok=True)
            else:
                tmp_file.replace(final_path)

            return DownloadResult(success=True, path=str(final_path))
    except requests.RequestException as exc:
        return DownloadResult(success=False, error=type(exc).__name__)


def download_web_content(url: str, dest_dir: str, max_size_mb: int = 10) -> DownloadResult:
    """Download web content as HTML and store with SHA256 filename.

    Detects and rejects HTML redirect pages to avoid ingesting placeholder content.
    """
    if not url:
        return DownloadResult(success=False, error="missing_url")

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=20, allow_redirects=True)
        if response.status_code != 200:
            return DownloadResult(success=False, error=f"http_{response.status_code}")

        content = response.text or ""
        if not content:
            return DownloadResult(success=False, error="empty_content")

        # Check if this is a redirect page that shouldn't be ingested
        if _is_redirect_page(content):
            return DownloadResult(success=False, error="redirect_page")
        max_bytes = max_size_mb * 1024 * 1024
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            return DownloadResult(success=False, error="max_size_exceeded")

        file_id = sha256(encoded).hexdigest()
        final_path = dest_path / f"{file_id}.html"
        if not final_path.exists():
            final_path.write_bytes(encoded)

        return DownloadResult(success=True, path=str(final_path))
    except requests.RequestException as exc:
        return DownloadResult(success=False, error=type(exc).__name__)
