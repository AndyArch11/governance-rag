"""Comprehensive unit tests for scripts.ingest.academic.downloader module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from scripts.ingest.academic.downloader import (
    DownloadResult,
    _is_redirect_page,
    download_reference_pdf,
    download_web_content,
)


class TestIsRedirectPage:
    """Tests for _is_redirect_page() function."""

    def test_empty_content(self) -> None:
        """Empty content is not a redirect page."""
        assert not _is_redirect_page("")

    def test_meta_refresh_redirect(self) -> None:
        """Detect meta refresh redirect."""
        html = '<html><head><meta http-equiv="refresh" content="0;url=http://example.com"></head></html>'
        assert _is_redirect_page(html)

    def test_meta_refresh_case_insensitive(self) -> None:
        """Meta refresh detection is case-insensitive."""
        html = '<html><head><META HTTP-EQUIV="REFRESH" CONTENT="0;url=http://example.com"></head></html>'
        assert _is_redirect_page(html)

    def test_javascript_location_replace(self) -> None:
        """Detect JavaScript location.replace redirect."""
        html = '<html><script>window.location.replace("http://example.com");</script></html>'
        assert _is_redirect_page(html)

    def test_javascript_location_href(self) -> None:
        """Detect JavaScript location.href redirect."""
        html = '<html><script>window.location.href = "http://example.com";</script></html>'
        assert _is_redirect_page(html)

    def test_javascript_location_assign(self) -> None:
        """Detect JavaScript window.location.assign redirect."""
        html = '<html><script>window.location.assign("http://example.com");</script></html>'
        assert _is_redirect_page(html)

    def test_frameset_redirect(self) -> None:
        """Detect frameset-based redirect."""
        html = '<html><frameset><frame src="http://example.com"></frameset></html>'
        assert _is_redirect_page(html)

    def test_empty_body_redirect(self) -> None:
        """Detect pages with empty body (common redirect pattern)."""
        html = "<html><head><title>Redirecting</title></head><body></body></html>"
        assert _is_redirect_page(html)

    def test_empty_body_with_whitespace(self) -> None:
        """Detect pages with whitespace-only body."""
        html = "<html><head><title>Redirecting</title></head><body>   \n  </body></html>"
        assert _is_redirect_page(html)

    def test_normal_content_accepted(self) -> None:
        """Normal HTML content is not flagged as redirect."""
        html = """<html>
            <head><title>Academic Paper</title></head>
            <body><p>This is real academic content with substantial text.</p></body>
        </html>"""
        assert not _is_redirect_page(html)

    def test_normal_content_with_meta_tags(self) -> None:
        """Normal content with non-redirect meta tags is accepted."""
        html = """<html>
            <head>
                <meta charset="utf-8">
                <meta name="description" content="Academic research">
                <title>Paper</title>
            </head>
            <body><p>Real content here.</p></body>
        </html>"""
        assert not _is_redirect_page(html)


class TestDownloadReferencePDF:
    """Tests for download_reference_pdf() function."""

    def test_missing_url(self) -> None:
        """Empty URL returns error."""
        result = download_reference_pdf("", "/tmp/test")
        assert not result.success
        assert result.error == "missing_url"
        assert result.path is None

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_http_error_404(self, mock_get: Mock) -> None:
        """HTTP 404 returns error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        result = download_reference_pdf("http://example.com/paper.pdf", "/tmp/test")
        assert not result.success
        assert result.error == "http_404"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_http_error_500(self, mock_get: Mock) -> None:
        """HTTP 500 returns error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        result = download_reference_pdf("http://example.com/paper.pdf", "/tmp/test")
        assert not result.success
        assert result.error == "http_500"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_wrong_content_type(self, mock_get: Mock) -> None:
        """Non-PDF content type returns error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.iter_content = Mock(return_value=[])  # Not reached, but required for context manager
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        # Use URL without .pdf extension to trigger content-type check
        result = download_reference_pdf("http://example.com/paper", "/tmp/test")
        assert not result.success
        assert result.error == "not_pdf"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_max_size_exceeded(self, mock_get: Mock) -> None:
        """Files exceeding max_size_mb are rejected."""
        # Create 2MB of PDF data
        chunk_size = 1024 * 1024  # 1MB
        pdf_data = b"%PDF-1.4" + (b"x" * chunk_size)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content = Mock(return_value=[pdf_data, pdf_data])  # 2MB total
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set max_size_mb to 1 (file is 2MB)
            result = download_reference_pdf("http://example.com/paper.pdf", tmpdir, max_size_mb=1)
            assert not result.success
            assert result.error == "max_size_exceeded"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_successful_download(self, mock_get: Mock) -> None:
        """Successful PDF download with SHA256 naming."""
        pdf_content = b"%PDF-1.4\nSample PDF content"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content = Mock(return_value=[pdf_content])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_reference_pdf("http://example.com/paper.pdf", tmpdir)
            assert result.success
            assert result.error is None
            assert result.path is not None
            
            # Verify file exists
            path = Path(result.path)
            assert path.exists()
            assert path.suffix == ".pdf"
            
            # Verify content matches
            assert path.read_bytes() == pdf_content

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_duplicate_file_not_redownloaded(self, mock_get: Mock) -> None:
        """Files with same SHA256 hash are not duplicated."""
        pdf_content = b"%PDF-1.4\nSample PDF content"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content = Mock(return_value=[pdf_content])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # First download
            result1 = download_reference_pdf("http://example.com/paper1.pdf", tmpdir)
            assert result1.success
            path1 = Path(result1.path)
            
            # Second download (same content, different URL)
            result2 = download_reference_pdf("http://example.com/paper2.pdf", tmpdir)
            assert result2.success
            path2 = Path(result2.path)
            
            # Should point to same file (same SHA256)
            assert path1 == path2
            
            # Only one file should exist
            pdf_files = list(Path(tmpdir).glob("*.pdf"))
            assert len(pdf_files) == 1

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_request_exception_captured(self, mock_get: Mock) -> None:
        """Network exceptions are captured and returned as error."""
        mock_get.side_effect = requests.ConnectionError("Network error")

        result = download_reference_pdf("http://example.com/paper.pdf", "/tmp/test")
        assert not result.success
        assert result.error == "ConnectionError"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_timeout_exception(self, mock_get: Mock) -> None:
        """Timeout exceptions are captured."""
        mock_get.side_effect = requests.Timeout("Connection timeout")

        result = download_reference_pdf("http://example.com/paper.pdf", "/tmp/test")
        assert not result.success
        assert result.error == "Timeout"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_content_type_case_insensitive(self, mock_get: Mock) -> None:
        """Content-Type header check is case-insensitive."""
        pdf_content = b"%PDF-1.4\nSample PDF content"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "APPLICATION/PDF"}  # Uppercase
        mock_response.iter_content = Mock(return_value=[pdf_content])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_reference_pdf("http://example.com/paper.pdf", tmpdir)
            assert result.success

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_content_type_with_charset(self, mock_get: Mock) -> None:
        """Content-Type with charset is accepted."""
        pdf_content = b"%PDF-1.4\nSample PDF content"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf; charset=utf-8"}
        mock_response.iter_content = Mock(return_value=[pdf_content])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_reference_pdf("http://example.com/paper.pdf", tmpdir)
            assert result.success


class TestDownloadWebContent:
    """Tests for download_web_content() function."""

    def test_missing_url(self) -> None:
        """Empty URL returns error."""
        result = download_web_content("", "/tmp/test")
        assert not result.success
        assert result.error == "missing_url"
        assert result.path is None

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_http_error_404(self, mock_get: Mock) -> None:
        """HTTP 404 returns error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = download_web_content("http://example.com/page.html", "/tmp/test")
        assert not result.success
        assert result.error == "http_404"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_empty_content(self, mock_get: Mock) -> None:
        """Empty response content returns error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_get.return_value = mock_response

        result = download_web_content("http://example.com/page.html", "/tmp/test")
        assert not result.success
        assert result.error == "empty_content"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_redirect_page_rejected(self, mock_get: Mock) -> None:
        """HTML redirect pages are rejected."""
        redirect_html = '<html><head><meta http-equiv="refresh" content="0;url=http://example.com"></head></html>'
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = redirect_html
        mock_get.return_value = mock_response

        result = download_web_content("http://example.com/redirect.html", "/tmp/test")
        assert not result.success
        assert result.error == "redirect_page"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_max_size_exceeded(self, mock_get: Mock) -> None:
        """Files exceeding max_size_mb are rejected."""
        # Create 2MB of HTML content
        large_html = "<html><body>" + ("x" * (2 * 1024 * 1024)) + "</body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = large_html
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set max_size_mb to 1 (content is ~2MB)
            result = download_web_content("http://example.com/large.html", tmpdir, max_size_mb=1)
            assert not result.success
            assert result.error == "max_size_exceeded"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_successful_download(self, mock_get: Mock) -> None:
        """Successful HTML download with SHA256 naming."""
        html_content = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_content
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_web_content("http://example.com/page.html", tmpdir)
            assert result.success
            assert result.error is None
            assert result.path is not None
            
            # Verify file exists
            path = Path(result.path)
            assert path.exists()
            assert path.suffix == ".html"
            
            # Verify content matches
            assert path.read_text(encoding="utf-8") == html_content

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_duplicate_content_not_redownloaded(self, mock_get: Mock) -> None:
        """Files with same SHA256 hash are not duplicated."""
        html_content = "<html><body>Academic content</body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_content
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # First download
            result1 = download_web_content("http://example.com/page1.html", tmpdir)
            assert result1.success
            path1 = Path(result1.path)
            
            # Second download (same content, different URL)
            result2 = download_web_content("http://example.com/page2.html", tmpdir)
            assert result2.success
            path2 = Path(result2.path)
            
            # Should point to same file (same SHA256)
            assert path1 == path2
            
            # Only one file should exist
            html_files = list(Path(tmpdir).glob("*.html"))
            assert len(html_files) == 1

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_request_exception_captured(self, mock_get: Mock) -> None:
        """Network exceptions are captured and returned as error."""
        mock_get.side_effect = requests.ConnectionError("Network error")

        result = download_web_content("http://example.com/page.html", "/tmp/test")
        assert not result.success
        assert result.error == "ConnectionError"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_timeout_exception(self, mock_get: Mock) -> None:
        """Timeout exceptions are captured."""
        mock_get.side_effect = requests.Timeout("Connection timeout")

        result = download_web_content("http://example.com/page.html", "/tmp/test")
        assert not result.success
        assert result.error == "Timeout"

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_unicode_content(self, mock_get: Mock) -> None:
        """Unicode content is correctly encoded and stored."""
        unicode_html = "<html><body><p>Résumé: naïve café</p></body></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = unicode_html
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_web_content("http://example.com/unicode.html", tmpdir)
            assert result.success
            
            # Verify unicode content preserved
            path = Path(result.path)
            assert path.read_text(encoding="utf-8") == unicode_html

    @patch("scripts.ingest.academic.downloader.requests.get")
    def test_custom_max_size(self, mock_get: Mock) -> None:
        """Custom max_size_mb parameter is respected."""
        html_content = "<html><body>" + ("x" * (500 * 1024)) + "</body></html>"  # ~500KB

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_content
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should succeed with default 10MB limit
            result1 = download_web_content("http://example.com/medium.html", tmpdir, max_size_mb=10)
            assert result1.success
            
            # Should fail with 0.1MB limit
            result2 = download_web_content("http://example.com/medium.html", tmpdir, max_size_mb=0)
            assert not result2.success
            assert result2.error == "max_size_exceeded"


class TestDownloadResultDataclass:
    """Tests for DownloadResult dataclass."""

    def test_success_result(self) -> None:
        """Success result with path."""
        result = DownloadResult(success=True, path="/tmp/test.pdf")
        assert result.success
        assert result.path == "/tmp/test.pdf"
        assert result.error is None

    def test_error_result(self) -> None:
        """Error result with error message."""
        result = DownloadResult(success=False, error="http_404")
        assert not result.success
        assert result.error == "http_404"
        assert result.path is None

    def test_default_values(self) -> None:
        """Default values for optional fields."""
        result = DownloadResult(success=True)
        assert result.success
        assert result.path is None
        assert result.error is None
