"""
Base provider interface for academic reference resolution.

Defines the common interface and retry logic for all metadata providers.

TODO: Expiry/Staleness thresholds based on reference type (eg):
- News articles: 30 days
- Blog posts: 60 days
- Online content: 90 days
- Academic PDFs: No staleness check (static)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class ReferenceStatus(str, Enum):
    """Status of a reference resolution."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    CACHED = "cached"


class RecoverableError(Exception):
    """Provider encountered recoverable error (retry)."""

    pass


class FatalError(Exception):
    """Provider encountered fatal error (skip)."""

    pass


@dataclass
class Reference:
    """Resolved academic reference."""

    ref_id: str
    raw_citation: str

    # Core metadata
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None

    # Venue information
    venue: Optional[str] = None  # Journal/conference name
    venue_type: Optional[str] = None  # "journal" | "conference" | "preprint" | "report"
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None

    # Authority & quality
    reference_type: str = (
        "online"  # "academic" | "government" | "preprint" | "report" | "news" | "blog" | "online"
    )
    resolved: bool = False
    status: ReferenceStatus = ReferenceStatus.UNRESOLVED
    quality_score: float = 0.0

    # Metadata provider info
    metadata_provider: str = ""  # "crossref" | "openalex" | "scholar" etc
    oa_available: bool = False
    oa_url: Optional[str] = None

    # Link availability tracking
    link_status: str = (
        "available"  # "available" | "stale_404" | "stale_timeout" | "stale_moved" | "unresolved"
    )

    # Citation metrics
    citation_count: Optional[int] = None  # Number of citations to this reference

    # Tracking
    doc_ids: List[str] = field(default_factory=list)  # Documents that cite this
    resolved_at: Optional[datetime] = None
    cached: bool = False

    def __post_init__(self):
        """Set resolved timestamp."""
        if self.resolved and not self.resolved_at:
            self.resolved_at = datetime.now(timezone.utc)


class BaseProvider(ABC):
    """Abstract base class for metadata providers."""

    name: str
    base_url: str
    rate_limit: int  # requests per second
    timeout: int  # seconds

    def __init__(self):
        """Initialise provider."""
        self.session = requests.Session()
        self.last_request_time = 0
        # Get provider logger but ensure it propagates to parent "academic_ingest" logger
        provider_logger = logging.getLogger(f"academic.providers.{self.name}")
        provider_logger.propagate = True  # Propagate to parent loggers for file/console handling
        self.logger = provider_logger

    @abstractmethod
    def resolve(
        self, citation_text: str, year: Optional[int] = None, doi: Optional[str] = None, logger=None
    ) -> Reference:
        """
        Resolve a citation through this provider.

        Args:
            citation_text: Raw citation text or title
            year: Publication year (optional)
            doi: DOI if available (optional)
            logger: Optional logger instance for status messages

        Returns:
            Reference object with resolved metadata

        Raises:
            RecoverableError: Temporary failure, try next provider
            FatalError: Permanent failure, skip provider
        """
        pass

    def _request_with_retry(
        self, method: str, url: str, max_retries: int = 2, backoff_factor: float = 1.0, **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST)
            url: Target URL
            max_retries: Number of retries
            backoff_factor: Backoff multiplier
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response object

        Raises:
            RecoverableError: On timeout/connection errors
            FatalError: On 4xx errors
        """
        # Rate limiting
        self._rate_limit()

        for attempt in range(max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)

                # Check for errors
                if response.status_code == 404:
                    raise FatalError(f"Not found: {url}")
                elif response.status_code >= 400 and response.status_code < 500:
                    raise FatalError(f"Client error {response.status_code}: {url}")
                elif response.status_code >= 500:
                    # Server error - might be recoverable
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2**attempt)
                        self.logger.warning(
                            f"[{self.name}] Server error {response.status_code} from {url}, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RecoverableError(
                            f"Server error {response.status_code} after {max_retries} retries"
                        )

                response.raise_for_status()
                return response

            except requests.Timeout:
                if attempt < max_retries:
                    wait_time = backoff_factor * (2**attempt)
                    self.logger.warning(
                        f"[{self.name}] Timeout connecting to {url}, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise RecoverableError(f"Timeout after {max_retries} retries")
            except requests.ConnectionError as e:
                if attempt < max_retries:
                    wait_time = backoff_factor * (2**attempt)
                    self.logger.warning(
                        f"[{self.name}] Connection error to {url}: {str(e)[:100]}, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise RecoverableError(f"Connection error after {max_retries} retries: {e}")

    def _rate_limit(self):
        """Enforce rate limiting (requests per second)."""
        if self.rate_limit <= 0:
            return

        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self.last_request_time

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def __del__(self):
        """Clean up session."""
        if hasattr(self, "session"):
            self.session.close()
