"""Academic ingestion configuration.

Centralises configuration with BaseConfig getters and CLI override support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scripts.utils.config import BaseConfig


@dataclass
class AcademicIngestConfig(BaseConfig):
    """Configuration for academic ingestion."""

    def __init__(self) -> None:
        super().__init__()

        # Deployment environment (Dev, Test, Prod)
        self.environment = self.get_str("ENVIRONMENT", "Dev")
        if self.environment not in ("Dev", "Test", "Prod"):
            raise ValueError(
                f"Invalid ENVIRONMENT '{self.environment}'. Must be one of: Dev, Test, Prod"
            )

        # Provider credentials
        self.crossref_email = self.get_str("CROSSREF_EMAIL", "")
        self.unpaywall_email = self.get_str("UNPAYWALL_EMAIL", "")
        self.semantic_scholar_key = self.get_str("SEMANTIC_SCHOLAR_API_KEY", "")
        self.orcid_client_id = self.get_str("ORCID_CLIENT_ID", "")
        self.orcid_client_secret = self.get_str("ORCID_CLIENT_SECRET", "")

        # Download configuration
        self.max_pdf_size_mb = self.get_int("ACADEMIC_INGEST_MAX_PDF_SIZE_MB", 50)
        self.concurrent_downloads = self.get_int("ACADEMIC_INGEST_CONCURRENT_DOWNLOADS", 3)
        self.cache_dir = self.get_path("ACADEMIC_INGEST_CACHE_DIR", "rag_data/academic_pdfs")
        self.dry_run = self.get_bool("ACADEMIC_INGEST_DRY_RUN", False)
        self.log_level = self.get_str("ACADEMIC_INGEST_LOG_LEVEL", "info")

        # BM25 keyword indexing (for hybrid search)
        self.bm25_indexing_enabled = self.get_bool("BM25_INDEXING_ENABLED", True)
        self.bm25_index_original_text = self.get_bool("BM25_INDEX_ORIGINAL_TEXT", True)

        # Parent-child chunking (for better context preservation)
        self.enable_parent_child_chunking = self.get_bool(
            "ACADEMIC_INGEST_ENABLE_PARENT_CHILD_CHUNKING", True
        )


_ACADEMIC_CONFIG: Optional[AcademicIngestConfig] = None


def get_academic_config(
    overrides: Optional[dict] = None, reset: bool = False
) -> AcademicIngestConfig:
    """Return singleton AcademicIngestConfig with optional overrides."""
    global _ACADEMIC_CONFIG
    if overrides:
        AcademicIngestConfig.set_overrides(overrides)
    if _ACADEMIC_CONFIG is None or reset:
        _ACADEMIC_CONFIG = AcademicIngestConfig()
    return _ACADEMIC_CONFIG
