"""Utility modules for the RAG project.

This package contains shared utility functions and classes used across
all modules (ingest, rag, consistency_graph).

Modules:
    logger: Unified logging with module-based loggers and audit trails
    retry_utils: Retry logic with exponential backoff for transient failures
    rate_limiter: Rate limiting for API calls using token bucket algorithm
    schemas: Pydantic validation schemas for data integrity
"""

__all__ = ["logger", "retry_utils", "rate_limiter", "schemas", "db_factory"]
