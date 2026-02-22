"""RAG query entry point.

Command-line interface for querying the RAG system with logging and error handling.
Provides a user-friendly interface to query retrieved documents with configurable
retrieval parameters, performance metrics, and source attribution.

Usage:
    python query.py "What is multi-factor authentication?"
    python query.py --k 10 "What are the security policies?"
    python query.py --verbose "How do we handle data retention?"
    python query.py --show-sources "Data retention requirements"
"""

import argparse
import getpass
import sys
from pathlib import Path
from typing import Any, Optional, Union

# Ensure project root is importable when running as a script path, e.g.:
# python scripts/rag/query.py --help
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.db_factory import get_default_vector_path, get_vector_client

# Optional typing helpers
try:
    from chromadb.api.models.Collection import (  # type: ignore  # noqa: WPS433,E402
        Collection as ChromaDBCollection,
    )
except Exception:
    ChromaDBCollection = Any  # type: ignore

try:
    from scripts.ingest.chromadb_sqlite import ChromaSQLiteCollection  # noqa: WPS433,E402
except Exception:
    ChromaSQLiteCollection = Any  # type: ignore

Collection = Union[ChromaDBCollection, ChromaSQLiteCollection, Any]

# Centralised backend selection (Chroma preferred)
PersistentClient, USING_SQLITE = get_vector_client(prefer="chroma")

from scripts.ingest.vectors import EMBEDDING_MODEL_NAME
from scripts.utils.logger import create_module_logger, flush_all_handlers
from scripts.utils.resource_monitor import ResourceMonitor
from scripts.utils.retry_utils import retry_chromadb_call

# Handle both package imports and direct script execution
if __name__ == "__main__" and __package__ is None:
    from scripts.rag.generate import answer, record_query_sample_for_adaptive_learning
    from scripts.rag.rag_config import RAGConfig

    get_logger, audit = create_module_logger("rag")
else:
    from .generate import answer, record_query_sample_for_adaptive_learning
    from .rag_config import RAGConfig

    get_logger, audit = create_module_logger("rag")

# Logger will be initialised in main() after purge logic
logger = None


@retry_chromadb_call(max_retries=3, initial_delay=1.0, operation_name="load_collection")
def _load_collection(config: RAGConfig) -> Collection:
    chroma_path = get_default_vector_path(Path(config.rag_data_path), USING_SQLITE)
    client = PersistentClient(path=chroma_path)
    return client.get_collection(config.chunk_collection_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query the RAG system for governance and security information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is MFA?"
  %(prog)s --k 10 "Security policies"
  %(prog)s --verbose --k 5 "Data retention requirements"
        """,
    )

    parser.add_argument("query", help="Question or query to ask the RAG system")

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (uses config default if not specified)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature for generation (0.0-1.0, uses config default if not specified)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output with metadata and performance info",
    )

    parser.add_argument(
        "--show-sources",
        action="store_true",
        default=False,
        help="Include source metadata for retrieved chunks",
    )

    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="Custom system role/prompt override (e.g., 'You are a security expert')",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain for domain-specific term expansion (e.g., 'aboriginal_torres_strait_islander')",
    )

    parser.add_argument(
        "--purge-logs",
        action="store_true",
        default=False,
        help="Purge all RAG log files before starting (disabled in Production environment)",
    )

    return parser.parse_args()


def get_collection(config: RAGConfig, logger=None) -> Collection:
    """Load ChromaDB collection for vector similarity search.

    Initialises a PersistentClient with the configured data path and retrieves
    the chunk collection for querying. Collection is retrieved without an embedding
    function since we handle embeddings separately in the retrieve() function.
    Logs and audits any failures.

    Args:
        config: RAG configuration containing data path and collection names.
        logger: Logger instance for logging messages.

    Returns:
        ChromaDB collection instance ready for semantic search.

    Raises:
        Exception: If collection cannot be loaded (e.g., path invalid, collection
                   doesn't exist, or ChromaDB connection fails).
    """
    # Prefer provided logger, fall back to module/global logger, then create a new one
    active_logger = logger or globals().get("logger") or get_logger()

    try:
        collection = _load_collection(config)
        active_logger.info(f"Loaded collection: {config.chunk_collection_name}")
        return collection
    except Exception as e:
        active_logger.error(f"Failed to load collection: {e}", exc_info=True)
        audit(
            "collection_load_error", {"collection": config.chunk_collection_name, "error": str(e)}
        )
        raise


def main() -> None:
    """Execute a single RAG query with full pipeline.

    Orchestrates the complete query workflow:
    1. Parse and validate command-line arguments
    2. Load RAG configuration from environment
    3. Purge logs if requested (Dev/Test only)
    4. Initialise ChromaDB collection
    5. Invoke the RAG pipeline (retrieve → generate)
    6. Format and display results with optional metadata

    Handles user interrupts gracefully and audits all outcomes.
    Exits with status 0 on success, 1 on failure.
    """
    global logger
    try:
        args = parse_args()
        config = RAGConfig()

        # Handle log purging BEFORE logger initialisation
        # This ensures we purge the old audit log before we write to a new one
        purge_logs_performed = False
        if args.purge_logs:
            if config.environment == "Prod":
                print("\n[ERROR] Log purging is disabled in Production environment for safety.")
                print("        Current environment: Prod")
                print("        To purge logs, set ENVIRONMENT=Dev or ENVIRONMENT=Test\n")
                if logger:
                    flush_all_handlers(logger)
                sys.exit(1)
            else:
                # Purge RAG logs ONLY (not ingest or consistency_graph logs)
                logs_dir = Path(__file__).parent.parent.parent / "logs"
                if logs_dir.exists():
                    rag_logs = [
                        "rag.log",
                        "rag_audit.jsonl",
                    ]

                    purged_count = 0
                    print(f"\n[PURGE LOGS] Environment: {config.environment}")
                    for log_name in rag_logs:
                        log_file = logs_dir / log_name
                        if log_file.exists():
                            try:
                                log_file.unlink()
                                purged_count += 1
                                print(f"  ✓ Removed: {log_file}")
                            except Exception as e:
                                print(f"  ✗ Failed to remove {log_file}: {e}")
                        else:
                            print(f"  - Not found: {log_name}")

                    print(f"[PURGE LOGS] Removed {purged_count} RAG log file(s)\n")
                    purge_logs_performed = True

        # Initialise logger AFTER purging so we don't write to a file we're about to delete
        # Respect any test-injected logger to keep behaviour deterministic
        if logger is None:
            logger = get_logger(log_to_console=True)

        # Log the purge event as the FIRST audit entry if logs were purged
        if purge_logs_performed:
            audit(
                "purge_logs",
                {
                    "environment": config.environment,
                    "module": "rag",
                    "user": getpass.getuser(),
                    "files_purged": ["rag.log", "rag_audit.jsonl"],
                },
            )

        # Validate query
        if not args.query or not args.query.strip():
            logger.error("Query cannot be empty")
            flush_all_handlers(logger)
            sys.exit(1)

        # Validate k if provided
        if args.k is not None and args.k < 1:
            logger.error(f"k must be >= 1, got {args.k}")
            flush_all_handlers(logger)
            sys.exit(1)

        # Validate temperature if provided
        if args.temperature is not None and not (0.0 <= args.temperature <= 1.0):
            logger.error(f"Temperature must be in range [0.0, 1.0], got {args.temperature}")
            flush_all_handlers(logger)
            sys.exit(1)

        # Load collection
        collection = get_collection(config)

        # Initialise resource monitoring
        resource_monitor = None
        if config.enable_resource_monitoring:
            resource_monitor = ResourceMonitor(
                operation_name=f"rag_query_{args.query[:30]}",
                interval=config.resource_monitoring_interval,
                enabled=True,
                monitor_ollama=config.monitor_ollama,
                monitor_chromadb=config.monitor_chromadb,
            )
            resource_monitor.start()
            logger.info("Resource monitoring started")

        try:
            # Generate answer
            logger.info(f"Query: {args.query}")
            audit(
                "query_start",
                {
                    "query": args.query[:200],
                    "k": args.k or config.k_results,
                    "temperature": args.temperature or config.temperature,
                    "custom_role": "provided" if args.role else "default",
                    "domain": args.domain,
                    "user": getpass.getuser(),
                    "llm_model": config.model_name,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "embedding_db_path": config.rag_data_path,
                    "chunk_collection": config.chunk_collection_name,
                    "doc_collection": config.doc_collection_name,
                },
            )

            answer_kwargs = {"k": args.k, "temperature": args.temperature}
            if args.role is not None:
                answer_kwargs["custom_role"] = args.role
            if args.domain is not None:
                answer_kwargs["domain"] = args.domain

            response = answer(args.query, collection, **answer_kwargs)

            # Record adaptive learning sample (non-blocking; won't break on failure)
            try:
                record_query_sample_for_adaptive_learning(args.query, response)
            except Exception as e:
                logger.debug(f"Failed to record adaptive learning sample: {e}")

            # Output results
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(response["answer"])
            print("=" * 80)

            # Verbose output
            if args.verbose:
                print(f"\nPerformance:")
                print(f"  Generation time: {response['generation_time']}s")
                print(f"  Total time: {response.get('total_time', 'N/A')}s")
                print(f"  Chunks retrieved: {response['retrieval_count']}")
                print(f"  Model: {response['model']}")

            # Show sources
            if args.show_sources and response["sources"]:
                print(f"\nSources ({len(response['sources'])} chunks):")
                for i, source in enumerate(response["sources"], 1):
                    print(
                        f"  {i}. {source.get('source', 'Unknown')} (v{source.get('version', '?')})"
                    )

            audit(
                "query_complete",
                {
                    "chunks_retrieved": response["retrieval_count"],
                    "total_time": response.get("total_time", response["generation_time"]),
                },
            )
        finally:
            # Stop resource monitoring
            if resource_monitor:
                resource_monitor.stop()
                resource_monitor.print_summary()
                stats_file = resource_monitor.export_json()
                logger.info(f"Resource statistics exported to {stats_file}")

    except KeyboardInterrupt:
        if logger:
            logger.info("Query interrupted by user")
            flush_all_handlers(logger)
        audit("query_interrupted", {})
        sys.exit(0)

    except Exception as e:
        if logger:
            logger.error(f"Query failed: {e}", exc_info=True)
            flush_all_handlers(logger)
        audit("query_failed", {"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
