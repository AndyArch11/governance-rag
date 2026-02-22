"""Academic paper ingestion pipeline entrypoint (initial scaffold).

TODO: Incorporate more of utils modules (e.g. resource monitor, retry_utils, etc)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is importable when running as a script path, e.g.:
# python scripts/ingest/ingest_academic.py --help
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.config import AcademicIngestConfig, get_academic_config
from scripts.ingest.academic.downloader import download_reference_pdf, download_web_content
from scripts.ingest.academic.graph import CitationGraph, add_references_to_graph
from scripts.ingest.academic.parser import extract_citations
from scripts.ingest.academic.providers import resolve_reference
from scripts.ingest.academic.terminology import DomainTerminologyExtractor, DomainTerminologyStore
from scripts.ingest.bm25_indexing import index_chunks_in_bm25
from scripts.ingest.chunk import chunk_text, create_parent_child_chunks
from scripts.ingest.htmlparser import extract_text_from_html
from scripts.ingest.ingest import compute_doc_id, compute_file_hash
from scripts.ingest.pdfparser import (
    extract_pdf_metadata,
    extract_structure_from_text,
    extract_text_from_pdf,
)
from scripts.ingest.vectors import (
    store_chunks_in_chroma,
    store_child_chunks,
    store_parent_chunks,
)
from scripts.ingest.word_frequency import WordFrequencyExtractor
from scripts.search.bm25_search import BM25Search
from scripts.utils.config import BaseConfig
from scripts.utils.db_factory import get_default_vector_path, get_vector_client
from scripts.utils.logger import create_module_logger
from scripts.utils.resource_monitor import ResourceMonitor

# Use shared "ingest" logger so all ingest activities are in the same log file
get_logger, audit = create_module_logger("ingest")


def _clean_doc_id(text: str, max_length: int = 150) -> str:
    """Generate clean doc_id from text with space insertion and truncation.

    Fixes concatenated text from metadata by:
    - Inserting spaces between lowercase-uppercase transitions
    - Inserting spaces between digit-letter transitions
    - Normalizing whitespace
    - Truncating to reasonable length

    Args:
        text: Raw text (could be citation, title, or concatenated content)
        max_length: Maximum length for resulting doc_id

    Returns:
        Clean doc_id with proper spacing, max 150 chars
    """
    import re

    if not text:
        return "unknown"

    # Insert space before capital letters that follow lowercase
    # "CruzP" -> "Cruz P"
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Insert space before capital letters that follow digits
    # "2019A" -> "2019 A"
    cleaned = re.sub(r"(\d)([A-Z])", r"\1 \2", cleaned)

    # Normalise whitespace (remove multiple spaces, convert to single space)
    cleaned = re.sub(r"\s+", " ", cleaned.strip())

    # Remove special chars that shouldn't be in doc_id
    cleaned = re.sub(r'[<>:"|?*\\]', "", cleaned)

    # Truncate to max length, trying to break at word boundary
    if len(cleaned) > max_length:
        # Truncate to max_length and backtrack to last space
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:  # If last space is reasonably far
            cleaned = truncated[:last_space]
        else:
            cleaned = truncated

    return cleaned or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic paper ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Document inputs
    parser.add_argument("papers_positional", nargs="*", help="Paths to academic papers (PDF)")
    parser.add_argument("--papers", action="append", default=[], help="PDF file paths")
    parser.add_argument("--papers-dir", type=str, default=None, help="Directory of PDFs")
    parser.add_argument("--batch", type=str, default=None, help="JSON batch manifest")

    # Document metadata
    parser.add_argument("--title", type=str, default=None, help="Document title")
    parser.add_argument("--domain", type=str, default=None, help="Primary domain tag")
    parser.add_argument("--topic", type=str, default=None, help="Topic label")
    parser.add_argument("--authors", type=str, default=None, help="Comma-separated authors")
    parser.add_argument("--institution", type=str, default=None, help="Institution name")
    parser.add_argument(
        "--skip-citations",
        action="store_true",
        default=False,
        help="Skip citation extraction/downloading (faster for thesis ingestion)",
    )
    parser.add_argument(
        "--skip-terminology",
        action="store_true",
        default=False,
        help="Skip domain terminology extraction (faster for thesis ingestion)",
    )

    # Pipeline controls
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--cache-reset", action="store_true", default=False)
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help=(
            "Clear ChromaDB storage, ingest caches, BM25 index, reference cache, terminology database, "
            "citation graph artefacts, academic PDF cache, and legacy artefacts before ingest"
        ),
    )
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument(
        "--purge-logs",
        action="store_true",
        default=False,
        help="Purge ingest logs before starting (requires ENVIRONMENT=Dev or Test)",
    )

    parser.add_argument(
        "--bm25-indexing",
        action="store_true",
        default=None,
        help="Enable BM25 keyword indexing during ingestion (default: from BM25_INDEXING_ENABLED env var)",
    )

    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        default=False,
        help="Disable BM25 keyword indexing during ingestion (overrides --bm25-indexing and env var)",
    )

    # Revalidation
    parser.add_argument(
        "--revalidate", choices=["stale", "online", "all", "failed", "ids"], default=None
    )
    parser.add_argument("--staleness-threshold", type=int, default=30)
    parser.add_argument("--ref-ids", nargs="*", default=[])

    # Provider credentials (CLI overrides)
    parser.add_argument("--crossref-email", type=str, default=None)
    parser.add_argument("--unpaywall-email", type=str, default=None)
    parser.add_argument("--semantic-scholar-key", type=str, default=None)
    parser.add_argument("--orcid-client-id", type=str, default=None)
    parser.add_argument("--orcid-client-secret", type=str, default=None)

    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides = {}

    if args.crossref_email:
        overrides["CROSSREF_EMAIL"] = args.crossref_email
    if args.unpaywall_email:
        overrides["UNPAYWALL_EMAIL"] = args.unpaywall_email
    if args.semantic_scholar_key:
        overrides["SEMANTIC_SCHOLAR_API_KEY"] = args.semantic_scholar_key
    if args.orcid_client_id:
        overrides["ORCID_CLIENT_ID"] = args.orcid_client_id
    if args.orcid_client_secret:
        overrides["ORCID_CLIENT_SECRET"] = args.orcid_client_secret
    if args.dry_run:
        overrides["ACADEMIC_INGEST_DRY_RUN"] = True

    # Priority: --skip-bm25 > --bm25-indexing > environment variable
    if getattr(args, "skip_bm25", False):
        overrides["BM25_INDEXING_ENABLED"] = False
    elif getattr(args, "bm25_indexing", None):
        overrides["BM25_INDEXING_ENABLED"] = True

    return overrides


def collect_documents(args: argparse.Namespace) -> List[Path]:
    """
    Collect list of documents based on the provided arguments.

    Args:
        args: Command line arguments

    Returns:
        List of document paths
    """
    paths: List[Path] = []

    # Positional
    for p in args.papers_positional:
        paths.append(Path(p).expanduser())

    # Named
    for p in args.papers or []:
        paths.append(Path(p).expanduser())

    # Directory
    if args.papers_dir:
        papers_dir = Path(args.papers_dir).expanduser()
        if papers_dir.exists():
            for pdf_path in papers_dir.rglob("*.pdf"):
                paths.append(pdf_path)

    # Batch manifest
    if args.batch:
        manifest = json.loads(Path(args.batch).expanduser().read_text())
        for doc in manifest.get("documents", []):
            if "path" in doc:
                paths.append(Path(doc["path"]).expanduser())

    # Deduplicate
    unique = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def stage_load_document(path: Path, config: AcademicIngestConfig, logger) -> Optional[str]:
    """
    Load a document from the given path, ensuring it meets the criteria for processing.

    Args:
        path: Path to the document
        config: Configuration for academic ingestion (e.g. max PDF size)
        logger: Logger for logging messages

    Returns:
        Extracted text from the document, or None if loading failed or document is invalid.
    """
    if not path.exists():
        logger.warning(f"Document not found: {path}")
        return None

    if path.suffix.lower() != ".pdf":
        logger.warning(f"Skipping non-PDF document: {path}")
        return None

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_pdf_size_mb:
        logger.warning(
            f"Skipping {path.name}: size {size_mb:.1f}MB exceeds limit {config.max_pdf_size_mb}MB"
        )
        return None

    try:
        return extract_text_from_pdf(str(path))
    except Exception as exc:
        logger.error(f"Failed to extract text from {path}: {type(exc).__name__}: {exc}")
        return None


def stage_extract_citations(text: str, logger) -> List[str]:
    """Extract citations from text using parser.

    Args:
        text: Full text of the document to extract citations from
        logger: Logger for logging messages

    Returns:
        List of raw citation strings extracted from the text
    """
    citations = extract_citations(text)
    if not citations:
        return []
    logger.info(f"Extracted {len(citations)} citations")
    return [c.raw_text for c in citations]


def stage_resolve_metadata(
    citations: List[str],
    cache: ReferenceCache,
    config: AcademicIngestConfig,
    logger,
) -> List[dict]:
    """
    Resolve citation metadata through provider chain or cache.

    Returns list of dicts with metadata for downstream processing.
    Args:
        citations: List of raw citation strings to resolve
        cache: ReferenceCache instance for caching resolved metadata
        config: AcademicIngestConfig for configuration options
        logger: Logger for logging messages

    Returns:
        List of dicts containing resolved metadata for each citation, including:
            - citation: The raw citation string
            - title: The resolved title of the reference
            - authors: List of authors for the reference
            - doi: The DOI of the reference
            - year: The publication year of the reference
            - reference_type: The type of reference (e.g., journal, book, online)
            - source: The source of the metadata (e.g., cached, provider)
            - url: The URL of the reference
            - oa_available: Whether the reference is openly accessible
            - confidence: The confidence score of the resolution
    """
    from scripts.ingest.academic.providers.base import Reference

    resolved = []
    for cit_idx, citation in enumerate(citations, 1):
        # Show progress every 50 citations or at end
        if cit_idx % 50 == 0 or cit_idx == len(citations):
            print(f"    Resolving metadata {cit_idx}/{len(citations)}...", end="\r")

        # Try cache first
        cached_ref = cache.get(citation)
        if cached_ref:
            record = {
                "citation": citation,
                "title": cached_ref.title,
                "doi": cached_ref.doi,
                "year": cached_ref.year,
                "reference_type": cached_ref.reference_type,
                "source": cached_ref.metadata_provider or "cached",
                "url": cached_ref.oa_url,
                "oa_available": cached_ref.oa_available,
                "confidence": None,  # Cached references don't have confidence from this ingestion
                "link_status": cached_ref.link_status,
                "venue_type": cached_ref.venue_type,
                "citation_count": cached_ref.citation_count,
            }
            resolved.append(record)
            continue

        # In dry-run mode, create placeholder Reference without calling expensive provider chain
        if config.dry_run:
            ref = Reference(
                ref_id=f"placeholder_{hash(citation) % 10000}",
                raw_citation=citation,
                resolved=False,
                reference_type="online",
            )
            confidence = 0.0
        else:
            # Call provider chain to resolve
            ref, confidence = resolve_reference(citation, logger=logger)

        # Cache the reference
        cache.put(citation, ref)

        # Convert to dict for downstream use
        record = {
            "citation": citation,
            "title": ref.title,
            "authors": ref.authors,  # Author list for author/year citations
            "doi": ref.doi,
            "year": ref.year,
            "reference_type": ref.reference_type,
            "source": ref.metadata_provider or "unresolved",
            "url": ref.oa_url,
            "oa_available": ref.oa_available,
            "confidence": confidence,
            "link_status": ref.link_status,
            "venue_type": ref.venue_type,
            "citation_count": ref.citation_count,
        }
        resolved.append(record)

    logger.info(f"Resolved {len(resolved)} references")
    return resolved


def stage_download_references(
    references: List[dict],
    config: AcademicIngestConfig,
    logger,
) -> List[dict]:
    """Download reference artifacts (PDFs or web content) based on resolved metadata.

    Args:
        references: List of dicts containing resolved metadata for each reference
        config: AcademicIngestConfig for configuration options (e.g., cache directory, max PDF size)
        logger: Logger for logging messages

    Returns:
        List of dicts with updated metadata including download status and artifact paths:
            - citation: The raw citation string
            - title: The resolved title of the reference
            - authors: List of authors for the reference
            - doi: The DOI of the reference
            - year: The publication year of the reference
            - reference_type: The type of reference (e.g., journal, book, online)
            - source: The source of the metadata (e.g., cached, provider)
            - url: The URL of the reference
            - oa_available: Whether the reference is openly accessible
            - confidence: The confidence score of the resolution
            - download_status: "success", "skipped", or error message
            - artifact_path: Local path to downloaded PDF or web content (if applicable)
    """
    if config.dry_run:
        logger.info("Dry run enabled: skipping downloads")
        updated = []
        for ref in references:
            ref_copy = dict(ref)
            ref_copy["download_status"] = "skipped"
            updated.append(ref_copy)
        return updated

    updated = []
    for ref in references:
        ref = dict(ref)
        ref_type = ref.get("reference_type")
        url = ref.get("url")
        pdf_url = ref.get("pdf_url")

        if pdf_url:
            result = download_reference_pdf(pdf_url, config.cache_dir, config.max_pdf_size_mb)
            ref["artifact_path"] = result.path
            ref["download_status"] = "success" if result.success else result.error
        elif ref_type in ("news", "blog", "online") and url:
            result = download_web_content(url, config.cache_dir)
            ref["artifact_path"] = result.path
            ref["download_status"] = "success" if result.success else result.error
        else:
            ref["download_status"] = "skipped"

        updated.append(ref)

    logger.info(f"Downloaded artifacts for {len(updated)} references")
    return updated


def stage_load_reference_text(artifact_path: str, logger) -> Optional[str]:
    """Load text content from reference artifact (PDF or web content).

    Args:
        artifact_path: Local path to the downloaded PDF or web content
        logger: Logger for logging messages
    Returns:
        Extracted text content from the artifact, or None if loading failed or unsupported type
    """
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.exists():
        logger.warning(f"Artifact missing: {artifact_path}")
        return None

    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(path))
    if path.suffix.lower() in (".html", ".htm"):
        return extract_text_from_html(str(path))

    logger.warning(f"Unsupported artifact type: {artifact_path}")
    return None


def stage_chunk_and_store(
    reference: dict,
    text: str,
    chunk_collection,
    doc_collection,
    config: AcademicIngestConfig,
    logger,
) -> bool:
    """Chunk reference text and store in ChromaDB with metadata.

    Args:
        reference: Dict containing reference metadata (including citation, title, authors, doi, year, type, source, url, oa_available, confidence)
        text: Full text content of the reference to be chunked and stored
        chunk_collection: ChromaDB collection for storing chunks
        doc_collection: ChromaDB collection for storing doc-level metadata and embeddings
        config: AcademicIngestConfig for configuration options (e.g., dry run)
        logger: Logger for logging messages

    Returns:
        True if storage succeeded or dry run, False if there was an error during storage
    """
    if not text:
        return False

    chunks = chunk_text(text, doc_type="academic_reference", adaptive=True)
    if not chunks:
        return False

    artifact_path = reference.get("artifact_path") or reference.get("path") or ""
    citation = reference.get("citation", "")

    # Extract document structure (chapter/section hierarchy) if PDF
    document_structure = None
    if artifact_path and artifact_path.endswith(".pdf"):
        try:
            document_structure = extract_structure_from_text(text)
            if document_structure:
                logger.info(f"Extracted {len(document_structure)} structural sections from PDF")
        except Exception as e:
            logger.warning(f"Failed to extract structure from {artifact_path}: {e}")

    # Extract PDF metadata if artifact exists
    pdf_metadata = {}
    if artifact_path and artifact_path.endswith(".pdf"):
        try:
            pdf_metadata = extract_pdf_metadata(artifact_path)
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata from {artifact_path}: {e}")

    # Build human-readable doc_id from available metadata (PDF or reference)
    # Priority: PDF metadata > reference metadata > citation text > hash
    title = pdf_metadata.get("title") or reference.get("title")
    author = pdf_metadata.get("author") or (
        reference.get("authors")[0] if reference.get("authors") else None
    )
    year = pdf_metadata.get("year") or reference.get("year")

    # Construct doc_id: "Author_Year_Title"
    doc_id_parts = []
    if author:
        # Use last name if comma-separated
        author_clean = (
            author.split(",")[0].strip()
            if "," in author
            else author.split()[0] if author.split() else author
        )
        doc_id_parts.append(author_clean[:30])
    if year:
        doc_id_parts.append(str(year))
    if title:
        # Clean title for doc_id (remove special chars, limit length)
        title_clean = re.sub(r"[^a-zA-Z0-9\s-]", "", title)[:80]
        title_clean = re.sub(r"\s+", "_", title_clean.strip())
        doc_id_parts.append(title_clean)

    # Generate doc_id with fallback chain
    if doc_id_parts:
        doc_id = "_".join(doc_id_parts)
    elif citation:
        doc_id = _clean_doc_id(citation)
    elif artifact_path:
        doc_id = compute_doc_id(artifact_path)
    else:
        logger.warning("No metadata available to generate doc_id")
        return False

    # Create display name for UI
    if title and author and year:
        display_name = f"{title} ({author}, {year})"
    elif title:
        display_name = title
    elif citation:
        display_name = citation[:150]
    elif artifact_path:
        display_name = Path(artifact_path).stem
    else:
        display_name = doc_id

    # Set file_hash and source_path
    if artifact_path:
        file_hash = compute_file_hash(artifact_path)
        source_path = artifact_path
    else:
        # No artifact - use citation hash
        import hashlib

        file_hash = hashlib.md5(citation.encode()).hexdigest()
        source_path = f"reference_metadata:{reference.get('doi') or doc_id}"

    metadata = {
        "doc_type": "academic_reference",
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "summary_scores": {"overall": 0},  # Dict format for store_chunks_in_chroma
        "key_topics": [],  # Empty topics list (could be enhanced later)
        "source_category": "academic_reference",
        "display_name": display_name,
    }
    # Add PDF metadata fields if extracted
    if pdf_metadata:
        if pdf_metadata.get("title"):
            metadata["title"] = pdf_metadata["title"]
        if pdf_metadata.get("author"):
            metadata["author"] = pdf_metadata["author"]
        if pdf_metadata.get("year"):
            metadata["year"] = pdf_metadata["year"]
        if pdf_metadata.get("keywords"):
            metadata["keywords"] = pdf_metadata["keywords"]
        if pdf_metadata.get("subject"):
            metadata["subject"] = pdf_metadata["subject"]

    # Only add optional fields if they're not None (reference metadata)
    if reference.get("reference_type"):
        metadata["reference_type"] = reference.get("reference_type")
    if reference.get("doi"):
        metadata["doi"] = reference.get("doi")
    if reference.get("year") and not metadata.get("year"):
        metadata["year"] = str(reference.get("year"))
    if reference.get("source"):
        metadata["source"] = reference.get("source")

    # Parent-child chunking (optional, improves retrieval context)
    parent_chunks = None
    child_chunks = None
    using_parent_child = bool(getattr(config, "enable_parent_child_chunking", True))

    if using_parent_child:
        try:
            # create_parent_child_chunks returns (child_chunks, parent_chunks)
            child_chunks, parent_chunks = create_parent_child_chunks(
                text=text, doc_type="academic_reference"
            )
            logger.debug(
                f"Created {len(child_chunks)} child chunks and {len(parent_chunks)} parent chunks for {doc_id}"
            )
        except Exception as e:
            logger.warning(f"Parent-child chunking failed: {e}")
            parent_chunks = None
            child_chunks = None

    if config.dry_run:
        title = reference.get("title") or reference.get("citation", "")[:100]
        doi = reference.get("doi") or "no-doi"
        source = reference.get("source") or "unknown-provider"
        if parent_chunks:
            logger.info(
                f"[DRY_RUN] Would store {len(child_chunks)} child chunks and {len(parent_chunks)} parent chunks for {doc_id} (title: {title[:50]}... doi: {doi} provider: {source})"
            )
        else:
            logger.info(
                f"[DRY_RUN] Would store {len(chunks)} chunks for {doc_id} (title: {title[:50]}... doi: {doi} provider: {source})"
            )
        return True

    # If using parent-child chunking, avoid storing duplicate child texts via generic store
    chunks_to_store = [] if parent_chunks else chunks

    # Store chunks using standard pipeline (stores both chunks and doc-level embedding)
    try:
        store_chunks_in_chroma(
            doc_id=doc_id,
            file_hash=file_hash,
            source_path=source_path,
            version=1,  # Academic references don't have versions
            chunks=chunks_to_store,
            metadata=metadata,
            chunk_collection=chunk_collection,
            doc_collection=doc_collection,
            preprocess_duration=0.0,
            ingest_duration=0.0,
            dry_run=False,  # Already handled dry_run check above
            enable_drift_detection=False,  # No drift detection for single-version refs
            enable_chunk_heuristic=True,
            full_text=text,
            document_structure=document_structure,  # Pass extracted structure
        )

        # If parent/child created, store child chunks (with embeddings) then parents
        # Following ingest_git.py pattern: child-first storage
        if parent_chunks:
            base_metadata = {
                "doc_id": doc_id,
                "source": source_path,
                "version": 1,
                "hash": file_hash,
                "doc_type": "academic_reference",
                "source_category": "academic_reference",
                "embedding_model": metadata.get("embedding_model", "unknown"),
            }

            # Store child chunks (searchable, with real embeddings) FIRST
            if child_chunks:
                try:
                    store_child_chunks(
                        doc_id=doc_id,
                        child_chunks=child_chunks,
                        chunk_collection=chunk_collection,
                        base_metadata=base_metadata,
                        dry_run=False,
                        full_text=text,
                        doc_type="academic_reference",
                    )
                    logger.debug(f"Stored {len(child_chunks)} child chunks for {doc_id}")
                except Exception as child_err:
                    logger.error(f"Failed to store child chunks for {doc_id}: {child_err}")
                    raise


            # Then store parent chunks (metadata/context only; non-fatal)
            try:
                store_parent_chunks(
                    doc_id=doc_id,
                    parent_chunks=parent_chunks,
                    chunk_collection=chunk_collection,
                    base_metadata=base_metadata,
                )
                logger.debug(f"Stored {len(parent_chunks)} parent chunks for {doc_id}")
            except Exception as parent_err:
                logger.warning(f"Failed to store parent chunks for {doc_id}: {parent_err}")
                # Non-fatal: parent storage shouldn't block the entire ingest

        # BM25 Keyword Indexing for chunks (at chunk-level granularity)
        # Uses common indexing utility function for consistency across modules
        # TODO: move db_factory import to top of page
        if config.bm25_indexing_enabled:
            try:
                from scripts.utils.db_factory import get_cache_client

                cache_db = get_cache_client(enable_cache=True)
                total_indexed = index_chunks_in_bm25(
                    doc_id=doc_id,
                    chunks=chunks,
                    child_chunks=child_chunks,
                    parent_chunks=parent_chunks,
                    config=config,
                    cache_db=cache_db,
                    logger=logger,
                )
                if total_indexed > 0:
                    logger.debug(
                        f"BM25 indexed {total_indexed} chunks for {doc_id} (granularity=chunk-level)"
                    )
            except Exception as e:
                logger.warning(f"BM25 indexing failed for {doc_id}: {e}")

        return True
    except Exception as e:
        if hasattr(logger, "error"):
            logger.error(f"Error storing chunks: {e}", exc_info=True)
        elif hasattr(logger, "warning"):
            logger.warning(f"Error storing chunks: {e}")
        return False


def main() -> int:
    import time
    start_time = time.perf_counter()
    
    args = parse_args()

    # Apply CLI overrides centrally
    BaseConfig.set_overrides(build_overrides(args))
    config = get_academic_config(reset=True)

    # Handle log purging BEFORE logger initialisation
    purge_logs_performed = False
    if args.purge_logs:
        if config.environment == "Prod":
            print("\n[ERROR] Log purging is disabled in Production environment for safety.")
            print("        Current environment: Prod")
            print("        To purge logs, set ENVIRONMENT=Dev or ENVIRONMENT=Test\n")
            return 1

        logs_dir = BaseConfig().logs_dir
        log_names = [
            "ingest.log",
            "ingest_audit.jsonl",
        ]

        purged_count = 0
        print(f"\n[PURGE LOGS] Environment: {config.environment}")
        for log_name in log_names:
            log_file = logs_dir / log_name
            if log_file.exists():
                try:
                    log_file.unlink()
                    purged_count += 1
                    print(f"  ✓ Removed: {log_file}")
                except Exception as exc:
                    print(f"  ✗ Failed to remove {log_file}: {exc}")
            else:
                print(f"  - Not found: {log_name}")

        print(f"[PURGE LOGS] Removed {purged_count} ingest log file(s)\n")
        purge_logs_performed = True

        # Clear logger cache so get_logger() creates fresh handlers
        from scripts.utils.logger import _loggers

        _loggers.pop("ingest", None)

    logger = get_logger("academic_ingest")

    # Configure academic.providers.* loggers to propagate to main logger
    from scripts.utils.logger import configure_child_logger_propagation

    configure_child_logger_propagation("academic_ingest", "academic.providers")

    audit(
        "start",
        {
            "dry_run": config.dry_run,
            "papers_dir": args.papers_dir,
            "batch": args.batch,
            "title": args.title,
            "domain": args.domain,
            "topic": args.topic,
            "authors": args.authors,
            "institution": args.institution,
            "skip_citations": args.skip_citations,
        },
    )
    if purge_logs_performed:
        audit("purge_logs", {"environment": config.environment})

    if args.reset:
        if config.dry_run:
            logger.info("[DRY_RUN] Would reset collections and caches")
        else:
            from scripts.utils.clear_databases import clear_for_ingestion

            logger.info("[RESET] Clearing collections and caches for academic ingestion")
            audit("reset_requested", {"dry_run": False})
            success = clear_for_ingestion(verbose=True, dry_run=False)
            if not success:
                logger.error("Reset failed - aborting academic ingestion")
                audit("reset_failed", {})
                return 1
            audit("reset_complete", {})

    if args.cache_reset:
        ReferenceCache().clear()
        audit("cache_reset", {})

    documents = collect_documents(args)
    if not documents and not args.revalidate:
        logger.error(
            "No documents provided. Use positional args, --papers, --papers-dir, or --batch."
        )
        audit(
            "no_documents",
            {
                "papers_dir": args.papers_dir,
                "batch": args.batch,
                "papers": args.papers,
            },
        )
        return 1

    logger.info(f"Academic ingestion starting. docs={len(documents)}, dry_run={config.dry_run}")
    audit("documents_discovered", {"count": len(documents)})

    # Initialise resource monitoring
    resource_monitor = ResourceMonitor(
        operation_name="academic_ingestion",
        interval=1.0,
        enabled=True,
    )
    resource_monitor.start()
    logger.info("Resource monitoring started")

    cache = ReferenceCache()

    primary_doc_id = compute_doc_id(str(documents[0])) if len(documents) == 1 else None

    # Initialise terminology extraction
    if args.skip_terminology:
        terminology_extractor = None
        terminology_store = None
        logger.info("Skipping terminology extraction (--skip-terminology enabled)")
    else:
        terminology_extractor = DomainTerminologyExtractor()
        terminology_store_path = Path(config.rag_data_path) / "academic_terminology.db"
        terminology_store = DomainTerminologyStore(terminology_store_path)

    # Initialise word frequency extraction for word cloud
    word_freq_extractor = WordFrequencyExtractor(min_word_length=2)
    accumulated_word_freqs: Counter = Counter()
    accumulated_word_doc_counts: Counter = Counter()

    # Vector store setup
    PersistentClient, _using_sqlite = get_vector_client(prefer="chroma")
    vector_path = get_default_vector_path(Path(config.rag_data_path), _using_sqlite)
    client = PersistentClient(path=vector_path)

    # Create collections with no auto-embedding (we provide embeddings manually)
    # This ensures consistent use of EMBEDDING_MODEL_NAME (mxbai-embed-large 1024D)
    # instead of ChromaDB's default 384D model
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    chunk_collection = client.get_or_create_collection(
        name=config.chunk_collection_name,
        embedding_function=None,  # Disable auto-embedding, we provide embeddings
    )
    doc_collection = client.get_or_create_collection(
        name=config.doc_collection_name,
        embedding_function=None,  # Disable auto-embedding, we provide embeddings
    )

    graph = CitationGraph()

    for doc_idx, doc_path in enumerate(documents, 1):
        try:
            import time
            doc_start_time = time.perf_counter()
            
            msg = f"[{doc_idx}/{len(documents)}] Processing: {doc_path.name}"
            print(msg)
            logger.info(msg)
            raw_text = stage_load_document(doc_path, config, logger)
            if not raw_text:
                continue

            # Extract word frequencies for word cloud
            doc_word_freqs = word_freq_extractor.extract_frequencies(raw_text)
            accumulated_word_freqs.update(doc_word_freqs)
            for word in doc_word_freqs:
                accumulated_word_doc_counts[word] += 1

            # Extract domain terminology from document
            doc_id = compute_doc_id(str(doc_path))
            if terminology_extractor:
                doc_terms = terminology_extractor.extract_terms(raw_text, doc_id=doc_id)
                if doc_terms:
                    domain = args.domain or "general"
                    inserted = terminology_store.insert_terms(doc_terms, domain, doc_id)
                    logger.info(
                        f"Extracted {len(doc_terms)} terminology terms ({inserted} new) from {doc_path.name}"
                    )
            # Chunk and store the thesis/paper itself (not just citations)
            # Extract PDF metadata for human-readable doc_id
            preprocess_start = time.perf_counter()
            pdf_metadata = extract_pdf_metadata(str(doc_path))
            thesis_ref = {
                "ref_id": doc_id,
                "title": pdf_metadata.get("title") or args.title or doc_path.stem,
                "authors": pdf_metadata.get("author") or args.authors or "Unknown",
                "year": pdf_metadata.get("year") or None,
                "doi": None,
                "citation": f"{pdf_metadata.get('author') or 'Unknown'} ({pdf_metadata.get('year') or 'n.d.'}). {pdf_metadata.get('title') or doc_path.stem}.",
                "source": "thesis_document",
                "artifact_path": str(doc_path),
            }
            preprocess_time = time.perf_counter() - preprocess_start
            
            # Store thesis document chunks
            ingest_start = time.perf_counter()
            if stage_chunk_and_store(
                thesis_ref, raw_text, chunk_collection, doc_collection, config, logger
            ):
                logger.info(f"Chunked and stored thesis document: {doc_path.name}")
            else:
                logger.warning(f"Failed to chunk/store thesis document: {doc_path.name}")
            ingest_time = time.perf_counter() - ingest_start

            # Skip citation extraction if requested (for faster thesis ingestion)
            if args.skip_citations:
                logger.info(
                    f"Skipping citation extraction for {doc_path.name} (--skip-citations enabled)"
                )
                print(f"  → Skipped citation extraction")
                continue

            citations = stage_extract_citations(raw_text, logger)
            if not citations:
                logger.warning(f"No citations found in {doc_path.name}")
                continue

            msg = f"  → Extracted {len(citations)} citations"
            print(msg)
            logger.info(f"Found {len(citations)} references in {doc_path.name}")
            resolved = stage_resolve_metadata(citations, cache, config, logger)
            msg = f"  → Resolved metadata for {len(resolved)} references"
            print(msg)
            logger.info(f"Metadata resolved for {len(resolved)} references in {doc_path.name}")
            downloaded = stage_download_references(resolved, config, logger)
            msg = f"  → Downloaded/processed {len(downloaded)} artifacts"
            print(msg)
            logger.info(f"Artifacts handled for {len(downloaded)} references in {doc_path.name}")

            stored_count = 0
            for ref_idx, ref in enumerate(downloaded, 1):
                try:
                    artifact = ref.get("artifact_path")
                    if artifact:
                        # Store artifact content if available
                        ref_text = stage_load_reference_text(artifact, logger)
                        if ref_text:
                            # Extract word frequencies for word cloud
                            ref_word_freqs = word_freq_extractor.extract_frequencies(ref_text)
                            accumulated_word_freqs.update(ref_word_freqs)
                            for word in ref_word_freqs:
                                accumulated_word_doc_counts[word] += 1

                            # Extract terminology from reference text
                            ref_doc_id = ref.get("ref_id", "unknown")
                            ref_terms = terminology_extractor.extract_terms(
                                ref_text, doc_id=ref_doc_id
                            )
                            if ref_terms:
                                domain = args.domain or "general"
                                terminology_store.insert_terms(
                                    ref_terms, domain, ref.get("ref_id", "unknown")
                                )

                            if stage_chunk_and_store(
                                ref, ref_text, chunk_collection, doc_collection, config, logger
                            ):
                                stored_count += 1
                    else:
                        # Store raw citation metadata even without artifact
                        # This ensures unresolved references are still searchable
                        citation_text = ref.get("citation", "")
                        title = ref.get("title")
                        if citation_text or title:
                            # Create a minimal chunk from the citation metadata
                            metadata_text = f"{title or citation_text}\n"
                            doi = ref.get("doi")
                            if doi and str(doi).lower() != "none":
                                metadata_text += f"DOI: {doi}\n"
                            year = ref.get("year")
                            if year and str(year).lower() != "none":
                                metadata_text += f"Year: {year}\n"
                            source = ref.get("source")
                            if source and str(source).lower() != "none":
                                metadata_text += f"Source: {source}"

                            if stage_chunk_and_store(
                                ref, metadata_text, chunk_collection, doc_collection, config, logger
                            ):
                                stored_count += 1
                    # Show progress every 50 references or at end
                    if ref_idx % 50 == 0 or ref_idx == len(downloaded):
                        print(
                            f"    Stored {stored_count}/{len(downloaded)} references...", end="\r"
                        )
                except Exception as e:
                    ref_id = ref.get("ref_id", "unknown")
                    title = ref.get("title", "unknown")
                    doi = ref.get("doi", "no-doi")
                    provider = ref.get("source", "unknown-provider")
                    citation = ref.get("citation", "")[:100]
                    logger.error(
                        f"Failed to store reference | "
                        f"ref_id={ref_id} | "
                        f"title={title[:60]}... | "
                        f"doi={doi} | "
                        f"provider={provider} | "
                        f"citation={citation}... | "
                        f"Error: {e}",
                        exc_info=True,
                    )
                    audit(
                        "reference_storage_failed",
                        {
                            "ref_id": ref_id,
                            "title": title,
                            "doi": doi,
                            "provider": provider,
                            "citation": citation,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    continue  # Continue processing other references
            print(f"  → Stored {stored_count}/{len(downloaded)} references")
            logger.info(f"Stored {stored_count} reference artifacts for {doc_path.name}")

            doc_id = compute_doc_id(str(doc_path))

            # Prepare document metadata for graph
            doc_metadata = {
                "title": args.title,
                "authors": args.authors,
                "year": None,
                "source": "document",
            }

            add_references_to_graph(graph, doc_id, resolved, doc_metadata=doc_metadata)
            
            # Record per-document timing
            doc_duration = time.perf_counter() - doc_start_time
            logger.info(f"Completed {doc_path.name} in {doc_duration:.2f}s")
            audit(
                "document_processed",
                {
                    "doc_path": str(doc_path),
                    "doc_id": doc_id,
                    "duration_seconds": doc_duration,
                    "preprocess_time": preprocess_time,
                    "ingest_time": ingest_time,
                },
            )

        except Exception as e:
            logger.error(f"Failed to process document {doc_path}: {e}", exc_info=True)
            audit(
                "document_processing_failed",
                {"doc_path": str(doc_path), "error": str(e), "error_type": type(e).__name__},
            )
            continue  # Continue processing other documents

    # Build citation graph
    # Write to SQLite (with JSON export for backward compatibility)
    graph_db_path = Path(config.rag_data_path) / "academic_citation_graph.db"

    if config.dry_run:
        logger.info(f"[DRY_RUN] Would write citation graph to {graph_db_path}")
    else:
        graph.write_sqlite(
            graph_db_path,
            doc_id=doc_id if len(documents) == 1 else None,
            export_json=True,  # Also export JSON for backward compatibility
        )
        logger.info(f"Citation graph written to {graph_db_path}")
        logger.info(
            f"JSON export written to {graph_db_path.parent / 'academic_citation_graph.json'}"
        )

    # Report terminology extraction results and record candidate terms
    if terminology_extractor is not None:
        vocabulary = terminology_extractor.get_vocabulary()
        domain_str = args.domain or "general"
        top_terms = terminology_store.get_terms_by_domain(
            domain_str,
            limit=20,
            doc_filter=primary_doc_id,
        )

        # Record candidate terms for domain term manager
        try:
            from scripts.rag.domain_terms import (
                DomainType,
                get_domain_term_manager,
                resolve_domain_type,
            )

            domain_type = None
            display_name = None
            if args.domain:
                domain_value, display_name = resolve_domain_type(args.domain)
                if domain_value:
                    try:
                        domain_type = DomainType(domain_value)
                    except ValueError:
                        # Should not happen after resolve_domain_type, but handle gracefully
                        logger.warning(f"Failed to resolve domain type: {args.domain}")
                        domain_type = DomainType.CUSTOM
                else:
                    # resolve_domain_type returns None for empty input
                    domain_type = DomainType.CUSTOM

            if domain_type and top_terms:
                manager = get_domain_term_manager()
                for term_dict in top_terms[:10]:  # Record top 10 as candidates
                    try:
                        term = term_dict.get("term")
                        if term:
                            manager.record_candidate_term(
                                term=term,
                                domain=domain_type,
                                source_doc_id=primary_doc_id,
                                context=f"Relevance: {term_dict.get('relevance', 0):.2f}, Freq: {term_dict.get('frequency', 0)}",
                                frequency_increment=term_dict.get("frequency", 1),
                            )
                    except Exception as e:
                        logger.debug(f"Failed to record candidate term '{term}': {e}")

                domain_display = domain_value if domain_value else "CUSTOM"
                logger.info(
                    f"Recorded {min(10, len(top_terms))} candidate terms for domain {domain_display} (from: {args.domain})"
                )
        except ImportError:
            logger.debug("Domain term manager not available")
        except Exception as e:
            logger.warning(f"Failed to record candidate terms: {e}")
    else:
        vocabulary = {}
        top_terms = []

    # Store word frequencies for word cloud visualisation
    if accumulated_word_freqs:
        if config.dry_run:
            logger.info(
                f"[DRY_RUN] Would store word frequencies for {len(accumulated_word_freqs)} unique words"
            )
            top_words_preview = sorted(
                accumulated_word_freqs.items(), key=lambda x: x[1], reverse=True
            )[:10]
            logger.info(f"[DRY_RUN] Top 10 words: {top_words_preview}")
        else:
            from scripts.utils.db_factory import get_cache_client

            cache_db = get_cache_client(enable_cache=True)
            cache_db.put_word_frequencies(
                dict(accumulated_word_freqs),
                doc_count=dict(accumulated_word_doc_counts),
            )

            # Report word frequency statistics
            word_stats = cache_db.get_word_frequency_stats()
            logger.info(
                f"Word frequency statistics: {word_stats['total_unique_words']} unique words, "
                f"total frequency {word_stats['total_frequency']}, "
                f"avg per word {word_stats['avg_frequency']}"
            )

            # Show top words for word cloud
            top_words = cache_db.get_top_words(limit=20, min_frequency=1)
            if top_words:
                logger.info("Top 20 words for word cloud:")
                for i, (word, freq, doc_count) in enumerate(top_words, 1):
                    logger.info(f"  {i:2d}. {word:30s} freq={freq:5d}, doc_count={doc_count:3d}")

    # BM25 Keyword Indexing (chunks now indexed at chunk-level in stage_chunk_and_store)
    # REFACTORED: Chunks are now indexed individually as they're stored (Option B)
    # This section now only updates corpus stats (IDF values) after all chunks are indexed
    if config.dry_run:
        logger.info("[DRY_RUN] Skipping BM25 corpus stats update")
    elif not config.bm25_indexing_enabled:
        logger.info("BM25 indexing disabled via config")
    else:
        logger.info("Updating BM25 corpus stats for academic artifacts...")
        try:
            from scripts.utils.db_factory import get_cache_client

            cache_db = get_cache_client(enable_cache=True)
            
            # Update corpus stats (IDF values) now that all chunks are indexed
            total_docs = cache_db.get_bm25_corpus_size()
            if total_docs > 0:
                cache_db.update_bm25_corpus_stats(total_docs)
                avg_doc_len = cache_db.get_bm25_avg_doc_length()
                logger.info(
                    f"BM25 corpus stats updated: {total_docs} documents, avg length {avg_doc_len:.1f} tokens"
                )
                audit(
                    "bm25_corpus_stats_updated",
                    {
                        "total_documents": total_docs,
                        "avg_doc_length": avg_doc_len,
                    },
                )

                print("\n  BM25 Keyword Indexing:")
                print(f"    Total corpus size: {total_docs}")
                print(f"    Average chunk length: {avg_doc_len:.0f} tokens")
        except Exception as e:
            logger.warning(f"BM25 corpus stats update failed: {e}")
            audit(
                "bm25_stats_update_failed",
                {"error": str(e)[:200], "error_type": type(e).__name__},
            )

    print("\n" + "=" * 80)
    print("Academic ingestion complete.")
    print("=" * 80)
    print(f"\nDomain Terminology Extraction:")
    print(f"  Total unique terms extracted: {len(vocabulary)}")
    if top_terms:
        print(f"  Top 20 terms for '{domain}':")
        for i, term in enumerate(top_terms[:20], 1):
            print(
                f"    {i:2}. {term['term']:40s} (freq={term['frequency']:3d}, relevance={term['relevance']:.2f})"
            )

    print()
    
    # Stop resource monitoring and export stats
    resource_monitor.stop()
    resource_monitor.print_summary()
    stats_file = resource_monitor.export_json()
    logger.info(f"Resource statistics exported to {stats_file}")
    
    # Calculate and log total duration
    import time
    total_duration = time.perf_counter() - start_time
    print(f"Total time: {total_duration:.2f}s\n")
    
    logger.info(f"Academic ingestion complete in {total_duration:.2f}s")
    logger.info(f"Domain terminology: {len(vocabulary)} unique terms extracted")
    audit(
        "complete",
        {
            "documents": len(documents),
            "dry_run": config.dry_run,
            "terminology_terms": len(vocabulary),
            "total_time_seconds": total_duration,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
