"""Academic paper ingestion pipeline entrypoint (initial scaffold).

TODO: Incorporate more of utils modules (e.g. resource monitor, retry_utils, etc)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable when running as a script path, e.g.:
# python scripts/ingest/ingest_academic.py --help
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ingest.academic.cache import Reference as CachedReference
from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.config import AcademicIngestConfig, get_academic_config
from scripts.ingest.academic.downloader import download_reference_pdf, download_web_content
from scripts.ingest.academic.graph import CitationGraph, add_references_to_graph
from scripts.ingest.academic.parser import DOI_PATTERN, extract_citations
from scripts.ingest.academic.providers import resolve_reference
from scripts.ingest.academic.providers.base import Reference
from scripts.ingest.academic.revalidation import revalidate_cached_references
from scripts.ingest.academic.terminology import DomainTerminologyExtractor, DomainTerminologyStore
from scripts.ingest.bm25_indexing import index_chunks_in_bm25
from scripts.ingest.chunk import chunk_text, create_parent_child_chunks
from scripts.ingest.htmlparser import extract_text_from_html
from scripts.ingest.ingest_utils import compute_doc_id, compute_file_hash
from scripts.ingest.pdfparser import (
    extract_pdf_metadata,
    extract_pdf_text_and_figures,
    extract_structure_from_text,
    extract_text_from_pdf,
    map_text_to_structure,
    validate_structure_against_toc,
)
from scripts.ingest.vectors import (
    delete_document_chunks,
    store_child_chunks,
    store_chunks_in_chroma,
    store_parent_chunks,
)
from scripts.ingest.word_frequency import WordFrequencyExtractor
from scripts.search.bm25_search import BM25Search
from scripts.thesis_graph.cultural_lenses import load_cultural_lens_profile
from scripts.thesis_graph.figure_assessment import assess_figure_locally
from scripts.thesis_graph.thesis_registry import build_and_register_thesis_graph
from scripts.utils import logger as logger_utils
from scripts.utils.clear_databases import clear_for_ingestion
from scripts.utils.config import BaseConfig
from scripts.utils.db_factory import get_cache_client, get_default_vector_path, get_vector_client
from scripts.utils.embedding_model_config import EMBEDDING_MODEL_NAME
from scripts.utils.logger import _loggers, configure_child_logger_propagation, create_module_logger
from scripts.utils.resource_monitor import ResourceMonitor

# Use shared "ingest" logger so all ingest activities are in the same log file
get_logger, audit = create_module_logger("ingest")


def _report_ingestion_progress(
    stage: str,
    items_done: int,
    items_total: int,
    succeeded: int,
    failed: int,
    skipped: int,
) -> None:
    """Write a run-correlated checkpoint for ingestion-stage progress."""
    audit(
        "progress_checkpoint",
        {
            "stage": stage,
            "items_done": items_done,
            "items_total": items_total,
            "percent": 100.0 * items_done / items_total if items_total else 100.0,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
        },
    )


DomainType: Any = None
get_domain_term_manager: Any = None
resolve_domain_type: Any = None
try:
    from scripts.rag.domain_terms import DomainType as _DomainType
    from scripts.rag.domain_terms import get_domain_term_manager as _get_domain_term_manager
    from scripts.rag.domain_terms import resolve_domain_type as _resolve_domain_type
except ImportError:
    pass
else:
    DomainType = _DomainType
    get_domain_term_manager = _get_domain_term_manager
    resolve_domain_type = _resolve_domain_type


def validate_thesis_chunk_structure(
    chunks: List[Dict[str, Any]],
    text: str,
    document_structure: List[Dict[str, Any]],
    min_coverage: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that thesis chunks have complete, non-crossing structural spans.

    Args:
        chunks: List of chunk dictionaries with "source_start" and "source_end".
        text: Full document text.
        document_structure: List of chapter entries with "chapter" and "start_pos".
        min_coverage: Minimum required coverage of chunks mapped to chapters.

    Returns:
        Tuple of (is_valid, details_dict) where is_valid is a boolean indicating
        if the chunks are valid, and details_dict contains validation metrics.
    """
    if not chunks or not document_structure:
        return False, {"reason": "No chunks or document structure available."}

    mapped_chunks = 0
    invalid_spans = 0
    cross_chapter_chunks = 0
    for chunk in chunks:
        source_start = chunk.get("source_start")
        source_end = chunk.get("source_end")
        if (
            not isinstance(source_start, int)
            or not isinstance(source_end, int)
            or source_start < 0
            or source_end <= source_start
            or source_end > len(text)
            or text[source_start:source_end] != chunk.get("text")
        ):
            invalid_spans += 1
            continue

        start_meta = map_text_to_structure(text, document_structure, source_start, source_start + 1)
        end_meta = map_text_to_structure(text, document_structure, source_end - 1, source_end)
        if start_meta.get("chapter"):
            mapped_chunks += 1
        if start_meta.get("chapter") != end_meta.get("chapter"):
            cross_chapter_chunks += 1

    total_chunks = len(chunks)
    coverage = mapped_chunks / total_chunks
    valid = invalid_spans == 0 and cross_chapter_chunks == 0 and coverage >= min_coverage
    toc_validation = validate_structure_against_toc(text, document_structure)
    return valid, {
        "total_chunks": total_chunks,
        "mapped_chunks": mapped_chunks,
        "chapter_coverage": coverage,
        "invalid_spans": invalid_spans,
        "cross_chapter_chunks": cross_chapter_chunks,
        "min_coverage": min_coverage,
        "toc_validation": toc_validation,
    }


def _format_toc_validation_warning(
    toc_validation: Dict[str, Any],
    minimum_coverage: float,
) -> Optional[str]:
    """Return a warning when ToC chapter extraction is incomplete or out of order.

    Args:
        toc_validation: Dictionary containing ToC validation results.
        minimum_coverage: Minimum required coverage for ToC chapters.

    Returns:
        A warning string if validation issues are found, otherwise None.
    """
    if not toc_validation.get("toc_present"):
        return None

    coverage = toc_validation.get("coverage")
    missing_chapters = toc_validation.get("missing_chapters", [])
    unexpected_chapters = toc_validation.get("unexpected_chapters", [])
    order_matches = toc_validation.get("order_matches")
    below_threshold = not isinstance(coverage, (int, float)) or coverage < minimum_coverage
    if not (below_threshold or missing_chapters or unexpected_chapters or order_matches is False):
        return None

    coverage_text = f"{coverage:.1%}" if isinstance(coverage, (int, float)) else "unknown"
    return (
        f"Table of Contents validation is incomplete: coverage={coverage_text} "
        f"(minimum={minimum_coverage:.1%}), order_matches={order_matches}, "
        f"missing={missing_chapters}, unexpected={unexpected_chapters}"
    )


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
    parser.add_argument(
        "--cultural-lens",
        type=str,
        default=None,
        help="Apply a cultural lens profile by ID (draft profiles are Dev-only)",
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
    parser.add_argument(
        "--thesis-id",
        type=str,
        default=None,
        help="Limit reference revalidation to citations from this thesis document ID",
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
    if args.refresh:
        overrides["ACADEMIC_INGEST_REPLACE_EXISTING_THESIS"] = True

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


def stage_load_document(
    path: Path,
    config: AcademicIngestConfig,
    logger,
    figures_out: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Load a document from the given path, ensuring it meets the criteria for processing.

    Args:
        path: Path to the document
        config: Configuration for academic ingestion (e.g. max PDF size)
        logger: Logger for logging messages
        figures_out: Optional list to receive figures from the same Docling conversion.

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
        if figures_out is not None:
            extracted_text, figures = extract_pdf_text_and_figures(str(path))
            figures_out.extend(figures)
            return extracted_text
        return extract_text_from_pdf(str(path))
    except Exception as exc:
        logger.error(f"Failed to extract text from {path}: {type(exc).__name__}: {exc}")
        return None


def _figure_reference_contexts(
    source_text: str,
    figure_data: Dict[str, Any],
    max_contexts: int = 3,
    context_chars: int = 420,
) -> List[str]:
    """Collect bounded prose excerpts around explicit figure references.

    Args:
        source_text: The full text of the document.
        figure_data: Metadata about the figure, including caption and caption number.
        max_contexts: Maximum number of context excerpts to return.
        context_chars: Maximum number of characters for each context excerpt.

    Returns:
        A list of context excerpts surrounding figure references.
    """
    caption_number = str(figure_data.get("caption_number") or "").strip()
    if not caption_number:
        return []

    reference_pattern = re.compile(
        rf"\b(?:figure|fig\.?)\s*{re.escape(caption_number)}(?![\w.])",
        re.IGNORECASE,
    )
    caption = str(figure_data.get("caption") or "")
    caption_spans = (
        [match.span() for match in re.finditer(re.escape(caption), source_text, re.IGNORECASE)]
        if caption
        else []
    )
    contexts = []
    for match in reference_pattern.finditer(source_text):
        if any(start <= match.start() < end for start, end in caption_spans):
            continue

        line_start = source_text.rfind("\n", 0, match.start()) + 1
        line_end = source_text.find("\n", match.end())
        if line_end < 0:
            line_end = len(source_text)
        source_line = source_text[line_start:line_end].strip()
        if (
            "|" in source_line
            or re.search(r"\.{2,}\s*\d+\s*$", source_line)
            or (
                re.match(
                    r"^\s*(?:fig(?:ure)?\.?)\s*\d+(?:\.\d+)*\s*[:.)-]\s*",
                    source_line,
                    re.IGNORECASE,
                )
                and not re.search(
                    r"\b(?:shows?|demonstrates?|illustrates?|depicts?|indicates?|suggests?)\b",
                    source_line,
                    re.IGNORECASE,
                )
            )
        ):
            continue

        paragraph_start = source_text.rfind("\n\n", 0, match.start())
        paragraph_start = 0 if paragraph_start < 0 else paragraph_start + 2
        paragraph_end = source_text.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(source_text)
        excerpt = source_text[paragraph_start:paragraph_end].strip()
        if len(excerpt) > context_chars:
            match_offset = match.start() - paragraph_start
            excerpt_start = max(0, match_offset - context_chars // 3)
            excerpt = excerpt[excerpt_start : excerpt_start + context_chars]
        excerpt = " ".join(excerpt.split())
        if excerpt and excerpt not in contexts:
            contexts.append(excerpt)
        if len(contexts) >= max_contexts:
            break
    return contexts


def _assess_thesis_figures(
    figures: List[Dict[str, Any]],
    thesis_text: str,
    cultural_lens_profile: Optional[Dict[str, Any]],
    config: AcademicIngestConfig,
    logger,
) -> None:
    """Assess thesis figures locally and flag profile-declared sensitivities for review.

    Args:
        figures: List of figures extracted from the thesis
        thesis_text: Full text of the thesis
        cultural_lens_profile: Optional cultural lens profile containing sensitivities
        config: Configuration for academic ingestion
        logger: Logger for logging messages
    """

    for figure in figures:
        visible_text = " ".join(
            str(figure.get(field) or "") for field in ("caption", "alt_text")
        ).casefold()
        sensitivity_ids = []
        image_sensitivity_ids_by_term: Dict[str, List[str]] = {}
        if cultural_lens_profile:
            for sensitivity in cultural_lens_profile.get("sensitivities", []):
                applies_to = set(sensitivity.get("applies_to") or ["images", "captions"])
                if not {"images", "captions"}.intersection(applies_to):
                    continue
                sensitivity_id = str(sensitivity.get("id", "sensitivity"))
                for term in sensitivity.get("detection_terms", []):
                    term_text = str(term).strip()
                    if term_text and "images" in applies_to:
                        image_sensitivity_ids_by_term.setdefault(term_text, []).append(
                            sensitivity_id
                        )
                if any(
                    str(term).casefold() in visible_text
                    for term in sensitivity.get("detection_terms", [])
                    if term
                ):
                    sensitivity_ids.append(sensitivity_id)

        if sensitivity_ids:
            figure["vision_status"] = "flagged_for_human_review"
            figure["sensitivity_ids"] = sensitivity_ids
            audit(
                "vision_figure_flagged",
                {
                    "figure_number": figure.get("figure_number"),
                    "sensitivity_ids": sensitivity_ids,
                    "automated_description_skipped": True,
                },
            )
            continue

        if not figure.get("image_bytes") and figure.get("image") is None:
            image_sensitivity_ids = list(
                dict.fromkeys(
                    sensitivity_id
                    for term_sensitivity_ids in image_sensitivity_ids_by_term.values()
                    for sensitivity_id in term_sensitivity_ids
                )
            )
            figure["vision_status"] = (
                "flagged_for_human_review" if image_sensitivity_ids else "image_unavailable"
            )
            figure.pop("description", None)
            if image_sensitivity_ids:
                figure["sensitivity_ids"] = image_sensitivity_ids
            figure["vision_assessment"] = {
                "human_review_required": True,
                "assessment_status": "image_unavailable",
                "sensitivity_screen": "image_unavailable",
                "sensitivity_ids": image_sensitivity_ids,
            }
            logger.warning(
                f"Figure {figure.get('figure_number')} has no local image data; "
                "vision assessment skipped"
            )
            audit(
                "vision_figure_image_unavailable",
                {
                    "figure_number": figure.get("figure_number"),
                    "sensitivity_ids": image_sensitivity_ids,
                    "human_review_required": True,
                },
            )
            continue

        caption = str(figure.get("caption") or "")
        context = ""
        if caption:
            caption_match = re.search(re.escape(caption), thesis_text, re.IGNORECASE)
            if caption_match:
                caption_start, caption_end = caption_match.span()
                context_before = thesis_text[max(0, caption_start - 600) : caption_start].strip()
                context_after = thesis_text[caption_end : caption_end + 1200].strip()
                context = "\n".join(part for part in (context_before, context_after) if part)
        reference_contexts = _figure_reference_contexts(thesis_text, figure)
        if reference_contexts:
            context_parts = [
                f"Explicit figure reference {index}: {excerpt}"
                for index, excerpt in enumerate(reference_contexts, 1)
            ]
            if context:
                context_parts.append(f"Caption-adjacent context: {context[:300]}")
            context = "\n\n".join(context_parts)
        try:
            assessment = assess_figure_locally(
                figure,
                model=config.vision_model_name,
                ollama_host=config.ollama_host,
                nearby_context=context,
                sensitivity_terms=list(image_sensitivity_ids_by_term),
                timeout=config.vision_assessment_timeout,
            )
            usage = assessment.get("token_usage", {})
            matched_sensitivity_ids = list(
                dict.fromkeys(
                    sensitivity_id
                    for term in assessment.get("sensitivity_term_matches", [])
                    for sensitivity_id in image_sensitivity_ids_by_term.get(str(term), [])
                )
            )
            incomplete_screen_ids = (
                list(
                    dict.fromkeys(
                        sensitivity_id
                        for sensitivity_ids_for_term in image_sensitivity_ids_by_term.values()
                        for sensitivity_id in sensitivity_ids_for_term
                    )
                )
                if assessment.get("sensitivity_screen_status") == "incomplete"
                else []
            )
            flagged_sensitivity_ids = matched_sensitivity_ids or incomplete_screen_ids
            visual_review_cues = assessment.get("non_text_visual_review_cues", [])
            if flagged_sensitivity_ids:
                sensitivity_screen = (
                    "legible_image_text"
                    if matched_sensitivity_ids
                    else "incomplete_legible_image_text"
                )
                figure["vision_status"] = "flagged_for_human_review"
                figure["sensitivity_ids"] = flagged_sensitivity_ids
                figure["vision_assessment"] = {
                    "human_review_required": True,
                    "sensitivity_screen": sensitivity_screen,
                    "sensitivity_ids": flagged_sensitivity_ids,
                    "token_usage": usage,
                }
                if visual_review_cues:
                    figure["vision_assessment"].update(
                        {
                            "visual_content_review": "possible_non_text_cue",
                            "non_text_visual_review_cues": visual_review_cues,
                        }
                    )
                figure.pop("description", None)
                audit(
                    "vision_figure_flagged",
                    {
                        "figure_number": figure.get("figure_number"),
                        "sensitivity_ids": flagged_sensitivity_ids,
                        "sensitivity_source": sensitivity_screen,
                        "automated_description_skipped": True,
                    },
                )
            elif visual_review_cues:
                visual_assessment = {
                    key: value
                    for key, value in assessment.items()
                    if key not in {"description", "findings_or_claims"}
                }
                visual_assessment["visual_content_review"] = "possible_non_text_cue"
                visual_assessment["human_review_required"] = True
                figure["vision_status"] = "flagged_for_human_review"
                figure["vision_assessment"] = visual_assessment
                figure.pop("description", None)
                audit(
                    "vision_figure_visual_review",
                    {
                        "figure_number": figure.get("figure_number"),
                        "visual_review_cues": visual_review_cues,
                        "automated_description_skipped": True,
                    },
                )
            else:
                figure["vision_assessment"] = assessment
                figure["description"] = assessment["description"]
                figure["vision_status"] = "human_review_required"
            audit(
                "llm_usage",
                {
                    "operation": "vision.figure_assessment",
                    "component": "academic_ingestion",
                    "model": config.vision_model_name,
                    "figure_number": figure.get("figure_number"),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "token_source": usage.get("token_source", "unavailable"),
                    "latency_ms": int(usage.get("total_duration_ns", 0) / 1_000_000),
                    "success": True,
                },
            )
        except Exception as exc:
            estimated_input_tokens = max(
                1,
                (len(caption) + len(str(figure.get("alt_text") or "")) + len(context)) // 4,
            )
            figure["vision_status"] = "assessment_failed"
            figure["assessment_error_type"] = type(exc).__name__
            error_detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            logger.warning(
                f"Local figure assessment failed for figure {figure.get('figure_number')} "
                f"({type(exc).__name__}: {error_detail})"
            )
            audit(
                "llm_usage",
                {
                    "operation": "vision.figure_assessment",
                    "component": "academic_ingestion",
                    "model": config.vision_model_name,
                    "figure_number": figure.get("figure_number"),
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": 0,
                    "token_source": "estimated",
                    "success": False,
                    "failure_reason": type(exc).__name__,
                },
            )


def stage_store_figure_chunks(
    chunk_collection,
    *,
    thesis_id: str,
    source_path: str,
    file_hash: str,
    figures: List[Dict[str, Any]],
    logger,
) -> int:
    """Embed figure text as typed thesis chunks for semantic retrieval.

    Args:
        chunk_collection: Collection to store the figure chunks
        thesis_id: ID of the thesis
        source_path: Path to the source file
        file_hash: Hash of the source file
        figures: List of figures extracted from the thesis
        logger: Logger for logging messages

    Returns:
        int: Number of figure chunks created
    """
    figure_chunks: List[Dict[str, Any]] = []
    for figure in figures:
        assessment = figure.get("vision_assessment") or {}
        text_parts = [
            f"Figure {figure.get('figure_number', '?')}",
            f"Caption: {figure.get('caption')}" if figure.get("caption") else "",
            f"Alt text: {figure.get('alt_text')}" if figure.get("alt_text") else "",
            f"Visual description: {figure.get('description')}" if figure.get("description") else "",
            (
                f"Figure type: {assessment.get('figure_type')}"
                if assessment.get("figure_type")
                else ""
            ),
            (
                f"Caption alignment: {assessment.get('caption_alignment')}"
                if assessment.get("caption_alignment")
                else ""
            ),
            (
                f"Caption-description alignment: {assessment.get('caption_description_alignment')}"
                if assessment.get("caption_description_alignment")
                else ""
            ),
            (
                f"Alt-text-description alignment: {assessment.get('alt_text_description_alignment')}"
                if assessment.get("alt_text_description_alignment")
                else ""
            ),
            (
                f"Body-text discussion: {assessment.get('body_text_discussion')}"
                if assessment.get("body_text_discussion")
                else ""
            ),
            (
                f"Body-text evidence: {assessment.get('body_text_evidence')}"
                if assessment.get("body_text_evidence")
                else ""
            ),
            (
                f"Findings or claims: {assessment.get('findings_or_claims')}"
                if assessment.get("findings_or_claims")
                else ""
            ),
            (
                f"Limitations: {assessment.get('limitations')}"
                if assessment.get("limitations")
                else ""
            ),
        ]
        figure_text = "\n".join(part for part in text_parts if part)
        if not figure_text.strip():
            continue
        figure_number = int(figure.get("figure_number") or len(figure_chunks) + 1)
        figure_chunks.append(
            {
                "id": f"figure_{figure_number}",
                "text": figure_text,
                "metadata": {
                    "figure_id": f"figure_{figure_number}",
                    "figure_number": figure_number,
                    "caption_number": str(figure.get("caption_number") or ""),
                    "page_number": (
                        int(figure["page_number"]) if figure.get("page_number") is not None else -1
                    ),
                    "chapter": str(figure.get("chapter") or ""),
                    "section_title": str(figure.get("section_title") or ""),
                    "heading_path": str(figure.get("heading_path") or ""),
                    "figure_source_start": int(figure.get("source_start", -1)),
                    "figure_source_end": int(figure.get("source_end", -1)),
                    "vision_status": str(figure.get("vision_status") or "pending"),
                },
            }
        )

    if not figure_chunks:
        return 0

    base_metadata = {
        "doc_id": thesis_id,
        "source": source_path,
        "source_kind": "thesis_document",
        "thesis_id": thesis_id,
        "version": 1,
        "hash": file_hash,
        "doc_type": "academic_reference",
        "source_category": "academic_reference",
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    try:
        store_child_chunks(
            doc_id=thesis_id,
            child_chunks=figure_chunks,
            chunk_collection=chunk_collection,
            base_metadata=base_metadata,
            doc_type="academic_reference",
            chunk_type="figure",
        )
        audit(
            "thesis_figure_chunks_stored",
            {"thesis_id": thesis_id, "figure_chunk_count": len(figure_chunks)},
        )
        return len(figure_chunks)
    except Exception as exc:
        logger.warning(
            f"Failed to store searchable figure chunks for {thesis_id}: {type(exc).__name__}"
        )
        audit(
            "thesis_figure_chunks_failed",
            {"thesis_id": thesis_id, "error_type": type(exc).__name__},
        )
        return 0


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
    resolved = []
    total = len(citations)
    succeeded = 0
    failed = 0
    skipped = 0
    _report_ingestion_progress("reference_resolution", 0, total, 0, 0, 0)
    for cit_idx, citation in enumerate(citations, 1):
        # Show progress every 50 citations or at end
        if cit_idx % 50 == 0 or cit_idx == len(citations):
            print(f"    Resolving metadata {cit_idx}/{len(citations)}...", end="\r")

        # Try cache first
        cached_ref = cache.get(citation)
        if cached_ref:
            record = {
                "citation": citation,
                "ref_id": cached_ref.ref_id,
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
            if cached_ref.resolved:
                succeeded += 1
            else:
                failed += 1
            if cit_idx % 25 == 0 or cit_idx == total:
                _report_ingestion_progress(
                    "reference_resolution", cit_idx, total, succeeded, failed, skipped
                )
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
            doi_match = DOI_PATTERN.search(citation)
            year_match = re.search(r"\b(?:19|20)\d{2}[a-z]?\b", citation)
            doi = doi_match.group(0).rstrip(".,;:-") if doi_match else None
            year = int(year_match.group(0)[:4]) if year_match else None
            ref, confidence = resolve_reference(
                citation,
                year=year,
                doi=doi,
                logger=logger,
            )

        # Cache uses its own SQLite persistence model. Convert the provider
        # result explicitly so provider enums and future fields cannot leak
        # across the storage boundary.
        cache.put(
            citation,
            CachedReference(
                ref_id=ref.ref_id,
                raw_citation=ref.raw_citation,
                doi=ref.doi,
                title=ref.title,
                authors=ref.authors,
                year=ref.year,
                abstract=ref.abstract,
                venue=ref.venue,
                venue_type=ref.venue_type,
                volume=ref.volume,
                issue=ref.issue,
                pages=ref.pages,
                reference_type=ref.reference_type,
                resolved=ref.resolved,
                status=str(ref.status.value if hasattr(ref.status, "value") else ref.status),
                quality_score=ref.quality_score,
                metadata_provider=ref.metadata_provider,
                oa_available=ref.oa_available,
                oa_url=ref.oa_url,
                link_status=ref.link_status,
                citation_count=ref.citation_count,
                doc_ids=ref.doc_ids,
                resolved_at=ref.resolved_at,
            ),
        )

        # Convert to dict for downstream use
        resolved_record: Dict[str, Any] = {
            "citation": citation,
            "ref_id": ref.ref_id,
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
        resolved.append(resolved_record)
        if config.dry_run:
            skipped += 1
        elif ref.resolved:
            succeeded += 1
        else:
            failed += 1
        if cit_idx % 25 == 0 or cit_idx == total:
            _report_ingestion_progress(
                "reference_resolution", cit_idx, total, succeeded, failed, skipped
            )

    logger.info(f"Resolved {len(resolved)} references")
    return resolved


def record_document_citations(
    cache: ReferenceCache,
    doc_id: str,
    references: List[Dict[str, Any]],
) -> int:
    """Record resolved reference IDs against the document that cites them.
    Args:
        cache: ReferenceCache instance for storing citation links
        doc_id: The ID of the document containing the citations
        references: List of resolved reference dicts

    Returns:
        The number of successfully linked references
    """
    linked_count = 0
    for reference in references:
        ref_id = reference.get("ref_id")
        citation = reference.get("citation")
        if not isinstance(ref_id, str) or not ref_id.strip():
            continue
        if not isinstance(citation, str) or not citation.strip():
            continue
        cache.add_citation(doc_id, ref_id, citation)
        linked_count += 1
    return linked_count


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
    total = len(references)
    succeeded = 0
    failed = 0
    skipped = 0
    _report_ingestion_progress("reference_download", 0, total, 0, 0, 0)
    if config.dry_run:
        logger.info("Dry run enabled: skipping downloads")
        updated = []
        for ref in references:
            ref_copy = dict(ref)
            ref_copy["download_status"] = "skipped"
            updated.append(ref_copy)
        _report_ingestion_progress("reference_download", total, total, 0, 0, total)
        return updated

    updated = []
    for ref_idx, ref in enumerate(references, 1):
        ref = dict(ref)
        ref_type = ref.get("reference_type")
        url = ref.get("url")
        pdf_url = ref.get("pdf_url")

        if pdf_url:
            result = download_reference_pdf(pdf_url, config.cache_dir, config.max_pdf_size_mb)
            ref["artifact_path"] = result.path
            ref["download_status"] = "success" if result.success else result.error or "failed"
        elif ref_type in ("news", "blog", "online") and url:
            result = download_web_content(url, config.cache_dir)
            ref["artifact_path"] = result.path
            ref["download_status"] = "success" if result.success else result.error or "failed"
        else:
            ref["download_status"] = "skipped"

        updated.append(ref)
        if ref["download_status"] == "success":
            succeeded += 1
        elif ref["download_status"] == "skipped":
            skipped += 1
        else:
            failed += 1
        if ref_idx % 25 == 0 or ref_idx == total:
            _report_ingestion_progress(
                "reference_download", ref_idx, total, succeeded, failed, skipped
            )

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
    reference_authors = reference.get("authors")
    first_reference_author = (
        reference_authors[0]
        if isinstance(reference_authors, list) and reference_authors
        else reference_authors if isinstance(reference_authors, str) else None
    )
    author = pdf_metadata.get("author") or first_reference_author
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

    # The source thesis supplies a canonical ref_id shared with its citation
    # graph. Preserve it so assessment, chunks, and citation graph agree.
    if reference.get("ref_id") and reference.get("source") == "thesis_document":
        doc_id = str(reference["ref_id"])
    elif doc_id_parts:
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
        file_hash = hashlib.sha256(citation.encode()).hexdigest()
        source_path = f"reference_metadata:{reference.get('doi') or doc_id}"

    metadata: Dict[str, Any] = {
        "doc_type": "academic_reference",
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "summary_scores": {"overall": 0},  # Dict format for store_chunks_in_chroma
        "key_topics": [],  # Empty topics list (could be enhanced later)
        "source_category": "academic_reference",
        "source_kind": reference.get("source", ""),
        "display_name": display_name,
    }
    if reference.get("ref_id"):
        metadata["ref_id"] = str(reference["ref_id"])
    if reference.get("source") == "thesis_document":
        metadata["thesis_id"] = doc_id
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
    parent_chunks: Optional[List[Dict[str, Any]]] = None
    child_chunks: Optional[List[Dict[str, Any]]] = None
    using_parent_child = bool(getattr(config, "enable_parent_child_chunking", True))

    if using_parent_child:
        try:
            # create_parent_child_chunks returns (child_chunks, parent_chunks)
            chunk_kwargs: Dict[str, Any] = {"text": text, "doc_type": "academic_reference"}
            if document_structure:
                chunk_kwargs["document_structure"] = document_structure
            child_chunks, parent_chunks = create_parent_child_chunks(**chunk_kwargs)
            logger.debug(
                f"Created {len(child_chunks)} child chunks and {len(parent_chunks)} parent chunks for {doc_id}"
            )
        except Exception as e:
            logger.warning(f"Parent-child chunking failed: {e}")
            parent_chunks = None
            child_chunks = None

    if reference.get("source") == "thesis_document" and child_chunks:
        min_coverage = float(getattr(config, "thesis_structure_min_coverage", 0.95))
        is_valid, structure_report = validate_thesis_chunk_structure(
            child_chunks,
            text,
            document_structure or [],
            min_coverage,
        )
        logger.info(f"Thesis structural metadata validation: {structure_report}")
        toc_warning = _format_toc_validation_warning(
            structure_report.get("toc_validation", {}),
            float(getattr(config, "thesis_toc_min_coverage", 0.95)),
        )
        if toc_warning:
            logger.warning(toc_warning)
        audit("thesis_structure_validation", {"doc_id": doc_id, **structure_report})
        if not is_valid:
            logger.error(f"Thesis chunk structure validation failed: {structure_report}")
            return False

        if getattr(config, "replace_existing_thesis", False) and not config.dry_run:
            delete_document_chunks(doc_id, chunk_collection)
            doc_collection.delete(where={"doc_id": doc_id})
            cache_db = get_cache_client(rag_data_path=Path(config.rag_data_path), enable_cache=True)
            cache_db.delete_bm25_document(doc_id)
            cache_db.close()
            logger.info(
                f"Replaced existing thesis chunks, document record, and BM25 index for {doc_id}"
            )

    if config.dry_run:
        title = reference.get("title") or reference.get("citation", "")[:100]
        doi = reference.get("doi") or "no-doi"
        source = reference.get("source") or "unknown-provider"
        if parent_chunks:
            logger.info(
                f"[DRY_RUN] Would store {len(child_chunks) if child_chunks else 0} child chunks and {len(parent_chunks)} parent chunks for {doc_id} (title: {title[:50]}... doi: {doi} provider: {source})"
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
        # Store child chunks before parent chunks.
        if parent_chunks:
            base_metadata = {
                "doc_id": doc_id,
                "source": source_path,
                "source_kind": reference.get("source", ""),
                "thesis_id": doc_id if reference.get("source") == "thesis_document" else "",
                "version": 1,
                "hash": file_hash,
                "doc_type": "academic_reference",
                "source_category": "academic_reference",
                "embedding_model": EMBEDDING_MODEL_NAME,
            }
            if reference.get("ref_id"):
                base_metadata["ref_id"] = str(reference["ref_id"])

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
                        document_structure=document_structure,
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
                    full_text=text,
                    doc_type="academic_reference",
                    document_structure=document_structure,
                )
                logger.debug(f"Stored {len(parent_chunks)} parent chunks for {doc_id}")
            except Exception as parent_err:
                logger.warning(f"Failed to store parent chunks for {doc_id}: {parent_err}")
                # Non-fatal: parent storage shouldn't block the entire ingest

        # BM25 Keyword Indexing for chunks (at chunk-level granularity)
        # Uses common indexing utility function for consistency across modules
        if config.bm25_indexing_enabled:
            try:
                cache_db = get_cache_client(
                    rag_data_path=Path(config.rag_data_path),
                    enable_cache=True,
                )
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

        logs_dir = logger_utils.LOGS_DIR
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

        # Clear the shared ingest logger cache so it creates a fresh file handler.
        _loggers.pop("ingest", None)

    logger = get_logger()

    cultural_lens_profile = None
    if args.cultural_lens:
        try:
            cultural_lens_profile = load_cultural_lens_profile(
                args.cultural_lens,
                Path(config.rag_data_path) / "cultural_lenses",
                config.environment,
            )
        except (OSError, ValueError) as exc:
            logger.error(f"Cultural lens profile rejected: {exc}")
            return 1

    # Configure academic.providers.* loggers to propagate to main logger
    configure_child_logger_propagation("ingest", "academic.providers")

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
            "cultural_lens_profile": args.cultural_lens,
        },
    )
    if purge_logs_performed:
        audit("purge_logs", {"environment": config.environment})

    if args.revalidate:
        has_document_input = bool(
            args.papers_positional or args.papers or args.papers_dir or args.batch
        )
        if args.reset or args.cache_reset or args.refresh or config.dry_run or has_document_input:
            logger.error(
                "Reference revalidation cannot be combined with ingestion, reset, refresh, or dry-run options."
            )
            audit("reference_revalidation_rejected", {"reason": "conflicting_options"})
            return 1
        if args.revalidate == "ids" and not args.ref_ids:
            logger.error("--ref-ids is required when --revalidate ids is selected.")
            audit("reference_revalidation_rejected", {"reason": "missing_ref_ids"})
            return 1
        if args.revalidate != "ids" and args.ref_ids:
            logger.error("--ref-ids can only be used with --revalidate ids.")
            audit("reference_revalidation_rejected", {"reason": "unexpected_ref_ids"})
            return 1

        rag_data_path = Path(config.rag_data_path)
        reference_cache_path = rag_data_path / "academic_references.db"
        if not reference_cache_path.is_file():
            logger.error("Reference cache not found: %s", reference_cache_path)
            audit("reference_revalidation_failed", {"reason": "cache_not_found"})
            return 1

        chunk_collection = None
        doc_collection = None
        try:
            vector_client_class, using_sqlite = get_vector_client(prefer="chroma")
            vector_client = vector_client_class(
                path=get_default_vector_path(rag_data_path, using_sqlite)
            )
            try:
                chunk_collection = vector_client.get_collection(name=config.chunk_collection_name)
            except Exception as exc:
                logger.warning("Reference chunk collection unavailable for embedding refresh: %s", exc)
            try:
                doc_collection = vector_client.get_collection(name=config.doc_collection_name)
            except Exception as exc:
                logger.warning("Reference document collection unavailable for embedding refresh: %s", exc)
        except Exception as exc:
            logger.warning("Vector store unavailable for reference embedding refresh: %s", exc)

        try:
            result = revalidate_cached_references(
                ReferenceCache(str(reference_cache_path)),
                rag_data_path / "academic_citation_graph.db",
                args.revalidate,
                args.staleness_threshold,
                args.ref_ids,
                logger,
                web_content_dir=Path(config.cache_dir),
                thesis_id=args.thesis_id,
                chunk_collection=chunk_collection,
                doc_collection=doc_collection,
            )
        except (OSError, ValueError) as exc:
            logger.error("Reference revalidation could not start: %s", exc)
            audit(
                "reference_revalidation_failed",
                {"reason": type(exc).__name__},
            )
            return 1
        return 1 if result.failed else 0

    if args.reset:
        if config.dry_run:
            logger.info("[DRY_RUN] Would reset collections and caches")
        else:
            logger.info("[RESET] Clearing collections and caches for academic ingestion")
            audit("reset_requested", {"dry_run": False})
            success = clear_for_ingestion(verbose=True, dry_run=False, config=config)
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
    chunk_collection = client.get_or_create_collection(
        name=config.chunk_collection_name,
        embedding_function=None,  # Disable auto-embedding, we provide embeddings
    )
    doc_collection = client.get_or_create_collection(
        name=config.doc_collection_name,
        embedding_function=None,  # Disable auto-embedding, we provide embeddings
    )

    graph = CitationGraph()
    ingested_theses: Dict[str, Dict[str, Any]] = {}

    for doc_idx, doc_path in enumerate(documents, 1):
        try:
            doc_start_time = time.perf_counter()

            msg = f"[{doc_idx}/{len(documents)}] Processing: {doc_path.name}"
            print(msg)
            logger.info(msg)
            thesis_figures: List[Dict[str, Any]] = []
            raw_text = stage_load_document(
                doc_path,
                config,
                logger,
                figures_out=thesis_figures,
            )
            if not raw_text:
                continue
            if thesis_figures:
                if config.dry_run:
                    for figure in thesis_figures:
                        figure["vision_status"] = "dry_run_skipped"
                elif config.figure_assessment_enabled:
                    _assess_thesis_figures(
                        thesis_figures,
                        raw_text,
                        cultural_lens_profile,
                        config,
                        logger,
                    )
                else:
                    for figure in thesis_figures:
                        figure["vision_status"] = "disabled"

            # Extract word frequencies for word cloud
            doc_word_freqs = word_freq_extractor.extract_frequencies(raw_text)
            accumulated_word_freqs.update(doc_word_freqs)
            for word in doc_word_freqs:
                accumulated_word_doc_counts[word] += 1

            # Extract domain terminology from document
            doc_id = compute_doc_id(str(doc_path))
            if terminology_extractor is not None and terminology_store is not None:
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
                if thesis_figures and not config.dry_run:
                    figure_chunk_count = stage_store_figure_chunks(
                        chunk_collection,
                        thesis_id=doc_id,
                        source_path=str(doc_path),
                        file_hash=compute_file_hash(str(doc_path)),
                        figures=thesis_figures,
                        logger=logger,
                    )
                    if figure_chunk_count:
                        logger.info(
                            f"Stored {figure_chunk_count} searchable figure chunks for {doc_path.name}"
                        )
                ingested_theses[doc_id] = {
                    "title": thesis_ref["title"],
                    "authors": thesis_ref["authors"],
                    "source_path": doc_path,
                    "figures": thesis_figures,
                }
                if thesis_figures:
                    logger.info(
                        f"Extracted {len(thesis_figures)} figure assets from {doc_path.name}"
                    )
                    audit(
                        "thesis_figures_extracted",
                        {"thesis_id": doc_id, "figure_count": len(thesis_figures)},
                    )
                graph.add_document(
                    doc_id,
                    metadata={
                        "title": thesis_ref["title"],
                        "authors": thesis_ref["authors"],
                        "year": thesis_ref["year"],
                        "source": "document",
                    },
                )
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
            if not config.dry_run:
                linked_count = record_document_citations(cache, doc_id, resolved)
                audit(
                    "document_citations_recorded",
                    {"doc_id": doc_id, "reference_count": linked_count},
                )
            downloaded = stage_download_references(resolved, config, logger)
            msg = f"  → Downloaded/processed {len(downloaded)} artifacts"
            print(msg)
            logger.info(f"Artifacts handled for {len(downloaded)} references in {doc_path.name}")

            stored_count = 0
            storage_succeeded = 0
            storage_failed = 0
            storage_skipped = 0
            _report_ingestion_progress("reference_storage", 0, len(downloaded), 0, 0, 0)
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
                            if terminology_extractor is not None and terminology_store is not None:
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
                                storage_succeeded += 1
                            else:
                                storage_skipped += 1
                        else:
                            storage_skipped += 1
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
                                storage_succeeded += 1
                            else:
                                storage_skipped += 1
                        else:
                            storage_skipped += 1
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
                    storage_failed += 1
                if ref_idx % 25 == 0 or ref_idx == len(downloaded):
                    _report_ingestion_progress(
                        "reference_storage",
                        ref_idx,
                        len(downloaded),
                        storage_succeeded,
                        storage_failed,
                        storage_skipped,
                    )
            print(f"  → Stored {stored_count}/{len(downloaded)} references")
            logger.info(f"Stored {stored_count} reference artifacts for {doc_path.name}")

            doc_id = compute_doc_id(str(doc_path))

            # Prepare document metadata for graph
            doc_metadata = {
                "title": thesis_ref.get("title") or args.title,
                "authors": thesis_ref.get("authors") or args.authors,
                "year": thesis_ref.get("year"),
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
    elif graph.nodes:
        graph.write_sqlite(
            graph_db_path,
            doc_id=primary_doc_id if len(documents) == 1 else None,
            export_json=True,  # Also export JSON for backward compatibility
        )
        logger.info(f"Citation graph written to {graph_db_path}")
        logger.info(
            f"JSON export written to {graph_db_path.parent / 'academic_citation_graph.json'}"
        )

        for thesis_id, thesis_metadata in ingested_theses.items():
            try:
                raw_authors = thesis_metadata["authors"]
                authors = (
                    [str(author) for author in raw_authors]
                    if isinstance(raw_authors, list)
                    else [str(raw_authors)] if raw_authors else []
                )
                thesis_graph = build_and_register_thesis_graph(
                    chunk_collection,
                    thesis_id=thesis_id,
                    title=str(thesis_metadata["title"] or thesis_id),
                    authors=authors,
                    source_path=Path(thesis_metadata["source_path"]),
                    graphs_dir=Path(config.rag_data_path) / "thesis_graphs",
                    registry_path=Path(config.rag_data_path) / "thesis_graphs" / "registry.sqlite",
                    cultural_lens_profile=cultural_lens_profile,
                    figures=thesis_metadata.get("figures", []),
                    citation_doc_node_id=(thesis_id if thesis_id in graph.nodes else None),
                )
                logger.info(
                    f"Built thesis evidence graph for {thesis_id}: "
                    f"nodes={thesis_graph.node_count}, edges={thesis_graph.edge_count}"
                )
                audit(
                    "thesis_graph_built",
                    {
                        "thesis_id": thesis_id,
                        "graph_path": str(thesis_graph.output_path),
                        "node_count": thesis_graph.node_count,
                        "edge_count": thesis_graph.edge_count,
                    },
                )
            except Exception as thesis_graph_error:
                logger.error(
                    f"Failed to build thesis graph for {thesis_id}: {thesis_graph_error}",
                    exc_info=True,
                )
                audit(
                    "thesis_graph_build_failed",
                    {"thesis_id": thesis_id, "error": str(thesis_graph_error)},
                )
    elif documents:
        logger.error("No document text was ingested; existing citation graph was left unchanged")
        audit(
            "citation_graph_write_skipped",
            {"reason": "no_document_text_ingested", "document_count": len(documents)},
        )

    # Report terminology extraction results and record candidate terms
    if terminology_extractor is not None and terminology_store is not None:
        vocabulary = terminology_extractor.get_vocabulary()
        domain_str = args.domain or "general"
        top_terms = terminology_store.get_terms_by_domain(
            domain_str,
            limit=20,
            doc_filter=primary_doc_id,
        )

        # Record candidate terms for domain term manager
        try:
            if DomainType is None or get_domain_term_manager is None or resolve_domain_type is None:
                raise ImportError("Domain term manager not available")

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
            cache_db = get_cache_client(
                rag_data_path=Path(config.rag_data_path),
                enable_cache=True,
            )

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
        print(f"  Top 20 terms for '{domain_str}':")
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

    return 1 if documents and not graph.nodes else 0


if __name__ == "__main__":
    raise SystemExit(main())
