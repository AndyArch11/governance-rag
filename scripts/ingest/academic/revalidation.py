"""Standalone refresh of cached academic reference metadata."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.ingest.academic.cache import Reference as CachedReference
from scripts.ingest.academic.cache import ReferenceCache
from scripts.ingest.academic.downloader import DownloadResult, download_web_content
from scripts.ingest.academic.graph import reference_id_from_metadata
from scripts.ingest.academic.models import RevalidationResult
from scripts.ingest.academic.providers import resolve_reference
from scripts.ingest.academic.providers.base import Reference as ProviderReference
from scripts.utils.logger import audit

ReferenceResolver = Callable[..., tuple[ProviderReference, float]]
WebContentDownloader = Callable[[str, str], DownloadResult]
_LINK_STATUS_PENALTIES = {
    "stale_404": 0.2,
    "stale_timeout": 0.15,
    "stale_moved": 0.05,
}
_REFERENCE_METADATA_MARKER = "\n\nRevalidated reference metadata:\n"


def _prefer_fresh(value: Any, previous: Any) -> Any:
    """Use a refreshed value when supplied, otherwise preserve cached metadata.

    Args:
        value: The potentially refreshed value.
        previous: The previously cached value.

    Returns:
        The value to use, preferring the fresh value if it is not None, empty string, or empty list.
    """
    return value if value not in (None, "", []) else previous


def _link_status_from_download(result: DownloadResult) -> str | None:
    """Map a bounded web download result to the persisted link-health status.

    Args:
        result: The result of a web content download attempt.

    Returns:
        A string representing the link status, or None if it cannot be determined.
    """
    if result.success:
        return "available"
    error = (result.error or "").casefold()
    if error in {"http_404", "http_410"}:
        return "stale_404"
    if error in {"http_301", "http_302", "http_303", "http_307", "http_308", "redirect_page"}:
        return "stale_moved"
    if "timeout" in error or "connection" in error:
        return "stale_timeout"
    return None


def _apply_link_status(reference: CachedReference, link_status: str | None) -> bool:
    """Persist a verified link state and apply its penalty once per stale period.

    Args:
        reference: The cached reference whose link status is to be updated.
        link_status: The new link status to apply.

    Returns:
        True if the link status was updated and a penalty was applied, False otherwise.
    """
    if not link_status or link_status == reference.link_status:
        return False
    if reference.link_status == "available" and link_status in _LINK_STATUS_PENALTIES:
        reference.quality_score = max(
            0.0,
            reference.quality_score - _LINK_STATUS_PENALTIES[link_status],
        )
    reference.link_status = link_status
    return True


def _merge_reference(existing: CachedReference, fresh: ProviderReference) -> CachedReference:
    """Merge refreshed metadata while preserving cache identity and associations.

    Args:
        existing: The cached reference to be updated.
        fresh: The freshly resolved reference metadata.

    Returns:
        A new CachedReference object with merged metadata.
    """
    status = fresh.status.value if hasattr(fresh.status, "value") else str(fresh.status)
    return CachedReference(
        ref_id=existing.ref_id,
        raw_citation=existing.raw_citation,
        doi=_prefer_fresh(fresh.doi, existing.doi),
        title=_prefer_fresh(fresh.title, existing.title),
        authors=_prefer_fresh(fresh.authors, existing.authors),
        year=_prefer_fresh(fresh.year, existing.year),
        abstract=_prefer_fresh(fresh.abstract, existing.abstract),
        venue=_prefer_fresh(fresh.venue, existing.venue),
        venue_type=_prefer_fresh(fresh.venue_type, existing.venue_type),
        volume=_prefer_fresh(fresh.volume, existing.volume),
        issue=_prefer_fresh(fresh.issue, existing.issue),
        pages=_prefer_fresh(fresh.pages, existing.pages),
        reference_type=_prefer_fresh(fresh.reference_type, existing.reference_type),
        resolved=True,
        status=status,
        quality_score=fresh.quality_score or existing.quality_score,
        metadata_provider=_prefer_fresh(fresh.metadata_provider, existing.metadata_provider),
        oa_available=fresh.oa_available or existing.oa_available,
        oa_url=_prefer_fresh(fresh.oa_url, existing.oa_url),
        link_status=existing.link_status,
        citation_count=(
            fresh.citation_count if fresh.citation_count is not None else existing.citation_count
        ),
        doc_ids=existing.doc_ids,
        resolved_at=fresh.resolved_at or datetime.now(timezone.utc),
    )


def _reference_embedding_text(reference: CachedReference) -> str:
    """Build concise, current bibliographic text for reference summary embeddings.
    Args:
        reference: The CachedReference object to build the embedding text for.

    Returns:
        A string containing the concise bibliographic text for the reference.
    """
    fields = [
        f"Title: {reference.title}" if reference.title else "",
        f"Authors: {', '.join(reference.authors)}" if reference.authors else "",
        f"Year: {reference.year}" if reference.year is not None else "",
        f"DOI: {reference.doi}" if reference.doi else "",
        f"Venue: {reference.venue}" if reference.venue else "",
        f"Volume: {reference.volume}" if reference.volume else "",
        f"Issue: {reference.issue}" if reference.issue else "",
        f"Pages: {reference.pages}" if reference.pages else "",
        f"Reference type: {reference.reference_type}" if reference.reference_type else "",
        f"Metadata provider: {reference.metadata_provider}" if reference.metadata_provider else "",
        f"Link status: {reference.link_status}" if reference.link_status else "",
        (
            f"Citation count: {reference.citation_count}"
            if reference.citation_count is not None
            else ""
        ),
        f"Open access available: {'yes' if reference.oa_available else 'no'}",
        f"Open access URL: {reference.oa_url}" if reference.oa_url else "",
        f"Quality score: {reference.quality_score:.2f}",
        f"Abstract: {reference.abstract[:1200]}" if reference.abstract else "",
        f"Citation: {reference.raw_citation}" if reference.raw_citation else "",
    ]
    return "\n".join(field for field in fields if field)


def _reference_vector_metadata(
    reference: CachedReference,
    existing_metadata: dict[str, Any],
    summary_text: str,
) -> dict[str, Any]:
    """Merge refreshed bibliographic fields into stored Chroma metadata.

    Args:
        reference: The CachedReference object containing the refreshed bibliographic fields.
        existing_metadata: The existing metadata dictionary to be updated.

    Returns:
        A dictionary containing the merged metadata.
    """
    metadata = dict(existing_metadata)
    refreshed_fields: dict[str, Any] = {
        "ref_id": reference.ref_id,
        "title": reference.title,
        "authors": json.dumps(reference.authors) if reference.authors else None,
        "year": reference.year,
        "doi": reference.doi,
        "abstract": reference.abstract,
        "venue": reference.venue,
        "venue_type": reference.venue_type,
        "reference_type": reference.reference_type,
        "link_status": reference.link_status,
        "quality_score": reference.quality_score,
        "metadata_provider": reference.metadata_provider,
        "citation_count": reference.citation_count,
        "oa_available": reference.oa_available,
        "oa_url": reference.oa_url,
        "summary": summary_text,
    }
    metadata.update({key: value for key, value in refreshed_fields.items() if value is not None})
    return metadata


def _refresh_reference_embeddings(
    reference: CachedReference,
    chunk_collection: Any | None,
    doc_collection: Any | None,
) -> int:
    """Refresh document summaries and metadata-only chunks for one reference.

    Args:
        reference: The CachedReference object to refresh embeddings for.
        chunk_collection: The collection containing metadata-only chunks.
        doc_collection: The collection containing document summaries.

    Returns:
        The total number of embeddings refreshed for the reference.
    """
    embedding_text = _reference_embedding_text(reference)
    if not embedding_text:
        return 0

    document_records = (
        doc_collection.get(
            where={"ref_id": reference.ref_id},
            include=["documents", "metadatas"],
        )
        if doc_collection is not None
        else {}
    )
    doc_ids = document_records.get("ids", [])
    doc_metadata_rows = document_records.get("metadatas", [])
    doc_text_rows = document_records.get("documents", [])
    doc_documents: list[str] = []
    doc_update_ids: list[str] = []
    doc_metadatas: list[dict[str, Any]] = []
    for index, metadata in enumerate(doc_metadata_rows):
        existing_metadata = metadata if isinstance(metadata, dict) else {}
        existing_text = (
            doc_text_rows[index]
            if index < len(doc_text_rows) and isinstance(doc_text_rows[index], str)
            else ""
        )
        metadata_only = str(existing_metadata.get("source") or "").startswith(
            "reference_metadata:"
        )
        if metadata_only:
            updated_text = embedding_text
        else:
            summary_without_previous_refresh = existing_text.partition(
                _REFERENCE_METADATA_MARKER
            )[0].strip()
            updated_text = (
                f"{summary_without_previous_refresh}{_REFERENCE_METADATA_MARKER}{embedding_text}"
                if summary_without_previous_refresh
                else embedding_text
            )
        if updated_text != existing_text:
            doc_update_ids.append(doc_ids[index])
            doc_documents.append(updated_text)
            doc_metadatas.append(
                _reference_vector_metadata(reference, existing_metadata, updated_text)
            )

    chunk_records = (
        chunk_collection.get(
            where={"ref_id": reference.ref_id},
            include=["documents", "metadatas"],
        )
        if chunk_collection is not None
        else {}
    )
    metadata_chunk_rows: list[tuple[str, dict[str, Any]]] = []
    chunk_documents: list[str] = []
    for chunk_id, document, metadata in zip(
        chunk_records.get("ids", []),
        chunk_records.get("documents", []),
        chunk_records.get("metadatas", []),
    ):
        if not isinstance(metadata, dict):
            continue
        if (
            str(metadata.get("source") or "").startswith("reference_metadata:")
            and metadata.get("chunk_type") in (None, "child")
            and document != embedding_text
        ):
            metadata_chunk_rows.append((chunk_id, metadata))
            chunk_documents.append(embedding_text)

    total_embeddings = len(doc_update_ids) + len(metadata_chunk_rows)
    if not total_embeddings:
        return 0

    document_count = len(doc_update_ids)
    texts_to_embed = doc_documents + chunk_documents
    from scripts.ingest.vectors import generate_chunk_embeddings_batch

    embeddings, _ = generate_chunk_embeddings_batch(texts_to_embed, batch_size=1)
    if not embeddings:
        return 0

    if doc_collection is not None and doc_update_ids:
        doc_collection.update(
            ids=doc_update_ids,
            documents=doc_documents,
            embeddings=embeddings[:document_count],
            metadatas=doc_metadatas,
        )

    if chunk_collection is not None and metadata_chunk_rows:
        chunk_ids = [chunk_id for chunk_id, _ in metadata_chunk_rows]
        chunk_collection.update(
            ids=chunk_ids,
            documents=[embedding_text] * len(chunk_ids),
            embeddings=embeddings[document_count:],
            metadatas=[
                _reference_vector_metadata(reference, metadata, embedding_text)
                for _, metadata in metadata_chunk_rows
            ],
        )

    audit(
        "ingest",
        "reference_embeddings_refreshed",
        {
            "ref_id": reference.ref_id,
            "document_records": len(doc_ids),
            "metadata_chunk_records": len(metadata_chunk_rows),
        },
    )
    return total_embeddings


def _update_citation_graph_node(
    graph_path: Path,
    previous: CachedReference,
    refreshed: CachedReference,
    confidence: float | None,
) -> None:
    """Refresh citation-node fields in place, retaining its ID and all edges.

    Args:
        graph_path: Path to the SQLite graph database file.
        previous: The previous cached reference metadata.
        refreshed: The refreshed cached reference metadata.
        confidence: The confidence score for the refreshed metadata.

    Returns:
        None.
    """
    if not graph_path.is_file():
        return

    node_id = reference_id_from_metadata(
        {
            "doi": previous.doi,
            "title": previous.title,
            "citation": previous.raw_citation,
        }
    )
    authors = json.dumps(refreshed.authors) if refreshed.authors else None
    with sqlite3.connect(graph_path) as connection:
        connection.execute(
            """
            UPDATE nodes
            SET title = COALESCE(?, title),
                authors = COALESCE(?, authors),
                doi = COALESCE(?, doi),
                year = COALESCE(?, year),
                year_verified = CASE WHEN ? IS NULL THEN year_verified ELSE 1 END,
                reference_type = COALESCE(?, reference_type),
                quality_score = COALESCE(?, quality_score),
                link_status = ?,
                venue_type = COALESCE(?, venue_type),
                source = COALESCE(?, source),
                confidence = COALESCE(?, confidence)
            WHERE node_id = ? AND node_type = 'reference'
            """,
            (
                refreshed.title,
                authors,
                refreshed.doi,
                refreshed.year,
                refreshed.year,
                refreshed.reference_type,
                refreshed.quality_score,
                previous.link_status,
                refreshed.venue_type,
                refreshed.metadata_provider,
                confidence,
                node_id,
            ),
        )


def _report_progress(
    items_done: int,
    items_total: int,
    succeeded: int,
    failed: int,
) -> None:
    """Write a run-correlated checkpoint consumed by the Pipelines tab.

    Args:
        items_done: The number of items that have been processed so far.
        items_total: The total number of items to be processed.
        succeeded: The number of items that have been successfully processed.
        failed: The number of items that have failed processing.

    Returns:
        None.
    """
    audit(
        "ingest",
        "progress_checkpoint",
        {
            "stage": "reference_revalidation",
            "items_done": items_done,
            "items_total": items_total,
            "percent": 100.0 * items_done / items_total if items_total else 100.0,
            "succeeded": succeeded,
            "failed": failed,
        },
    )


def revalidate_cached_references(
    cache: ReferenceCache,
    citation_graph_path: Path,
    mode: str,
    staleness_threshold_days: int = 30,
    ref_ids: list[str] | None = None,
    logger: logging.Logger | None = None,
    resolver: ReferenceResolver = resolve_reference,
    web_content_dir: Path | None = None,
    web_downloader: WebContentDownloader = download_web_content,
    thesis_id: str | None = None,
    chunk_collection: Any | None = None,
    doc_collection: Any | None = None,
) -> RevalidationResult:
    """Refresh selected cached references without re-ingesting source documents.

    Args:
        cache: The reference cache containing cached references.
        citation_graph_path: Path to the SQLite citation graph database file.
        mode: The revalidation mode to use.
        staleness_threshold_days: The number of days after which a cached reference is considered stale.
        ref_ids: Optional list of specific reference IDs to revalidate.
        logger: Optional logger for logging progress and errors.
        resolver: The reference resolver function to use for fetching fresh metadata.
        web_content_dir: Directory for bounded web-content link checks.
        web_downloader: Downloader used to check eligible online references.
        thesis_id: Optional source document ID used to limit references to one thesis.

    Returns:
        A RevalidationResult object summarizing the outcome of the revalidation process.
    """
    candidates = cache.select_for_revalidation(
        mode,
        staleness_threshold_days=staleness_threshold_days,
        ref_ids=ref_ids,
        thesis_id=thesis_id,
    )
    started_at = time.perf_counter()
    selected_ref_ids = {reference.ref_id for _, reference in candidates}
    embedding_records_refreshed = 0
    embedding_refresh_failures = 0
    if chunk_collection is not None or doc_collection is not None:
        resolved_references = cache.select_for_revalidation(
            "all",
            staleness_threshold_days=staleness_threshold_days,
            thesis_id=thesis_id,
        )
        audit(
            "ingest",
            "reference_embedding_backfill_started",
            {"candidate_count": len(resolved_references), "thesis_id": thesis_id},
        )
        for _, cached_reference in resolved_references:
            if cached_reference.ref_id in selected_ref_ids:
                continue
            try:
                embedding_records_refreshed += _refresh_reference_embeddings(
                    cached_reference,
                    chunk_collection,
                    doc_collection,
                )
            except Exception as exc:
                embedding_refresh_failures += 1
                if logger:
                    logger.warning(
                        "Reference embedding backfill failed for ref_id=%s: %s",
                        cached_reference.ref_id,
                        exc,
                    )
                audit(
                    "ingest",
                    "reference_embedding_refresh_failed",
                    {
                        "ref_id": cached_reference.ref_id,
                        "failure_reason": type(exc).__name__,
                    },
                )

    total = len(candidates)
    updated = 0
    unchanged = 0
    failed = 0
    details: dict[str, list[dict[str, Any]]] = {
        "updated": [],
        "unchanged": [],
        "failed": [],
    }
    progress_interval = max(1, (total + 99) // 100)
    _report_progress(0, total, 0, 0)
    audit("ingest", "reference_revalidation_started", {"mode": mode, "total": total})

    for index, (cache_key, existing) in enumerate(candidates, start=1):
        fresh: ProviderReference | None = None
        confidence = 0.0
        resolution_error: Exception | None = None
        try:
            fresh, confidence = resolver(
                existing.raw_citation,
                year=existing.year,
                doi=existing.doi,
                logger=logger,
            )
        except Exception as exc:
            resolution_error = exc

        checked_link_status = None
        check_url = (fresh.oa_url if fresh and fresh.oa_url else None) or existing.oa_url
        reference_type = (
            fresh.reference_type if fresh and fresh.resolved else existing.reference_type
        )
        if (
            web_content_dir is not None
            and reference_type in {"news", "blog", "online"}
            and check_url
        ):
            try:
                download_result = web_downloader(check_url, str(web_content_dir))
                checked_link_status = _link_status_from_download(download_result)
                if checked_link_status:
                    audit(
                        "ingest",
                        "reference_link_checked",
                        {"ref_id": existing.ref_id, "link_status": checked_link_status},
                    )
            except Exception as exc:
                if logger:
                    logger.warning(
                        "Reference link check failed for ref_id=%s: %s",
                        existing.ref_id,
                        exc,
                    )

        try:
            if fresh is None or not fresh.resolved:
                if _apply_link_status(existing, checked_link_status):
                    cache.put(cache_key, existing)
                    _update_citation_graph_node(
                        citation_graph_path,
                        existing,
                        existing,
                        None,
                    )
                failed += 1
                details["failed"].append({"ref_id": existing.ref_id})
                if resolution_error and logger:
                    logger.warning(
                        "Reference revalidation failed for ref_id=%s: %s",
                        existing.ref_id,
                        resolution_error,
                    )
            else:
                refreshed = _merge_reference(existing, fresh)
                _apply_link_status(refreshed, checked_link_status)
                changed = any(
                    getattr(existing, field) != getattr(refreshed, field)
                    for field in (
                        "doi",
                        "title",
                        "authors",
                        "year",
                        "abstract",
                        "venue",
                        "venue_type",
                        "citation_count",
                        "oa_available",
                        "oa_url",
                        "link_status",
                        "quality_score",
                    )
                )
                embedding_records_refreshed += _refresh_reference_embeddings(
                    refreshed,
                    chunk_collection,
                    doc_collection,
                )
                cache.put(cache_key, refreshed)
                _update_citation_graph_node(
                    citation_graph_path,
                    existing,
                    refreshed,
                    confidence,
                )
                result_key = "updated" if changed else "unchanged"
                details[result_key].append({"ref_id": existing.ref_id})
                if changed:
                    updated += 1
                else:
                    unchanged += 1
        except Exception as exc:
            failed += 1
            details["failed"].append({"ref_id": existing.ref_id})
            if logger:
                logger.warning(
                    "Reference revalidation failed for ref_id=%s: %s",
                    existing.ref_id,
                    exc,
                )

        if index % progress_interval == 0 or index == total:
            _report_progress(index, total, updated + unchanged, failed)

    result = RevalidationResult(
        total=total,
        updated=updated,
        unchanged=unchanged,
        failed=failed,
        details=details,
        duration_sec=time.perf_counter() - started_at,
    )
    audit(
        "ingest",
        "reference_revalidation_complete",
        {
            "mode": mode,
            "total": result.total,
            "updated": result.updated,
            "unchanged": result.unchanged,
            "failed": result.failed,
            "embedding_records_refreshed": embedding_records_refreshed,
            "embedding_refresh_failures": embedding_refresh_failures,
            "duration_sec": result.duration_sec,
        },
    )
    if logger:
        logger.info(result.summary().strip())
    return result
