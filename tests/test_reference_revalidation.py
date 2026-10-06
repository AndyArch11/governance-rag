"""Tests for standalone academic reference revalidation."""

import json
import sqlite3
from math import isclose

import pytest

from scripts.ingest.academic import revalidation
from scripts.ingest.academic.cache import Reference, ReferenceCache
from scripts.ingest.academic.citation_graph_schema import ensure_schema
from scripts.ingest.academic.downloader import DownloadResult
from scripts.ingest.academic.graph import reference_id_from_metadata
from scripts.ingest.academic.providers.base import Reference as ProviderReference
from scripts.ingest.academic.providers.base import ReferenceStatus
from scripts.ingest.academic.revalidation import revalidate_cached_references


def _seed_cache(cache: ReferenceCache) -> None:
    records = [
        Reference(
            ref_id="online-stale",
            raw_citation="Online article (2020)",
            title="Online article",
            authors=["Original Author"],
            year=2020,
            reference_type="online",
            resolved=True,
            status="resolved",
            link_status="stale_404",
            doc_ids=["doc-1"],
        ),
        Reference(
            ref_id="online-fresh",
            raw_citation="Fresh article (2021)",
            title="Fresh article",
            reference_type="online",
            resolved=True,
            status="resolved",
        ),
        Reference(
            ref_id="academic",
            raw_citation="Journal article (2022)",
            title="Journal article",
            reference_type="academic",
            resolved=True,
            status="resolved",
        ),
        Reference(
            ref_id="failed",
            raw_citation="Unresolved citation",
            title=None,
            reference_type="online",
            resolved=False,
            status="unresolved",
        ),
    ]
    for reference in records:
        cache.put(f"key-{reference.ref_id}", reference)

    with cache._get_connection() as connection:
        connection.execute(
            "UPDATE cached_references SET cached_at = datetime('now', '-31 days') "
            "WHERE ref_id = 'online-stale'"
        )
        connection.commit()


def test_reference_cache_selects_documented_revalidation_modes(tmp_path):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)

    def selected_ids(mode: str, ref_ids: list[str] | None = None) -> set[str]:
        return {
            reference.ref_id
            for _, reference in cache.select_for_revalidation(
                mode,
                staleness_threshold_days=30,
                ref_ids=ref_ids,
            )
        }

    assert selected_ids("stale") == {"online-stale"}
    assert selected_ids("online") == {"online-stale", "online-fresh", "failed"}
    assert selected_ids("all") == {"online-stale", "online-fresh", "academic"}
    assert selected_ids("failed") == {"failed"}
    assert selected_ids("ids", ["academic", "failed"]) == {"academic", "failed"}


def test_reference_cache_scopes_revalidation_to_citing_thesis(tmp_path):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    cache.add_citation("thesis-1", "online-stale", "Online article (2020)")
    cache.add_citation("thesis-2", "academic", "Journal article (2022)")

    thesis_one_refs = cache.select_for_revalidation("all", thesis_id="thesis-1")
    thesis_two_refs = cache.select_for_revalidation("all", thesis_id="thesis-2")
    unlinked_refs = cache.select_for_revalidation("all", thesis_id="thesis-3")

    assert {reference.ref_id for _, reference in thesis_one_refs} == {"online-stale"}
    assert {reference.ref_id for _, reference in thesis_two_refs} == {"academic"}
    assert unlinked_refs == []


def test_revalidation_service_passes_thesis_scope_to_cache(tmp_path, monkeypatch):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    cache.add_citation("thesis-1", "online-stale", "Online article (2020)")
    cache.add_citation("thesis-2", "academic", "Journal article (2022)")
    resolved_ids = []

    def resolve(citation_text, year=None, doi=None, logger=None):
        resolved_ids.append(citation_text)
        return (
            ProviderReference(
                ref_id="refreshed",
                raw_citation=citation_text,
                title="Refreshed title",
                resolved=True,
                status=ReferenceStatus.RESOLVED,
            ),
            0.9,
        )

    result = revalidate_cached_references(
        cache,
        tmp_path / "missing-graph.db",
        "all",
        resolver=resolve,
        thesis_id="thesis-1",
    )

    assert result.total == 1
    assert resolved_ids == ["Online article (2020)"]


def test_failed_mode_backfills_embeddings_for_previously_resolved_references(
    tmp_path, monkeypatch
):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    refreshed_ids = []
    monkeypatch.setattr(
        revalidation,
        "_refresh_reference_embeddings",
        lambda reference, _chunks, _documents: refreshed_ids.append(reference.ref_id) or 1,
    )

    def unresolved(citation_text, year=None, doi=None, logger=None):
        return (
            ProviderReference(
                ref_id="unresolved",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
            ),
            0.0,
        )

    result = revalidate_cached_references(
        cache,
        tmp_path / "missing-graph.db",
        "failed",
        resolver=unresolved,
        chunk_collection=object(),
        doc_collection=object(),
    )

    assert result.total == 1
    assert result.failed == 1
    assert set(refreshed_ids) == {"online-stale", "online-fresh", "academic"}
    assert "failed" not in refreshed_ids


def test_revalidation_updates_metadata_without_changing_ids_or_edges(tmp_path, monkeypatch):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    graph_path = tmp_path / "academic_citation_graph.db"
    old_reference = {
        "doi": None,
        "title": "Online article",
        "citation": "Online article (2020)",
    }
    old_node_id = reference_id_from_metadata(old_reference)
    with sqlite3.connect(graph_path) as connection:
        ensure_schema(connection)
        connection.execute(
            "INSERT INTO nodes (node_id, node_type, title) VALUES (?, 'document', ?)",
            ("doc-1", "Citing thesis"),
        )
        connection.execute(
            """
            INSERT INTO nodes
                (node_id, node_type, title, authors, reference_type, link_status)
            VALUES (?, 'reference', ?, ?, ?, ?)
            """,
            (
                old_node_id,
                "Online article",
                json.dumps(["Original Author"]),
                "online",
                "stale_404",
            ),
        )
        connection.execute(
            "INSERT INTO edges (source, target, relation) VALUES (?, ?, 'cites')",
            ("doc-1", old_node_id),
        )

    progress_events = []
    monkeypatch.setattr(
        revalidation,
        "audit",
        lambda module, event, data: progress_events.append((event, data)),
    )

    def resolve(citation_text, year=None, doi=None, logger=None):
        return (
            ProviderReference(
                ref_id="new-provider-id",
                raw_citation=citation_text,
                doi="10.1234/refreshed",
                title="Refreshed online article",
                authors=["Updated Author"],
                year=2024,
                reference_type="online",
                resolved=True,
                status=ReferenceStatus.RESOLVED,
                quality_score=0.9,
                metadata_provider="crossref",
            ),
            0.95,
        )

    result = revalidate_cached_references(
        cache,
        graph_path,
        "ids",
        ref_ids=["online-stale"],
        resolver=resolve,
    )

    assert result.total == 1
    assert result.updated == 1
    assert result.failed == 0
    refreshed = cache.select_for_revalidation("ids", ref_ids=["online-stale"])[0][1]
    assert refreshed.ref_id == "online-stale"
    assert refreshed.doc_ids == ["doc-1"]
    assert refreshed.title == "Refreshed online article"
    assert refreshed.link_status == "stale_404"
    with sqlite3.connect(graph_path) as connection:
        node = connection.execute(
            "SELECT node_id, title, doi, link_status FROM nodes WHERE node_id = ?",
            (old_node_id,),
        ).fetchone()
        edge = connection.execute(
            "SELECT source, target FROM edges WHERE source = 'doc-1'"
        ).fetchone()
    assert node == (old_node_id, "Refreshed online article", "10.1234/refreshed", "stale_404")
    assert edge == ("doc-1", old_node_id)
    checkpoint = [data for event, data in progress_events if event == "progress_checkpoint"][-1]
    assert checkpoint["items_done"] == 1
    assert checkpoint["items_total"] == 1
    assert checkpoint["succeeded"] == 1


def test_revalidation_reembeds_reference_summary_and_metadata_chunk(tmp_path, monkeypatch):
    from scripts.ingest import vectors

    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)

    class FakeCollection:
        def __init__(self, records):
            self.records = records
            self.updates = []

        def get(self, *, where, include):
            matching = [
                record
                for record in self.records
                if record["metadatas"][0].get("ref_id") == where["ref_id"]
            ]
            return {
                key: [record[key][0] for record in matching]
                for key in ("ids", "documents", "metadatas")
            }

        def update(self, **kwargs):
            self.updates.append(kwargs)

    old_metadata = {
        "ref_id": "online-stale",
        "source": "/references/article.pdf",
        "doc_id": "Original_Author_2020_Online_article",
    }
    document_collection = FakeCollection(
        [{"ids": ["Original_Author_2020_Online_article_v1"], "documents": ["old summary"], "metadatas": [old_metadata]}]
    )
    assert document_collection.get(
        where={"ref_id": "online-stale"}, include=["documents", "metadatas"]
    )["metadatas"][0]["source"] == "/references/article.pdf"
    chunk_collection = FakeCollection(
        [
            {
                "ids": ["Original_Author_2020_Online_article-chunk-0"],
                "documents": ["Online article\nYear: 2020"],
                "metadatas": [
                    {
                        **old_metadata,
                        "source": "reference_metadata:10.1234/old",
                        "chunk_type": "child",
                    }
                ],
            },
            {
                "ids": ["Original_Author_2020_Online_article-chunk-1"],
                "documents": ["Downloaded article body"],
                "metadatas": {
                    0: {**old_metadata, "source": "/references/article.pdf", "chunk_type": "child"}
                },
            },
        ]
    )

    embedding_inputs = []

    def fake_generate_embeddings(texts, **_kwargs):
        embedding_inputs.extend(texts)
        return [[float(len(text)), 0.2, 0.3] for text in texts], {}

    monkeypatch.setattr(
        vectors,
        "generate_chunk_embeddings_batch",
        fake_generate_embeddings,
    )

    def resolve(citation_text, year=None, doi=None, logger=None):
        return (
            ProviderReference(
                ref_id="provider-id",
                raw_citation=citation_text,
                doi="10.1234/new",
                title="Refreshed article",
                authors=["Updated Author"],
                year=2024,
                abstract="A refreshed abstract.",
                reference_type="online",
                resolved=True,
                status=ReferenceStatus.RESOLVED,
                quality_score=0.9,
                metadata_provider="crossref",
            ),
            0.95,
        )

    result = revalidate_cached_references(
        cache,
        tmp_path / "missing-graph.db",
        "ids",
        ref_ids=["online-stale"],
        resolver=resolve,
        chunk_collection=chunk_collection,
        doc_collection=document_collection,
    )

    assert result.updated == 1
    assert len(embedding_inputs) == 2
    assert all("Title: Refreshed article" in text for text in embedding_inputs)
    assert all("DOI: 10.1234/new" in text for text in embedding_inputs)
    assert document_collection.updates[0]["ids"] == ["Original_Author_2020_Online_article_v1"]
    assert "old summary" in document_collection.updates[0]["documents"][0], document_collection.updates
    assert "Title: Refreshed article" in document_collection.updates[0]["documents"][0]
    assert document_collection.updates[0]["embeddings"] == [
        [float(len(document_collection.updates[0]["documents"][0])), 0.2, 0.3]
    ]
    metadata_chunk_update = next(
        update for update in chunk_collection.updates if "documents" in update
    )
    assert metadata_chunk_update["ids"] == ["Original_Author_2020_Online_article-chunk-0"]
    assert "DOI: 10.1234/new" in metadata_chunk_update["documents"][0]
    assert metadata_chunk_update["embeddings"] == [
        [float(len(metadata_chunk_update["documents"][0])), 0.2, 0.3]
    ]


def test_failed_revalidation_preserves_cached_metadata_and_reports_failure(tmp_path, monkeypatch):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    progress_events = []
    monkeypatch.setattr(
        revalidation,
        "audit",
        lambda module, event, data: progress_events.append((event, data)),
    )

    def unresolved(citation_text, year=None, doi=None, logger=None):
        return (
            ProviderReference(
                ref_id="unresolved",
                raw_citation=citation_text,
                resolved=False,
                status=ReferenceStatus.UNRESOLVED,
            ),
            0.0,
        )

    result = revalidate_cached_references(
        cache,
        tmp_path / "missing-graph.db",
        "ids",
        ref_ids=["online-stale"],
        resolver=unresolved,
    )

    assert result.total == 1
    assert result.failed == 1
    preserved = cache.select_for_revalidation("ids", ref_ids=["online-stale"])[0][1]
    assert preserved.title == "Online article"
    assert preserved.doc_ids == ["doc-1"]
    checkpoint = [data for event, data in progress_events if event == "progress_checkpoint"][-1]
    assert checkpoint["failed"] == 1


def test_revalidation_checks_online_link_and_applies_404_status(tmp_path):
    cache = ReferenceCache(str(tmp_path / "references.sqlite"))
    _seed_cache(cache)
    with cache._get_connection() as connection:
        connection.execute(
            "UPDATE cached_references SET link_status = 'available', "
            "oa_url = 'https://example.test/article', quality_score = 0.8 "
            "WHERE ref_id = 'online-stale'"
        )
        connection.commit()

    def resolve(citation_text, year=None, doi=None, logger=None):
        return (
            ProviderReference(
                ref_id="refreshed",
                raw_citation=citation_text,
                title="Online article",
                authors=["Original Author"],
                year=2020,
                reference_type="online",
                resolved=True,
                status=ReferenceStatus.RESOLVED,
                quality_score=0.8,
                oa_url="https://example.test/article",
            ),
            0.9,
        )

    downloads = []

    def download(url, dest_dir, max_size_mb=10):
        downloads.append((url, dest_dir, max_size_mb))
        return DownloadResult(success=False, error="http_404")

    revalidate_cached_references(
        cache,
        tmp_path / "missing-graph.db",
        "ids",
        ref_ids=["online-stale"],
        resolver=resolve,
        web_content_dir=tmp_path / "web_cache",
        web_downloader=download,
    )

    refreshed = cache.select_for_revalidation("ids", ref_ids=["online-stale"])[0][1]
    assert downloads == [("https://example.test/article", str(tmp_path / "web_cache"), 10)]
    assert refreshed.link_status == "stale_404"
    assert isclose(refreshed.quality_score, 0.6)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (None, "available"),
        ("http_404", "stale_404"),
        ("http_410", "stale_404"),
        ("ReadTimeout", "stale_timeout"),
        ("ConnectionError", "stale_timeout"),
        ("redirect_page", "stale_moved"),
        ("http_503", None),
    ],
)
def test_link_download_result_mapping(error, expected_status):
    result = DownloadResult(success=error is None, error=error)

    assert revalidation._link_status_from_download(result) == expected_status


def test_stale_link_quality_penalty_is_applied_only_once():
    reference = Reference(
        ref_id="ref-1",
        title="Online article",
        quality_score=0.8,
        link_status="available",
    )

    assert revalidation._apply_link_status(reference, "stale_404") is True
    assert revalidation._apply_link_status(reference, "stale_404") is False
    assert isclose(reference.quality_score, 0.6)
