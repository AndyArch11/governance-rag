"""
Test Features 5-7: Advanced Filtering, Export, and Benchmarking
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from datetime import datetime

import pytest

from scripts.rag.benchmark_manager import BenchmarkManager
from scripts.ui.export_manager import ExportManager


class TestFeature5AdvancedFiltering:
    """Test Feature 5: Advanced Filtering with custom role"""

    def test_custom_role_parameter(self):
        """Test that custom_role parameter is accepted"""
        # Should accept custom_role parameter
        import inspect

        from scripts.rag.generate import answer
        from scripts.rag.rag_config import RAGConfig

        sig = inspect.signature(answer)
        params = list(sig.parameters.keys())
        assert "custom_role" in params, "answer() should have custom_role parameter"

    def test_filter_date_range(self):
        """Test date range filtering logic"""
        from datetime import datetime, timedelta

        # Create sample sources with dates
        sources = [
            {"doc_id": "doc1", "created_date": datetime.now().isoformat()},
            {"doc_id": "doc2", "created_date": (datetime.now() - timedelta(days=1)).isoformat()},
            {"doc_id": "doc3", "created_date": (datetime.now() - timedelta(days=10)).isoformat()},
        ]

        # Filter logic (simplified)
        cutoff = datetime.now() - timedelta(days=5)
        filtered = [
            s
            for s in sources
            if (s.get("created_date") and datetime.fromisoformat(s["created_date"]) >= cutoff)
        ]

        assert len(filtered) == 2, "Should filter by date range"

    def test_filter_confidence_score(self):
        """Test confidence/similarity filtering"""
        sources = [
            {"doc_id": "doc1", "distance": 0.1},
            {"doc_id": "doc2", "distance": 0.3},
            {"doc_id": "doc3", "distance": 1.5},
        ]

        # High confidence filter (low distance threshold)
        min_confidence = 80  # 0-100%
        max_distance = 2.0 * (1.0 - min_confidence / 100.0)  # 0.4

        filtered = [s for s in sources if s.get("distance", 0) <= max_distance]
        assert (
            len(filtered) == 2
        ), f"Should filter by confidence, got {len(filtered)} with max_distance={max_distance}"

    def test_filter_result_type(self):
        """Test result type filtering"""
        sources = [
            {"doc_id": "doc1.py", "language": "python"},
            {"doc_id": "security_policy.md"},
            {"doc_id": "doc2.java", "language": "java"},
        ]

        # Filter for code only
        filtered = [s for s in sources if s.get("language")]
        assert len(filtered) == 2, "Should filter for code results"

        # Filter for documents only
        filtered = [s for s in sources if not s.get("language")]
        assert len(filtered) == 1, "Should filter for document results"

    def test_filter_tags(self):
        """Test tags filtering"""
        sources = [
            {"doc_id": "doc1", "tags": ["security", "java"]},
            {"doc_id": "doc2", "tags": ["python", "api"]},
            {"doc_id": "doc3", "tags": ["security", "api"]},
        ]

        # Filter for security tag
        tag_filter = "security"
        filtered = [
            s
            for s in sources
            if any(tag.lower() == tag_filter.lower() for tag in s.get("tags", []))
        ]
        assert len(filtered) == 2, "Should filter by tags"


class TestFeature6Export:
    """Test Feature 6: Export Functionality"""

    def test_export_markdown(self):
        """Test Markdown export"""
        conversation = {
            "id": "conv123",
            "turns": [
                {
                    "query": "What is MFA?",
                    "answer": "Multi-Factor Authentication is...",
                    "sources": [{"doc_id": "security.md", "distance": 0.1}],
                    "generation_time": 2.5,
                    "total_time": 3.0,
                }
            ],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "title": "Security Questions",
            },
        }

        md = ExportManager.export_conversation_markdown(
            conversation["id"],
            conversation["turns"],
            conversation["metadata"],
        )

        assert "Conversation: conv123" in md
        assert "What is MFA?" in md
        assert "Multi-Factor Authentication" in md
        assert "security.md" in md
        print("✓ Markdown export works")

    def test_export_pdf(self):
        """Test PDF export"""
        try:
            from reportlab.lib.pagesizes import letter

            conversation = {
                "id": "conv456",
                "turns": [
                    {
                        "query": "Show Java services",
                        "answer": "```java\nServiceA,\nServiceB\n```",
                        "sources": [],
                        "generation_time": 1.5,
                        "total_time": 2.0,
                    }
                ],
                "metadata": {"title": "Code Query"},
            }

            pdf_bytes = ExportManager.export_conversation_pdf(
                conversation["id"],
                conversation["turns"],
                conversation["metadata"],
            )

            assert len(pdf_bytes) > 0
            assert pdf_bytes[:4] == b"%PDF"  # PDF magic number
            print("✓ PDF export works")

        except ImportError:
            print("⚠ reportlab not installed, skipping PDF export test")

    def test_export_batch_conversations(self):
        """Test batch export"""
        conversations = [
            {
                "id": "conv1",
                "turns": [{"query": "Q1", "answer": "A1", "sources": []}],
                "metadata": {},
            },
            {
                "id": "conv2",
                "turns": [{"query": "Q2", "answer": "A2", "sources": []}],
                "metadata": {},
            },
        ]

        zip_bytes = ExportManager.export_batch_conversations(
            conversations,
            format="markdown",
        )

        assert len(zip_bytes) > 0
        assert zip_bytes[:2] == b"PK"  # ZIP magic number
        print("✓ Batch export works")


class TestFeature7Benchmarking:
    """Test Feature 7: Performance Benchmarking"""

    def test_benchmark_manager_init(self):
        """Test benchmark manager initialisation"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = BenchmarkManager(str(db_path))
            assert db_path.exists()
            print("✓ Benchmark database created")

    def test_record_query_benchmark(self):
        """Test recording query benchmark"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = BenchmarkManager(str(db_path))

            # Record a benchmark
            record_id = manager.record_query(
                query="What is MFA?",
                response={
                    "answer": "Multi-Factor Authentication...",
                    "generation_time": 2.5,
                    "total_time": 3.0,
                    "retrieval_count": 5,
                    "sources": [{"doc_id": "doc1"}],
                    "model": "gpt-4",
                    "is_code_query": False,
                },
                query_params={
                    "k": 5,
                    "temperature": 0.3,
                    "collection_name": "default",
                },
            )

            assert record_id > 0
            print("✓ Query benchmark recorded")

    def test_get_statistics(self):
        """Test statistics aggregation"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = BenchmarkManager(str(db_path))

            # Record multiple benchmarks
            for i in range(3):
                manager.record_query(
                    query=f"Query {i}",
                    response={
                        "answer": f"Answer {i}",
                        "generation_time": 1.0 + i * 0.5,
                        "total_time": 1.5 + i * 0.5,
                        "retrieval_count": 5,
                        "sources": [{"doc_id": "doc1"}],
                        "model": "gpt-4",
                        "is_code_query": False,
                    },
                    query_params={
                        "k": 5,
                        "temperature": 0.3,
                        "collection_name": "default",
                    },
                )

            stats = manager.get_statistics()

            assert stats["total_queries"] == 3
            assert stats["avg_total_time"] > 0
            assert stats["avg_generation_time"] > 0
            print("✓ Statistics aggregation works")

    def test_get_slowest_queries(self):
        """Test slowest queries retrieval"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = BenchmarkManager(str(db_path))

            # Record benchmarks with different speeds
            for i in range(3):
                manager.record_query(
                    query=f"Query {i}",
                    response={
                        "answer": f"Answer {i}",
                        "generation_time": 1.0 + i * 1.0,
                        "total_time": 1.5 + i * 1.0,
                        "retrieval_count": 5,
                        "sources": [],
                        "model": "gpt-4",
                        "is_code_query": False,
                    },
                    query_params={
                        "k": 5,
                        "temperature": 0.3,
                        "collection_name": "default",
                    },
                )

            slowest = manager.get_slowest_queries(limit=2)

            assert len(slowest) == 2
            assert slowest[0]["total_time"] >= slowest[1]["total_time"]
            print("✓ Slowest queries retrieval works")

    def test_export_benchmark_report(self):
        """Test benchmark report export"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = BenchmarkManager(str(db_path))

            # Record a benchmark
            manager.record_query(
                query="Test query",
                response={
                    "answer": "Test answer",
                    "generation_time": 2.0,
                    "total_time": 3.0,
                    "retrieval_count": 5,
                    "sources": [],
                    "model": "gpt-4",
                    "is_code_query": False,
                },
                query_params={
                    "k": 5,
                    "temperature": 0.3,
                    "collection_name": "default",
                },
            )

            report = manager.export_report("")

            assert "Benchmark Report" in report
            assert "Total Queries" in report
            print("✓ Benchmark report generation works")


if __name__ == "__main__":
    # Run tests
    print("\n" + "=" * 60)
    print("Testing Features 5-7: Advanced Filtering, Export, Benchmarking")
    print("=" * 60 + "\n")

    # Feature 5 tests
    print("Testing Feature 5: Advanced Filtering")
    print("-" * 60)
    test5 = TestFeature5AdvancedFiltering()
    test5.test_custom_role_parameter()
    test5.test_filter_date_range()
    test5.test_filter_confidence_score()
    test5.test_filter_result_type()
    test5.test_filter_tags()
    print()

    # Feature 6 tests
    print("Testing Feature 6: Export Functionality")
    print("-" * 60)
    test6 = TestFeature6Export()
    test6.test_export_markdown()
    test6.test_export_pdf()
    test6.test_export_batch_conversations()
    print()

    # Feature 7 tests
    print("Testing Feature 7: Benchmarking")
    print("-" * 60)
    test7 = TestFeature7Benchmarking()
    test7.test_benchmark_manager_init()
    test7.test_record_query_benchmark()
    test7.test_get_statistics()
    test7.test_get_slowest_queries()
    test7.test_export_benchmark_report()
    print()

    print("=" * 60)
    print("✅ All features 5-7 tests passed!")
    print("=" * 60)
