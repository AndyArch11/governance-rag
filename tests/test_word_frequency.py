"""Unit tests for word frequency extraction and caching."""

import pytest

from scripts.ingest.cache_db import CacheDB
from scripts.ingest.word_frequency import WordFrequencyExtractor, extract_word_frequencies
from scripts.ui.word_cloud_provider import get_word_cloud_data, get_word_cloud_stats


class TestWordFrequencyExtractor:
    """Test word frequency extraction functionality."""

    def test_extract_basic_frequencies(self):
        """Test basic word frequency extraction."""
        text = "leadership leadership workplace workplace indigenous research"
        extractor = WordFrequencyExtractor()
        freqs = extractor.extract_frequencies(text)

        assert freqs["leadership"] == 2
        assert freqs["workplace"] == 2
        assert freqs["indigenous"] == 1
        assert "research" in freqs  # Should be included if not a stopword

    def test_extract_filters_stopwords(self):
        """Test that common stopwords are filtered."""
        text = "the leadership is important and the workplace is collaborative with many of the researchers"
        extractor = WordFrequencyExtractor()
        freqs = extractor.extract_frequencies(text)

        # Common words should be filtered
        assert "the" not in freqs
        assert "is" not in freqs
        assert "and" not in freqs
        assert "of" not in freqs

        # Domain words should remain
        assert "leadership" in freqs
        assert "workplace" in freqs

    def test_extract_with_min_length(self):
        """Test minimum word length filtering."""
        text = "a executive leadership is critical"
        extractor = WordFrequencyExtractor(min_word_length=3)
        freqs = extractor.extract_frequencies(text)

        # Single-letter and two-letter stopwords should be excluded
        assert "a" not in freqs
        # 'is' is short (2 chars)
        assert "executive" in freqs
        assert "leadership" in freqs

    def test_extract_filters_numeric_and_noise_tokens(self):
        """Test that numeric and noise tokens are filtered."""
        text = "leadership 10 2026 doi https org"
        extractor = WordFrequencyExtractor()
        freqs = extractor.extract_frequencies(text)

        assert "leadership" in freqs
        assert "10" not in freqs
        assert "2026" not in freqs
        assert "doi" not in freqs
        assert "https" not in freqs
        assert "org" not in freqs

    def test_extract_with_doc_count(self):
        """Test extraction with document count tracking."""
        text = "leadership and workplace collaboration"
        extractor = WordFrequencyExtractor()
        freqs, doc_counts = extractor.extract_frequencies_with_doc_count(text)

        # All words from single document should have doc_count=1
        for word in freqs:
            assert doc_counts[word] == 1

    def test_convenience_function(self):
        """Test convenience function."""
        text = "research methodology and triangulation are important"
        freqs = extract_word_frequencies(text, min_word_length=2)

        assert "research" in freqs
        assert "methodology" in freqs
        assert "triangulation" in freqs


class TestCacheDBWordFrequency:
    """Test word frequency storage and retrieval in cache database."""

    def test_put_word_frequencies(self, tmp_path):
        """Test storing word frequencies."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        word_freqs = {"leadership": 10, "workplace": 5, "research": 3}
        cache.put_word_frequencies(word_freqs)

        # Verify stored
        stats = cache.get_word_frequency_stats()
        assert stats["total_unique_words"] == 3
        assert stats["total_frequency"] == 18

        cache.close()

    def test_get_top_words(self, tmp_path):
        """Test retrieving top words by frequency."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        word_freqs = {"leadership": 100, "workplace": 50, "research": 30, "methodology": 20}
        cache.put_word_frequencies(word_freqs)

        top_words = cache.get_top_words(limit=2)
        assert len(top_words) == 2
        assert top_words[0][0] == "leadership"  # Should be first (highest freq)
        assert top_words[0][1] == 100

        cache.close()

    def test_accumulate_frequencies(self, tmp_path):
        """Test accumulating frequencies across multiple documents."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        # First document
        cache.put_word_frequencies({"leadership": 50, "workplace": 30})

        # Second document
        cache.put_word_frequencies({"leadership": 30, "research": 25})

        # Should accumulate
        top_words = cache.get_top_words(limit=10)
        words_dict = {word: freq for word, freq, _ in top_words}

        assert words_dict["leadership"] == 80  # 50 + 30
        assert words_dict["workplace"] == 30
        assert words_dict["research"] == 25

        cache.close()

    def test_accumulate_doc_counts(self, tmp_path):
        """Test accumulating document counts across multiple documents."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        cache.put_word_frequencies({"leadership": 5}, doc_count={"leadership": 1})
        cache.put_word_frequencies({"leadership": 3}, doc_count={"leadership": 1})

        top_words = cache.get_top_words(limit=1)
        assert top_words[0][0] == "leadership"
        assert top_words[0][1] == 8
        assert top_words[0][2] == 2

        cache.close()

    def test_get_frequency_stats(self, tmp_path):
        """Test word frequency statistics."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        word_freqs = {"leadership": 100, "workplace": 50, "research": 30}
        cache.put_word_frequencies(word_freqs)

        stats = cache.get_word_frequency_stats()
        assert stats["total_unique_words"] == 3
        assert stats["total_frequency"] == 180
        assert stats["avg_frequency"] == 60.0

        cache.close()

    def test_clear_word_frequencies(self, tmp_path):
        """Test clearing word frequency data."""
        cache = CacheDB(db_path=tmp_path / "test.db", enable_cache=True)

        word_freqs = {"leadership": 100, "workplace": 50}
        cache.put_word_frequencies(word_freqs)

        stats = cache.get_word_frequency_stats()
        assert stats["total_unique_words"] > 0

        cache.clear_word_frequencies()
        stats = cache.get_word_frequency_stats()
        assert stats["total_unique_words"] == 0

        cache.close()


class TestWordCloudProvider:
    """Test word cloud data provider for dashboard."""

    def test_word_cloud_data_array_format_direct(self, tmp_path):
        """Test word cloud data formatting (direct cache test)."""
        cache = CacheDB(db_path=tmp_path / "test_cloud.db", enable_cache=True)
        cache.put_word_frequencies({"leadership": 100, "workplace": 50, "research": 30})

        # Get raw data and format manually
        top_words = cache.get_top_words(limit=10)
        data = [
            {"word": word, "frequency": frequency, "doc_count": doc_count}
            for word, frequency, doc_count in top_words
        ]

        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["word"] == "leadership"
        assert data[0]["frequency"] == 100
        assert "doc_count" in data[0]

        cache.close()

    def test_word_cloud_data_object_format_direct(self, tmp_path):
        """Test word cloud object format (direct cache test)."""
        cache = CacheDB(db_path=tmp_path / "test_cloud2.db", enable_cache=True)
        cache.put_word_frequencies({"leadership": 100, "workplace": 50})

        # Format as object
        top_words = cache.get_top_words(limit=10)
        data = {
            word: {"frequency": frequency, "doc_count": doc_count}
            for word, frequency, doc_count in top_words
        }

        assert isinstance(data, dict)
        assert "leadership" in data
        assert data["leadership"]["frequency"] == 100

        cache.close()

    def test_word_cloud_data_weighted_format_direct(self, tmp_path):
        """Test word cloud weighted format (direct cache test)."""
        cache = CacheDB(db_path=tmp_path / "test_cloud3.db", enable_cache=True)
        cache.put_word_frequencies({"leadership": 100, "workplace": 50, "research": 25})

        # Get data with normalised sizes
        top_words = cache.get_top_words(limit=10)
        max_freq = max(freq for _, freq, _ in top_words) if top_words else 1
        data = [
            {
                "word": word,
                "size": frequency / max_freq,
                "frequency": frequency,
                "doc_count": doc_count,
            }
            for word, frequency, doc_count in top_words
        ]

        assert isinstance(data, list)
        assert data[0]["size"] == 1.0  # Max
        assert data[1]["size"] == 0.5  # 50/100
        assert data[2]["size"] == 0.25  # 25/100

        cache.close()

    def test_word_cloud_data_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError):
            get_word_cloud_data(format="invalid")

    def test_word_cloud_stats_direct(self, tmp_path):
        """Test word cloud statistics (direct cache test)."""
        cache = CacheDB(db_path=tmp_path / "test_cloud4.db", enable_cache=True)
        cache.put_word_frequencies({"leadership": 100, "workplace": 50, "research": 25})

        stats = cache.get_word_frequency_stats()
        assert stats["total_unique_words"] == 3
        assert stats["total_frequency"] == 175

        # Get min/max
        top_words = cache.get_top_words(limit=1)
        max_freq = top_words[0][1] if top_words else 0

        all_words = cache.get_top_words(limit=999999)
        min_freq = min(freq for _, freq, _ in all_words) if all_words else 0

        assert max_freq == 100
        assert min_freq == 25

        cache.close()

    def test_word_cloud_data_array_format_provider(self, monkeypatch):
        """Test provider array format with mocked cache."""

        class FakeCache:
            def __init__(self):
                self.calls = []

            def get_top_words(self, limit=100, min_frequency=1):
                self.calls.append((limit, min_frequency))
                return [("leadership", 12, 3), ("workplace", 5, 2)]

        fake_cache = FakeCache()
        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: fake_cache,
        )

        data = get_word_cloud_data(limit=2, min_frequency=5, format="array")

        assert fake_cache.calls == [(2, 5)]
        assert data == [
            {"word": "leadership", "frequency": 12, "doc_count": 3},
            {"word": "workplace", "frequency": 5, "doc_count": 2},
        ]

    def test_word_cloud_data_object_format_provider(self, monkeypatch):
        """Test provider object format with mocked cache."""

        class FakeCache:
            def get_top_words(self, limit=100, min_frequency=1):
                return [("leadership", 12, 3), ("workplace", 5, 2)]

        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: FakeCache(),
        )

        data = get_word_cloud_data(format="object")

        assert data == {
            "leadership": {"frequency": 12, "doc_count": 3},
            "workplace": {"frequency": 5, "doc_count": 2},
        }

    def test_word_cloud_data_weighted_format_provider(self, monkeypatch):
        """Test provider weighted format with mocked cache."""

        class FakeCache:
            def get_top_words(self, limit=100, min_frequency=1):
                return [("leadership", 12, 3), ("workplace", 6, 2), ("research", 3, 1)]

        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: FakeCache(),
        )

        data = get_word_cloud_data(format="weighted")

        assert data[0]["size"] == 1.0
        assert data[1]["size"] == 0.5
        assert data[2]["size"] == 0.25

    def test_word_cloud_data_empty(self, monkeypatch):
        """Test provider returns empty data when no words exist."""

        class FakeCache:
            def get_top_words(self, limit=100, min_frequency=1):
                return []

        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: FakeCache(),
        )

        assert get_word_cloud_data(format="array") == []
        assert get_word_cloud_data(format="object") == {}
        assert get_word_cloud_data(format="weighted") == {}

    def test_word_cloud_stats_empty(self, monkeypatch):
        """Test provider stats when no words exist."""

        class FakeCache:
            def __init__(self):
                self.calls = 0

            def get_word_frequency_stats(self):
                return {
                    "total_unique_words": 0,
                    "total_frequency": 0,
                    "avg_frequency": 0.0,
                }

            def get_top_words(self, limit=100, min_frequency=1):
                self.calls += 1
                return []

        fake_cache = FakeCache()
        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: fake_cache,
        )

        stats = get_word_cloud_stats()

        assert stats == {
            "total_unique_words": 0,
            "total_frequency": 0,
            "avg_frequency": 0.0,
            "max_frequency": 0,
            "min_frequency": 0,
        }
        assert fake_cache.calls == 0

    def test_word_cloud_stats_populated(self, monkeypatch):
        """Test provider stats with min/max derived from cache."""

        class FakeCache:
            def get_word_frequency_stats(self):
                return {
                    "total_unique_words": 3,
                    "total_frequency": 21,
                    "avg_frequency": 7.0,
                }

            def get_top_words(self, limit=100, min_frequency=1):
                if limit == 1:
                    return [("leadership", 12, 3)]
                return [("leadership", 12, 3), ("workplace", 6, 2), ("research", 3, 1)]

        monkeypatch.setattr(
            "scripts.ui.word_cloud_provider.get_cache_client",
            lambda enable_cache=True: FakeCache(),
        )

        stats = get_word_cloud_stats()

        assert stats["total_unique_words"] == 3
        assert stats["total_frequency"] == 21
        assert stats["avg_frequency"] == 7.0
        assert stats["max_frequency"] == 12
        assert stats["min_frequency"] == 3
