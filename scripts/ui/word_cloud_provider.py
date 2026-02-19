"""Word cloud data provider for dashboard visualisation.

Provides formatted word frequency data from the cache database
for rendering word clouds on the dashboard.

Usage:
    from scripts.ui.word_cloud_provider import get_word_cloud_data

    # Get data for word cloud visualisation
    word_cloud_data = get_word_cloud_data(limit=100, min_frequency=2)
    # Returns: [{"word": "leadership", "frequency": 2194}, ...]
"""

from __future__ import annotations

from typing import Any, Dict, List

from scripts.utils.db_factory import get_cache_client
from scripts.utils.logger import get_logger

logger = get_logger()


def get_word_cloud_data(
    limit: int = 100, min_frequency: int = 1, format: str = "array"
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Get word frequency data formatted for word cloud visualisation.

    Retrieves top N words by frequency from the cache database and formats
    them for use in D3.js, Plotly.js, or other word cloud visualisation libraries.

    Args:
        limit: Maximum number of words to return (default 100)
        min_frequency: Minimum frequency threshold for inclusion (default 1)
        format: Output format - 'array' (list of dicts), 'object' (dict by word),
                or 'weighted' (with normalised weight for sizing)

    Returns:
        Formatted word frequency data:
        - 'array': [{"word": "leadership", "frequency": 2194, "doc_count": 45}, ...]
        - 'object': {"leadership": {"frequency": 2194, "doc_count": 45}, ...}
        - 'weighted': [{"word": "leadership", "size": 0.95, "frequency": 2194}, ...]

    Raises:
        ValueError: If format is not recognised

    Example:
        >>> data = get_word_cloud_data(limit=50, format='weighted')
        >>> # Use in Plotly word cloud: fig.update_traces(word=data[*].word, size=data[*].size)
    """
    if format not in ("array", "object", "weighted"):
        raise ValueError(f"Unknown format: {format}. Must be 'array', 'object', or 'weighted'")

    cache_db = get_cache_client(enable_cache=True)
    top_words = cache_db.get_top_words(limit=limit, min_frequency=min_frequency)

    if not top_words:
        logger.warning("No word frequency data available for word cloud")
        return [] if format == "array" else {}

    # Find max frequency for normalisation
    max_freq = max(freq for _, freq, _ in top_words) if top_words else 1

    if format == "array":
        result = [
            {"word": word, "frequency": frequency, "doc_count": doc_count}
            for word, frequency, doc_count in top_words
        ]

    elif format == "object":
        result = {
            word: {"frequency": frequency, "doc_count": doc_count}
            for word, frequency, doc_count in top_words
        }

    elif format == "weighted":
        # Normalise frequencies to 0-1 range for sizing
        result = [
            {
                "word": word,
                "size": frequency / max_freq,  # Normalised for sizing (0-1)
                "frequency": frequency,
                "doc_count": doc_count,
            }
            for word, frequency, doc_count in top_words
        ]

    return result


def get_word_cloud_stats() -> Dict[str, Any]:
    """Get overall statistics about word frequencies for dashboard.

    Returns:
        Dictionary with stats: total_unique_words, total_frequency, avg_frequency,
        max_frequency, min_frequency
    """
    cache_db = get_cache_client(enable_cache=True)
    base_stats = cache_db.get_word_frequency_stats()

    if base_stats.get("total_unique_words", 0) == 0:
        return {
            "total_unique_words": 0,
            "total_frequency": 0,
            "avg_frequency": 0.0,
            "max_frequency": 0,
            "min_frequency": 0,
        }

    # Get min/max frequencies
    top_words = cache_db.get_top_words(limit=1)  # Get max
    max_freq = top_words[0][1] if top_words else 0

    all_words = cache_db.get_top_words(limit=999999)
    min_freq = min(freq for _, freq, _ in all_words) if all_words else 0

    return {
        "total_unique_words": base_stats.get("total_unique_words", 0),
        "total_frequency": base_stats.get("total_frequency", 0),
        "avg_frequency": base_stats.get("avg_frequency", 0.0),
        "max_frequency": max_freq,
        "min_frequency": min_freq,
    }
