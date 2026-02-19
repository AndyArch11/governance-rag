"""Word frequency extraction and cloud generation utilities.

Provides functionality to extract word frequencies from text and store them
in the cache database for word cloud visualisation on the dashboard.

Usage:
    from scripts.ingest.word_frequency import WordFrequencyExtractor

    # Extract frequencies from document
    extractor = WordFrequencyExtractor()
    word_freqs = extractor.extract_frequencies(document_text)

    # Store in cache
    cache_db.put_word_frequencies(word_freqs, doc_count={w: 1 for w in word_freqs})
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Set

from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor

WORD_CLOUD_STOP_WORDS: Set[str] = {
    "doi",
    "http",
    "https",
    "www",
    "org",
    "com",
    "net",
    "edu",
    "gov",
    "crossref",
    "google",
    "scholar",
    "et",
    "al",
}


class WordFrequencyExtractor:
    """Extract word frequencies from text for word cloud visualisation."""

    def __init__(self, min_word_length: int = 2):
        """Initialise word frequency extractor.

        Args:
            min_word_length: Minimum word length to include (default 2)
        """
        self.min_word_length = min_word_length
        # Use LOWERCASE strategy to preserve word identity (no stemming for word clouds)
        self.preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=min_word_length,
            additional_stopwords=WORD_CLOUD_STOP_WORDS,
        )

    def extract_frequencies(self, text: str) -> Dict[str, int]:
        """Extract word frequencies from text.

        Tokenises and filters text, then counts word occurrences.
        Applies stop word filtering and minimum length constraints.

        Args:
            text: Text to extract frequencies from

        Returns:
            Dictionary of word -> frequency
        """
        if not text:
            return {}

        # Tokenise using TextPreprocessor (applies stop word filtering)
        tokens = self.preprocessor.tokenise(text)

        # Filter tokens without alphabetic characters (drops pure numbers/IDs)
        # TODO: Consider whether to keep short alphanumeric tokens (e.g., "3d", "x2") which may be relevant in academic texts
        # TODO: For word clouds, we may want to preserve certain short tokens if they are meaningful (e.g., "AI", "IoT"). Consider adding a whitelist of such tokens.
        # TODO: Consider domain-specific stop words or preservation rules (e.g., preserve chemical formulas, gene names, etc. in scientific texts)
        # TODO: Currently only English stop words are removed. For multilingual documents, consider language detection and applying appropriate stop word lists.
        tokens = [token for token in tokens if re.search(r"[a-z]", token)]

        # Count frequencies
        return dict(Counter(tokens))

    def extract_frequencies_with_doc_count(
        self, text: str
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Extract word frequencies and document occurrence counts.

        For word clouds, it's useful to know both:
        - Absolute frequency: How many times a word appears
        - Document frequency: How many unique documents contain the word

        This method returns both. When called for a single document,
        doc_count will be 1 for all words (they each appear in 1 doc).
        TODO: When processing multiple documents, we can aggregate frequencies and doc counts across the corpus.

        Args:
            text: Text to extract frequencies from

        Returns:
            Tuple of (word_frequencies, doc_count) dicts where doc_count is 1 for all words
        """
        word_freqs = self.extract_frequencies(text)
        doc_count = {word: 1 for word in word_freqs}
        return word_freqs, doc_count


def extract_word_frequencies(text: str, min_word_length: int = 2) -> Dict[str, int]:
    """Convenience function to extract word frequencies from text.

    Args:
        text: Text to extract frequencies from
        min_word_length: Minimum word length to include (default 2)

    Returns:
        Dictionary of word -> frequency
    """
    extractor = WordFrequencyExtractor(min_word_length=min_word_length)
    return extractor.extract_frequencies(text)
