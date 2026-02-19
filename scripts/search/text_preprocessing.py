"""Text preprocessing utilities for search: stemming, lemmatization, and tokenisation.

Provides multiple preprocessing strategies:
- Porter Stemmer: Fast, aggressive stemming
- Snowball Stemmer: Multi-language support
- WordNet Lemmatizer: Dictionary-based lemmatization
- Custom token filters: stopword removal, n-grams, etc.
"""

import logging
import re
from enum import Enum
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Try importing NLTK components
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")


class PreprocessingStrategy(Enum):
    """Text preprocessing strategies."""

    NONE = "none"  # No preprocessing
    LOWERCASE = "lowercase"  # Lowercase only
    STEM_PORTER = "stem_porter"  # Porter stemmer
    STEM_SNOWBALL = "stem_snowball"  # Snowball stemmer
    LEMMATIZE = "lemmatize"  # WordNet lemmatizer


class TextPreprocessor:
    """Configurable text preprocessing for search."""

    def __init__(
        self,
        strategy: PreprocessingStrategy = PreprocessingStrategy.STEM_PORTER,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        language: str = "english",
        additional_stopwords: Optional[Set[str]] = None,
    ):
        """Initialise text preprocessor.

        Args:
            strategy: Preprocessing strategy to use
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum token length to keep
            language: Language for stemming/stopwords
            additional_stopwords: Optional set of additional stopwords to filter
        """
        self.strategy = strategy
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.language = language

        # Initialise NLTK components if available
        if NLTK_AVAILABLE and strategy != PreprocessingStrategy.NONE:
            self._init_nltk_components()
        elif not NLTK_AVAILABLE and strategy not in [
            PreprocessingStrategy.NONE,
            PreprocessingStrategy.LOWERCASE,
        ]:
            raise RuntimeError(f"NLTK required for strategy '{strategy.value}' but not installed")

        # Initialise stemmer/lemmatizer
        self.stemmer = None
        self.lemmatizer = None

        if NLTK_AVAILABLE:
            if strategy == PreprocessingStrategy.STEM_PORTER:
                self.stemmer = PorterStemmer()
            elif strategy == PreprocessingStrategy.STEM_SNOWBALL:
                self.stemmer = SnowballStemmer(language)
            elif strategy == PreprocessingStrategy.LEMMATIZE:
                self.lemmatizer = WordNetLemmatizer()

        # Load stopwords
        self.stopword_set: Set[str] = set()
        if NLTK_AVAILABLE and remove_stopwords:
            try:
                self.stopword_set = set(stopwords.words(language))
            except LookupError:
                logger.warning(f"Stopwords for '{language}' not found. Downloading...")
                nltk.download("stopwords", quiet=True)
                self.stopword_set = set(stopwords.words(language))

        # Add additional domain-specific stopwords
        if additional_stopwords:
            self.stopword_set |= additional_stopwords

    def _init_nltk_components(self):
        """Download required NLTK data if not present."""
        required_data = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ]

        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info(f"Downloading NLTK data: {name}")
                nltk.download(name, quiet=True)

    def tokenise(self, text: str) -> List[str]:
        """Tokenise text into tokens.

        TODO: address repeated words in n-grams; eg "report annual report"

        Args:
            text: Input text

        Returns:
            List of tokens (with stopwords already removed if remove_stopwords=True)
        """
        # Basic regex tokenisation as fallback
        if not NLTK_AVAILABLE or self.strategy in [
            PreprocessingStrategy.NONE,
            PreprocessingStrategy.LOWERCASE,
        ]:
            tokens = re.findall(r"\b\w+\b", text.lower())
        else:
            # Use NLTK word tokeniser
            tokens = word_tokenize(text.lower())

        # Filter and preprocess each token
        processed_tokens = []
        for token in tokens:
            processed = self.preprocess_token(token)
            if processed is not None:
                processed_tokens.append(processed)

        return processed_tokens

    def preprocess_token(self, token: str) -> Optional[str]:
        """Preprocess a single token.

        Args:
            token: Input token

        Returns:
            Preprocessed token or None if filtered out
        """
        # Lowercase
        token = token.lower()

        # Filter by length
        if len(token) < self.min_token_length:
            return None

        # Remove stopwords
        if self.remove_stopwords and token in self.stopword_set:
            return None

        # Apply stemming/lemmatization
        if (
            self.strategy == PreprocessingStrategy.STEM_PORTER
            or self.strategy == PreprocessingStrategy.STEM_SNOWBALL
        ):
            if self.stemmer:
                token = self.stemmer.stem(token)
        elif self.strategy == PreprocessingStrategy.LEMMATIZE:
            if self.lemmatizer:
                # Get POS tag for better lemmatization
                pos = self._get_wordnet_pos(token)
                token = self.lemmatizer.lemmatize(token, pos=pos)

        return token

    def _get_wordnet_pos(self, token: str) -> str:
        """Get WordNet POS tag for a token.

        Args:
            token: Input token

        Returns:
            WordNet POS tag
        """
        # Simplified POS tagging for lemmatization
        # In production, use nltk.pos_tag for better accuracy
        return wordnet.NOUN  # Default to noun

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text into preprocessed tokens.

        Args:
            text: Input text

        Returns:
            List of preprocessed tokens
        """
        # Toknise
        tokens = self.tokenise(text)

        # Preprocess each token
        processed = []
        for token in tokens:
            processed_token = self.preprocess_token(token)
            if processed_token:
                processed.append(processed_token)

        return processed


def compare_strategies(text: str) -> None:
    """Compare different preprocessing strategies.

    Args:
        text: Sample text to compare
    """
    print(f"\nOriginal text: {text}\n")
    print("=" * 60)

    strategies = [
        PreprocessingStrategy.NONE,
        PreprocessingStrategy.LOWERCASE,
        PreprocessingStrategy.STEM_PORTER,
        PreprocessingStrategy.STEM_SNOWBALL,
        PreprocessingStrategy.LEMMATIZE,
    ]

    for strategy in strategies:
        if not NLTK_AVAILABLE and strategy not in [
            PreprocessingStrategy.NONE,
            PreprocessingStrategy.LOWERCASE,
        ]:
            print(f"{strategy.value:20s}: [NLTK not available]")
            continue

        try:
            preprocessor = TextPreprocessor(strategy=strategy, remove_stopwords=True)
            tokens = preprocessor.preprocess(text)
            print(f"{strategy.value:20s}: {' '.join(tokens)}")
        except Exception as e:
            print(f"{strategy.value:20s}: [Error: {e}]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Check NLTK availability
    if not NLTK_AVAILABLE:
        print("⚠️  NLTK not installed. Install with: pip install nltk")
        print("Demonstrating basic preprocessing only...\n")

    # Sample texts for comparison
    samples = [
        "The quick brown foxes are jumping over the lazy dogs",
        "Running, runs, ran, and runner are different forms of run",
        "Machine learning algorithms process natural language efficiently",
    ]

    for sample in samples:
        compare_strategies(sample)
        print()
