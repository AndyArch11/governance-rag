"""
Domain terminology extraction from academic documents.

Extracts domain-specific terms from text with:
- N-gram analysis (unigrams, bigrams, trigrams)
- BM25 keyword ranking (consistent with hybrid RAG indexing)
- TF-IDF scoring
- Length-based penalty to reduce over-emphasis on longer n-grams
- Stop word filtering
- Term relationship detection
- Incremental vocabulary building

BM25 integration ensures source-agnostic RAG queries:
- Academic artefacts indexed with BM25 (like regular ingestion)
- Query layer doesn't need source awareness
- Consistent ranking across document types

N-gram length penalty:
- Unigrams (1 word): No penalty
- Bigrams (2 words): 30% penalty, reduced for high-frequency terms
- Trigrams (3+ words): 55% penalty, reduced for high-frequency terms
- Prevents low-frequency longer n-grams from dominating rankings
- High-frequency longer terms (4x+ min frequency) receive reduced penalty


TODO: N-grams are currently extracted using a simple sliding window and not doing a particularly good job at capturing meaningful phrases.
Consider using a more sophisticated method like RAKE or TextRank for better phrase extraction.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from nltk.tokenize import sent_tokenize

# Import BM25 for consistent keyword indexing
from scripts.search.bm25_search import BM25Index, BM25Search

# Import shared text preprocessing for consistent stopword handling
from scripts.search.text_preprocessing import PreprocessingStrategy, TextPreprocessor

# Domain-specific stopwords (academic and PDF artifacts)
# These extend the standard NLTK stopwords with context-specific noise
DOMAIN_STOP_WORDS = {
    # Academic paper boilerplate
    "paper",
    "article",
    "abstract",
    "introduction",
    "conclusion",
    "references",
    "table",
    "figure",
    "section",
    "chapter",
    "page",
    "pages",
    "number",
    "numbers",
    "method",
    "methodology",
    "result",
    "results",
    "analysis",
    "analyses",
    "based",
    "based on",
    "using",
    "used",
    "approach",
    "study",
    "studies",
    "research",
    "researcher",
    "proposed",
    "presented",
    "described",
    "show",
    "shows",
    "demonstrate",
    "demonstrates",
    "evaluate",
    "evaluated",
    # Metadata artifacts from PDF extraction
    "none",
    "null",
    "unknown",
    "unresolved",
    # PDF UI artifacts (print buttons, share widgets, navigation)
    "print",
    "save",
    "share",
    "post",
    "click",
    "download",
    "expand",
    "collapse",
    # Common PDF footer/header artifacts
    "pp",
    "vol",
    "doi",
    "retrieved",
    "available",
    "accessed",
    "online",
    # URLs and web artifacts
    "http",
    "https",
    "www",
    "url",
    "link",
    "href",
    # Publisher/website domains and academic platforms
    "sagepub",
    "com",
    "org",
    "edu",
    "net",
    "gov",
    "pdf",
    "html",
    "jsp",
    "google",
    "scholar",
    "press",
    "journal",
    "journals",
    "springer",
    "elsevier",
    "wiley",
    "taylor",
    "francis",
    "nature",
    "science",
    "acm",
    "ieee",
    # Citation/reference artifacts (e.g., from CrossRef API responses or PDF metadata)
    "crossref",
    "citation",
    "citations",
    "cited",
    "cite",
    "reference",
    "issn",
    "isbn",
    "registered",
    "issn registered",
    "crossref DOI registered",
    "et al",
    "et",
    "al",  # Common citation notation
    # Single letters (often from broken extraction)
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
}


def get_all_stopwords() -> Set[str]:
    """Get combined NLTK + domain-specific stopwords.

    Uses shared TextPreprocessor for NLTK stopwords, adds domain-specific ones.
    This ensures consistency with search pipeline stopword filtering.

    Returns:
        Set of stopwords to filter from extracted terms.
    """
    preprocessor = TextPreprocessor(
        strategy=PreprocessingStrategy.LOWERCASE,
        remove_stopwords=True,
        additional_stopwords=DOMAIN_STOP_WORDS,
    )
    return preprocessor.stopword_set


# Cache combined stopwords at module level for efficiency
_ALL_STOPWORDS = get_all_stopwords()


@dataclass
class TermScore:
    """Scoring result for a term."""

    term: str
    frequency: int
    tf_idf: float
    domain_relevance: float = 0.0  # Relevance to domain (0-1)
    term_type: str = "concept"  # concept, method, tool, dataset, etc.
    related_terms: List[str] = field(default_factory=list)
    bm25_score: float = 0.0  # BM25 keyword ranking score (for hybrid RAG consistency)


class DomainTerminologyExtractor:
    """
    Extract domain-specific terminology from academic documents.

    Features:
    - Extracts n-grams (1-3 words)
    - Scores by TF-IDF and domain relevance
    - Filters stop words
    - Detects term relationships
    - Builds incremental vocabulary
    """

    def __init__(
        self, min_term_freq: int = 2, max_terms: int = 500, ngram_range: Tuple[int, int] = (1, 4)
    ):
        """
        Initialise extractor.

        Args:
            min_term_freq: Minimum frequency for a term to be considered
            max_terms: Maximum terms to return
            ngram_range: Tuple of (min_ngram, max_ngram) sizes
        """
        self.min_term_freq = min_term_freq
        self.max_terms = max_terms
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.doc_frequencies: Dict[str, int] = {}  # For IDF calculation
        self.num_docs = 0

        # Shared text preprocessor (consistent with search pipeline)
        self.preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )

        # Combine NLTK stopwords with domain-specific ones
        self.all_stopwords = self.preprocessor.stopword_set | DOMAIN_STOP_WORDS

        # BM25 index for keyword ranking (consistent with hybrid RAG indexing)
        self.bm25 = BM25Search()
        self.bm25_scores: Dict[str, float] = {}  # Cache BM25 scores for terms

    def _normalise_term(self, term: str) -> str:
        """Normalise term for consistency.

        Args:
            term: Raw term to normalise

        Returns:
            Normalised term string
        """
        return term.lower().strip()

    def _is_valid_term(self, term: str) -> bool:
        """Check if term should be included.

        Args:
            term: Term to validate

        Returns:
            True if term is valid, False if it should be filtered out
        """
        normalised = self._normalise_term(term)

        # Filter by length (minimum 3 chars for single words, 2 for phrases)
        words = normalised.split()
        if len(words) == 1 and len(normalised) < 3:
            return False
        elif len(normalised) < 2:
            return False

        # Allowlist key Indigenous phrases that include stopwords like "and"
        allowed_phrases = {
            "aboriginal and torres strait islander",
            "aboriginal and torres strait islanders",
        }
        if normalised in allowed_phrases:
            return True

        # Filter stop words using combined set (NLTK + domain-specific)
        if normalised in self.all_stopwords:
            return False
        if any(word in self.all_stopwords for word in words):
            return False

        # Filter pure numbers or punctuation
        if not re.search(r"[a-z]", normalised):
            return False

        # Filter terms that are mostly domains/urls
        if any(
            domain in normalised
            for domain in [".com", ".org", ".edu", ".net", ".gov", "http", "www"]
        ):
            return False

        # Filter non-words: check vowel-consonant balance
        # Real English words have reasonable vowel/consonant ratios
        # Acronyms and non-words often have skewed ratios
        for word in words:
            vowels = len(re.findall(r"[aeiou]", word.lower()))
            consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", word.lower()))
            total_alpha = vowels + consonants

            # Skip validation for very short words (2-3 chars) as they're too small to evaluate
            if total_alpha >= 4:
                vowel_ratio = vowels / total_alpha if total_alpha > 0 else 0
                # Real words have 30-60% vowels. Less means it's probably an acronym/non-word
                # Greater means it might be a name or non-English word
                if vowel_ratio < 0.25 or vowel_ratio > 0.65:
                    # Exceptions: allow all-uppercase (likely acronyms we want to keep)
                    if not word.isupper():
                        return False

        return True

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text.

        Args:
            text: Input text to extract from
            n: N-gram size (1 for unigrams, 2 for bigrams, etc.)

        Returns:
            List of extracted n-grams
        """
        # Use shared preprocessor for tokenisation (consistent with search)
        words = self.preprocessor.tokenise(text)

        # Filter valid words
        valid_words = [w for w in words if re.match(r"[a-z]+", w)]

        # Skip plural normalisation - it's too aggressive and breaks proper nouns
        # like 'Torres' (becoming 'Torre') and domain terms like 'Indigenous'
        # Standard n-gram extraction is sufficient for terminology analysis

        # Extract n-grams
        ngrams = []
        for i in range(len(valid_words) - n + 1):
            ngram = " ".join(valid_words[i : i + n])
            if self._is_valid_term(ngram):
                ngrams.append(ngram)

        return ngrams

    def _compute_bm25_scores(
        self, candidate_terms: Dict[str, int], doc_id: str
    ) -> Dict[str, float]:
        """
        Compute BM25 scores for candidate terms using BM25 index.

        This ensures terminology extraction uses the same keyword ranking as hybrid RAG,
        making queries source-agnostic (academic and regular documents use same indexing).

        Args:
            candidate_terms: Dictionary of term -> frequency
            doc_id: Document identifier for BM25 indexing

        Returns:
            Dictionary of term -> BM25 score
        """
        # Build pseudo-query from candidate terms (weighted by frequency)
        weighted_terms = " ".join(
            [term for term, freq in candidate_terms.items() for _ in range(min(freq, 3))]
        )

        # Add document to BM25 index
        self.bm25.add_document(doc_id, weighted_terms)

        # Score each term using BM25
        bm25_scores = {}
        for term in candidate_terms:
            # Query with single term to get BM25 score
            results = self.bm25.search(term, top_k=1)

            # Find score for this document
            bm25_score = 0.0
            for result_doc_id, score in results:
                if result_doc_id == doc_id:
                    bm25_score = score
                    break

            bm25_scores[term] = bm25_score

        return bm25_scores

    def extract_terms(self, text: str, doc_id: Optional[str] = None) -> Dict[str, TermScore]:
        """
        Extract domain terms from text with BM25 and TF-IDF scoring.

        Uses both BM25 (for keyword ranking consistency with hybrid RAG)
        and TF-IDF (for domain relevance). This ensures source-agnostic queries.

        Args:
            text: Document text to analyse
            doc_id: Optional document ID for BM25 scoring (defaults to doc_{num_docs})

        Returns:
            Dictionary of term -> TermScore
        """
        if not text:
            return {}

        # Use default doc_id if not provided
        if doc_id is None:
            doc_id = f"doc_{self.num_docs}"

        term_freq: Counter = Counter()
        term_sentences: Dict[str, List[str]] = defaultdict(list)

        # Extract n-grams of various sizes
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams = self._extract_ngrams(text, n)
            term_freq.update(ngrams)

            # Track which sentences contain each term
            sentences = sent_tokenize(text.lower())
            for sent in sentences:
                for term in ngrams:
                    if term in sent:
                        term_sentences[term].append(sent)

        # Filter by minimum frequency
        unigram_freq = {term: freq for term, freq in term_freq.items() if len(term.split()) == 1}
        phrase_word_min_freq = max(3, self.min_term_freq)
        filtered_terms: Dict[str, int] = {}
        for term, freq in term_freq.items():
            if freq >= self.min_term_freq:
                filtered_terms[term] = freq
                continue
            words = term.split()
            if len(words) > 1 and freq >= 1:
                if all(unigram_freq.get(word, 0) >= phrase_word_min_freq for word in words):
                    filtered_terms[term] = freq

        # Compute BM25 scores for keyword ranking (consistent with hybrid RAG)
        bm25_scores = self._compute_bm25_scores(filtered_terms, doc_id)

        # Calculate combined scores for domain relevance
        scores = {}
        total_terms = sum(filtered_terms.values())
        max_freq = max(filtered_terms.values()) if filtered_terms else 1
        phrase_word_max_freq: Dict[str, int] = defaultdict(int)
        for term, freq in filtered_terms.items():
            if len(term.split()) > 1:
                for word in term.split():
                    phrase_word_max_freq[word] = max(phrase_word_max_freq[word], freq)

        for term, freq in sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)[
            : self.max_terms
        ]:
            # TF (raw frequency ratio)
            tf = freq / total_terms if total_terms > 0 else 0

            # Traditional IDF (estimate based on frequency)
            # This penalises common terms, which is not ideal for domain extraction
            idf = 1.0 / (1 + (freq / max(1, len(filtered_terms))))
            tf_idf = tf * idf

            # Frequency-based domain score: normalised frequency
            # High-frequency terms are likely core domain concepts
            freq_score = freq / max_freq if max_freq > 0 else 0

            # BM25 score (from keyword ranking)
            bm25_score = bm25_scores.get(term, 0.0)

            # Normalise BM25 to 0-1 range
            # CRITICAL: Rare terms naturally have high BM25 scores, but they're less likely to be domain relevant
            # Use inverse normalisation: penalise rare terms by using their frequency-adjusted BM25
            normalised_bm25 = min(bm25_score / max(0.1, max(bm25_scores.values() or [0.1])), 1.0)
            # Apply frequency penalty to BM25: rare terms should NOT get high relevance from BM25
            normalised_bm25 = normalised_bm25 * freq_score  # Weight by frequency

            # Domain relevance for core concepts (weighted heavily towards frequency)
            # For domain extraction, we want:
            # - HIGH-frequency terms (>= mean frequency) to rank very high
            # - MEDIUM-frequency terms to rank medium
            # - LOW-frequency terms (minimum threshold) to rank very low
            # - Avoid letting BM25 inflate scores for rare terms
            #
            # Weights: 85% frequency + 15% BM25
            # This ensures frequency dominates, preventing noise terms from ranking high
            base_relevance = (freq_score * 0.85) + (normalised_bm25 * 0.15)

            # Apply length penalty to reduce over-emphasis on longer n-grams
            # Longer n-grams are naturally rarer, but should still surface when relevant
            word_count = len(term.split())
            if word_count == 1:
                length_penalty = 1.0  # No penalty for unigrams
            elif word_count == 2:
                length_penalty = 1.0  # No penalty for bigrams
            else:  # 3+ words
                length_penalty = 0.95  # Light penalty for trigrams and longer

            # Frequency boost: reduce penalty for high-frequency longer terms
            # If frequency is significantly higher than minimum, boost the score for longer phrases
            freq_ratio = freq / max(1, self.min_term_freq)
            if freq_ratio >= 5:  # 5x or more the minimum frequency
                length_penalty = min(
                    length_penalty + 0.15, 1.0
                )  # Strong boost for very frequent terms
            elif freq_ratio >= 3:
                length_penalty = min(length_penalty + 0.1, 1.0)  # Medium boost for frequent terms

            unigram_penalty = 1.0
            phrase_bonus = 0.0
            if word_count == 1:
                phrase_freq = phrase_word_max_freq.get(term, 0)
                if phrase_freq >= self.min_term_freq:
                    phrase_ratio = phrase_freq / max_freq if max_freq > 0 else 0
                    unigram_penalty = max(0.3, 1.0 - (phrase_ratio * 0.7))
            else:
                bonus_scale = min(freq_ratio / 3.0, 1.0)
                phrase_bonus = (0.15 if word_count == 2 else 0.2) * bonus_scale

            domain_relevance = (base_relevance * length_penalty * unigram_penalty) + phrase_bonus
            # Clamp to ensure domain relevance stays within 0-1
            domain_relevance = max(0.0, min(1.0, domain_relevance))

            # Classify term type
            term_type = self._classify_term_type(term)

            scores[term] = TermScore(
                term=term,
                frequency=freq,
                tf_idf=tf_idf,
                domain_relevance=domain_relevance,
                term_type=term_type,
                related_terms=self._find_related_terms(term, filtered_terms),
                bm25_score=bm25_score,  # Store BM25 score for debugging/analysis
            )

        # Update vocabulary
        self.vocabulary.update(filtered_terms)
        self.num_docs += 1
        for term in filtered_terms:
            self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1

        # Cache BM25 scores for this extraction
        self.bm25_scores.update(bm25_scores)

        return scores

    def _classify_term_type(self, term: str) -> str:
        """Classify term into category.

        Args:
            term: Term to classify

        Returns:
            Term type string (concept, method, tool, dataset, etc.)
        """
        term_lower = term.lower()

        # Method indicators
        if any(word in term_lower for word in ["algorithm", "method", "approach", "technique"]):
            return "method"

        # Tool/system indicators
        if any(word in term_lower for word in ["system", "framework", "tool", "platform", "model"]):
            return "tool"

        # Dataset indicators
        if any(word in term_lower for word in ["dataset", "corpus", "database", "benchmark"]):
            return "dataset"

        # Measurement/metric indicators
        if any(
            word in term_lower for word in ["metric", "measure", "score", "accuracy", "precision"]
        ):
            return "metric"

        # Default to concept
        return "concept"

    def _find_related_terms(self, term: str, vocabulary: Dict[str, int]) -> List[str]:
        """Find related terms (simple heuristic based on word overlap).

        Args:
            term: Term to find relationships for
            vocabulary: Current vocabulary to search for related terms

        Returns:
            List of related terms (limited to top 5)
        """
        words = set(term.split())
        related = []

        for other_term in vocabulary:
            if other_term == term:
                continue

            other_words = set(other_term.split())

            # Terms with word overlap are considered related
            overlap = len(words & other_words)
            if overlap > 0 and len(words) <= 2:
                related.append(other_term)

        return sorted(related)[:5]  # Limit to top 5

    def get_vocabulary(self) -> Dict[str, int]:
        """Get accumulated vocabulary.

        Returns:
            Dictionary of term -> frequency across all processed documents
        """
        return dict(self.vocabulary)

    def merge_extractions(self, other: DomainTerminologyExtractor) -> None:
        """Merge another extractor's vocabulary (for incremental building).

        Args:
            other: Another DomainTerminologyExtractor to merge from

        Returns:
            None (merges in place)
        """
        self.vocabulary.update(other.vocabulary)
        for term, freq in other.doc_frequencies.items():
            self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + freq
        self.num_docs += other.num_docs


class DomainTerminologyStore:
    """Store and query domain terminology in SQLite.

    Args:
    - sqlite_path: Path to SQLite database file

    Features:
    - Stores terms with frequency, TF-IDF, BM25 score, domain relevance, type, and relationships
    - Query by domain, relevance, frequency, and document filters
    """

    def __init__(self, sqlite_path: Path | str):
        """Initialise store with SQLite database."""
        self.sqlite_path = str(sqlite_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create or validate database schema."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        # Terminology table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_terms (
                term_id TEXT PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                domain TEXT,
                frequency INTEGER DEFAULT 0,
                tf_idf_score REAL DEFAULT 0,
                domain_relevance_score REAL DEFAULT 0,
                term_type TEXT,
                doc_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Term relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS term_relationships (
                source_term TEXT NOT NULL,
                target_term TEXT NOT NULL,
                relationship_type TEXT,
                PRIMARY KEY (source_term, target_term, relationship_type),
                FOREIGN KEY (source_term) REFERENCES domain_terms(term),
                FOREIGN KEY (target_term) REFERENCES domain_terms(term)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_domain ON domain_terms(domain)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_terms_relevance ON domain_terms(domain_relevance_score)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_frequency ON domain_terms(frequency)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON term_relationships(source_term)"
        )

        conn.commit()
        conn.close()

    def insert_terms(self, terms: Dict[str, TermScore], domain: str, doc_id: str) -> int:
        """
        Insert extracted terms into database.

        Args:
            terms: Dictionary of term -> TermScore
            domain: Domain classification
            doc_id: Document ID containing these terms

        Returns:
            Count of inserted terms
        """
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        inserted = 0
        for term, score in terms.items():
            # Defensive filtering using combined NLTK + domain stopwords
            term_lower = term.lower()
            words = term_lower.split()
            if any(word in _ALL_STOPWORDS for word in words):
                continue
            if term_lower in _ALL_STOPWORDS:
                continue

            term_id = f"term_{hash(term) % 10**10}"

            try:
                # Insert or update term
                cursor.execute(
                    """
                    INSERT INTO domain_terms 
                    (term_id, term, domain, frequency, tf_idf_score, domain_relevance_score, term_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(term) DO UPDATE SET
                        frequency = frequency + ?,
                        tf_idf_score = (tf_idf_score + ?) / 2,
                        domain_relevance_score = (domain_relevance_score + ?) / 2,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (
                        term_id,
                        term,
                        domain,
                        score.frequency,
                        score.tf_idf,
                        score.domain_relevance,
                        score.term_type,
                        score.frequency,
                        score.tf_idf,
                        score.domain_relevance,
                    ),
                )

                # Update doc_ids
                cursor.execute("SELECT doc_ids FROM domain_terms WHERE term = ?", (term,))
                row = cursor.fetchone()
                if row and row[0]:
                    doc_ids = row[0].split(",")
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
                    new_doc_ids = ",".join(doc_ids)
                else:
                    new_doc_ids = doc_id

                cursor.execute(
                    "UPDATE domain_terms SET doc_ids = ? WHERE term = ?", (new_doc_ids, term)
                )

                # Insert relationships
                for related in score.related_terms:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO term_relationships 
                        (source_term, target_term, relationship_type)
                        VALUES (?, ?, 'related')
                    """,
                        (term, related),
                    )

                inserted += 1

            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()

        return inserted

    def get_terms_by_domain(
        self, domain: str, limit: int = 100, doc_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get top terms for a domain.

        Args:
            domain: Domain to filter by
            limit: Maximum number of terms to return
            doc_filter: Optional document ID substring to filter by (e.g., "Author_2025" for thesis only)
        """
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        if doc_filter:
            # Filter to only terms that appear in documents matching the filter
            cursor.execute(
                """
                SELECT term, frequency, tf_idf_score, domain_relevance_score, term_type, doc_ids
                FROM domain_terms
                WHERE domain = ? AND doc_ids LIKE ?
                ORDER BY domain_relevance_score DESC, frequency DESC
                LIMIT ?
            """,
                (domain, f"%{doc_filter}%", limit),
            )
        else:
            cursor.execute(
                """
                SELECT term, frequency, tf_idf_score, domain_relevance_score, term_type, doc_ids
                FROM domain_terms
                WHERE domain = ?
                ORDER BY domain_relevance_score DESC, frequency DESC
                LIMIT ?
            """,
                (domain, limit),
            )

        terms = [
            {
                "term": row[0],
                "frequency": row[1],
                "tf_idf": row[2],
                "relevance": row[3],
                "type": row[4],
                "doc_ids": row[5] if len(row) > 5 else None,
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return terms

    def get_term_relationships(self, term: str) -> List[Tuple[str, str]]:
        """Get related terms for a given term.

        Args:
            term: Term to find relationships for

        Returns:
            List of tuples (related_term, relationship_type)
        """
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT target_term, relationship_type
            FROM term_relationships
            WHERE source_term = ?
            ORDER BY target_term
        """,
            (term,),
        )

        relationships = [(row[0], row[1]) for row in cursor.fetchall()]

        conn.close()
        return relationships

    def close(self) -> None:
        """Close connection (placeholder)."""
        pass
