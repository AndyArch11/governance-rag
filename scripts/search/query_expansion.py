"""Query expansion for improved search recall using synonyms and related terms.

Provides multiple expansion strategies:
- WordNet synsets: Expand queries with synonyms from WordNet
- Custom dictionaries: Domain-specific synonym mappings
- Pseudo-relevance feedback: Expand using top result terms
- N-gram expansion: Add common phrase variations

TODO: - merge with rag/query_expansion.py and refactor to be more modular and reusable across search and RAG contexts

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Try importing NLTK for WordNet
try:
    from nltk.corpus import wordnet

    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    logger.warning("NLTK WordNet not available. Install with: pip install nltk")


@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""

    # WordNet expansion settings
    use_wordnet: bool = True
    max_synonyms_per_term: int = 3
    include_hypernyms: bool = False  # Include more general terms
    include_hyponyms: bool = False  # Include more specific terms

    # Custom dictionary
    custom_synonyms: Dict[str, List[str]] = field(default_factory=dict)

    # Weighting
    original_term_weight: float = 1.0
    synonym_weight: float = 0.5


class QueryExpander:
    """Expand queries with synonyms and related terms."""

    def __init__(self, config: Optional[ExpansionConfig] = None):
        """Initialise query expander.

        Args:
            config: Expansion configuration
        """
        self.config = config or ExpansionConfig()

        if self.config.use_wordnet and not WORDNET_AVAILABLE:
            logger.warning("WordNet requested but not available. Disabling WordNet expansion.")
            self.config.use_wordnet = False

    def _get_wordnet_synonyms(self, term: str, max_synonyms: int = 3) -> Set[str]:
        """Get synonyms from WordNet.

        Args:
            term: Input term
            max_synonyms: Maximum number of synonyms to return

        Returns:
            Set of synonym strings
        """
        if not WORDNET_AVAILABLE:
            return set()

        synonyms = set()

        # Get all synsets for the term
        synsets = wordnet.synsets(term)

        for synset in synsets[:3]:  # Limit to first 3 synsets to avoid noise
            # Get lemmas (word forms) from synset
            for lemma in synset.lemmas()[:max_synonyms]:
                # Get lemma name and replace underscores
                synonym = lemma.name().lower().replace("_", " ")

                # Only add if different from original term
                if synonym != term.lower():
                    synonyms.add(synonym)

        return synonyms

    def _get_wordnet_hypernyms(self, term: str, max_hypernyms: int = 2) -> Set[str]:
        """Get more general terms (hypernyms) from WordNet.

        Args:
            term: Input term
            max_hypernyms: Maximum number of hypernyms to return

        Returns:
            Set of hypernym strings
        """
        if not WORDNET_AVAILABLE:
            return set()

        hypernyms = set()

        synsets = wordnet.synsets(term)
        for synset in synsets[:2]:
            for hypernym_synset in synset.hypernyms()[:max_hypernyms]:
                for lemma in hypernym_synset.lemmas()[:1]:
                    hypernym = lemma.name().lower().replace("_", " ")
                    if hypernym != term.lower():
                        hypernyms.add(hypernym)

        return hypernyms

    def _get_wordnet_hyponyms(self, term: str, max_hyponyms: int = 2) -> Set[str]:
        """Get more specific terms (hyponyms) from WordNet.

        Args:
            term: Input term
            max_hyponyms: Maximum number of hyponyms to return

        Returns:
            Set of hyponym strings
        """
        if not WORDNET_AVAILABLE:
            return set()

        hyponyms = set()

        synsets = wordnet.synsets(term)
        for synset in synsets[:2]:
            for hyponym_synset in synset.hyponyms()[:max_hyponyms]:
                for lemma in hyponym_synset.lemmas()[:1]:
                    hyponym = lemma.name().lower().replace("_", " ")
                    if hyponym != term.lower():
                        hyponyms.add(hyponym)

        return hyponyms

    def _get_custom_synonyms(self, term: str) -> Set[str]:
        """Get synonyms from custom dictionary.

        Args:
            term: Input term

        Returns:
            Set of synonym strings
        """
        term_lower = term.lower()
        return set(self.config.custom_synonyms.get(term_lower, []))

    def expand_term(self, term: str) -> Set[str]:
        """Expand a single term with synonyms.

        Args:
            term: Input term

        Returns:
            Set of expanded terms including the original
        """
        expanded = {term}  # Always include original term

        # Add WordNet synonyms
        if self.config.use_wordnet:
            synonyms = self._get_wordnet_synonyms(term, self.config.max_synonyms_per_term)
            expanded.update(synonyms)

            if self.config.include_hypernyms:
                hypernyms = self._get_wordnet_hypernyms(term)
                expanded.update(hypernyms)

            if self.config.include_hyponyms:
                hyponyms = self._get_wordnet_hyponyms(term)
                expanded.update(hyponyms)

        # Add custom synonyms
        custom_syns = self._get_custom_synonyms(term)
        expanded.update(custom_syns)

        return expanded

    def expand_query(self, query: str, tokenise_fn=None) -> str:
        """Expand a query string with synonyms.

        Args:
            query: Input query string
            tokenise_fn: Optional tokenisation function. If None, splits on whitespace.

        Returns:
            Expanded query string with original and synonym terms
        """
        # Tokenis query
        if tokenise_fn:
            tokens = tokenise_fn(query)
        else:
            tokens = query.lower().split()

        # Expand each token
        all_terms = []
        for token in tokens:
            expanded = self.expand_term(token)
            all_terms.extend(expanded)

        # Join into expanded query (removing duplicates while preserving order)
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        expanded_query = " ".join(unique_terms)

        logger.info(f"Expanded '{query}' to '{expanded_query}'")
        return expanded_query

    def expand_query_weighted(self, query: str, tokenise_fn=None) -> List[tuple[str, float]]:
        """Expand query with weighted terms.

        Args:
            query: Input query string
            tokenise_fn: Optional tokenisation function

        Returns:
            List of (term, weight) tuples
        """
        if tokenise_fn:
            tokens = tokenise_fn(query)
        else:
            tokens = query.lower().split()

        weighted_terms = []

        for token in tokens:
            # Add original term with full weight
            weighted_terms.append((token, self.config.original_term_weight))

            # Expand and add synonyms with reduced weight
            expanded = self.expand_term(token)
            for term in expanded:
                if term != token:  # Don't duplicate original
                    weighted_terms.append((term, self.config.synonym_weight))

        return weighted_terms


def load_domain_synonyms(domain: str = "technical") -> Dict[str, List[str]]:
    """Load domain-specific synonym dictionaries.

    Args:
        domain: Domain name (e.g., 'technical', 'medical', 'legal')

    Returns:
        Dictionary mapping terms to synonym lists
    """
    # Example technical domain synonyms
    technical_synonyms = {
        "algorithm": ["method", "procedure", "process", "technique"],
        "data": ["information", "records", "dataset"],
        "code": ["source", "program", "script"],
        "bug": ["error", "defect", "issue", "problem"],
        "feature": ["functionality", "capability", "function"],
        "deploy": ["release", "publish", "launch"],
        "build": ["compile", "construct", "create"],
        "test": ["verify", "validate", "check"],
        "api": ["interface", "endpoint", "service"],
        "database": ["datastore", "repository", "storage"],
        "server": ["host", "backend", "service"],
        "client": ["frontend", "user", "consumer"],
        "cache": ["buffer", "storage", "memory"],
        "optimise": ["optimize", "improve", "enhance", "refine"],
        "performance": ["speed", "efficiency", "throughput"],
    }

    if domain == "technical":
        return technical_synonyms

    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("Query Expansion Demo")
    print("=" * 70)

    # Test WordNet expansion
    if WORDNET_AVAILABLE:
        print("\n1. WordNet Synonym Expansion")
        print("-" * 70)

        config = ExpansionConfig(
            use_wordnet=True,
            max_synonyms_per_term=3,
            include_hypernyms=False,
        )
        expander = QueryExpander(config)

        queries = [
            "fast algorithm",
            "machine learning",
            "database optimisation",
            "error handling",
        ]

        for query in queries:
            expanded = expander.expand_query(query)
            print(f"  '{query}' → '{expanded}'")

        # Test with hypernyms/hyponyms
        print("\n2. WordNet with Hypernyms/Hyponyms")
        print("-" * 70)

        config_extended = ExpansionConfig(
            use_wordnet=True,
            max_synonyms_per_term=2,
            include_hypernyms=True,
            include_hyponyms=True,
        )
        expander_extended = QueryExpander(config_extended)

        term = "dog"
        expanded = expander_extended.expand_term(term)
        print(f"  '{term}' → {expanded}")
    else:
        print("\n⚠️  WordNet not available. Install NLTK and download wordnet.")

    # Test custom domain synonyms
    print("\n3. Custom Domain Synonyms (Technical)")
    print("-" * 70)

    domain_synonyms = load_domain_synonyms("technical")
    config_custom = ExpansionConfig(
        use_wordnet=False,
        custom_synonyms=domain_synonyms,
    )
    expander_custom = QueryExpander(config_custom)

    technical_queries = [
        "algorithm performance",
        "database optimisation",
        "api deployment",
        "code testing",
    ]

    for query in technical_queries:
        expanded = expander_custom.expand_query(query)
        print(f"  '{query}' → '{expanded}'")

    # Test weighted expansion
    print("\n4. Weighted Term Expansion")
    print("-" * 70)

    config_weighted = ExpansionConfig(
        use_wordnet=False,
        custom_synonyms=domain_synonyms,
        original_term_weight=1.0,
        synonym_weight=0.5,
    )
    expander_weighted = QueryExpander(config_weighted)

    query = "fast algorithm"
    weighted_terms = expander_weighted.expand_query_weighted(query)
    print(f"  Query: '{query}'")
    for term, weight in weighted_terms:
        print(f"    - {term}: {weight:.1f}")

    print("\n" + "=" * 70)
