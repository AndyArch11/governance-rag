"""Query expansion for improved keyword search coverage.

Handles:
- Synonym expansion (related terms)
- US/British spelling variants
- Common abbreviations and their expansions
- Domain-specific term equivalents

TODO: Option to integrate with multiple domain term managers
TODO: Use a JSON/YAML config for term lists, allowing easier updates
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("rag")

# Import domain term manager - will be available if domain_terms.py exists
try:
    from scripts.rag.domain_terms import DomainType, get_domain_term_manager

    DOMAIN_TERMS_AVAILABLE = True
except ImportError:
    DOMAIN_TERMS_AVAILABLE = False
    DomainType = None


# Spelling variants: US spelling -> British spelling (and vice versa)
# Supports optional domain scoping via dict entries:
#   "fiber": {"variant": "fibre", "domains": ["network", "nutrition"]}
# For terms without domain restrictions, just use string mapping.
# For terms with no variant, do not include an entry.
# See https://en.wikipedia.org/wiki/American_and_British_English_spelling_differences for a more comprehensive list.
SPELLING_VARIANTS: Dict[str, Union[str, Dict[str, Union[str, List[str]]]]] = {
    # US -> British
    "aluminum": "aluminium",
    "analyze": "analyse",
    "authorize": "authorise",
    "authorization": "authorisation",
    "behavior": "behaviour",
    "catalog": "catalogue",
    "categorize": "categorise",
    "categorization": "categorisation",
    "centralize": "centralise",
    "centralization": "centralisation",
    "center": "centre",
    "check": "cheque",
    "color": "colour",
    "criticize": "criticise",
    "customize": "customise",
    "customization": "customisation",
    "dialog": "dialogue",
    "emphasize": "emphasise",
    "favor": "favour",
    "favorite": "favourite",
    "fiber": {"variant": "fibre", "domains": ["network", "nutrition"]},
    "finalize": "finalise",
    "flavor": "flavour",
    "harbor": "harbour",
    "gray": "grey",
    "initialize": "initialise",
    "initialization": "initialisation",
    "labor": "labour",
    "license": "licence",
    "liter": "litre",
    "maximize": "maximise",
    "meter": "metre",
    "minimize": "minimise",
    "mobilize": "mobilise",
    "mobilization": "mobilisation",
    "mom": "mum",
    "neighbor": "neighbour",
    "normalize": "normalise",
    "normalization": "normalisation",
    "optimize": "optimise",
    "optimization": "optimisation",
    "organize": "organise",
    "organization": "organisation",
    "parameterize": "parameterise",
    "parameterization": "parameterisation",
    "practice": "practise",
    "prioritize": "prioritise",
    "prioritization": "prioritisation",
    "realize": "realise",
    "realization": "realisation",
    "recognize": "recognise",
    "sanitize": "sanitise",
    "sanitization": "sanitisation",
    "serialize": "serialise",
    "serialization": "serialisation",
    "specialize": "specialise",
    "specialization": "specialisation",
    "standardize": "standardise",
    "standardization": "standardisation",
    "summarize": "summarise",
    "tokenize": "tokenise",
    "utilize": "utilise",
    "utilization": "utilisation",
    "visualize": "visualise",
    "visualization": "visualisation",
    # British -> US (add reverse mappings)
    "aluminium": "aluminum",
    "analyse": "analyze",
    "authorise": "authorize",
    "authorisation": "authorization",
    "behaviour": "behavior",
    "catalogue": "catalog",
    "categorise": "categorize",
    "categorisation": "categorization",
    "centralise": "centralize",
    "centralisation": "centralization",
    "centre": "center",
    "cheque": "check",
    "colour": "color",
    "criticise": "criticize",
    "customise": "customize",
    "customisation": "customization",
    "dialogue": "dialog",
    "emphasise": "emphasize",
    "favour": "favor",
    "favourite": "favorite",
    "fibre": {"variant": "fiber", "domains": ["network", "nutrition"]},
    "finalise": "finalize",
    "flavour": "flavor",
    "grey": "gray",
    "harbour": "harbor",
    "initialise": "initialize",
    "initialisation": "initialization",
    "labour": "labor",
    "licence": "license",
    "litre": "liter",
    "maximise": "maximize",
    "metre": "meter",
    "minimise": "minimize",
    "mobilise": "mobilize",
    "mobilisation": "mobilization",
    "mum": "mom",
    "neighbour": "neighbor",
    "normalise": "normalize",
    "normalisation": "normalization",
    "optimise": "optimize",
    "optimisation": "optimization",
    "organise": "organize",
    "organisation": "organization",
    "parameterise": "parameterize",
    "parameterisation": "parameterization",
    "practise": "practice",
    "prioritise": "prioritize",
    "prioritisation": "prioritization",
    "realise": "realize",
    "realisation": "realization",
    "recognise": "recognize",
    "sanitise": "sanitize",
    "sanitisation": "sanitization",
    "serialise": "serialize",
    "serialisation": "serialization",
    "specialise": "specialize",
    "specialisation": "specialization",
    "standardise": "standardize",
    "standardisation": "standardization",
    "summarise": "summarize",
    "tokenise": "tokenize",
    "utilise": "utilize",
    "utilisation": "utilization",
    "visualise": "visualize",
    "visualisation": "visualization",
}

# Common synonyms and related terms in security/governance domain
DOMAIN_SYNONYMS = {
    # Authentication/Authorization
    "auth": ["authentication", "authorize", "authorise", "access", "login", "credential"],
    "authentication": ["auth", "login", "credential", "authorize", "identity"],
    "authenticate": ["auth", "login", "credential", "identity", "mtls", "mfa", "sso"],
    "authorize": ["auth", "permission", "access", "grant"],
    "access_control": ["iam", "identity", "permission", "rbac"],
    "identity": ["iam", "authentication", "user", "account"],
    # Security concepts
    "security": [
        "confidentiality",
        "integrity",
        "protection",
        "encryption",
        "control",
        "access",
        "availability",
        "audit",
    ],
    "encryption": [
        "cipher",
        "crypto",
        "secure",
        "protection",
        "tls",
        "ssl",
        "certificate",
        "data",
        "confidentiality",
    ],
    "compliance": ["regulation", "policy", "audit", "governance"],
    "audit": ["compliance", "log", "trace", "record"],
    "threat": ["risk", "vulnerability", "attack", "exploit"],
    "secret": ["credential", "password", "key", "token", "certificate", "sensitivity"],
    # Network/Infrastructure
    "vpn": [
        "virtual",
        "private",
        "network",
        "tunnel",
        "ipsec",
        "ike",
        "l2tp",
        "wireguard",
        "openvpn",
        "pptp",
    ],
    "firewall": ["network", "protection", "filter", "rule"],
    "gateway": ["proxy", "edge", "filter", "transit"],
    # Data Protection
    "dlp": ["data", "loss", "prevention", "protection"],
    "backup": ["restore", "recovery", "redundancy", "snapshot"],
    "classification": ["label", "sensitivity", "category", "tag"],
    # Common abbreviations
    "iam": ["identity", "access", "management", "authentication", "authorization"],
    "mfa": ["multi-factor authentication", "authentication"],
    "sso": ["single sign-on", "authentication"],
    "rbac": ["role-based access control", "authorization", "permission"],
    "saml": ["authentication", "authorization", "security assertion markup language", "standard"],
    "oauth": ["authentication", "authorization", "protocol"],
}

# Abbreviation expansions
ABBREVIATIONS = {
    "auth": "authentication",
    "auth.": "authentication",
    "dlp": "data loss prevention",
    "iam": "identity and access management",
    "mfa": "multi-factor authentication",
    "sso": "single sign-on",
    "rbac": "role-based access control",
    "saml": "security assertion markup language",
    "oauth": "open authorization",
    "api": "application programming interface",
    "http": "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "vpn": "virtual private network",
    "url": "uniform resource locator",
    "xml": "extensible markup language",
    "json": "javascript object notation",
    "jwt": "json web token",
    "mtls": "mutual tls",
    "tls": "transport layer security",
    "ssl": "secure sockets layer",
    "ldap": "lightweight directory access protocol",
    "ad": "active directory",
    "okta": "okta identity platform",
    "siem": "security information and event management",
    "soar": "security orchestration automation response",
    "casb": "cloud access security broker",
    "fwass": "firewall as a service",
}


class QueryExpander:
    """Expands queries with synonyms and spelling variants."""

    def __init__(self, domain: Optional[str] = None):
        """Initialise query expander.

        Args:
            domain: Optional domain for domain-specific term expansion
        """
        self.logger = get_logger()
        self.domain = domain
        self.domain_manager = None

        if DOMAIN_TERMS_AVAILABLE and domain:
            try:
                self.domain_manager = get_domain_term_manager()
            except Exception as e:
                self.logger.warning(f"Could not load domain term manager: {e}")

    def expand_query(
        self,
        query: str,
        include_variants: bool = True,
        include_synonyms: bool = True,
        include_domain_terms: bool = True,
    ) -> List[str]:
        """Expand a query with variants and synonyms.

        Args:
            query: Original query string
            include_variants: Include spelling variants
            include_synonyms: Include domain synonyms
            include_domain_terms: Include domain-specific terms

        Returns:
            List of expanded query terms (includes original)
        """
        expanded_terms = set()
        original_terms = self._tokenize(query)

        for term in original_terms:
            expanded_terms.add(term)  # Add original

            # Add spelling variants
            if include_variants:
                variant = self.get_spelling_variants(term)
                if variant:
                    expanded_terms.add(variant)

            # Add synonyms
            if include_synonyms:
                synonyms = DOMAIN_SYNONYMS.get(term.lower(), [])
                expanded_terms.update(synonyms)

                # Also check abbreviations
                abbrev_expansion = ABBREVIATIONS.get(term.lower())
                if abbrev_expansion:
                    expanded_terms.update(self._tokenize(abbrev_expansion))

            # Add domain-specific terms
            if include_domain_terms and self.domain_manager and self.domain:
                try:
                    # Get domain vocabulary and look for related terms
                    domain_type = DomainType[self.domain.upper()] if self.domain else None
                    if domain_type:
                        vocab = self.domain_manager.get_vocabulary(domain_type)
                        # Add terms that match category or are closely related
                        for domain_term in vocab.terms.values():
                            if term.lower() in domain_term.term.lower() or (
                                domain_term.acronym and term.lower() == domain_term.acronym.lower()
                            ):
                                expanded_terms.add(domain_term.term)
                                if domain_term.acronym:
                                    expanded_terms.add(domain_term.acronym)
                except Exception as e:
                    self.logger.debug(f"Error expanding domain terms: {e}")

        return list(expanded_terms)

    def expand_query_with_weights(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """Expand query and get domain-specific term weights.

        Args:
            query: Original query

        Returns:
            Tuple of (term_weights dict, expanded_terms list)
        """
        expanded = self.expand_query(query)
        term_weights = {}

        # Add domain term weights if available
        if self.domain_manager and self.domain:
            try:
                for term in expanded:
                    # Get base weight (1.0 for non-domain terms)
                    weight = self.domain_manager.get_term_boost(term, self.domain)
                    term_weights[term] = weight
            except Exception as e:
                self.logger.debug(f"Error getting domain term weights: {e}")
                # Default all to 1.0 if error
                for term in expanded:
                    term_weights[term] = 1.0
        else:
            # No domain terms, default weight
            for term in expanded:
                term_weights[term] = 1.0

        return term_weights, expanded

    def expand_query_string(self, query: str) -> str:
        """Expand query into a combined search string.

        Args:
            query: Original query

        Returns:
            Space-separated combined query with expansions
        """
        expanded = self.expand_query(query)
        unique_terms = set(self._tokenize(query)) | set(expanded)
        return " ".join(sorted(unique_terms))

    def expand_query_semantic(
        self,
        query: str,
        include_variants: bool = True,
        include_synonyms: bool = True,
        similarity_threshold: float = 0.75,
        max_synonyms_per_term: int = 3,
    ) -> List[str]:
        """Expand query using semantic clustering for intelligent synonym detection.

        Enhances traditional expansion with embedding-based semantic similarity
        to find conceptually related terms beyond hardcoded synonym lists.

        Args:
            query: Original query string
            include_variants: Include spelling variants
            include_synonyms: Include domain synonyms
            similarity_threshold: Minimum similarity for synonyms (0.0-1.0)
            max_synonyms_per_term: Maximum synonyms to add per term

        Returns:
            List of semantically expanded query terms
        """
        # Start with traditional expansion
        expanded_terms = set(
            self.expand_query(
                query,
                include_variants=include_variants,
                include_synonyms=include_synonyms,
                include_domain_terms=False,  # Handle separately
            )
        )

        # Use semantic clustering for additional synonym detection
        try:
            from scripts.rag.semantic_clustering import get_semantic_clusterer

            clusterer = get_semantic_clusterer(similarity_threshold=similarity_threshold)

            # Get domain vocabulary as candidate terms for synonym matching
            candidate_terms = set()

            # Add domain-specific candidates
            if self.domain_manager and self.domain:
                try:
                    domain_type = DomainType[self.domain.upper()]
                    vocab = self.domain_manager.get_vocabulary(domain_type)
                    candidate_terms.update(t.term for t in vocab.terms.values())
                    # Add acronyms too
                    candidate_terms.update(t.acronym for t in vocab.terms.values() if t.acronym)
                except Exception as e:
                    self.logger.debug(f"Could not load domain vocabulary: {e}")

            # Add terms from hardcoded synonyms as candidates
            for syn_list in DOMAIN_SYNONYMS.values():
                candidate_terms.update(syn_list)

            # For each term in the query, find semantic synonyms
            original_terms = self._tokenize(query)
            for term in original_terms:
                try:
                    synonyms = clusterer.find_synonyms(
                        term,
                        list(candidate_terms - {term}),  # Exclude term itself
                        top_k=max_synonyms_per_term,
                    )

                    # Add synonyms that meet the threshold
                    for syn_term, similarity in synonyms:
                        if similarity >= similarity_threshold:
                            expanded_terms.add(syn_term)

                except Exception as e:
                    self.logger.debug(f"Semantic expansion failed for '{term}': {e}")

            audit(
                "semantic_expansion",
                {
                    "original_terms": len(original_terms),
                    "expanded_terms": len(expanded_terms),
                    "similarity_threshold": similarity_threshold,
                },
            )

        except ImportError:
            self.logger.debug("Semantic clustering not available")
        except Exception as e:
            self.logger.warning(f"Semantic expansion failed: {e}")

        return list(expanded_terms)

    def get_spelling_variants(self, term: str) -> Optional[str]:
        """Get spelling variant for a term.

        Args:
            term: Term to get variant for

        Returns:
            Variant spelling or None
        """
        entry = SPELLING_VARIANTS.get(term.lower())
        if not entry:
            return None
        if isinstance(entry, str):
            return entry
        variant = entry.get("variant")
        domains = entry.get("domains", [])
        if not domains:
            return variant
        if self.domain and self.domain.lower() in {d.lower() for d in domains}:
            return variant
        return None

    def get_synonyms(self, term: str) -> List[str]:
        """Get domain synonyms for a term.

        Args:
            term: Term to get synonyms for

        Returns:
            List of related terms
        """
        return DOMAIN_SYNONYMS.get(term.lower(), [])

    def get_abbreviation_expansion(self, abbrev: str) -> Optional[str]:
        """Get full expansion of an abbreviation.

        Args:
            abbrev: Abbreviation to expand

        Returns:
            Expanded form or None
        """
        return ABBREVIATIONS.get(abbrev.lower())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise text into terms.

        Args:
            text: Text to tokenise

        Returns:
            List of terms (lowercase, >2 chars)
        """
        # Tokenisation: split on whitespace, hyphens, underscores, and camelCase
        import re

        # First convert camelCase to words (multifactor -> multi factor)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Replace hyphens and underscores with spaces for splitting
        text = text.replace("-", " ").replace("_", " ")
        # Split on whitespace
        terms = text.lower().split()
        return [t for t in terms if len(t) > 2]


class QueryExpansionCache:
    """Cache for expanded queries to avoid recomputation."""

    def __init__(self, max_entries: int = 1000, domain: Optional[str] = None):
        """Initialise cache.

        Args:
            max_entries: Maximum cache entries
            domain: Optional domain for domain-specific expansion
        """
        self.cache: Dict[str, List[str]] = {}
        self.max_entries = max_entries
        self.logger = get_logger()
        self.expander = QueryExpander(domain=domain)
        self.domain = domain

    def get_expanded(self, query: str) -> List[str]:
        """Get expanded query, using cache if available.

        Args:
            query: Query to expand

        Returns:
            List of expanded terms
        """
        cache_key = query.lower().strip()

        if cache_key in self.cache:
            self.logger.debug(f"Query expansion cache hit for: {query[:50]}")
            return self.cache[cache_key]

        # Compute expansion
        expanded = self.expander.expand_query(query)

        # Add to cache (with simple LRU: remove oldest if at capacity)
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry (first in dict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = expanded
        return expanded

    def get_expanded_with_weights(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """Get expanded query with domain term weights.

        Args:
            query: Query to expand

        Returns:
            Tuple of (term_weights, expanded_terms)
        """
        # Note: weights not cached, only terms
        expanded = self.get_expanded(query)

        if self.expander.domain_manager:
            weights, _ = self.expander.expand_query_with_weights(query)
            return weights, expanded
        else:
            # Default weights
            weights = {term: 1.0 for term in expanded}
            return weights, expanded

    def clear(self):
        """Clear cache."""
        self.cache.clear()


# Global instances
_query_expander: Optional[QueryExpander] = None
_expansion_cache: Optional[QueryExpansionCache] = None


def get_query_expander(domain: Optional[str] = None) -> QueryExpander:
    """Get or create global query expander instance.

    Args:
        domain: Optional domain for domain-specific expansion

    Returns:
        QueryExpander instance
    """
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander(domain=domain)
    return _query_expander


def get_expansion_cache(
    max_entries: int = 1000, domain: Optional[str] = None
) -> QueryExpansionCache:
    """Get or create global expansion cache instance.

    Args:
        max_entries: Maximum cache entries
        domain: Optional domain for domain-specific expansion

    Returns:
        QueryExpansionCache instance
    """
    global _expansion_cache
    if _expansion_cache is None:
        _expansion_cache = QueryExpansionCache(max_entries, domain=domain)
    return _expansion_cache
