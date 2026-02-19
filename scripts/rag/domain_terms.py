"""Domain-specific term management for RAG indexing and search.

Manages domain vocabularies, acronyms, and specialised terms to ensure
proper search weight and relevance for domain-specific content.

Supports multiple domains with automatic source discovery and caching.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

from scripts.utils.logger import create_module_logger

get_logger, audit = create_module_logger("rag")


class DomainType(str, Enum):
    """Supported domain types."""

    CYBERSECURITY = "cybersecurity"
    CLOUD_INFRASTRUCTURE = "cloud_infrastructure"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    EDUCATION = "education"
    ABORIGINAL_TORRES_STRAIT_ISLANDER = "aboriginal_torres_strait_islander"
    CUSTOM = "custom"


# Friendly name mappings for domain types
# TODO: - Consider loading these from a config file for easier updates without code changes
DOMAIN_FRIENDLY_NAMES: Dict[str, str] = {
    "australian aboriginal and torres strait cultural leadership": "aboriginal_torres_strait_islander",
    "aboriginal torres strait islander": "aboriginal_torres_strait_islander",
    "indigenous": "aboriginal_torres_strait_islander",
    "cybersecurity": "cybersecurity",
    "cloud infrastructure": "cloud_infrastructure",
    "cloud": "cloud_infrastructure",
    "finance": "finance",
    "healthcare": "healthcare",
    "health": "healthcare",
    "legal": "legal",
    "manufacturing": "manufacturing",
    "retail": "retail",
    "education": "education",
    "custom": "custom",
}


def resolve_domain_type(domain_input: str | None) -> tuple[str | None, str | None]:
    """Resolve domain type from friendly name or enum value.

    Maps friendly domain names to DomainType enum values. Falls back to
    attempting direct enum lookup, then defaults to CUSTOM for unrecognised domains.

    Args:
        domain_input: Domain name (friendly or enum value)

    Returns:
        Tuple of (domain_type_value, display_name) or (None, None) on empty input

    Examples:
        >>> resolve_domain_type("Australian Aboriginal and Torres Strait Cultural Leadership")
        ('aboriginal_torres_strait_islander', 'australian aboriginal and torres strait cultural leadership')
        >>> resolve_domain_type("cloud infrastructure")
        ('cloud_infrastructure', 'cloud infrastructure')
        >>> resolve_domain_type("invalid")
        ('custom', 'invalid (mapped to CUSTOM)')
    """
    if not domain_input:
        return None, None

    # First, try friendly name lookup (case-insensitive)
    lookup_key = domain_input.lower().strip()
    if lookup_key in DOMAIN_FRIENDLY_NAMES:
        return DOMAIN_FRIENDLY_NAMES[lookup_key], lookup_key

    # Second, try direct enum value lookup (already lowercase expected)
    try:
        domain_type = DomainType(lookup_key)
        return domain_type.value, domain_input
    except ValueError:
        pass

    # Default to CUSTOM for unrecognised domains
    return "custom", f"{domain_input} (mapped to CUSTOM)"


@dataclass
class DomainTerm:
    """Represents a domain-specific term or acronym."""

    term: str
    acronym: Optional[str] = None
    expansion: Optional[str] = None
    category: Optional[str] = None
    weight: float = 1.0  # Search weight boost (1.0 = normal, 2.0 = 2x boost)
    description: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "acronym": self.acronym,
            "expansion": self.expansion,
            "category": self.category,
            "weight": self.weight,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainTerm":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CandidateTerm:
    """Candidate term extracted from ingested documents for human curation."""

    term: str
    domain: Optional[DomainType] = None
    source_doc_id: Optional[str] = None
    frequency: int = 1  # How many times encountered
    context: Optional[str] = None  # Example usage context
    suggested_weight: float = 1.5
    status: str = "pending"  # pending, approved, rejected
    curated_by: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "domain": self.domain.value if self.domain else None,
            "source_doc_id": self.source_doc_id,
            "frequency": self.frequency,
            "context": self.context,
            "suggested_weight": self.suggested_weight,
            "status": self.status,
            "curated_by": self.curated_by,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateTerm":
        """Create from dictionary."""
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        domain_value = payload.get("domain")
        if domain_value:
            try:
                payload["domain"] = DomainType(domain_value)
            except ValueError:
                payload["domain"] = None
        return cls(**payload)


@dataclass
class DomainVocabulary:
    """Collection of domain-specific terms and acronyms."""

    domain: DomainType
    name: str
    description: str
    terms: Dict[str, DomainTerm] = field(default_factory=dict)
    acronyms: Dict[str, str] = field(default_factory=dict)  # acronym -> full term
    categories: Dict[str, List[str]] = field(default_factory=dict)  # category -> terms
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    source_url: Optional[str] = None

    def add_term(self, term: DomainTerm) -> None:
        """Add a term to the vocabulary."""
        self.terms[term.term.lower()] = term

        if term.acronym:
            self.acronyms[term.acronym.lower()] = term.term.lower()

        if term.category:
            if term.category not in self.categories:
                self.categories[term.category] = []
            self.categories[term.category].append(term.term.lower())

    def get_term(self, term: str) -> Optional[DomainTerm]:
        """Get a term by name."""
        return self.terms.get(term.lower())

    def get_by_acronym(self, acronym: str) -> Optional[DomainTerm]:
        """Get term by acronym."""
        full_term = self.acronyms.get(acronym.lower())
        if full_term:
            return self.terms.get(full_term)
        return None

    def get_by_category(self, category: str) -> List[DomainTerm]:
        """Get all terms in a category."""
        term_names = self.categories.get(category, [])
        return [self.terms[name] for name in term_names if name in self.terms]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain.value,
            "name": self.name,
            "description": self.description,
            "terms": {k: v.to_dict() for k, v in self.terms.items()},
            "acronyms": self.acronyms,
            "categories": self.categories,
            "last_updated": self.last_updated,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainVocabulary":
        """Create from dictionary."""
        vocab = cls(
            domain=DomainType(data.get("domain", "custom")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            source_url=data.get("source_url"),
        )

        for term_data in data.get("terms", {}).values():
            term = DomainTerm.from_dict(term_data)
            vocab.add_term(term)

        return vocab


class DomainTermManager:
    """Manages domain-specific vocabularies and term indexing."""

    # Standard domain term sources
    DOMAIN_SOURCES = {
        DomainType.CYBERSECURITY: {
            "local": "rag_data/domain_terms/cybersecurity.json",
            "remote": "https://raw.githubusercontent.com/github/gitignore/main/.gitignore",  # Placeholder
            "standards": ["NIST", "OWASP", "CIS"],
        },
        DomainType.CLOUD_INFRASTRUCTURE: {
            "local": "rag_data/domain_terms/cloud_infrastructure.json",
            "standards": ["AWS", "Azure", "GCP"],
        },
        DomainType.FINANCE: {
            "local": "rag_data/domain_terms/finance.json",
            "standards": ["SEC", "FINRA"],
        },
        DomainType.HEALTHCARE: {
            "local": "rag_data/domain_terms/healthcare.json",
            "standards": ["HIPAA", "HL7", "SNOMED"],
        },
        DomainType.LEGAL: {
            "local": "rag_data/domain_terms/legal.json",
            "standards": ["Legal", "Compliance"],
        },
        DomainType.ABORIGINAL_TORRES_STRAIT_ISLANDER: {
            "local": "rag_data/domain_terms/aboriginal_torres_strait_islander.json",
            "standards": ["AIATSIS", "Reconciliation Australia", "Indigenous Affairs"],
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        """Initialise domain term manager.

        Args:
            config_path: Path to domain vocabularies directory
        """
        if config_path is None:
            from scripts.rag.rag_config import RAGConfig

            config = RAGConfig()
            config_path = Path(config.rag_data_path) / "domain_terms"

        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        # Cache loaded vocabularies
        self.vocabularies: Dict[DomainType, DomainVocabulary] = {}

        # Candidate terms for human curation
        self.candidate_terms: Dict[str, CandidateTerm] = {}  # term -> CandidateTerm
        self.candidate_terms_path = self.config_path / "candidate_terms.json"

        self._load_available_vocabularies()
        self._load_candidate_terms()

    def _load_available_vocabularies(self) -> None:
        """Load all available vocabularies from disk."""
        vocab_files = self.config_path.glob("*.json")
        for vocab_file in vocab_files:
            if vocab_file.name == "candidate_terms.json":
                continue
            try:
                with open(vocab_file) as f:
                    data = json.load(f)
                    vocab = DomainVocabulary.from_dict(data)
                    self.vocabularies[vocab.domain] = vocab
                    self.logger.debug(
                        f"Loaded vocabulary for {vocab.domain.value}: {len(vocab.terms)} terms"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load vocabulary {vocab_file}: {e}")

    def create_vocabulary(
        self,
        domain: DomainType,
        name: str,
        description: str,
        source_url: Optional[str] = None,
    ) -> DomainVocabulary:
        """Create a new domain vocabulary.

        Args:
            domain: Domain type
            name: Human-readable name
            description: Domain description
            source_url: Optional URL for the term source

        Returns:
            New vocabulary instance
        """
        vocab = DomainVocabulary(
            domain=domain,
            name=name,
            description=description,
            source_url=source_url,
        )
        self.vocabularies[domain] = vocab
        return vocab

    def get_vocabulary(self, domain: DomainType) -> Optional[DomainVocabulary]:
        """Get vocabulary for a domain.

        Args:
            domain: Domain type

        Returns:
            Vocabulary or None if not loaded
        """
        return self.vocabularies.get(domain)

    def load_from_csv(
        self,
        domain: DomainType,
        csv_path: Path,
        term_column: str = "term",
        acronym_column: Optional[str] = "acronym",
        category_column: Optional[str] = "category",
        weight_column: Optional[str] = "weight",
    ) -> DomainVocabulary:
        """Load domain terms from CSV file.

        Args:
            domain: Domain type
            csv_path: Path to CSV file
            term_column: Column name for terms
            acronym_column: Column name for acronyms (optional)
            category_column: Column name for categories (optional)
            weight_column: Column name for weights (optional)

        Returns:
            Loaded vocabulary
        """
        vocab = self.get_vocabulary(domain)
        if not vocab:
            vocab = self.create_vocabulary(domain, domain.value, f"Terms for {domain.value}")

        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = DomainTerm(
                        term=row.get(term_column, "").strip(),
                        acronym=row.get(acronym_column, "").strip() if acronym_column else None,
                        category=row.get(category_column, "").strip() if category_column else None,
                        weight=float(row.get(weight_column, 1.0)) if weight_column else 1.0,
                    )
                    if term.term:
                        vocab.add_term(term)

            self.logger.info(f"Loaded {len(vocab.terms)} terms from {csv_path}")
            return vocab
        except Exception as e:
            self.logger.error(f"Failed to load CSV {csv_path}: {e}")
            return vocab

    def load_from_json(
        self,
        domain: DomainType,
        json_path: Path,
    ) -> DomainVocabulary:
        """Load domain terms from JSON file.

        Expected JSON formats:
        - Array format: {"name": "...", "description": "...", "terms": [{"term": "...", ...}]}
        - Dict format: {"name": "...", "description": "...", "terms": {"term_key": {"term": "...", ...}}}

        Args:
            domain: Domain type
            json_path: Path to JSON file

        Returns:
            Loaded vocabulary
        """
        try:
            with open(json_path) as f:
                data = json.load(f)

            vocab = self.create_vocabulary(
                domain,
                data.get("name", domain.value),
                data.get("description", f"Terms for {domain.value}"),
                data.get("source_url"),
            )

            terms_data = data.get("terms", [])

            # Handle both dict and array formats
            if isinstance(terms_data, dict):
                # Dict format: convert values to list
                terms_data = terms_data.values()

            for term_data in terms_data:
                term = DomainTerm(
                    term=term_data.get("term", "").strip(),
                    acronym=term_data.get("acronym"),
                    expansion=term_data.get("expansion"),
                    category=term_data.get("category"),
                    weight=term_data.get("weight", 1.0),
                    description=term_data.get("description"),
                    source=term_data.get("source"),
                )
                if term.term:
                    vocab.add_term(term)

            self.logger.info(f"Loaded {len(vocab.terms)} terms from {json_path}")
            return vocab
        except Exception as e:
            self.logger.error(f"Failed to load JSON {json_path}: {e}")
            return self.get_vocabulary(domain) or self.create_vocabulary(
                domain, domain.value, f"Terms for {domain.value}"
            )

    def save_vocabulary(self, vocab: DomainVocabulary) -> bool:
        """Save vocabulary to disk.

        Args:
            vocab: Vocabulary to save

        Returns:
            True if successful
        """
        try:
            vocab_file = self.config_path / f"{vocab.domain.value}.json"
            with open(vocab_file, "w") as f:
                json.dump(vocab.to_dict(), f, indent=2)
            self.logger.debug(f"Saved vocabulary for {vocab.domain.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save vocabulary: {e}")
            return False

    def merge_vocabularies(
        self,
        target: DomainType,
        source: DomainType,
    ) -> bool:
        """Merge one vocabulary into another.

        Args:
            target: Target domain
            source: Source domain

        Returns:
            True if successful
        """
        target_vocab = self.get_vocabulary(target)
        source_vocab = self.get_vocabulary(source)

        if not target_vocab or not source_vocab:
            return False

        for term in source_vocab.terms.values():
            target_vocab.add_term(term)

        return self.save_vocabulary(target_vocab)

    def get_term_boost(self, term: str, domain: Optional[DomainType] = None) -> float:
        """Get search weight boost for a term.

        Args:
            term: Term to look up
            domain: Domain to search (or all if None)

        Returns:
            Weight boost (1.0 = normal weight)
        """
        if domain:
            vocab = self.get_vocabulary(domain)
            if vocab:
                domain_term = vocab.get_term(term) or vocab.get_by_acronym(term)
                if domain_term:
                    return domain_term.weight
        else:
            # Search all vocabularies
            for vocab in self.vocabularies.values():
                domain_term = vocab.get_term(term) or vocab.get_by_acronym(term)
                if domain_term:
                    return domain_term.weight

        return 1.0  # Default weight

    def expand_with_domain_terms(
        self,
        query: str,
        domain: Optional[DomainType] = None,
    ) -> Tuple[List[str], List[float]]:
        """Expand query with domain terms and acronyms.

        Args:
            query: Query string
            domain: Domain to use (or all if None)

        Returns:
            Tuple of (expanded_terms, weights)
        """
        terms = []
        weights = []

        query_lower = query.lower()

        # Check all vocabularies (or specific domain)
        vocabs = [self.get_vocabulary(domain)] if domain else list(self.vocabularies.values())
        vocabs = [v for v in vocabs if v is not None]

        for vocab in vocabs:
            for term_str, domain_term in vocab.terms.items():
                if term_str in query_lower or (
                    domain_term.acronym and domain_term.acronym.lower() in query_lower
                ):
                    terms.append(term_str)
                    weights.append(domain_term.weight)

                # Add acronym expansion
                if domain_term.acronym:
                    terms.append(domain_term.acronym.lower())
                    weights.append(domain_term.weight)

        return terms, weights

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded vocabularies.

        Returns:
            Statistics dictionary
        """
        return {
            "total_domains": len(self.vocabularies),
            "domains": {
                domain.value: {
                    "terms": len(vocab.terms),
                    "categories": len(vocab.categories),
                    "last_updated": vocab.last_updated,
                }
                for domain, vocab in self.vocabularies.items()
            },
            "candidate_terms": {
                "pending": len(self.get_candidate_terms("pending")),
                "approved": len(self.get_candidate_terms("approved")),
                "rejected": len(self.get_candidate_terms("rejected")),
            },
        }

    def _load_candidate_terms(self) -> None:
        """Load candidate terms from JSON file."""
        if not self.candidate_terms_path.exists():
            return

        try:
            with open(self.candidate_terms_path, "r") as f:
                data = json.load(f)
                for term_str, term_data in data.items():
                    self.candidate_terms[term_str] = CandidateTerm.from_dict(term_data)
        except Exception as e:
            self.logger.warning(f"Failed to load candidate terms: {e}")

    def record_candidate_term(
        self,
        term: str,
        domain: Optional[DomainType] = None,
        source_doc_id: Optional[str] = None,
        context: Optional[str] = None,
        frequency_increment: int = 1,
    ) -> CandidateTerm:
        """Record a candidate term from ingested material.

        Args:
            term: The candidate term
            domain: Optional domain type
            source_doc_id: Source document ID
            context: Context where term was found
            frequency_increment: How many times to increment frequency

        Returns:
            The CandidateTerm object
        """
        term_lower = term.lower().strip()
        if not term_lower:
            raise ValueError("Candidate term cannot be empty")

        if term_lower in self.candidate_terms:
            # Update existing
            candidate = self.candidate_terms[term_lower]
            candidate.frequency += frequency_increment
            if source_doc_id and source_doc_id not in (candidate.source_doc_id or ""):
                candidate.source_doc_id = source_doc_id
            if context:
                candidate.context = context
        else:
            # Create new
            candidate = CandidateTerm(
                term=term,
                domain=domain,
                source_doc_id=source_doc_id,
                frequency=frequency_increment,
                context=context,
                created_at=datetime.now().isoformat(),
            )
            self.candidate_terms[term_lower] = candidate

        self.save_candidate_terms()
        return candidate

    def get_candidate_terms(
        self, status: Optional[str] = "pending", domain: Optional[DomainType] = None
    ) -> List[CandidateTerm]:
        """Get candidate terms by status and domain.

        Args:
            status: Filter by status (pending, approved, rejected, or None for all)
            domain: Filter by domain, or None for all domains

        Returns:
            List of matching candidate terms
        """
        results = []
        for candidate in self.candidate_terms.values():
            if status and candidate.status != status:
                continue
            if domain and candidate.domain != domain:
                continue
            results.append(candidate)

        # Sort by frequency descending
        return sorted(results, key=lambda x: x.frequency, reverse=True)

    def approve_candidate_term(
        self, term: str, category: str, weight: float = 1.5, curated_by: str = "system"
    ) -> bool:
        """Approve a candidate term and add to vocabulary.

        Args:
            term: The candidate term
            category: Category to add to
            weight: Weight for the term
            curated_by: Who approved it

        Returns:
            True if approved successfully
        """
        term_lower = term.lower().strip()

        if term_lower not in self.candidate_terms:
            self.logger.warning(f"Candidate term not found: {term}")
            return False

        candidate = self.candidate_terms[term_lower]

        # Add to vocabulary
        if candidate.domain:
            vocab = self.get_vocabulary(candidate.domain)
            if vocab:
                # Create domain term
                domain_term = DomainTerm(
                    term=candidate.term,
                    category=category,
                    description=candidate.context or "",
                    weight=weight,
                    acronym=None,
                )
                vocab.add_term(domain_term)
                vocab.last_updated = datetime.now().isoformat()

        # Mark as approved
        candidate.status = "approved"
        candidate.curated_by = curated_by
        candidate.suggested_weight = weight

        self.save_candidate_terms()
        return True

    def reject_candidate_term(
        self, term: str, reason: str = "", curated_by: str = "system"
    ) -> bool:
        """Reject a candidate term.

        Args:
            term: The candidate term
            reason: Reason for rejection
            curated_by: Who rejected it

        Returns:
            True if rejected successfully
        """
        term_lower = term.lower().strip()

        if term_lower not in self.candidate_terms:
            self.logger.warning(f"Candidate term not found: {term}")
            return False

        candidate = self.candidate_terms[term_lower]
        candidate.status = "rejected"
        candidate.curated_by = curated_by
        candidate.notes = reason

        self.save_candidate_terms()
        return True

    def save_candidate_terms(self) -> bool:
        """Save candidate terms to JSON file.

        Returns:
            True if saved successfully
        """
        try:
            self.candidate_terms_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                term_str: candidate.to_dict()
                for term_str, candidate in self.candidate_terms.items()
            }

            with open(self.candidate_terms_path, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save candidate terms: {e}")
            return False

    def cluster_domain_terms(
        self,
        domain: DomainType,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Cluster domain terms by semantic similarity.

        Groups related domain terms into semantic clusters for better
        organisation and discovery of term relationships.

        Args:
            domain: Domain type to cluster
            similarity_threshold: Minimum similarity for clustering (0.0-1.0)

        Returns:
            List of cluster dictionaries with terms and metadata
        """
        try:
            from scripts.rag.semantic_clustering import get_semantic_clusterer

            vocab = self.get_vocabulary(domain)
            if not vocab or not vocab.terms:
                return []

            # Extract term strings
            term_strings = [term.term for term in vocab.terms.values()]

            if not term_strings:
                return []

            # Cluster using semantic similarity
            clusterer = get_semantic_clusterer(similarity_threshold=similarity_threshold)

            clusters = clusterer.cluster_terms(term_strings)

            # Convert to dictionary format with metadata
            cluster_dicts = []
            for cluster in clusters:
                # Get full term objects for cluster members
                cluster_term_objects = [vocab.terms[t] for t in cluster.terms if t in vocab.terms]

                cluster_dicts.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "terms": cluster.terms,
                        "confidence": cluster.confidence,
                        "size": len(cluster.terms),
                        "categories": list(
                            set(t.category for t in cluster_term_objects if t.category)
                        ),
                        "avg_weight": (
                            sum(t.weight for t in cluster_term_objects) / len(cluster_term_objects)
                            if cluster_term_objects
                            else 1.0
                        ),
                    }
                )

            audit(
                "cluster_domain_terms",
                {
                    "domain": domain.value,
                    "total_terms": len(term_strings),
                    "num_clusters": len(clusters),
                    "similarity_threshold": similarity_threshold,
                },
            )

            return cluster_dicts

        except ImportError:
            self.logger.debug("Semantic clustering not available")
            return []
        except Exception as e:
            self.logger.error(f"Failed to cluster domain terms: {e}")
            return []

    def find_related_terms(
        self,
        term: str,
        domain: DomainType,
        top_k: int = 5,
        similarity_threshold: float = 0.70,
    ) -> List[Tuple[str, float]]:
        """Find semantically related domain terms.

        Uses embedding-based similarity to find terms related to the query term
        within the specified domain vocabulary.

        Args:
            term: Query term
            domain: Domain type to search
            top_k: Maximum number of related terms to return
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of (term, similarity_score) tuples, sorted by similarity descending
        """
        try:
            from scripts.rag.semantic_clustering import get_semantic_clusterer

            vocab = self.get_vocabulary(domain)
            if not vocab or not vocab.terms:
                return []

            # Get candidate terms (all terms in vocabulary except query)
            candidate_terms = [
                t.term for t in vocab.terms.values() if t.term.lower() != term.lower()
            ]

            if not candidate_terms:
                return []

            # Find similar terms
            clusterer = get_semantic_clusterer(similarity_threshold=similarity_threshold)

            related = clusterer.find_synonyms(
                term,
                candidate_terms,
                top_k=top_k,
            )

            # Filter by threshold
            filtered = [(t, score) for t, score in related if score >= similarity_threshold]

            audit(
                "find_related_terms",
                {
                    "query_term": term,
                    "domain": domain.value,
                    "found_count": len(filtered),
                    "top_similarity": filtered[0][1] if filtered else 0.0,
                },
            )

            return filtered

        except ImportError:
            self.logger.debug("Semantic clustering not available")
            return []
        except Exception as e:
            self.logger.error(f"Failed to find related terms: {e}")
            return []


# Global instance
_domain_manager: Optional[DomainTermManager] = None


def get_domain_term_manager(config_path: Optional[Path] = None) -> DomainTermManager:
    """Get or create global domain term manager instance."""
    global _domain_manager
    if _domain_manager is None:
        _domain_manager = DomainTermManager(config_path)
    return _domain_manager
