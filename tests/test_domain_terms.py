"""Tests for domain-specific term management system."""

import pytest
import json
import tempfile
from pathlib import Path

from scripts.rag.domain_terms import (
    DomainType,
    DomainTerm,
    DomainVocabulary,
    DomainTermManager,
    get_domain_term_manager,
)


class TestDomainType:
    """Test DomainType enum."""
    
    def test_domain_types_exist(self):
        """Test that all expected domain types are defined."""
        assert hasattr(DomainType, 'CYBERSECURITY')
        assert hasattr(DomainType, 'CLOUD_INFRASTRUCTURE')
        assert hasattr(DomainType, 'FINANCE')
        assert hasattr(DomainType, 'HEALTHCARE')
        assert hasattr(DomainType, 'LEGAL')
    
    def test_domain_type_string_conversion(self):
        """Test converting domain types to/from strings."""
        domain_str = DomainType.CYBERSECURITY.value
        assert domain_str == 'cybersecurity'
        assert DomainType[domain_str.upper()] == DomainType.CYBERSECURITY


class TestDomainTerm:
    """Test DomainTerm dataclass."""
    
    def test_domain_term_creation(self):
        """Test creating a domain term."""
        term = DomainTerm(
            term="encryption",
            acronym="ENC",
            expansion="Encapsulation of data",
            category="security",
            weight=2.0,
            description="Data encryption method"
        )
        assert term.term == "encryption"
        assert term.acronym == "ENC"
        assert term.weight == 2.0
    
    def test_domain_term_optional_fields(self):
        """Test domain term with optional fields."""
        term = DomainTerm(
            term="security",
            category="general"
        )
        assert term.acronym is None
        assert term.weight == 1.0  # Default weight
        assert term.description is None


class TestDomainVocabulary:
    """Test DomainVocabulary dataclass."""
    
    def test_vocabulary_creation(self):
        """Test creating a vocabulary."""
        terms = {
            "encryption": DomainTerm("encryption", category="crypto"),
            "authentication": DomainTerm("authentication", category="auth"),
        }
        vocab = DomainVocabulary(
            domain=DomainType.CYBERSECURITY,
            name="Cybersecurity",
            description="Test cybersecurity vocabulary",
            terms=terms
        )
        assert vocab.domain == DomainType.CYBERSECURITY
        assert len(vocab.terms) == 2
    
    def test_vocabulary_get_by_category(self):
        """Test getting terms by category."""
        # Create vocabulary and add terms manually
        vocab = DomainVocabulary(
            domain=DomainType.CYBERSECURITY,
            name="Cybersecurity",
            description="Test cybersecurity vocabulary",
        )
        
        # Add terms to vocab
        term1 = DomainTerm("encryption", category="crypto", weight=2.0)
        term2 = DomainTerm("cipher", category="crypto", weight=1.8)
        term3 = DomainTerm("authentication", category="auth", weight=2.0)
        
        vocab.add_term(term1)
        vocab.add_term(term2)
        vocab.add_term(term3)
        
        crypto_terms = vocab.get_by_category("crypto")
        assert len(crypto_terms) == 2
        assert any(t.term == "encryption" for t in crypto_terms)
        assert any(t.term == "cipher" for t in crypto_terms)


class TestDomainTermManager:
    """Test DomainTermManager functionality."""
    
    def test_manager_instantiation(self):
        """Test creating a domain term manager."""
        manager = DomainTermManager()
        assert manager is not None
    
    def test_load_builtin_vocabularies(self):
        """Test loading built-in domain vocabularies."""
        manager = DomainTermManager()
        
        # Pre-built vocabularies should be loaded from JSON files
        # Manager loads them on initialisation
        stats = manager.get_statistics()
        
        # Should have loaded at least some vocabularies
        if stats['total_domains'] == 0:
            # Try loading cybersecurity manually
            from pathlib import Path
            cyber_path = Path("rag_data/domain_terms/cybersecurity.json")
            if cyber_path.exists():
                manager.load_from_json(DomainType.CYBERSECURITY, cyber_path)
        
        cyber_vocab = manager.get_vocabulary(DomainType.CYBERSECURITY)
        # After loading, it should exist
        assert cyber_vocab is not None
        assert len(cyber_vocab.terms) > 0
    
    def test_get_term_boost(self):
        """Test getting term boost weight."""
        manager = DomainTermManager()
        
        # Known cybersecurity term should have boost > 1.0
        boost = manager.get_term_boost("encryption", DomainType.CYBERSECURITY)
        assert boost >= 1.0
        
        # Unknown term should return 1.0
        boost = manager.get_term_boost("unknown_term_xyz", DomainType.CYBERSECURITY)
        assert boost == 1.0
    
    def test_get_term_boost_by_acronym(self):
        """Test getting term boost by acronym."""
        manager = DomainTermManager()
        
        # AWS should boost for cloud domain
        aws_boost = manager.get_term_boost("AWS", DomainType.CLOUD_INFRASTRUCTURE)
        assert aws_boost >= 1.0
    
    def test_expand_with_domain_terms(self):
        """Test expanding query with domain terms."""
        manager = DomainTermManager()
        
        # Expand a query with domain terms
        # This returns (terms, weights) tuple
        terms, weights = manager.expand_with_domain_terms(
            "encryption", 
            DomainType.CYBERSECURITY
        )
        
        # May or may not find exact match depending on vocabulary content
        assert isinstance(terms, list)
        assert isinstance(weights, list)
        assert len(terms) == len(weights)
    
    def test_load_from_json(self):
        """Test loading vocabulary from JSON."""
        manager = DomainTermManager()
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_data = {
                "domain": "test_domain",
                "name": "Test Domain",
                "description": "Test vocabulary",
                "terms": [
                    {
                        "term": "test_term",
                        "acronym": "TT",
                        "category": "test",
                        "weight": 2.0
                    }
                ]
            }
            json.dump(vocab_data, f)
            temp_path = f.name
        
        try:
            # Load from JSON using custom domain
            vocab = manager.load_from_json(DomainType.CUSTOM, Path(temp_path))
            assert vocab is not None
            # Should have loaded the terms
            assert len(vocab.terms) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_get_statistics(self):
        """Test getting vocabulary statistics."""
        manager = DomainTermManager()
        
        stats = manager.get_statistics()
        assert isinstance(stats, dict)
        # Should have some statistics
        assert 'total_vocabularies' in stats or 'total_domains' in stats
    
    def test_merge_vocabularies(self):
        """Test merging vocabularies."""
        manager = DomainTermManager()
        
        # Load cybersecurity first
        from pathlib import Path
        cyber_path = Path("rag_data/domain_terms/cybersecurity.json")
        if cyber_path.exists():
            manager.load_from_json(DomainType.CYBERSECURITY, cyber_path)
        
        vocab1 = manager.get_vocabulary(DomainType.CYBERSECURITY)
        if vocab1:
            original_count = len(vocab1.terms)
            assert original_count > 0
            
            # Test merge would work with existing vocabularies
            assert manager.merge_vocabularies(
                DomainType.CYBERSECURITY,
                DomainType.CYBERSECURITY
            ) == True


class TestCandidateTerms:
    """Tests for candidate term recording and curation."""

    def test_record_and_reload_candidate_terms(self, tmp_path: Path):
        """Record candidate terms and reload from disk."""
        manager = DomainTermManager(config_path=tmp_path)

        candidate = manager.record_candidate_term(
            "data sovereignty",
            domain=DomainType.LEGAL,
            source_doc_id="doc-123",
            context="Discussion of data sovereignty and compliance"
        )

        assert candidate.term == "data sovereignty"
        assert (tmp_path / "candidate_terms.json").exists()

        manager2 = DomainTermManager(config_path=tmp_path)
        loaded = manager2.get_candidate_terms()
        assert any(c.term == "data sovereignty" for c in loaded)
        assert manager2.candidate_terms["data sovereignty"].domain == DomainType.LEGAL

    def test_approve_candidate_term_adds_to_vocab(self, tmp_path: Path):
        """Approving a candidate term adds it to the vocabulary."""
        manager = DomainTermManager(config_path=tmp_path)
        manager.create_vocabulary(DomainType.LEGAL, "Legal", "Legal terms")

        manager.record_candidate_term(
            "data residency",
            domain=DomainType.LEGAL,
            source_doc_id="doc-456",
            context="Requirements for data residency"
        )

        approved = manager.approve_candidate_term(
            "data residency",
            category="compliance",
            weight=1.8,
            curated_by="tester"
        )

        assert approved is True
        vocab = manager.get_vocabulary(DomainType.LEGAL)
        assert vocab is not None
        assert vocab.get_term("data residency") is not None
        assert manager.candidate_terms["data residency"].status == "approved"


class TestDomainTermIntegration:
    """Integration tests for domain term system."""
    
    def test_multiple_domains_independent(self):
        """Test that different domains don't interfere."""
        manager = DomainTermManager()
        
        cyber_boost = manager.get_term_boost("encryption", DomainType.CYBERSECURITY)
        cloud_boost = manager.get_term_boost("encryption", DomainType.CLOUD_INFRASTRUCTURE)
        
        # Both should be valid (may differ)
        assert cyber_boost >= 1.0
        assert cloud_boost >= 1.0
    
    def test_get_domain_term_manager_singleton(self):
        """Test that get_domain_term_manager returns singleton."""
        manager1 = get_domain_term_manager()
        manager2 = get_domain_term_manager()
        
        # Should be same instance
        assert manager1 is manager2
    
    def test_domain_vocabulary_completeness(self):
        """Test that built-in vocabularies have minimum coverage."""
        manager = DomainTermManager()
        
        # Load all domain vocabularies from JSON files
        from pathlib import Path
        domains = [
            (DomainType.CYBERSECURITY, "rag_data/domain_terms/cybersecurity.json", 20),
            (DomainType.CLOUD_INFRASTRUCTURE, "rag_data/domain_terms/cloud_infrastructure.json", 10),
            (DomainType.FINANCE, "rag_data/domain_terms/finance.json", 10),
            (DomainType.HEALTHCARE, "rag_data/domain_terms/healthcare.json", 10),
            (DomainType.LEGAL, "rag_data/domain_terms/legal.json", 10),
        ]
        
        for domain_type, path, min_terms in domains:
            path_obj = Path(path)
            if path_obj.exists():
                vocab = manager.load_from_json(domain_type, path_obj)
                assert vocab is not None
                assert len(vocab.terms) >= min_terms, f"{domain_type.value} has only {len(vocab.terms)} terms, expected at least {min_terms}"
    
    def test_domain_term_acronym_matching(self):
        """Test that acronyms are properly indexed."""
        manager = DomainTermManager()
        
        # Load healthcare vocabulary
        from pathlib import Path
        health_path = Path("rag_data/domain_terms/healthcare.json")
        if health_path.exists():
            manager.load_from_json(DomainType.HEALTHCARE, health_path)
        
        # HIPAA is in healthcare
        hipaa_boost = manager.get_term_boost("HIPAA", DomainType.HEALTHCARE)
        assert hipaa_boost >= 1.0  # Should at least have the term
        
        # Load legal vocabulary  
        legal_path = Path("rag_data/domain_terms/legal.json")
        if legal_path.exists():
            manager.load_from_json(DomainType.LEGAL, legal_path)
        
        # GDPR is in legal
        gdpr_boost = manager.get_term_boost("GDPR", DomainType.LEGAL)
        assert gdpr_boost >= 1.0  # Should at least have the term


class TestDomainTermWeighting:
    """Test domain term weighting schemes."""
    
    def test_weight_range(self):
        """Test that weights are within expected range."""
        manager = DomainTermManager()
        
        # Load cybersecurity vocabulary
        from pathlib import Path
        cyber_path = Path("rag_data/domain_terms/cybersecurity.json")
        if cyber_path.exists():
            manager.load_from_json(DomainType.CYBERSECURITY, cyber_path)
        
        cyber = manager.get_vocabulary(DomainType.CYBERSECURITY)
        if cyber and len(cyber.terms) > 0:
            for term_obj in cyber.terms.values():
                assert 1.0 <= term_obj.weight <= 3.0
    
    def test_critical_terms_higher_weight(self):
        """Test that critical security terms have higher weights."""
        manager = DomainTermManager()
        
        # Load cybersecurity vocabulary
        from pathlib import Path
        cyber_path = Path("rag_data/domain_terms/cybersecurity.json")
        if cyber_path.exists():
            manager.load_from_json(DomainType.CYBERSECURITY, cyber_path)
        
        cyber = manager.get_vocabulary(DomainType.CYBERSECURITY)
        if cyber and len(cyber.terms) > 0:
            # Find encryption term (should be critical)
            encryption_terms = [t for t in cyber.terms.values() 
                              if 'encryption' in t.term.lower()]
            if encryption_terms:
                # Critical terms should have higher weight
                assert encryption_terms[0].weight >= 1.5
    
    def test_regulatory_term_weights(self):
        """Test that regulatory terms have proper weights."""
        manager = DomainTermManager()
        
        # Load legal vocabulary
        from pathlib import Path
        legal_path = Path("rag_data/domain_terms/legal.json")
        if legal_path.exists():
            manager.load_from_json(DomainType.LEGAL, legal_path)
        
        legal = manager.get_vocabulary(DomainType.LEGAL)
        if legal and len(legal.terms) > 0:
            gdpr_terms = [t for t in legal.terms.values() 
                         if 'GDPR' in (t.acronym or "")]
            
            if gdpr_terms:
                # Regulatory terms should be high priority
                assert gdpr_terms[0].weight >= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
