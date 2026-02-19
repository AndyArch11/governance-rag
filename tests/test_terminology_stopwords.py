"""
Unit tests for terminology extraction stop word filtering.

Ensures that domain-specific stop words (publishers, PDF artifacts, etc.)
are properly excluded from terminology extraction results.
"""

import pytest
from scripts.search.text_preprocessing import TextPreprocessor, PreprocessingStrategy
from scripts.ingest.academic.terminology import (
    DOMAIN_STOP_WORDS,
    DomainTerminologyExtractor,
)


class TestDomainStopWords:
    """Test that DOMAIN_STOP_WORDS contains expected terms."""

    def test_pdf_ui_artifacts_excluded(self):
        """Verify PDF UI artifacts are in stop words."""
        pdf_artifacts = {'print', 'save', 'share', 'post', 'click', 'download', 'expand', 'collapse'}
        assert pdf_artifacts.issubset(DOMAIN_STOP_WORDS), \
            f"Missing PDF artifacts: {pdf_artifacts - DOMAIN_STOP_WORDS}"

    def test_metadata_artifacts_excluded(self):
        """Verify metadata extraction artifacts are in stop words."""
        metadata_artifacts = {'none', 'null', 'unknown', 'unresolved'}
        assert metadata_artifacts.issubset(DOMAIN_STOP_WORDS), \
            f"Missing metadata artifacts: {metadata_artifacts - DOMAIN_STOP_WORDS}"

    def test_publisher_platforms_excluded(self):
        """Verify major publishers and platforms are in stop words."""
        publishers = {
            'sagepub', 'google', 'scholar', 'press', 'journal', 'journals',
            'springer', 'elsevier', 'wiley', 'taylor', 'francis',
            'nature', 'science', 'acm', 'ieee'
        }
        assert publishers.issubset(DOMAIN_STOP_WORDS), \
            f"Missing publishers: {publishers - DOMAIN_STOP_WORDS}"

    def test_domain_extensions_excluded(self):
        """Verify domain extensions are in stop words."""
        domains = {'com', 'org', 'edu', 'net', 'gov'}
        assert domains.issubset(DOMAIN_STOP_WORDS), \
            f"Missing domain extensions: {domains - DOMAIN_STOP_WORDS}"

    def test_web_protocol_excluded(self):
        """Verify web protocols are in stop words."""
        protocols = {'http', 'https', 'www', 'url', 'link', 'href'}
        assert protocols.issubset(DOMAIN_STOP_WORDS), \
            f"Missing protocols: {protocols - DOMAIN_STOP_WORDS}"

    def test_pdf_footer_artifacts_excluded(self):
        """Verify common PDF footer artifacts are in stop words."""
        footer_artifacts = {'pp', 'vol', 'doi', 'retrieved', 'available', 'accessed', 'online'}
        assert footer_artifacts.issubset(DOMAIN_STOP_WORDS), \
            f"Missing footer artifacts: {footer_artifacts - DOMAIN_STOP_WORDS}"

    def test_academic_boilerplate_excluded(self):
        """Verify academic paper boilerplate is in stop words."""
        boilerplate = {
            'paper', 'article', 'abstract', 'introduction', 'conclusion',
            'table', 'figure', 'section', 'chapter', 'page', 'pages'
        }
        assert boilerplate.issubset(DOMAIN_STOP_WORDS), \
            f"Missing boilerplate: {boilerplate - DOMAIN_STOP_WORDS}"


class TestTextPreprocessorStopWords:
    """Test TextPreprocessor with domain stop words."""

    def test_tokenise_filters_stop_words(self):
        """Verify tokenise() filters both NLTK and domain stop words."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )
        
        # Text with mixed stop words and meaningful terms
        text = 'Click here to save print post share documents none unknown null research paper study'
        
        tokens = preprocessor.tokenise(text)
        
        # Only 'documents' should remain
        assert tokens == ['documents'], f"Expected ['documents'], got {tokens}"

    def test_tokenise_keeps_meaningful_terms(self):
        """Verify meaningful academic terms are kept."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )
        
        text = 'neural network deep learning transformer architecture optimisation'
        tokens = preprocessor.tokenise(text)
        
        expected = ['neural', 'network', 'deep', 'learning', 'transformer', 'architecture', 'optimisation']
        assert tokens == expected, f"Expected {expected}, got {tokens}"

    def test_tokenise_removes_publisher_names(self):
        """Verify publisher names are filtered."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )
        
        text = 'Published by Springer Elsevier Nature Science Google Scholar'
        tokens = preprocessor.tokenise(text)
        
        # Should only keep 'published' and maybe 'by'
        filtered_publishers = [t for t in tokens if t in 
                             {'springer', 'elsevier', 'nature', 'science', 'google', 'scholar'}]
        assert len(filtered_publishers) == 0, f"Publishers not filtered: {filtered_publishers}"

    def test_tokenise_removes_web_artifacts(self):
        """Verify web artifacts are filtered."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )
        
        text = 'Available online at http www.example.com/download'
        tokens = preprocessor.tokenise(text)
        
        web_artifacts = [t for t in tokens if t in {'online', 'http', 'www', 'download'}]
        assert len(web_artifacts) == 0, f"Web artifacts not filtered: {web_artifacts}"


class TestDomainTerminologyExtractor:
    """Test DomainTerminologyExtractor filters stop words."""

    def test_extract_filters_stop_words_from_bigrams(self):
        """Verify n-grams don't contain stop words."""
        extractor = DomainTerminologyExtractor()
        
        # Text that would produce "none none", "save print", "post share" if stop words not filtered
        text = '''
        This is none none save print post share way. The download and online 
        method on Google Scholar and Springer Press journals.
        '''
        
        term_dict = extractor.extract_terms(text)
        terms = [(term, score) for term, score in term_dict.items()]
        
        # Check no problematic stop word combinations
        problematic_terms = ['none none', 'save print', 'post share', 'download', 'online', 'google']
        found = [term for term in problematic_terms if term in [t[0] for t in terms]]
        assert len(found) == 0, f"Stop words found in terms: {found}"

    def test_extract_keeps_meaningful_terms(self):
        """Verify meaningful terms are extracted."""
        extractor = DomainTerminologyExtractor()
        
        text = '''
        Machine learning algorithms use neural networks for deep learning tasks.
        The transformer architecture revolutionised natural language processing.
        '''
        
        term_dict = extractor.extract_terms(text)
        term_strings = [term.lower() for term in term_dict.keys()]
        
        # Check for meaningful individual words (stop words like 'the', 'for', 'use' should be absent)
        meaningful_words = ['learning', 'neural', 'networks', 'transformer', 'architecture']
        found = [word for word in meaningful_words if word in term_strings]
        assert len(found) > 0, f"No meaningful terms found. Got: {term_strings[:10]}"
        
        # Ensure stop words are not in the terms
        stop_words_in_terms = [t for t in term_strings if t in {'the', 'for', 'use', 'used'}]
        assert len(stop_words_in_terms) == 0, f"Stop words found in terms: {stop_words_in_terms}"

    def test_extract_from_academic_text(self):
        """Test extraction from realistic academic text."""
        extractor = DomainTerminologyExtractor()
        
        academic_text = '''
        Abstract: This paper presents a novel approach to natural language understanding
        using transformer models. We demonstrate improved performance on benchmark datasets.
        
        Introduction: Deep learning has revolutionised artificial intelligence. 
        Recent advances in neural networks have shown promise.
        
        Methods: We propose a new architecture combining attention mechanisms with
        positional encoding. Our model was trained on large-scale corpus.
        
        Results: The evaluation shows significant improvements in semantic similarity tasks.
        Conclusion: Future work should explore knowledge distillation techniques.
        '''
        
        term_dict = extractor.extract_terms(academic_text)
        terms = [(term, score) for term, score in term_dict.items()]
        term_strings = [t[0] for t in terms]
        
        # Verify stop words are NOT in top terms
        stop_word_terms = [t for t in term_strings if t in DOMAIN_STOP_WORDS or 
                          any(sw in t.split() for sw in DOMAIN_STOP_WORDS)]
        
        # Allow only a few stop words due to n-gram extraction edge cases
        assert len(stop_word_terms) < 3, \
            f"Too many stop words in terms: {stop_word_terms}"

    def test_extract_specific_problematic_phrases(self):
        """Test that previously problematic phrases no longer appear."""
        extractor = DomainTerminologyExtractor()
        
        # Text that previously produced bad results
        text = 'none none save print post share download online google scholar press'
        
        term_dict = extractor.extract_terms(text)
        terms = [(term, score) for term, score in term_dict.items()]
        term_strings = [t[0] for t in terms]
        
        problematic = {
            'none none', 'save print', 'post share', 'download online',
            'google scholar', 'scholar press', 'press journal'
        }
        
        found = [term for term in problematic if term in term_strings]
        assert len(found) == 0, f"Problematic stop word phrases found: {found}"


class TestStopWordIntegration:
    """Integration tests for stop word filtering across components."""

    def test_preprocessor_with_additional_stopwords(self):
        """Verify TextPreprocessor correctly merges NLTK + domain stop words."""
        preprocessor = TextPreprocessor(
            strategy=PreprocessingStrategy.LOWERCASE,
            remove_stopwords=True,
            min_token_length=2,
            additional_stopwords=DOMAIN_STOP_WORDS,
        )
        
        # Combine NLTK stop words with domain stop words
        combined_text = 'the and or save print google scholar'
        tokens = preprocessor.tokenise(combined_text)
        
        # Should be empty (all stop words)
        assert len(tokens) == 0, f"Expected no tokens, got {tokens}"

    def test_terminology_extractor_uses_additional_stopwords(self):
        """Verify DomainTerminologyExtractor passes stop words to preprocessor."""
        extractor = DomainTerminologyExtractor()
        
        # Verify preprocessor has DOMAIN_STOP_WORDS
        assert DOMAIN_STOP_WORDS.issubset(extractor.preprocessor.stopword_set), \
            "DomainTerminologyExtractor's preprocessor missing DOMAIN_STOP_WORDS"

    def test_all_stopwords_property(self):
        """Verify all_stopwords property includes domain stop words."""
        extractor = DomainTerminologyExtractor()
        
        assert DOMAIN_STOP_WORDS.issubset(extractor.all_stopwords), \
            "all_stopwords property missing DOMAIN_STOP_WORDS"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
