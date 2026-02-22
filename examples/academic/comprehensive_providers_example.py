"""
Comprehensive example demonstrating all ten metadata providers.

This example shows how to use:
- ArxivProvider for preprints
- PubMedProvider for biomedical literature
- DataCiteProvider for research data
- CrossrefProvider for scholarly metadata
- OpenAlexProvider for broad coverage
- SemanticScholarProvider for AI-backed search
- ORCIDProvider for author metadata
- GoogleScholarProvider for broad web-indexed coverage
- UnpaywallProvider for open access links
- URLFetchProvider for metadata from URLs
- The complete provider chain with all providers
"""

from scripts.ingest.academic.providers import (
    ArxivProvider,
    CrossrefProvider,
    DataCiteProvider,
    GoogleScholarProvider,
    OpenAlexProvider,
    ORCIDProvider,
    ProviderChain,
    PubMedProvider,
    SemanticScholarProvider,
    UnpaywallProvider,
    URLFetchProvider,
    create_default_chain,
)


def example_1_arxiv_provider():
    """Example 1: Using arXiv provider for preprints."""
    print("\n" + "=" * 80)
    print("Example 1: ArxivProvider - Resolving CS/Physics/Math Preprints")
    print("=" * 80)

    provider = ArxivProvider()

    # Example 1a: Resolve by arXiv ID
    print("\n1a. Resolving by arXiv ID:")
    ref = provider.resolve("arXiv:2103.14030")  # Famous Attention paper

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Authors: {', '.join(ref.authors[:3])}...")
        print(f"  Year: {ref.year}")
        print(f"  Venue: {ref.venue}")
        print(f"  PDF: {ref.oa_url}")
        print(f"  Quality Score: {ref.quality_score:.2f}")

    # Example 1b: Resolve by title
    print("\n1b. Resolving by title:")
    ref = provider.resolve("Attention Is All You Need", year=2017, authors=["Vaswani"])

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Category: {ref.venue}")
        print(f"  Abstract: {ref.abstract[:100]}...")
    else:
        print("✗ Not found in arXiv")


def example_2_pubmed_provider():
    """Example 2: Using PubMed provider for biomedical literature."""
    print("\n" + "=" * 80)
    print("Example 2: PubMedProvider - Resolving Biomedical Literature")
    print("=" * 80)

    # Note: You can provide an API key for higher rate limits
    # provider = PubMedProvider(api_key="your_ncbi_api_key")
    provider = PubMedProvider()

    # Example 2a: Resolve by PMID
    print("\n2a. Resolving by PMID:")
    ref = provider.resolve("PMID: 33958682")  # COVID-19 paper

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Journal: {ref.venue}")
        print(f"  DOI: {ref.doi}")
        print(f"  Open Access: {ref.oa_available}")
        if ref.oa_url:
            print(f"  PMC URL: {ref.oa_url}")

    # Example 2b: Resolve by title and author
    print("\n2b. Resolving by title:")
    ref = provider.resolve("Cancer immunotherapy", year=2020, authors=["Smith"])

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Reference Type: {ref.reference_type}")
        print(f"  Quality Score: {ref.quality_score:.2f}")
    else:
        print("✗ Not found in PubMed")


def example_3_datacite_provider():
    """Example 3: Using DataCite provider for research data."""
    print("\n" + "=" * 80)
    print("Example 3: DataCiteProvider - Resolving Research Datasets")
    print("=" * 80)

    provider = DataCiteProvider()

    # Example 3a: Resolve dataset by DOI
    print("\n3a. Resolving dataset by DOI:")
    ref = provider.resolve(
        "10.5061/dryad.12345",
        doi="10.5061/dryad.12345",  # Example Dryad dataset
    )

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Resource Type: {ref.reference_type}")
        print(f"  Venue Type: {ref.venue_type}")
        print(f"  Publisher: {ref.venue}")
        print(f"  Open Access: {ref.oa_available}")
    else:
        print("✗ Dataset not found")

    # Example 3b: Resolve by title
    print("\n3b. Resolving by title:")
    ref = provider.resolve("Climate Data Repository", year=2021, authors=["Climate Research Team"])

    if ref.resolved:
        print(f"✓ Resolved: {ref.title}")
        print(f"  Creators: {', '.join(ref.authors)}")
        print(f"  Quality Score: {ref.quality_score:.2f}")
    else:
        print("✗ Not found")


def example_4_complete_provider_chain():
    """Example 4: Using the complete provider chain with all 10 providers."""
    print("\n" + "=" * 80)
    print("Example 4: Complete Provider Chain (All 10 Providers)")
    print("=" * 80)

    # Create default chain with all providers
    chain = create_default_chain()

    print(f"\nChain includes {len(chain.providers)} providers:")
    for i, provider in enumerate(chain.providers, 1):
        print(f"  {i}. {provider.name.title()} (rate: {provider.rate_limit} req/sec)")

    # Test with different types of references
    test_citations = [
        {
            "text": "arXiv:2103.14030",
            "type": "Preprint",
        },
        {
            "text": "Nature Medicine COVID-19 vaccine efficacy",
            "year": 2021,
            "type": "Biomedical Journal",
        },
        {
            "text": "Climate dataset from Dryad",
            "doi": "10.5061/dryad.example",
            "type": "Research Data",
        },
    ]

    print("\n4a. Testing chain with various citation types:")

    for i, citation in enumerate(test_citations, 1):
        print(f"\n  Test {i}: {citation['type']}")
        print(f"  Citation: {citation['text']}")

        try:
            result = chain.resolve(
                citation_text=citation["text"],
                year=citation.get("year"),
                doi=citation.get("doi"),
            )

            if result.reference.resolved:
                print(f"  ✓ Resolved by: {result.provider}")
                print(f"    Confidence: {result.confidence:.2f}")
                print(f"    Attempts: {result.attempt_count}")
                print(f"    Title: {result.reference.title[:60]}...")
            else:
                print(f"  ✗ Unresolved after {result.attempt_count} attempts")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Show statistics
    print("\n4b. Chain statistics:")
    stats = chain.get_stats()
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Resolved: {stats['resolved']} ({stats['resolution_rate']:.1%})")
    print(f"  Unresolved: {stats['unresolved']}")

    print("\n  Per-provider stats:")
    for provider_name, provider_stats in stats["by_provider"].items():
        total = provider_stats["success"] + provider_stats["failure"]
        if total > 0:
            success_rate = provider_stats["success"] / total
            print(f"    {provider_name}: {provider_stats['success']}/{total} ({success_rate:.1%})")


def example_4c_additional_providers():
    """Example 4c: Brief demos for ORCID, Google Scholar, Unpaywall, and URL providers."""
    print("\n" + "=" * 80)
    print("Example 4c: Additional Providers (ORCID, Google Scholar, Unpaywall, URL)")
    print("=" * 80)

    # ORCID provider (author metadata)
    print("\n4c.1 ORCID provider:")
    orcid_provider = ORCIDProvider()
    orcid_ref = orcid_provider.resolve("0000-0002-1825-0097")
    if orcid_ref.resolved:
        print(f"✓ Resolved: {orcid_ref.title}")
        print(f"  Authors: {', '.join(orcid_ref.authors[:3])}...")
    else:
        print("✗ Not found in ORCID")

    # Google Scholar provider (broad web-indexed coverage)
    print("\n4c.2 Google Scholar provider:")
    scholar_provider = GoogleScholarProvider()
    scholar_ref = scholar_provider.resolve("Attention Is All You Need")
    if scholar_ref.resolved:
        print(f"✓ Resolved: {scholar_ref.title}")
        print(f"  Year: {scholar_ref.year}")
    else:
        print("✗ Not found in Google Scholar")

    # Unpaywall provider (open access links)
    print("\n4c.3 Unpaywall provider:")
    unpaywall_provider = UnpaywallProvider()
    unpaywall_ref = unpaywall_provider.resolve(
        "10.1038/s41586-020-2649-2",
        doi="10.1038/s41586-020-2649-2",
    )
    if unpaywall_ref.resolved:
        print(f"✓ Resolved: {unpaywall_ref.title}")
        print(f"  Open Access: {unpaywall_ref.oa_available}")
        if unpaywall_ref.oa_url:
            print(f"  OA URL: {unpaywall_ref.oa_url}")
    else:
        print("✗ Not found in Unpaywall")

    # URL fetch provider (metadata from URLs)
    print("\n4c.4 URL fetch provider:")
    url_provider = URLFetchProvider()
    url_ref = url_provider.resolve("https://arxiv.org/abs/2103.14030")
    if url_ref.resolved:
        print(f"✓ Resolved: {url_ref.title}")
        print(f"  Venue: {url_ref.venue}")
    else:
        print("✗ Not found via URL fetch")


def example_5_custom_chain_configuration():
    """Example 5: Creating custom provider chains for specific domains."""
    print("\n" + "=" * 80)
    print("Example 5: Domain-Specific Provider Chains")
    print("=" * 80)

    # Example 5a: Biomedical-focused chain
    print("\n5a. Biomedical-focused chain:")
    biomedical_chain = ProviderChain(
        providers=[
            PubMedProvider(),  # Primary for biomedical
            CrossrefProvider(),  # Fallback for journals
            SemanticScholarProvider(),  # AI-powered search
        ],
        min_confidence=0.80,  # Lower threshold for better recall
    )

    print(f"  Providers: {[p.name for p in biomedical_chain.providers]}")
    print("  Use case: Medical research, clinical trials, health sciences")

    # Example 5b: Computer Science chain
    print("\n5b. Computer Science chain:")
    cs_chain = ProviderChain(
        providers=[
            ArxivProvider(),  # Primary for CS preprints
            SemanticScholarProvider(),  # AI/CS focus
            CrossrefProvider(),  # Conference papers
            OpenAlexProvider(),  # Comprehensive backup
        ],
        min_confidence=0.85,
    )

    print(f"  Providers: {[p.name for p in cs_chain.providers]}")
    print("  Use case: AI/ML research, computer science papers")

    # Example 5c: Data-focused chain
    print("\n5c. Research data chain:")
    data_chain = ProviderChain(
        providers=[
            DataCiteProvider(),  # Primary for datasets
            OpenAlexProvider(),  # Data papers
            CrossrefProvider(),  # Published data papers
        ],
        min_confidence=0.85,
    )

    print(f"  Providers: {[p.name for p in data_chain.providers]}")
    print("  Use case: Dataset discovery, research reproducibility")


def example_6_batch_resolution():
    """Example 6: Batch resolution with progress tracking."""
    print("\n" + "=" * 80)
    print("Example 6: Batch Resolution with All Providers")
    print("=" * 80)

    chain = create_default_chain()

    # Sample citations from different domains
    citations = [
        ("Deep Learning for Computer Vision", {"year": 2015, "authors": ["LeCun"]}),
        ("CRISPR gene editing breakthrough", {"year": 2020, "authors": ["Doudna"]}),
        ("Climate change temperature dataset", {"year": 2021, "doi": "10.5061/dryad.example"}),
        ("Quantum computing algorithm", {"year": 2019, "authors": ["Preskill"]}),
        ("COVID-19 vaccine efficacy study", {"year": 2021, "authors": ["Polack"]}),
    ]

    print(f"\nResolving {len(citations)} citations...")

    def progress_callback(current: int, total: int):
        """Progress callback for batch processing."""
        print(f"  Progress: {current}/{total} ({current/total:.1%})")

    # Batch resolve (chain.resolve_batch only accepts citation strings)
    batch_citations = []
    for text, meta in citations:
        parts = [text]
        if meta.get("doi"):
            parts.append(f"doi:{meta['doi']}")
        if meta.get("year"):
            parts.append(str(meta["year"]))
        if meta.get("authors"):
            parts.append(" ".join(meta["authors"]))
        batch_citations.append(" ".join(parts))

    results = chain.resolve_batch(
        citations=batch_citations,
        progress_callback=progress_callback,
    )

    # Analyse results
    print("\nResults by provider:")
    provider_counts = {}
    for result in results:
        if result.reference.resolved:
            provider = result.provider
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

    for provider, count in provider_counts.items():
        print(f"  {provider}: {count} resolutions")

    # Show average confidence
    confidences = [r.confidence for r in results if r.reference.resolved]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage confidence: {avg_confidence:.2f}")

    # Show final stats
    print("\nFinal statistics:")
    stats = chain.get_stats()
    print(f"  Resolution rate: {stats['resolution_rate']:.1%}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PROVIDER EXAMPLES")
    print("Demonstrating all 10 metadata providers")
    print("=" * 80)

    # Run examples
    example_1_arxiv_provider()
    example_2_pubmed_provider()
    example_3_datacite_provider()
    example_4_complete_provider_chain()
    example_4c_additional_providers()
    example_5_custom_chain_configuration()
    example_6_batch_resolution()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
