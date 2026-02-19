#!/usr/bin/env python
"""Date-based code extraction and drift analysis from Bitbucket.

This script demonstrates how to:
1. Extract files modified before a specific date
2. Compare code versions between two dates for drift analysis
3. Analyse changes over time with rate limit handling

Important: Bitbucket API Rate Limits
------------------------------------
Bitbucket Cloud enforces API request limits:
- ~1000 requests/hour per user (Bitbucket Cloud)
- Server limits vary by configuration

The BitbucketConnector automatically handles rate limiting with:
- Request throttling (100ms delay between requests)
- Automatic retry with exponential backoff for 429 errors
- Respects Retry-After headers

For large repositories or bulk operations, consider:
- Processing in smaller batches
- Using date-based filtering to reduce file counts
- Adjusting max_retries and retry_delay parameters

Reference: https://support.atlassian.com/bitbucket-cloud/docs/api-request-limits/

Usage:
    cd ~/rag-project
    python examples/git/bitbucket_dated_example.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.git.bitbucket_code_ingestion import BitbucketCodeIngestion


def example_date_based_extraction():
    """Example: Extract files modified before a specific date."""
    print("=" * 70)
    print("Example 1: Date-Based File Extraction")
    print("=" * 70)
    
    # Configure with rate limiting parameters
    ingestion = BitbucketCodeIngestion(
        host="https://bitbucket.company.com",
        username="user@company.com",
        password="app-password",
        is_cloud=False,
        max_retries=5,  # Increase retries for rate-limited requests
        retry_delay=2.0,  # Initial delay before retry (exponential backoff)
    )
    
    # Extract files modified before 6 months ago
    six_months_ago = datetime.now() - timedelta(days=180)
    
    result = ingestion.ingest_repository_dated(
        project_key="PROJ",
        repo_slug="my-repo",
        modified_before=six_months_ago,
        file_types=["groovy", "gradle"],
    )
    
    print(f"\nFiles modified before {six_months_ago.date()}:")
    print(f"  Total files: {result['summary']['total_files']}")
    print(f"  Successfully parsed: {result['summary']['parsed_successfully']}")
    print(f"  External dependencies: {len(result['summary']['external_dependencies'])}")
    print(f"  Service types: {result['summary']['service_types']}")
    
    # Print sample files with dates
    if result.get("parsed_files"):
        print(f"\nSample files:")
        for f in result["parsed_files"][:5]:
            print(f"  - {f['file_path']} (modified: {f['modified_date']})")


def example_version_drift_analysis():
    """Example: Compare code versions for drift analysis."""
    print("\n" + "=" * 70)
    print("Example 2: Version Drift Analysis")
    print("=" * 70)
    
    ingestion = BitbucketCodeIngestion(
        host="https://bitbucket.company.com",
        username="user@company.com",
        password="app-password",
        is_cloud=False,
    )
    
    # Compare code between two versions
    v1_date = datetime(2025, 1, 1)  # Version 1 baseline
    v2_date = datetime(2026, 1, 1)  # Version 2 current
    
    drift_report = ingestion.analyse_version_drift(
        project_key="PROJ",
        repo_slug="my-repo",
        v1_date=v1_date,
        v2_date=v2_date,
    )
    
    print(f"\nDrift Analysis: {v1_date.date()} → {v2_date.date()}")
    summary = drift_report["summary"]
    print(f"  V1 files: {summary['v1_file_count']}")
    print(f"  V2 files: {summary['v2_file_count']}")
    print(f"  Files added: {summary['total_added']}")
    print(f"  Files removed: {summary['total_removed']}")
    print(f"  Files modified: {summary['total_modified']}")
    print(f"  Files unchanged: {summary['total_unchanged']}")
    print(f"  Overall drift: {summary['drift_percentage']}%")
    
    # Show parsed changes
    changes = drift_report["parsed_changes"]
    if changes["added"]:
        print(f"\n  New files added ({len(changes['added'])}):")
        for f in changes["added"][:3]:
            print(f"    - {f['file_path']}")
    
    if changes["modified"]:
        print(f"\n  Files modified ({len(changes['modified'])}):")
        for f in changes["modified"][:3]:
            size_change = f.get("size_change", 0)
            print(f"    - {f['file_path']} (size change: {size_change:+d} bytes)")
    
    if changes["removed"]:
        print(f"\n  Files removed ({len(changes['removed'])}):")
        for f in changes["removed"][:3]:
            print(f"    - {f}")


def example_rolling_window_analysis():
    """Example: Analyse drift over rolling time windows."""
    print("\n" + "=" * 70)
    print("Example 3: Rolling Window Analysis")
    print("=" * 70)
    
    ingestion = BitbucketCodeIngestion(
        host="https://bitbucket.company.com",
        username="user@company.com",
        password="app-password",
        is_cloud=False,
    )
    
    # Analyse drift month-by-month
    print("\nMonthly drift analysis (last 6 months):")
    
    base_date = datetime.now()
    for month in range(6, 0, -1):
        v1 = base_date - timedelta(days=30*month)
        v2 = base_date - timedelta(days=30*(month-1))
        
        print(f"\n  {v1.date()} → {v2.date()}:")
        
        try:
            drift_report = ingestion.analyse_version_drift(
                project_key="PROJ",
                repo_slug="my-repo",
                v1_date=v1,
                v2_date=v2,
            )
            
            summary = drift_report["summary"]
            print(f"    - Added: {summary['total_added']}, " 
                  f"Modified: {summary['total_modified']}, "
                  f"Removed: {summary['total_removed']}, "
                  f"Drift: {summary['drift_percentage']}%")
        except Exception as e:
            print(f"    - Error: {e}")


if __name__ == "__main__":
    print("""
    Bitbucket Date-Based Code Extraction & Drift Analysis Examples
    ==============================================================
    
    These examples show how to:
    1. Extract files modified before a specific date
    2. Compare code versions for drift analysis
    3. Perform rolling window analysis over time
    
    NOTE: These are template examples. Configure with your Bitbucket credentials
    and repository details before running.
    """)
    
    # Uncomment to run examples:
    # example_date_based_extraction()
    # example_version_drift_analysis()
    # example_rolling_window_analysis()
    
    print("\nSee script source for full examples.")
