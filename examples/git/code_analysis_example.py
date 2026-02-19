"""Example: Using code_parser to analyse Bitbucket repositories.

This demonstrates how to use the CodeParser to extract architectural
information from a codebase for migration planning.

Usage:
    python examples/git/code_analysis_example.py /path/to/bitbucket/repo
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest.git.code_parser import CodeParser, FileType


def analyse_repository(repo_path: str) -> dict:
    """Analyse a repository for dependencies and service structure.
    
    Args:
        repo_path: Path to the repository to analyse.
        
    Returns:
        Dictionary containing analysis results with:
        - services: List of detected services
        - dependencies: External dependencies by service
        - internal_calls: Cross-service dependencies
        - endpoints: Public endpoints
        - message_flows: Message queue integrations
    """
    parser = CodeParser()
    repo_path = Path(repo_path)
    
    analysis = {
        "repo": str(repo_path),
        "services": {},
        "external_dependencies": defaultdict(list),
        "internal_calls": defaultdict(list),
        "endpoints": defaultdict(list),
        "message_flows": defaultdict(list),
        "parsed_files": 0,
        "errors": [],
    }
    
    # Find all parseable code files
    patterns = [
        "**/*.java",
        "**/*.groovy",
        "**/pom.xml",
        "**/build.gradle",
        "**/*-mule.xml",
        "**/*.properties",
        "**/*.yml",
        "**/*.yaml",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(repo_path.glob(pattern))
    
    if not files:
        print(f"No code files found in {repo_path}")
        return analysis
    
    print(f"Found {len(files)} files to analyse")
    
    for file_path in sorted(files):
        # Skip certain paths
        if any(skip in str(file_path) for skip in [".git", "target", "build", "node_modules"]):
            continue
        
        try:
            result = parser.parse_file(str(file_path))
            analysis["parsed_files"] += 1
            
            # Store service information
            if result.service_name:
                if result.service_name not in analysis["services"]:
                    analysis["services"][result.service_name] = {
                        "type": result.service_type,
                        "files": [],
                        "exports": [],
                    }
                analysis["services"][result.service_name]["files"].append(str(file_path.relative_to(repo_path)))
                analysis["services"][result.service_name]["exports"].extend(result.exports)
            
            # Collect external dependencies
            for dep in result.external_dependencies:
                analysis["external_dependencies"][result.service_name or "unknown"].append(dep)
            
            # Collect internal calls
            for call in result.internal_calls:
                analysis["internal_calls"][result.service_name or "unknown"].append(call)
            
            # Collect endpoints
            for ep in result.endpoints:
                analysis["endpoints"][result.service_name or "unknown"].append(ep)
            
            # Collect message flows
            for queue in result.message_queues:
                analysis["message_flows"][result.service_name or "unknown"].append(queue)
            
            if result.errors:
                analysis["errors"].extend(result.errors)
        
        except Exception as e:
            analysis["errors"].append(f"Failed to parse {file_path}: {str(e)}")
    
    # Convert defaultdicts to regular dicts for JSON serialisation
    analysis["external_dependencies"] = dict(analysis["external_dependencies"])
    analysis["internal_calls"] = dict(analysis["internal_calls"])
    analysis["endpoints"] = dict(analysis["endpoints"])
    analysis["message_flows"] = dict(analysis["message_flows"])
    
    return analysis


def print_analysis(analysis: dict) -> None:
    """Pretty print analysis results."""
    print("\n" + "=" * 80)
    print(f"REPOSITORY ANALYSIS: {analysis['repo']}")
    print("=" * 80)
    
    print(f"\nFiles analysed: {analysis['parsed_files']}")
    print(f"Services detected: {len(analysis['services'])}")
    
    if analysis['services']:
        print("\n--- SERVICES ---")
        for service_name, info in sorted(analysis['services'].items()):
            print(f"\n  {service_name}")
            print(f"    Type: {info['type'] or 'unknown'}")
            print(f"    Files: {len(info['files'])}")
            if info['exports']:
                print(f"    Exports: {', '.join(info['exports'][:5])}")
    
    if analysis['external_dependencies']:
        print("\n--- EXTERNAL DEPENDENCIES (by service) ---")
        for service, deps in sorted(analysis['external_dependencies'].items()):
            unique_deps = list(set(deps))
            if unique_deps:
                print(f"\n  {service}:")
                for dep in unique_deps[:5]:
                    print(f"    - {dep}")
                if len(unique_deps) > 5:
                    print(f"    ... and {len(unique_deps) - 5} more")
    
    if analysis['endpoints']:
        print("\n--- REST ENDPOINTS (by service) ---")
        for service, endpoints in sorted(analysis['endpoints'].items()):
            if endpoints:
                print(f"\n  {service}:")
                for ep in set(endpoints)[:3]:
                    print(f"    - {ep}")
    
    if analysis['message_flows']:
        print("\n--- MESSAGE QUEUES/TOPICS (by service) ---")
        for service, queues in sorted(analysis['message_flows'].items()):
            if queues:
                print(f"\n  {service}:")
                for queue in set(queues):
                    print(f"    - {queue}")
    
    if analysis['internal_calls']:
        print("\n--- INTERNAL SERVICE CALLS (potential dependencies) ---")
        for service, calls in sorted(analysis['internal_calls'].items()):
            unique_calls = list(set(calls))
            if unique_calls:
                print(f"\n  {service} calls:")
                for call in unique_calls[:5]:
                    print(f"    - {call}")
                if len(unique_calls) > 5:
                    print(f"    ... and {len(unique_calls) - 5} more")
    
    if analysis['errors']:
        print(f"\n--- ERRORS ({len(analysis['errors'])}) ---")
        for error in analysis['errors'][:5]:
            print(f"  - {error}")
        if len(analysis['errors']) > 5:
            print(f"  ... and {len(analysis['errors']) - 5} more")
    
    print("\n" + "=" * 80)


def save_analysis_json(analysis: dict, output_path: str) -> None:
    """Save analysis to JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nAnalysis saved to: {output}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/git/code_analysis_example.py <repo_path> [output.json]")
        print("\nExample:")
        print("  python examples/git/code_analysis_example.py ~/projects/my-service")
        print("  python examples/git/code_analysis_example.py ~/projects/my-service analysis.json")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "code_analysis.json"
    
    print(f"Analysing repository: {repo_path}")
    analysis = analyse_repository(repo_path)
    
    print_analysis(analysis)
    save_analysis_json(analysis, output_path)
