"""End-to-end Bitbucket code ingestion pipeline.

Integrates BitbucketConnector and CodeParser to:
1. Connect to a Bitbucket instance
2. Discover projects and repositories
3. Clone or walk repositories
4. Parse Groovy, Java, and config files for metadata
5. Store parsed results for downstream RAG processing
6. Support date-based extraction and version drift analysis

Standalone code for testing and development.

Usage:
    python3 bitbucket_code_ingestion.py \\
        --host https://bitbucket.company.com \\
        --username user@company.com \\
        --password app-password \\
        --project PROJ \\
        --repo my-repo \\
        --output results.json \\
        --file-types groovy java gradle

    Or for date-based extraction:
    
    from examples.git.bitbucket_code_ingestion import BitbucketCodeIngestion
    from datetime import datetime
    
    ingestion = BitbucketCodeIngestion(
        host="https://bitbucket.company.com",
        username="user@company.com",
        password="app-password",
    )
    
    # Extract files modified before a date
    results = ingestion.ingest_repository_dated(
        "PROJ", "my-repo",
        modified_before=datetime(2025, 6, 1)
    )
    
    # Compare versions for drift analysis
    drift = ingestion.analyse_version_drift(
        "PROJ", "my-repo",
        v1_date=datetime(2025, 1, 1),
        v2_date=datetime(2026, 1, 1)
    )
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.ingest.git.bitbucket_connector import BitbucketConnector, RepositoryWalker
from scripts.ingest.git.code_parser import CodeParser
from scripts.utils.logger import create_module_logger

get_logger, _ = create_module_logger("ingest")

logger = get_logger()


@dataclass
class ParsedFile:
    """Result of parsing a single file from repository."""

    file_path: str
    repository: str
    project_key: str
    parse_result: Dict  # Serialised ParseResult


# ============================================================================
# Helper Methods
# ============================================================================


def _parse_and_aggregate_files(
    file_iterator,
    repo_slug: str,
    project_key: str,
    parser,
    include_dates: bool = False,
) -> Tuple[List[Dict], set, Dict, int]:
    """Parse files and aggregate metadata (helper function for both dated and non-dated ingestion).

    Shared logic for parsing file iterators and aggregating:
    - External dependencies
    - Service types
    - Parse errors

    Args:
        file_iterator: Iterator yielding (file_path, content[, mod_date])
        repo_slug: Repository slug
        project_key: Project key
        parser: CodeParser instance
        include_dates: If True, expects iterator to yield 3-tuples with mod_date

    Returns:
        Tuple of (parsed_files, external_deps set, service_types dict, parse_error_count)
    """
    parsed_files = []
    external_deps = set()
    service_types = {}
    parse_errors = 0

    for item in file_iterator:
        if include_dates:
            file_path, content, mod_date = item
        else:
            file_path, content = item
            mod_date = None

        try:
            logger.debug(f"Parsing {file_path}")

            # Create temp file reference for parser
            # Parser needs actual file path, not content
            temp_file = Path(file_path) if isinstance(file_path, (str, Path)) else file_path

            parse_result = parser.parse_file(str(temp_file))

            # Build parsed file entry
            parsed_entry = {
                "file_path": str(file_path),
                "repository": repo_slug,
                "project_key": project_key,
                "parse_result": parse_result.to_dict(),
            }

            if include_dates and mod_date:
                parsed_entry["modified_date"] = (
                    mod_date.isoformat() if hasattr(mod_date, "isoformat") else str(mod_date)
                )

            parsed_files.append(parsed_entry)

            # Aggregate metadata
            external_deps.update(parse_result.external_dependencies)
            if parse_result.service_type:
                service_types[parse_result.service_type] = (
                    service_types.get(parse_result.service_type, 0) + 1
                )

            if parse_result.errors:
                parse_errors += 1

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            parse_errors += 1

    return parsed_files, external_deps, service_types, parse_errors


def _clone_repository_safe(
    connector,
    project_key: str,
    repo_slug: str,
    clone_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict]]:
    """Clone repository with consistent error handling.

    Args:
        connector: BitbucketConnector instance
        project_key: Project key
        repo_slug: Repository slug
        clone_path: Optional override for clone path

    Returns:
        Tuple of (repo_path, error_dict) where error_dict is None on success
    """
    try:
        logger.info(f"Cloning repository {repo_slug}...")
        repo_path = connector.clone_repository(project_key, repo_slug, target_dir=clone_path)
        return repo_path, None
    except Exception as e:
        error_dict = {
            "repository": repo_slug,
            "project_key": project_key,
            "error": f"Clone failed: {str(e)}",
        }
        logger.error(f"Failed to clone repository: {e}")
        return None, error_dict


def _get_file_iterator(
    walker,
    file_types: List[str],
    dated: bool = False,
    modified_before: Optional[datetime] = None,
    modified_after: Optional[datetime] = None,
):
    """Route to appropriate file walker based on file types and date filters.

    Args:
        walker: RepositoryWalker instance
        file_types: List of file types (groovy, java, gradle, all, or patterns)
        dated: If True, use dated walkers
        modified_before: Upper date bound (for dated walkers)
        modified_after: Lower date bound (for dated walkers)

    Returns:
        Iterator yielding (file_path, content[, mod_date])
    """
    if "all" in file_types:
        if dated:
            return walker.walk_all_code_files_dated(
                modified_before=modified_before,
                modified_after=modified_after,
            )
        else:
            return walker.walk_all_code_files()
    elif "groovy" in file_types or "gradle" in file_types:
        if dated:
            return walker.walk_groovy_files_dated(
                modified_before=modified_before,
                modified_after=modified_after,
            )
        else:
            return walker.walk_groovy_files()
    elif "java" in file_types:
        if dated:
            return walker.walk_java_files_dated(
                modified_before=modified_before,
                modified_after=modified_after,
            )
        else:
            return walker.walk_java_files()
    else:
        # Custom patterns (fallback to all for dated, specific for non-dated)
        if dated:
            return walker.walk_all_code_files_dated(
                modified_before=modified_before,
                modified_after=modified_after,
            )
        else:
            patterns = [f"**/*{ft}" if ft.startswith(".") else f"**/*{ft}" for ft in file_types]
            return walker.walk_by_pattern(patterns)


class BitbucketCodeIngestion:
    """Pipeline for ingesting code from Bitbucket repositories."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        is_cloud: bool = False,
        verify_ssl: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialise ingestion pipeline.

        Args:
            host: Bitbucket host URL
            username: Username or email
            password: Password or app password
            is_cloud: True for Bitbucket Cloud, False for Server
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum retry attempts for rate-limited requests
            retry_delay: Initial delay before retrying (uses exponential backoff)
        """
        self.connector = BitbucketConnector(
            host=host,
            username=username,
            password=password,
            is_cloud=is_cloud,
            verify_ssl=verify_ssl,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.parser = CodeParser()
        self.logger = get_logger()

    def list_projects(self) -> List[Dict]:
        """List all projects in the Bitbucket instance.

        Returns:
            List of project dictionaries
        """
        projects = self.connector.list_projects()
        return [
            {
                "key": p.key,
                "name": p.name,
                "description": p.description,
                "url": p.url,
            }
            for p in projects
        ]

    def list_repositories(self, project_key: str) -> List[Dict]:
        """List repositories in a project.

        Args:
            project_key: Project key/slug

        Returns:
            List of repository dictionaries
        """
        repos = self.connector.list_repositories(project_key)
        return [
            {
                "slug": r.slug,
                "name": r.name,
                "project_key": r.project_key,
                "description": r.description,
                "url": r.url,
            }
            for r in repos
        ]

    def ingest_repository(
        self,
        project_key: str,
        repo_slug: str,
        file_types: Optional[List[str]] = None,
        clone_path: Optional[str] = None,
    ) -> Dict:
        """Ingest a repository: clone, parse, and extract metadata.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            file_types: File types to parse (groovy, java, gradle, all). Defaults to all.
            clone_path: Override clone path (defaults to temp directory)

        Returns:
            Dictionary with ingestion results:
            {
                "repository": "repo-slug",
                "project_key": "PROJ",
                "clone_path": "/path/to/repo",
                "directory_structure": {...},
                "parsed_files": [
                    {"file_path": "...", "parse_result": {...}},
                    ...
                ],
                "summary": {
                    "total_files": N,
                    "parsed_successfully": N,
                    "parse_errors": N,
                    "external_dependencies": [...],
                    "service_types": {...},
                },
            }
        """
        self.logger.info(f"Starting ingestion of {project_key}/{repo_slug}")

        if file_types is None:
            file_types = ["all"]

        # Clone repository
        repo_path, error_dict = _clone_repository_safe(
            self.connector, project_key, repo_slug, clone_path
        )
        if error_dict:
            return error_dict

        # Walk and parse files
        walker = RepositoryWalker(repo_path)
        directory_structure = walker.get_directory_structure()

        self.logger.info(f"Walking repository structure from {repo_path}")
        self.logger.info(f"Directory stats: {directory_structure}")

        # Get file iterator
        files_iter = _get_file_iterator(walker, file_types, dated=False)

        # Parse and aggregate
        self.logger.info(f"Parsing files (types: {file_types})...")
        parsed_files, external_deps, service_types, parse_errors = _parse_and_aggregate_files(
            files_iter, repo_slug, project_key, self.parser, include_dates=False
        )

        self.logger.info(f"Parsed {len(parsed_files)} files, {parse_errors} errors")

        return {
            "repository": repo_slug,
            "project_key": project_key,
            "clone_path": repo_path,
            "directory_structure": directory_structure,
            "parsed_files": parsed_files,
            "summary": {
                "total_files": len(parsed_files),
                "parsed_successfully": len(parsed_files) - parse_errors,
                "parse_errors": parse_errors,
                "external_dependencies": sorted(list(external_deps)),
                "service_types": service_types,
                "groovy_files_found": len(directory_structure.get("groovy_files", [])),
                "java_files_found": len(directory_structure.get("java_files", [])),
                "gradle_files_found": len(directory_structure.get("gradle_files", [])),
                "jenkins_files_found": len(directory_structure.get("jenkins_files", [])),
            },
        }

    def ingest_pull_request(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        clone_path: Optional[str] = None,
    ) -> Dict:
        """Ingest code changes from a pull request.

        Clones the repository, checks out the PR branch, and parses changed files.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            pr_id: Pull request ID
            clone_path: Override clone path

        Returns:
            Dictionary with PR ingestion results
        """
        self.logger.info(f"Starting PR ingestion {project_key}/{repo_slug}#PR-{pr_id}")

        # Get PR metadata
        prs = self.connector.list_pull_requests(project_key, repo_slug, state="ALL")
        pr = next((p for p in prs if p.id == pr_id), None)

        if not pr:
            self.logger.error(f"Pull request {pr_id} not found")
            return {
                "repository": repo_slug,
                "project_key": project_key,
                "pr_id": pr_id,
                "error": "Pull request not found",
            }

        self.logger.info(f"PR {pr_id}: {pr.title} ({pr.state})")

        # Clone repository
        repo_path = self.connector.clone_repository(project_key, repo_slug, target_dir=clone_path)

        # Get list of changed files in PR
        changed_files = self.connector.get_pull_request_files(project_key, repo_slug, pr_id)
        self.logger.info(f"PR changes {len(changed_files)} files")

        # Parse only changed files
        parsed_files = []
        for change in changed_files:
            file_path = change["path"]
            status = change["status"]

            # Skip deleted files
            if status == "DELETED":
                continue

            full_path = Path(repo_path) / file_path

            if not full_path.exists():
                self.logger.warning(f"File not found in PR branch: {file_path}")
                continue

            # Parse only code files
            if full_path.suffix not in [".groovy", ".java", ".gradle", ".xml", ".properties"]:
                continue

            try:
                self.logger.debug(f"Parsing PR change: {file_path} ({status})")
                parse_result = self.parser.parse_file(str(full_path))

                parsed_files.append(
                    {
                        "file_path": file_path,
                        "status": status,
                        "parse_result": parse_result.to_dict(),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to parse PR file {file_path}: {e}")

        return {
            "repository": repo_slug,
            "project_key": project_key,
            "pull_request": {
                "id": pr.id,
                "title": pr.title,
                "state": pr.state,
                "source_branch": pr.source_branch,
                "target_branch": pr.target_branch,
                "author": pr.author,
            },
            "clone_path": repo_path,
            "parsed_files": parsed_files,
            "summary": {
                "total_changes": len(changed_files),
                "parsed_successfully": len(parsed_files),
                "added": len([c for c in changed_files if c["status"] == "ADDED"]),
                "modified": len([c for c in changed_files if c["status"] == "MODIFIED"]),
                "deleted": len([c for c in changed_files if c["status"] == "DELETED"]),
            },
        }

    def ingest_multiple_repositories(
        self,
        project_key: str,
        repo_filters: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Ingest multiple repositories in a project.

        Args:
            project_key: Project key/slug
            repo_filters: List of repo slugs to filter (None = all repos)
            file_types: File types to parse

        Returns:
            List of ingestion results for each repository
        """
        repos = self.connector.list_repositories(project_key)

        if repo_filters:
            repos = [r for r in repos if r.slug in repo_filters]

        self.logger.info(f"Ingesting {len(repos)} repositories from project {project_key}")

        results = []
        for repo in repos:
            try:
                result = self.ingest_repository(project_key, repo.slug, file_types=file_types)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to ingest {repo.slug}: {e}")
                results.append(
                    {
                        "repository": repo.slug,
                        "project_key": project_key,
                        "error": str(e),
                    }
                )

        return results

    def ingest_repository_dated(
        self,
        project_key: str,
        repo_slug: str,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
        file_types: Optional[List[str]] = None,
        clone_path: Optional[str] = None,
    ) -> Dict:
        """Ingest repository files filtered by modification date.

        Useful for extracting only files modified within a specific time window.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            modified_before: Only include files modified before this date
            modified_after: Only include files modified after this date
            file_types: File types to parse (groovy, java, gradle, all)
            clone_path: Override clone path

        Returns:
            Dictionary with dated ingestion results including modification dates
        """
        self.logger.info(
            f"Starting dated ingestion of {project_key}/{repo_slug} "
            f"(before={modified_before}, after={modified_after})"
        )

        if file_types is None:
            file_types = ["all"]

        # Clone repository
        repo_path, error_dict = _clone_repository_safe(
            self.connector, project_key, repo_slug, clone_path
        )
        if error_dict:
            return error_dict

        walker = RepositoryWalker(repo_path)

        # Get dated file iterator
        files_iter = _get_file_iterator(
            walker,
            file_types,
            dated=True,
            modified_before=modified_before,
            modified_after=modified_after,
        )

        # Parse and aggregate (include_dates=True for dated files)
        self.logger.info(f"Parsing dated files (types: {file_types})")
        parsed_files, external_deps, service_types, parse_errors = _parse_and_aggregate_files(
            files_iter, repo_slug, project_key, self.parser, include_dates=True
        )

        self.logger.info(f"Parsed {len(parsed_files)} dated files, {parse_errors} errors")

        return {
            "repository": repo_slug,
            "project_key": project_key,
            "clone_path": repo_path,
            "date_filter": {
                "modified_before": modified_before.isoformat() if modified_before else None,
                "modified_after": modified_after.isoformat() if modified_after else None,
            },
            "parsed_files": parsed_files,
            "summary": {
                "total_files": len(parsed_files),
                "parsed_successfully": len(parsed_files) - parse_errors,
                "parse_errors": parse_errors,
                "external_dependencies": sorted(list(external_deps)),
                "service_types": service_types,
            },
        }

    def analyse_version_drift(
        self,
        project_key: str,
        repo_slug: str,
        v1_date: datetime,
        v2_date: datetime,
        clone_path: Optional[str] = None,
    ) -> Dict:
        """Analyse code drift between two versions (dates).

        Compares code at two different points in time and reports:
        - Files added/removed/modified
        - Drift percentage
        - Parse results for changed files

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            v1_date: First version date (typically older)
            v2_date: Second version date (typically newer)
            clone_path: Override clone path

        Returns:
            Dictionary with version comparison and drift analysis
        """
        self.logger.info(
            f"Starting version drift analysis {project_key}/{repo_slug} "
            f"v1={v1_date} vs v2={v2_date}"
        )

        # Clone repository
        try:
            repo_path = self.connector.clone_repository(
                project_key, repo_slug, target_dir=clone_path
            )
        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}")
            return {
                "repository": repo_slug,
                "project_key": project_key,
                "error": f"Clone failed: {str(e)}",
            }

        walker = RepositoryWalker(repo_path)

        # Compare versions
        comparison = walker.compare_versions(v1_date, v2_date)

        self.logger.info(f"Drift analysis complete: {comparison['summary']}")

        # Parse modified/added files in V2
        parsed_changes = {
            "added": [],
            "modified": [],
            "removed": [],
        }

        # Parse added files
        for file_path in comparison["summary"]["added"]:
            if file_path in comparison["v2_files"]:
                try:
                    content, mod_date = comparison["v2_files"][file_path]
                    # Create temp file for parsing
                    temp_file = Path(repo_path) / file_path
                    parse_result = self.parser.parse_file(str(temp_file))
                    parsed_changes["added"].append(
                        {
                            "file_path": file_path,
                            "modified_date": mod_date.isoformat(),
                            "parse_result": parse_result.to_dict(),
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to parse added file {file_path}: {e}")

        # Parse modified files
        for file_path in comparison["summary"]["modified"]:
            if file_path in comparison["v2_files"]:
                try:
                    v1_content, v1_date_mod = comparison["v1_files"][file_path]
                    v2_content, v2_date_mod = comparison["v2_files"][file_path]
                    temp_file = Path(repo_path) / file_path
                    parse_result = self.parser.parse_file(str(temp_file))

                    parsed_changes["modified"].append(
                        {
                            "file_path": file_path,
                            "v1_modified_date": v1_date_mod.isoformat(),
                            "v2_modified_date": v2_date_mod.isoformat(),
                            "v1_size": len(v1_content),
                            "v2_size": len(v2_content),
                            "size_change": len(v2_content) - len(v1_content),
                            "parse_result": parse_result.to_dict(),
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to parse modified file {file_path}: {e}")

        return {
            "repository": repo_slug,
            "project_key": project_key,
            "clone_path": repo_path,
            "version_comparison": {
                "v1_date": v1_date.isoformat(),
                "v2_date": v2_date.isoformat(),
            },
            "summary": comparison["summary"],
            "parsed_changes": parsed_changes,
        }


def main():
    """Command-line entry point for Bitbucket code ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest Groovy/Java code from Bitbucket repositories"
    )

    parser.add_argument("--host", required=True, help="Bitbucket host URL")
    parser.add_argument("--username", required=True, help="Username or email")
    parser.add_argument("--password", required=True, help="Password or app password")
    parser.add_argument("--is-cloud", action="store_true", help="Use Bitbucket Cloud")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL verification")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--list-projects", action="store_true", help="List all projects")
    mode_group.add_argument(
        "--list-repos", help="List repositories in project (provide PROJECT_KEY)"
    )
    mode_group.add_argument(
        "--ingest-repo",
        nargs=2,
        metavar=("PROJECT_KEY", "REPO_SLUG"),
        help="Ingest a repository",
    )
    mode_group.add_argument(
        "--ingest-pr",
        nargs=3,
        metavar=("PROJECT_KEY", "REPO_SLUG", "PR_ID"),
        help="Ingest a pull request",
    )
    mode_group.add_argument(
        "--ingest-project",
        help="Ingest all repositories in a project (provide PROJECT_KEY)",
    )

    parser.add_argument(
        "--file-types",
        nargs="+",
        default=["all"],
        help="File types to parse (groovy, java, gradle, all, etc.)",
    )
    parser.add_argument(
        "--clone-path",
        help="Override default clone path",
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Initialise ingestion pipeline
    ingestion = BitbucketCodeIngestion(
        host=args.host,
        username=args.username,
        password=args.password,
        is_cloud=args.is_cloud,
        verify_ssl=not args.no_verify_ssl,
    )

    result = None

    try:
        if args.list_projects:
            logger.info("Listing projects...")
            result = {"projects": ingestion.list_projects()}

        elif args.list_repos:
            logger.info(f"Listing repositories in {args.list_repos}...")
            result = {"repositories": ingestion.list_repositories(args.list_repos)}

        elif args.ingest_repo:
            project_key, repo_slug = args.ingest_repo
            logger.info(f"Ingesting repository {project_key}/{repo_slug}...")
            result = ingestion.ingest_repository(
                project_key,
                repo_slug,
                file_types=args.file_types,
                clone_path=args.clone_path,
            )

        elif args.ingest_pr:
            project_key, repo_slug, pr_id = args.ingest_pr
            logger.info(f"Ingesting PR {project_key}/{repo_slug}#{pr_id}...")
            result = ingestion.ingest_pull_request(
                project_key,
                repo_slug,
                int(pr_id),
                clone_path=args.clone_path,
            )

        elif args.ingest_project:
            logger.info(f"Ingesting all repositories in {args.ingest_project}...")
            result = {
                "project_key": args.ingest_project,
                "repositories": ingestion.ingest_multiple_repositories(
                    args.ingest_project,
                    file_types=args.file_types,
                ),
            }

        # Output results
        if result:
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results written to {args.output}")
            else:
                print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
