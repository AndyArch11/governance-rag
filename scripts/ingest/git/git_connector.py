"""Abstract Git connector interface for multi-platform code ingestion.

This module defines the abstract interface that all Git hosting platforms
(Bitbucket, GitHub, GitLab, Azure DevOps, etc.) must implement to participate
in the unified Git ingestion framework.

Each concrete implementation handles:
- Platform-specific authentication
- API interactions
- Repository discovery
- File walking and cloning
- Pull request listing

The abstract interface ensures consistent behaviour across all platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class GitProject:
    """Common project metadata across Git platforms."""

    key: str  # Bitbucket key, GitHub org, GitLab group, Azure project
    name: str
    description: Optional[str] = None
    url: Optional[str] = None


@dataclass
class GitRepository:
    """Common repository metadata across Git platforms."""

    slug: str  # Bitbucket/GitHub/GitLab slug or Azure repo name
    name: str
    project_key: str  # Parent project identifier
    description: Optional[str] = None
    url: Optional[str] = None
    default_branch: str = "main"


@dataclass
class GitPullRequest:
    """Common pull request metadata."""

    id: str
    title: str
    state: str  # open|closed|merged|declined
    source_branch: str
    target_branch: str
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    url: Optional[str] = None


class RepositoryWalker(ABC):
    """Abstract base for walking repository file structure across platforms.

    Each platform implements file discovery, filtering, and version comparison
    in a standardised way.
    """

    def __init__(self, repo_path: str):
        """Initialise walker for local repository path.

        Args:
            repo_path: Local path to cloned repository
        """
        self.repo_path = Path(repo_path)

    @abstractmethod
    def walk_files(
        self,
        extensions: Optional[List[str]] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, Optional[datetime]], None, None]:
        """Walk repository files with optional filtering.

        Args:
            extensions: File extensions to include (e.g., ['.java', '.groovy'])
            modified_before: Only include files modified before this date
            modified_after: Only include files modified after this date

        Yields:
            (file_path, content, modification_date) tuples
        """
        pass

    @abstractmethod
    def compare_versions(
        self,
        date1: datetime,
        date2: datetime,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare repository state across two dates.

        Useful for drift analysis: what changed between two points in time?

        Args:
            date1: First point in time
            date2: Second point in time
            extensions: File types to compare

        Returns:
            Dict with 'added', 'removed', 'modified', 'unchanged' file lists
        """
        pass


class GitConnector(ABC):
    """Abstract Git connector for multi-platform code ingestion.

    Implementations handle authentication, repository discovery, cloning,
    and file operations for specific Git hosting platforms.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and test connection to Git platform.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If authentication fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (sessions, connections, etc.)."""
        pass

    @abstractmethod
    def list_projects(self) -> List[GitProject]:
        """List all accessible projects/organisations.

        Returns:
            List of GitProject objects
        """
        pass

    @abstractmethod
    def list_repositories(self, project_key: str) -> List[GitRepository]:
        """List all repositories in a project.

        Args:
            project_key: Project identifier (key, org, group, etc.)

        Returns:
            List of GitRepository objects
        """
        pass

    @abstractmethod
    def clone_repository(
        self,
        project_key: str,
        repo_slug: str,
        target_dir: str,
        branch: str = "main",
    ) -> str:
        """Clone repository to local directory.

        Args:
            project_key: Project identifier
            repo_slug: Repository identifier
            target_dir: Base directory for clones
            branch: Branch to clone (default: main/master)

        Returns:
            Local path to cloned repository

        Raises:
            RuntimeError: If clone fails

        Note:
            Repository reset/re-cloning is handled at ingestion start via
            GitConnectorFactory or the main ingest script, not per-repository.
        """
        pass

    @abstractmethod
    def get_repository_walker(self, repo_path: str) -> RepositoryWalker:
        """Get file walker for repository.

        Args:
            repo_path: Local path to cloned repository

        Returns:
            RepositoryWalker instance for this platform
        """
        pass

    @abstractmethod
    def list_pull_requests(
        self,
        project_key: str,
        repo_slug: str,
        state: str = "open",
        limit: int = 50,
    ) -> List[GitPullRequest]:
        """List pull requests in repository.

        Args:
            project_key: Project identifier
            repo_slug: Repository identifier
            state: Filter by state (open, closed, merged, etc.)
            limit: Maximum number to return

        Returns:
            List of GitPullRequest objects
        """
        pass

    @abstractmethod
    def get_pull_request_files(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """List files changed in a pull request.

        Args:
            project_key: Project identifier
            repo_slug: Repository identifier
            pr_id: Pull request ID
            limit: Optional maximum number of files to return

        Returns:
            List of file change dicts with keys: path, status, old_path
        """
        pass

    @abstractmethod
    def get_file_content(
        self,
        project_key: str,
        repo_slug: str,
        file_path: str,
        branch: str = "main",
    ) -> str:
        """Get file content directly from repository (without cloning).

        Useful for small files or quick lookups.

        Args:
            project_key: Project identifier
            repo_slug: Repository identifier
            file_path: Path to file in repository
            branch: Branch to fetch from

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        pass

    def __enter__(self):
        """Context manager entry - establishes connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connection."""
        self.close()
        return False
