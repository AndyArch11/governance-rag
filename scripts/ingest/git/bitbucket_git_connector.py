"""Bitbucket Git connector implementing unified GitConnector interface.

Adapts existing BitbucketConnector to conform to the abstract GitConnector
interface, enabling use with the unified multi-provider ingestion framework.

Supports:
- Bitbucket Cloud
- Bitbucket Server (self-hosted)
- All existing functionality from bitbucket_connector.py

This module bridges the existing Bitbucket implementation with the subsequent
unified architecture.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.ingest.git.bitbucket_connector import (
    BitbucketConnector,
    BitbucketProject,
    BitbucketRepository,
)
from scripts.ingest.git.bitbucket_connector import PullRequest as BitbucketPR
from scripts.ingest.git.bitbucket_connector import RepositoryWalker as BitbucketRepositoryWalker
from scripts.ingest.git.git_connector import (
    GitConnector,
    GitProject,
    GitPullRequest,
    GitRepository,
    RepositoryWalker,
)
from scripts.utils.logger import create_module_logger

get_logger, _ = create_module_logger("ingest")
logger = get_logger()


class BitbucketRepositoryWalkerAdapter(RepositoryWalker):
    """Adapts BitbucketRepositoryWalker to RepositoryWalker interface."""

    def __init__(self, repo_path: str):
        """Initialise walker.

        Args:
            repo_path: Local path to cloned repository
        """
        super().__init__(repo_path)
        self._walker = BitbucketRepositoryWalker(Path(repo_path))

    def walk_files(
        self,
        extensions: Optional[List[str]] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ):
        """Walk repository files using underlying BitbucketRepositoryWalker.

        Args:
            extensions: File extensions to include
            modified_before: Only files modified before this date
            modified_after: Only files modified after this date

        Yields:
            (relative_path, content, filesystem_mtime) tuples
        """
        # Delegate to existing walker - it handles filtering internally
        # The BitbucketRepositoryWalker.walk_files matches our interface
        yield from self._walker.walk_files(
            extensions=extensions,
            modified_before=modified_before,
            modified_after=modified_after,
        )

    def compare_versions(
        self,
        date1: datetime,
        date2: datetime,
        extensions: Optional[List[str]] = None,
    ):
        """Compare repository state across two dates.

        Args:
            date1: First point in time
            date2: Second point in time
            extensions: File types to compare

        Returns:
            Dict with 'added', 'removed', 'modified' file lists
        """
        # Delegate to existing walker
        return self._walker.compare_versions(date1, date2, extensions)


class BitbucketGitConnector(GitConnector):
    """Bitbucket implementation of unified GitConnector interface.

    Wraps existing BitbucketConnector to conform to the abstract interface,
    enabling integration with the unified multi-provider framework.
    """

    def __init__(
        self,
        host: str = "https://bitbucket.org",
        username: str = "",
        password: str = "",
        is_cloud: bool = True,
        verify_ssl: bool = True,
        api_username: str = "",
    ):
        """Initialise Bitbucket connector.

        Args:
            host: Bitbucket host URL
            username: Bitbucket username for git operations
            password: Bitbucket password or app password
            is_cloud: True for Bitbucket Cloud, False for Server
            verify_ssl: Whether to verify SSL certificates
            api_username: Email/username for API calls (Cloud typically needs email)
        """
        self.host = host
        self.username = username
        self.api_username = api_username or username  # Fall back to username
        self.password = password
        self.is_cloud = is_cloud
        self.verify_ssl = verify_ssl
        self._connector = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Bitbucket instance.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If authentication fails
        """
        try:
            # Create connector if not already created
            if self._connector is None:
                self._connector = BitbucketConnector(
                    host=self.host,
                    username=self.username,
                    password=self.password,
                    is_cloud=self.is_cloud,
                    verify_ssl=self.verify_ssl,
                    api_username=self.api_username,
                )

            # Call connect on the underlying connector if it has the method
            if hasattr(self._connector, "connect"):
                self._connector.connect()

            self._connected = True
            logger.info(f"Connected to Bitbucket ({'Cloud' if self.is_cloud else 'Server'})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Bitbucket: {e}")
            raise ConnectionError(f"Bitbucket authentication failed: {e}")

    def close(self) -> None:
        """Close connection and clean up resources."""
        if self._connector:
            # Call close on the underlying connector if it has the method
            if hasattr(self._connector, "close"):
                self._connector.close()
            # Otherwise try to close the session if it exists
            elif hasattr(self._connector, "session"):
                if hasattr(self._connector.session, "close"):
                    self._connector.session.close()
        self._connected = False

    def list_projects(self) -> List[GitProject]:
        """List all projects in Bitbucket instance.

        Returns:
            List of GitProject objects
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            bitbucket_projects = self._connector.list_projects()
            projects = [
                GitProject(
                    key=p.key,
                    name=p.name,
                    description=p.description,
                    url=p.url,
                )
                for p in bitbucket_projects
            ]
            logger.info(f"Found {len(projects)} Bitbucket projects")
            return projects
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return []

    def list_repositories(self, project_key: str) -> List[GitRepository]:
        """List repositories in a Bitbucket project.

        Args:
            project_key: Bitbucket project key

        Returns:
            List of GitRepository objects
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            bitbucket_repos = self._connector.list_repositories(project_key)
            repos = [
                GitRepository(
                    slug=r.slug,
                    name=r.name,
                    project_key=r.project_key,
                    description=r.description,
                    url=r.url,
                    default_branch=self._get_default_branch(r),
                )
                for r in bitbucket_repos
            ]
            logger.info(f"Found {len(repos)} repositories in {project_key}")
            return repos
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return []

    @staticmethod
    def _get_default_branch(repo: BitbucketRepository) -> str:
        """Extract default branch from Bitbucket repository.

        Falls back to 'master' when the repository does not provide a value.
        """
        return repo.default_branch or "master"

    def clone_repository(
        self,
        project_key: str,
        repo_slug: str,
        target_dir: str,
        branch: str = "master",
    ) -> str:
        """Clone Bitbucket repository.

        Args:
            project_key: Bitbucket project key
            repo_slug: Repository slug
            target_dir: Base directory for clones (will create project/repo subdirectory)
            branch: Branch to clone

        Returns:
            Local path to cloned repository

        Note:
            Repository reset/re-cloning should be handled at ingestion start,
            not per-repository. See GitIngestConfig.git_reset_repo
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            # Create subdirectory for this repo: base_dir/bitbucket/project_key/repo_slug
            base_dir = Path(target_dir)
            if base_dir.name != "bitbucket":
                base_dir = base_dir / "bitbucket"
            repo_clone_dir = str(base_dir / project_key / repo_slug)

            repo_path = self._connector.clone_repository(
                project_key=project_key,
                repo_slug=repo_slug,
                target_dir=repo_clone_dir,
                branch=branch,
            )
            logger.info(f"Cloned {project_key}/{repo_slug} to {repo_path}")
            return repo_path
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise

    def get_repository_walker(self, repo_path: str) -> RepositoryWalker:
        """Get file walker for Bitbucket repository.

        Args:
            repo_path: Local path to cloned repository

        Returns:
            BitbucketRepositoryWalkerAdapter instance
        """
        return BitbucketRepositoryWalkerAdapter(repo_path)

    def list_pull_requests(
        self,
        project_key: str,
        repo_slug: str,
        state: str = "OPEN",
        limit: int = 50,
    ) -> List[GitPullRequest]:
        """List pull requests in Bitbucket repository.

        Args:
            project_key: Bitbucket project key
            repo_slug: Repository slug
            state: Filter by state (OPEN, MERGED, DECLINED, ALL)
            limit: Maximum number to return

        Returns:
            List of GitPullRequest objects
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            bitbucket_prs = self._connector.list_pull_requests(
                project_key=project_key,
                repo_slug=repo_slug,
                state=state,
                limit=limit,
            )

            prs = [
                GitPullRequest(
                    id=str(pr.id),
                    title=pr.title,
                    state=pr.state,
                    source_branch=pr.source_branch,
                    target_branch=pr.target_branch,
                    description=pr.description,
                    author=pr.author,
                    created_at=(
                        datetime.fromisoformat(pr.created_at)
                        if isinstance(pr.created_at, str)
                        else pr.created_at
                    ),
                    updated_at=(
                        datetime.fromisoformat(pr.updated_at)
                        if isinstance(pr.updated_at, str)
                        else pr.updated_at
                    ),
                )
                for pr in bitbucket_prs
            ]
            logger.info(f"Found {len(prs)} PRs in {project_key}/{repo_slug}")
            return prs
        except Exception as e:
            logger.warning(f"Error listing PRs: {e}")
            return []

    def get_file_content(
        self,
        project_key: str,
        repo_slug: str,
        file_path: str,
        branch: str = "master",
    ) -> str:
        """Get file content from Bitbucket repository.

        Args:
            project_key: Bitbucket project key
            repo_slug: Repository slug
            file_path: Path to file in repository
            branch: Branch to fetch from

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            content = self._connector.get_file_content(
                project_key=project_key,
                repo_slug=repo_slug,
                file_path=file_path,
                branch=branch,
            )
            return content
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise FileNotFoundError(f"File not found: {file_path}")
            raise

    def get_pull_request_files(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """List files changed in a pull request.

        Args:
            project_key: Bitbucket project key
            repo_slug: Repository slug
            pr_id: Pull request ID
            limit: Optional maximum number of files to return

        Returns:
            List of file change dicts with keys: path, status, old_path
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            files = self._connector.get_pull_request_files(
                project_key=project_key,
                repo_slug=repo_slug,
                pr_id=pr_id,
                limit=limit,
            )
            return files
        except Exception as e:
            logger.warning(f"Error listing PR files: {e}")
            return []
