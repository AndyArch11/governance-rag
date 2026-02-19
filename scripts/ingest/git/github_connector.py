"""GitHub REST API connector for code ingestion.

Implements the GitConnector interface for GitHub (Cloud and Enterprise).
Handles authentication via Personal Access Tokens (PAT), repository
discovery, cloning, and file operations.

Supports:
- GitHub Cloud (github.com)
- GitHub Enterprise Server (self-hosted)
- Rate limiting and exponential backoff
- Pagination for large result sets

Example:
    connector = GitHubGitConnector(
        host="https://github.com",
        token="ghp_xxxxxxxxxxxx",
        api_url="https://api.github.com"
    )
    with connector:
        projects = connector.list_projects()
        repos = connector.list_repositories("myorg")
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from scripts.ingest.git.git_connector import (
    GitConnector,
    GitProject,
    GitPullRequest,
    GitRepository,
    RepositoryWalker,
)
from scripts.utils.logger import create_module_logger
from scripts.utils.retry_utils import retry_with_backoff

get_logger, _ = create_module_logger("ingest")
logger = get_logger()


class GitHubRepositoryWalker(RepositoryWalker):
    """Walk GitHub repository file structure (local clone).

    Operates on locally cloned repositories using filesystem operations.
    Git history queries require separate API calls.
    """

    def walk_files(
        self,
        extensions: Optional[List[str]] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, Optional[datetime]], None, None]:
        """Walk repository files with optional filtering.

        Since we're working with a local clone, we use filesystem modification
        times as a proxy for GitHub commit dates. For accurate history,
        use GitHubGitConnector.list_pull_requests() or query commits via API.

        Args:
            extensions: File extensions to include
            modified_before: Only files modified before this date
            modified_after: Only files modified after this date

        Yields:
            (relative_path, content, filesystem_mtime) tuples
        """
        if extensions is None:
            extensions = []

        # Normalise extensions to include leading dot
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        for file_path in self.repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check extension filter
            if extensions and file_path.suffix not in extensions:
                continue

            # Skip common non-code files
            if any(part.startswith(".") for part in file_path.parts):
                continue

            try:
                # Get file modification time
                stat = file_path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)

                # Apply date filters
                if modified_before and mtime > modified_before:
                    continue
                if modified_after and mtime < modified_after:
                    continue

                # Read file content
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(self.repo_path))

                yield rel_path, content, mtime
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue

    def compare_versions(
        self,
        date1: datetime,
        date2: datetime,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare repository state across two dates using Git history.

        Args:
            date1: First point in time
            date2: Second point in time (should be after date1)
            extensions: File types to compare

        Returns:
            Dict with 'added', 'removed', 'modified' file lists
        """
        # Format dates for git commands (ISO format)
        date1_str = date1.isoformat()
        date2_str = date2.isoformat()

        result = {
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": [],
        }

        try:
            # Get all commits between dates
            cmd = [
                "git",
                "log",
                "--oneline",
                f"--since={date1_str}",
                f"--until={date2_str}",
            ]
            commits = subprocess.check_output(cmd, cwd=self.repo_path).decode().strip()

            if not commits:
                return result

            # For each commit, get file changes
            for commit_line in commits.split("\n"):
                if not commit_line.strip():
                    continue

                commit_hash = commit_line.split()[0]
                # Get files changed in this commit
                cmd = [
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    commit_hash,
                ]
                changes = subprocess.check_output(cmd, cwd=self.repo_path).decode().strip()

                for line in changes.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    status = parts[0]
                    filepath = parts[1] if len(parts) > 1 else ""

                    if extensions and not any(filepath.endswith(ext) for ext in extensions):
                        continue

                    if status == "A":
                        if filepath not in result["added"]:
                            result["added"].append(filepath)
                    elif status == "D":
                        if filepath not in result["removed"]:
                            result["removed"].append(filepath)
                    elif status == "M":
                        if filepath not in result["modified"]:
                            result["modified"].append(filepath)

            return result
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git command failed: {e}")
            return result


class GitHubGitConnector(GitConnector):
    """GitHub REST API connector for code ingestion.

    Supports GitHub Cloud (github.com) and GitHub Enterprise Server.
    Authentication via Personal Access Tokens (PAT).
    """

    def __init__(
        self,
        host: str = "https://github.com",
        token: str = "",
        api_url: str = "https://api.github.com",
        verify_ssl: bool = True,
    ):
        """Initialise GitHub connector.

        Args:
            host: GitHub host (https://github.com or self-hosted URL)
            token: Personal Access Token for authentication
            api_url: GitHub API URL (auto-determined from host if not provided)
            verify_ssl: Whether to verify SSL certificates
        """
        self.host = host
        self.token = token
        self.api_url = api_url
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "RAG-Git-Ingestion",
            }
        )
        self._connected = False

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        transient_types=(ConnectionError, requests.Timeout),
        operation_name="github_connect",
    )
    def connect(self) -> bool:
        """Authenticate and test connection to GitHub.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If authentication fails
        """
        try:
            url = f"{self.api_url}/user"
            resp = self.session.get(url, verify=self.verify_ssl, timeout=10)
            resp.raise_for_status()
            user_data = resp.json()
            logger.info(f"Connected to GitHub as: {user_data.get('login', 'unknown')}")
            self._connected = True
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to GitHub: {e}")
            raise ConnectionError(f"GitHub authentication failed: {e}")

    def close(self) -> None:
        """Close session and clean up resources."""
        if self.session:
            self.session.close()
        self._connected = False

    def list_projects(self) -> List[GitProject]:
        """List organisations/accounts accessible to authenticated user.

        Returns:
            List of GitProject objects (GitHub orgs/users)
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        projects = []
        try:
            # Include user account as a project for parity with other providers
            user_url = f"{self.api_url}/user"
            user_resp = self.session.get(user_url, verify=self.verify_ssl, timeout=10)
            user_resp.raise_for_status()
            user_data = user_resp.json()
            user_login = user_data.get("login")
            if user_login:
                projects.append(
                    GitProject(
                        key=user_login,
                        name=user_data.get("name", user_login),
                        description=None,
                        url=user_data.get("html_url"),
                    )
                )

            # Get user's organisations
            url = f"{self.api_url}/user/orgs"
            page = 1
            while True:
                resp = self.session.get(
                    url,
                    params={"page": page, "per_page": 100},
                    verify=self.verify_ssl,
                    timeout=10,
                )
                resp.raise_for_status()
                orgs = resp.json()

                if not orgs:
                    break

                for org in orgs:
                    projects.append(
                        GitProject(
                            key=org["login"],
                            name=org.get("name", org["login"]),
                            description=org.get("description"),
                            url=org.get("html_url"),
                        )
                    )
                page += 1

            # De-duplicate by project key to avoid duplicates when user is also an org
            unique_projects = {project.key: project for project in projects}
            projects = list(unique_projects.values())
            logger.info(f"Found {len(projects)} GitHub organisations")
            return projects
        except Exception as e:
            logger.warning(f"Error listing GitHub organisations: {e}")
            return []

    def list_repositories(self, project_key: str) -> List[GitRepository]:
        """List repositories in GitHub organisation/user.

        Args:
            project_key: GitHub organisation or username

        Returns:
            List of GitRepository objects
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        repos = []
        try:
            url = f"{self.api_url}/orgs/{project_key}/repos"
            page = 1
            while True:
                resp = self.session.get(
                    url,
                    params={"page": page, "per_page": 100, "type": "all"},
                    verify=self.verify_ssl,
                    timeout=10,
                )
                resp.raise_for_status()
                repo_list = resp.json()

                if not repo_list:
                    break

                for repo in repo_list:
                    repos.append(
                        GitRepository(
                            slug=repo["name"],
                            name=repo["full_name"],
                            project_key=project_key,
                            description=repo.get("description"),
                            url=repo.get("html_url"),
                            default_branch=repo.get("default_branch", "main"),
                        )
                    )
                page += 1

            logger.info(f"Found {len(repos)} repositories in {project_key}")
            return repos
        except requests.exceptions.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 404:
                # Fallback to user endpoint for personal accounts
                logger.info(
                    f"GitHub org endpoint not found for '{project_key}', trying user endpoint"
                )
                return self._list_user_repositories(project_key)
            logger.warning(f"Error listing repositories for {project_key}: {http_err}")
            return []
        except Exception as e:
            logger.warning(f"Error listing repositories for {project_key}: {e}")
            return []

    def _list_user_repositories(self, username: str) -> List[GitRepository]:
        repos = []
        try:
            url = f"{self.api_url}/users/{username}/repos"
            page = 1
            while True:
                resp = self.session.get(
                    url,
                    params={"page": page, "per_page": 100, "type": "owner"},
                    verify=self.verify_ssl,
                    timeout=10,
                )
                resp.raise_for_status()
                repo_list = resp.json()

                if not repo_list:
                    break

                for repo in repo_list:
                    repos.append(
                        GitRepository(
                            slug=repo["name"],
                            name=repo["full_name"],
                            project_key=username,
                            description=repo.get("description"),
                            url=repo.get("html_url"),
                            default_branch=repo.get("default_branch", "main"),
                        )
                    )
                page += 1

            logger.info(f"Found {len(repos)} repositories for GitHub user {username}")
            return repos
        except Exception as e:
            logger.warning(f"Error listing repositories for GitHub user {username}: {e}")
            return []

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        transient_types=(subprocess.CalledProcessError,),
        operation_name="github_clone",
    )
    def clone_repository(
        self,
        project_key: str,
        repo_slug: str,
        target_dir: str,
        branch: str = "main",
    ) -> str:
        """Clone GitHub repository.

        Args:
            project_key: Organisation/user name
            repo_slug: Repository name
            target_dir: Base directory for clones (will create org/repo subdirectory)
            branch: Branch to clone

        Returns:
            Local path to cloned repository

        Note:
            Repository reset/re-cloning is handled at ingestion start via
            the main ingest script, not per-repository.
        """
        # Create subdirectory structure: base_dir/github/project_key/repo_slug
        base_dir = Path(target_dir)
        if base_dir.name != "github":
            base_dir = base_dir / "github"
        repo_path = base_dir / project_key / repo_slug

        if not repo_path.exists():
            clone_url = f"{self.host}/{project_key}/{repo_slug}.git"
            logger.info(f"Cloning {clone_url} to {repo_path}")

            try:
                repo_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "-b", branch, clone_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                logger.info(f"Successfully cloned to {repo_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Clone failed: {e.stderr.decode()}")
                raise

        return str(repo_path)

    def get_repository_walker(self, repo_path: str) -> RepositoryWalker:
        """Get file walker for GitHub repository.

        Args:
            repo_path: Local path to cloned repository

        Returns:
            GitHubRepositoryWalker instance
        """
        return GitHubRepositoryWalker(repo_path)

    def list_pull_requests(
        self,
        project_key: str,
        repo_slug: str,
        state: str = "open",
        limit: int = 50,
    ) -> List[GitPullRequest]:
        """List pull requests in GitHub repository.

        Args:
            project_key: Organisation/user
            repo_slug: Repository name
            state: Filter by state (open, closed, all)
            limit: Maximum number to return

        Returns:
            List of GitPullRequest objects
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        prs = []
        try:
            url = f"{self.api_url}/repos/{project_key}/{repo_slug}/pulls"
            resp = self.session.get(
                url,
                params={"state": state, "per_page": min(limit, 100)},
                verify=self.verify_ssl,
                timeout=10,
            )
            resp.raise_for_status()
            pr_list = resp.json()

            for pr in pr_list[:limit]:
                prs.append(
                    GitPullRequest(
                        id=str(pr["number"]),
                        title=pr["title"],
                        state=pr["state"],
                        source_branch=pr["head"]["ref"],
                        target_branch=pr["base"]["ref"],
                        description=pr.get("body"),
                        author=pr.get("user", {}).get("login"),
                        created_at=datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00")),
                        url=pr.get("html_url"),
                    )
                )
        except Exception as e:
            logger.warning(f"Error listing PRs: {e}")
            return []

    def get_pull_request_files(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """List files changed in a pull request.

        Args:
            project_key: Organisation/user
            repo_slug: Repository name
            pr_id: Pull request number
            limit: Optional maximum number of files to return

        Returns:
            List of file change dicts with keys: path, status, old_path
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        files: List[Dict[str, Optional[str]]] = []
        try:
            url = f"{self.api_url}/repos/{project_key}/{repo_slug}/pulls/{pr_id}/files"
            page = 1
            per_page = 100

            while True:
                resp = self.session.get(
                    url,
                    params={"page": page, "per_page": per_page},
                    verify=self.verify_ssl,
                    timeout=10,
                )
                resp.raise_for_status()
                file_list = resp.json()

                if not file_list:
                    break

                for file_item in file_list:
                    files.append(
                        {
                            "path": file_item.get("filename"),
                            "status": (file_item.get("status") or "").upper(),
                            "old_path": file_item.get("previous_filename"),
                        }
                    )
                    if limit is not None and limit > 0 and len(files) >= limit:
                        return files

                if len(file_list) < per_page:
                    break
                page += 1

            return files
        except Exception as e:
            logger.warning(f"Error listing PR files: {e}")
            return []

    def get_file_content(
        self,
        project_key: str,
        repo_slug: str,
        file_path: str,
        branch: str = "main",
    ) -> str:
        """Get file content from GitHub repository (without cloning).

        Args:
            project_key: Organisation/user
            repo_slug: Repository name
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
            url = f"{self.api_url}/repos/{project_key}/{repo_slug}/contents/{file_path}"
            resp = self.session.get(
                url,
                params={"ref": branch},
                verify=self.verify_ssl,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if "content" not in data:
                raise FileNotFoundError(f"File not found: {file_path}")

            import base64

            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return content
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(f"File not found: {file_path}")
            raise
