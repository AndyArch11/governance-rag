"""Bitbucket repository connector for cloning, browsing, and pulling code.

This module provides utilities to:
- Connect to Bitbucket Server/Cloud instances
- List and discover projects and repositories
- Clone repositories locally
- Walk repository file structure with date-based filtering
- List and handle pull requests
- Extract code for parsing
- Compare file versions across dates for drift analysis

Supports:
- Bitbucket Server (self-hosted)
- Bitbucket Cloud
- Date-based file extraction and version comparison

Example Usage:
    from scripts.ingest.git.bitbucket_connector import BitbucketConnector, RepositoryWalker
    from datetime import datetime, timedelta

    # Connect to Bitbucket
    connector = BitbucketConnector(
        host="https://bitbucket.company.com",
        username="user@company.com",
        password="app-password",
        is_cloud=False
    )

    # Clone a repository
    repo_path = connector.clone_repository("PROJECT_KEY", "repo-slug")

    # Walk repository files older than a specific date
    walker = RepositoryWalker(repo_path)
    cutoff_date = datetime.now() - timedelta(days=30)
    for file_path, content, mod_date in walker.walk_groovy_files(modified_before=cutoff_date):
        print(f"Found (modified {mod_date}): {file_path}")

    # Compare versions across dates for drift analysis
    v1_date = datetime(2025, 1, 1)
    v2_date = datetime(2026, 1, 1)
    comparison = walker.compare_versions(v1_date, v2_date, extensions=[".groovy"])
"""

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse, urlunparse

import requests

from scripts.utils.logger import create_module_logger
from scripts.utils.retry_utils import retry_with_backoff

get_logger, _ = create_module_logger("ingest")

logger = get_logger()


@dataclass
class BitbucketProject:
    """Bitbucket project metadata."""

    key: str
    name: str
    description: Optional[str] = None
    url: Optional[str] = None


@dataclass
class BitbucketRepository:
    """Bitbucket repository metadata."""

    slug: str
    name: str
    project_key: str
    description: Optional[str] = None
    clone_url: Optional[str] = None
    clone_ssh: Optional[str] = None
    url: Optional[str] = None
    default_branch: Optional[str] = None


@dataclass
class PullRequest:
    """Bitbucket pull request metadata."""

    id: int
    title: str
    state: str  # OPEN, MERGED, DECLINED
    source_branch: str
    target_branch: str
    author: str
    created_at: str
    updated_at: Optional[str] = None
    description: Optional[str] = None


class BitbucketConnector:
    """Connect to Bitbucket Server or Cloud and manage repositories."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        is_cloud: bool = False,
        verify_ssl: bool = True,
        api_username: Optional[str] = None,
    ):
        """Initialise Bitbucket connector.

        Args:
            host: Bitbucket host URL (e.g., https://bitbucket.company.com or https://api.bitbucket.org)
            username: Username for git operations (Bitbucket username)
            password: Password or app password
            is_cloud: True for Bitbucket Cloud, False for Bitbucket Server
            verify_ssl: Whether to verify SSL certificates
            api_username: Username for API calls (typically email for Cloud, falls back to username if not provided)

        Note:
            For Bitbucket Cloud, API calls typically require the account email address,
            while git operations use the Bitbucket username. If api_username is not provided,
            username will be used for both operations.

        Retry Configuration:
            - Uses standard retry_utils with exponential backoff
            - Automatically retries transient errors (connection, timeout, rate limits)
            - Max 5 retries for API calls with initial 1.0s delay
            - Rate limiting via standard rate_limiter (100ms throttle)
        """
        self.host = host.rstrip("/")
        self.username = username  # For git operations
        self.api_username = api_username or username  # For API calls (email for Cloud)
        self.password = password
        self.is_cloud = is_cloud
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.auth = (self.api_username, password)  # Use api_username for API calls
        self.session.verify = verify_ssl
        self.logger = get_logger()

        # Normalise Bitbucket Cloud host (git uses bitbucket.org, API uses api.bitbucket.org)
        if self.is_cloud and "bitbucket.org" in self.host and "api.bitbucket.org" not in self.host:
            self.host = "https://api.bitbucket.org"

        # Set API base URL (keep trailing path for manual join later)
        if is_cloud:
            self.api_base = f"{self.host}/2.0"
        else:
            self.api_base = f"{self.host}/rest/api/1.0"

    def _throttle_request(self):
        """Throttle requests to respect rate limits.

        Note: For Bitbucket Cloud rate limiting, use standard rate_limiter utility
        which provides 100ms throttling between requests.
        For Bitbucket Server: typically higher limits, depends on configuration.
        """
        pass  # Rate limiting handled by retry_utils and rate_limiter in calling code

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict:
        """Make HTTP request to Bitbucket API with retry logic and error handling.

        Delegates to _request_with_retry which uses retry_utils for:
        - Exponential backoff on transient errors
        - 429 rate limiting handling
        - Connection/timeout errors
        - Comprehensive audit logging

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (relative to api_base)
            **kwargs: Additional arguments to pass to requests method

        Returns:
            Response JSON as dictionary

        Raises:
            requests.RequestException: On non-recoverable errors
        """
        # Safe join to avoid dropping the /2.0 path when endpoint lacks leading slash
        url = f"{self.api_base.rstrip('/')}/{endpoint.lstrip('/')}"
        return self._request_with_retry(method, url, **kwargs)

    @retry_with_backoff(
        max_retries=5,
        initial_delay=1.0,
        backoff_factor=2.0,
        max_delay=20.0,
        jitter=True,
        operation_name="bitbucket_api_call",
    )
    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict:
        """Execute HTTP request with retry decorator handling transient failures.

        The @retry_with_backoff decorator automatically:
        - Classifies errors as transient vs hard failures
        - Retries transient errors with exponential backoff
        - Fails immediately on hard failures (validation, not found, etc.)
        - Logs all attempts and final outcomes

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to call
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response JSON as dictionary

        Raises:
            requests.RequestException: On final failure after all retries
        """
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def list_projects(self) -> List[BitbucketProject]:
        """List all projects in the Bitbucket instance.

        Returns:
            List of BitbucketProject objects
        """
        projects = []

        if self.is_cloud:
            # Bitbucket Cloud
            resp = self._make_request("GET", "workspaces")
            for workspace in resp.get("values", []):
                projects.append(
                    BitbucketProject(
                        key=workspace["slug"],
                        name=workspace["name"],
                        description=workspace.get("description"),
                        url=workspace.get("links", {}).get("html", {}).get("href"),
                    )
                )
        else:
            # Bitbucket Server
            start = 0
            limit = 25

            while True:
                resp = self._make_request(
                    "GET", "projects", params={"start": start, "limit": limit}
                )

                for project in resp.get("values", []):
                    projects.append(
                        BitbucketProject(
                            key=project["key"],
                            name=project["name"],
                            description=project.get("description"),
                            url=project.get("links", {}).get("self", [{}])[0].get("href"),
                        )
                    )

                # Check if there are more pages
                if resp.get("isLastPage", True):
                    break
                start = resp.get("start", 0) + len(resp.get("values", []))

        return projects

    def list_repositories(
        self,
        project_key: str,
        name_pattern: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BitbucketRepository]:
        """List repositories in a project with optional filtering.

        Args:
            project_key: Project key/slug
            name_pattern: Optional substring to filter repository names (case-insensitive)
            limit: Optional maximum number of repositories to return

        Returns:
            List of BitbucketRepository objects
        """
        repositories = []

        if self.is_cloud:
            # Bitbucket Cloud - workspace/repo_slug pattern
            page = 1
            pagelen = 100

            while True:
                try:
                    resp = self._make_request(
                        "GET",
                        f"repositories/{project_key}",
                        params={"page": page, "pagelen": pagelen},
                    )
                except Exception as e:
                    # If workspace listing fails (401/403), try user repositories endpoint
                    if "401" in str(e) or "403" in str(e):
                        import logging

                        logger = logging.getLogger("ingest")
                        logger.warning(
                            f"Cannot list workspace repositories (permission denied). "
                            f"Trying user repositories endpoint instead."
                        )
                        # Fallback to user repositories (repos the authenticated user has access to)
                        return self._list_user_repositories_cloud(name_pattern, limit)
                    raise

                for repo in resp.get("values", []):
                    # Apply name pattern filter
                    if name_pattern and name_pattern.lower() not in repo["slug"].lower():
                        continue

                    clone_urls = {
                        link["name"]: link["href"]
                        for link in repo.get("links", {}).get("clone", [])
                    }
                    repositories.append(
                        BitbucketRepository(
                            slug=repo["slug"],
                            name=repo["name"],
                            project_key=project_key,
                            description=repo.get("description"),
                            clone_url=clone_urls.get("http"),
                            clone_ssh=clone_urls.get("ssh"),
                            url=repo.get("links", {}).get("html", {}).get("href"),
                            default_branch=(repo.get("mainbranch") or {}).get("name"),
                        )
                    )

                    # Check if we've reached the limit
                    if limit and len(repositories) >= limit:
                        return repositories

                # Check for next page
                if "next" not in resp:
                    break
                page += 1

        else:
            # Bitbucket Server - project key pattern
            start = 0
            page_limit = 25

            while True:
                resp = self._make_request(
                    "GET",
                    f"projects/{project_key}/repos",
                    params={"start": start, "limit": page_limit},
                )

                for repo in resp.get("values", []):
                    # Apply name pattern filter
                    if name_pattern and name_pattern.lower() not in repo["slug"].lower():
                        continue

                    clone_links = repo.get("links", {}).get("clone", [])
                    clone_url = next((c["href"] for c in clone_links if c["name"] == "http"), None)
                    clone_ssh = next((c["href"] for c in clone_links if c["name"] == "ssh"), None)

                    repositories.append(
                        BitbucketRepository(
                            slug=repo["slug"],
                            name=repo["name"],
                            project_key=project_key,
                            description=repo.get("description"),
                            clone_url=clone_url,
                            clone_ssh=clone_ssh,
                            url=repo.get("links", {}).get("self", [{}])[0].get("href"),
                            default_branch=(repo.get("defaultBranch") or {}).get("displayId"),
                        )
                    )

                    # Check if we've reached the limit
                    if limit and len(repositories) >= limit:
                        return repositories

                # Check if there are more pages
                if resp.get("isLastPage", True):
                    break
                start = resp.get("start", 0) + len(resp.get("values", []))

        return repositories

    def _list_user_repositories_cloud(
        self,
        name_pattern: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BitbucketRepository]:
        """List repositories the authenticated user has access to (Bitbucket Cloud fallback).

        This endpoint lists all repositories the user has explicit access to,
        which works when workspace listing permissions are not available.

        Args:
            name_pattern: Optional substring to filter repository names (case-insensitive)
            limit: Optional maximum number of repositories to return

        Returns:
            List of BitbucketRepository objects
        """
        repositories = []
        page = 1
        pagelen = 100

        while True:
            # Use user repositories endpoint instead of workspace endpoint
            resp = self._make_request(
                "GET",
                "user/permissions/repositories",
                params={
                    "page": page,
                    "pagelen": pagelen,
                    "q": 'permission="write" OR permission="admin"',
                },
            )

            for item in resp.get("values", []):
                repo = item.get("repository", {})

                # Apply name pattern filter
                if name_pattern and name_pattern.lower() not in repo["slug"].lower():
                    continue

                # Extract workspace from full_name (format: "workspace/repo")
                full_name = repo.get("full_name", "")
                workspace = full_name.split("/")[0] if "/" in full_name else ""

                clone_urls = {
                    link["name"]: link["href"] for link in repo.get("links", {}).get("clone", [])
                }
                repositories.append(
                    BitbucketRepository(
                        slug=repo["slug"],
                        name=repo["name"],
                        project_key=workspace,  # Use workspace from full_name
                        description=repo.get("description"),
                        clone_url=clone_urls.get("https"),
                        clone_ssh=clone_urls.get("ssh"),
                        url=repo.get("links", {}).get("html", {}).get("href"),
                        default_branch=(repo.get("mainbranch") or {}).get("name"),
                    )
                )

                # Check if we've reached the limit
                if limit and len(repositories) >= limit:
                    return repositories

            # Check for next page
            if "next" not in resp:
                break
            page += 1

        return repositories

    def clone_repository(
        self,
        project_key: str,
        repo_slug: str,
        target_dir: Optional[str] = None,
        use_ssh: bool = False,
        branch: Optional[str] = None,
        refresh_if_exists: bool = True,
    ) -> str:
        """Clone a repository locally.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            target_dir: Target directory for clone (defaults to rag-project/repos/bitbucket/{project_key}/{repo_slug})
            use_ssh: Use SSH for cloning (requires SSH key setup)
            branch: Optional branch to checkout (defaults to default branch)
            refresh_if_exists: If True, will fetch latest changes if repo already exists locally; otherwise reuses existing clone without updating

        Returns:
            Path to cloned repository
        """
        # First attempt: list repositories via API to get clone URLs
        try:
            repos = self.list_repositories(project_key)
            repo = next((r for r in repos if r.slug == repo_slug), None)
        except requests.HTTPError as e:
            # For Bitbucket Cloud, 401 here often means auth OK for git but not for API.
            # Fall back to direct git clone using constructed HTTPS URL.
            if e.response is not None and e.response.status_code == 401 and self.is_cloud:
                self.logger.warning(
                    "API listing returned 401; falling back to direct git clone URL construction"
                )
                repo = BitbucketRepository(
                    slug=repo_slug,
                    name=repo_slug,
                    project_key=project_key,
                    clone_url=f"https://bitbucket.org/{project_key}/{repo_slug}.git",
                )
            else:
                raise

        if not repo:
            raise ValueError(f"Repository {repo_slug} not found in project/workspace {project_key}")

        # Choose clone URL
        clone_url = repo.clone_ssh if use_ssh else repo.clone_url
        if not clone_url and use_ssh and self.is_cloud:
            # Build a default SSH URL for Bitbucket Cloud
            clone_url = f"git@bitbucket.org:{project_key}/{repo_slug}.git"

        if not clone_url and self.is_cloud:
            # Build a default HTTPS URL for Bitbucket Cloud
            clone_url = f"https://bitbucket.org/{project_key}/{repo_slug}.git"

        if not clone_url:
            raise ValueError(f"No {'SSH' if use_ssh else 'HTTP'} clone URL available")

        # Prepare target directory
        if target_dir is None:
            target_dir = f"rag-project/repos/bitbucket/{project_key}/{repo_slug}"

        target_path = Path(target_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone or refresh repository
        try:
            auth_clone_url = clone_url

            # For HTTPS: inject credentials into URL to avoid interactive prompts
            if not use_ssh and self.username and self.password:
                parsed = urlparse(clone_url)
                if parsed.scheme in ("http", "https") and not parsed.username:
                    host = parsed.hostname or ""
                    port = f":{parsed.port}" if parsed.port else ""
                    netloc = f"{quote(self.username)}:{quote(self.password)}@{host}{port}"
                    auth_clone_url = urlunparse(parsed._replace(netloc=netloc))

            display_url = clone_url  # avoid logging credentials
            # If the target already exists and is a git repo, refresh instead of re-cloning
            git_dir = target_path / ".git"
            if target_path.exists() and git_dir.exists():
                if not refresh_if_exists:
                    self.logger.info(
                        f"Reusing existing repository at {target_dir} (refresh disabled)"
                    )
                    return str(target_path)

                self.logger.info(f"Refreshing existing repository at {target_dir}")

                # Build environment (SSH options if needed)
                git_env = os.environ.copy()
                if use_ssh:
                    git_env["GIT_SSH_COMMAND"] = (
                        "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
                    )

                # Ensure remote URL is correct (use auth URL for HTTPS to avoid prompts)
                try:
                    subprocess.run(
                        ["git", "remote", "set-url", "origin", auth_clone_url],
                        cwd=str(target_path),
                        check=True,
                        capture_output=True,
                        timeout=60,
                        env=git_env,
                    )
                except subprocess.CalledProcessError as e:
                    # Log but proceed; remote might already be set appropriately
                    self.logger.debug(f"remote set-url warning: {e.stderr.decode(errors='ignore')}")

                # Fetch latest with shallow depth
                fetch_args = ["git", "fetch", "--prune", "--depth", "1", "origin"]
                if branch:
                    fetch_args.append(branch)
                subprocess.run(
                    fetch_args,
                    cwd=str(target_path),
                    check=True,
                    capture_output=True,
                    timeout=600,
                    env=git_env,
                )

                # Checkout/reset to requested branch or origin/HEAD
                if branch:
                    subprocess.run(
                        ["git", "checkout", "-B", branch, f"origin/{branch}"],
                        cwd=str(target_path),
                        check=True,
                        capture_output=True,
                        timeout=120,
                        env=git_env,
                    )
                else:
                    # Fallback: keep current branch and fast-forward
                    subprocess.run(
                        ["git", "pull", "--ff-only", "origin"],
                        cwd=str(target_path),
                        check=True,
                        capture_output=True,
                        timeout=300,
                        env=git_env,
                    )

                self.logger.info(f"Repository refreshed at {target_dir}")
                return str(target_path)

            # Otherwise perform a fresh clone
            self.logger.info(f"Cloning {display_url} to {target_dir}")

            git_cmd = ["git", "clone", "--depth", "1"]
            if branch:
                git_cmd += ["--branch", branch]

            if use_ssh:
                git_env = os.environ.copy()
                git_env["GIT_SSH_COMMAND"] = (
                    "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null"
                )
                subprocess.run(
                    git_cmd + [auth_clone_url, str(target_path)],
                    check=True,
                    capture_output=True,
                    timeout=900,
                    env=git_env,
                )
            else:
                subprocess.run(
                    git_cmd + [auth_clone_url, str(target_path)],
                    check=True,
                    capture_output=True,
                    timeout=900,
                )

            self.logger.info(f"Repository cloned successfully to {target_dir}")
            return str(target_path)
        except subprocess.CalledProcessError as e:
            # Redact credentials from error message
            error_msg = e.stderr.decode()
            if self.username and self.password:
                error_msg = error_msg.replace(self.username, "***")
                error_msg = error_msg.replace(self.password, "***")
            self.logger.error(f"Git clone failed: {error_msg}")
            raise RuntimeError(f"Failed to clone repository: {error_msg}")

    def list_pull_requests(
        self,
        project_key: str,
        repo_slug: str,
        state: str = "OPEN",
        limit: Optional[int] = None,
    ) -> List[PullRequest]:
        """List pull requests in a repository.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            state: PR state filter (OPEN, MERGED, DECLINED, ALL)
            limit: Optional maximum number of pull requests to return

        Returns:
            List of PullRequest objects
        """
        pull_requests = []

        if self.is_cloud:
            # Bitbucket Cloud
            state_param = state.lower() if state != "ALL" else None
            params: Dict[str, Any] = {}
            if state_param:
                params["state"] = state_param

            page = 1
            pagelen = 50
            remaining = limit if limit is not None and limit > 0 else None

            while True:
                if remaining is not None:
                    params["pagelen"] = min(pagelen, remaining)
                else:
                    params["pagelen"] = pagelen
                params["page"] = page

                resp = self._make_request(
                    "GET",
                    f"repositories/{project_key}/{repo_slug}/pullrequests",
                    params=params,
                )

                for pr in resp.get("values", []):
                    pull_requests.append(
                        PullRequest(
                            id=pr["id"],
                            title=pr["title"],
                            state=pr["state"].upper(),
                            source_branch=pr["source"]["branch"]["name"],
                            target_branch=pr["destination"]["branch"]["name"],
                            author=pr["author"]["display_name"],
                            created_at=pr["created_on"],
                            updated_at=pr.get("updated_on"),
                            description=pr.get("description"),
                        )
                    )
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            return pull_requests

                if "next" not in resp:
                    break
                page += 1
        else:
            # Bitbucket Server
            start = 0
            page_limit = 25 if limit is None or limit <= 0 else min(25, limit)

            while True:
                server_params: Dict[str, Any] = {"start": start, "limit": page_limit}
                if state != "ALL":
                    server_params["state"] = state

                resp = self._make_request(
                    "GET",
                    f"projects/{project_key}/repos/{repo_slug}/pull-requests",
                    params=server_params,
                )

                for pr in resp.get("values", []):
                    pull_requests.append(
                        PullRequest(
                            id=pr["id"],
                            title=pr["title"],
                            state=pr["state"],
                            source_branch=pr["fromRef"]["repository"]["name"],
                            target_branch=pr["toRef"]["repository"]["name"],
                            author=pr["author"]["user"]["name"],
                            created_at=pr["createdDate"],
                            updated_at=pr.get("updatedDate"),
                            description=pr.get("description"),
                        )
                    )
                    if limit is not None and limit > 0 and len(pull_requests) >= limit:
                        return pull_requests

                # Check if there are more pages
                if resp.get("isLastPage", True):
                    break
                start = resp.get("start", 0) + len(resp.get("values", []))

        return pull_requests

    @retry_with_backoff(
        max_retries=5,
        initial_delay=1.0,
        backoff_factor=2.0,
        max_delay=20.0,
        jitter=True,
        operation_name="bitbucket_file_content",
    )
    def _request_text_with_retry(self, method: str, url: str, **kwargs) -> str:
        """Execute HTTP request with retry handling and return response text."""
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.text

    def get_file_content(
        self,
        project_key: str,
        repo_slug: str,
        file_path: str,
        branch: str = "master",
    ) -> str:
        """Get file content from Bitbucket repository.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            file_path: Path to file in repository
            branch: Branch to fetch from

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file does not exist
        """
        safe_path = quote(file_path.lstrip("/"), safe="/")

        if self.is_cloud:
            url = f"{self.api_base}/repositories/{project_key}/{repo_slug}/src/{branch}/{safe_path}"
            params = None
        else:
            url = f"{self.api_base}/projects/{project_key}/repos/{repo_slug}/raw/{safe_path}"
            params = {"at": branch}

        try:
            return self._request_text_with_retry("GET", url, params=params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise FileNotFoundError(f"File not found: {file_path}")
            raise

    def get_pull_request_files(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """Get list of files changed in a pull request.

        Args:
            project_key: Project key/slug
            repo_slug: Repository slug
            pr_id: Pull request ID
            limit: Optional maximum number of files to return

        Returns:
            List of file change dicts with keys: path, status, old_path
        """
        files = []

        if self.is_cloud:
            # Bitbucket Cloud
            resp = self._make_request(
                "GET",
                f"repositories/{project_key}/{repo_slug}/pullrequests/{pr_id}/diffstat",
            )
            for change in resp.get("values", []):
                files.append(
                    {
                        "path": change["new"]["path"] if change["new"] else change["old"]["path"],
                        "status": change["status"].upper(),
                        "old_path": change["old"]["path"] if change["old"] else None,
                    }
                )
                if limit is not None and limit > 0 and len(files) >= limit:
                    return files
        else:
            # Bitbucket Server
            start = 0
            page_limit = 25 if limit is None or limit <= 0 else min(25, limit)

            while True:
                resp = self._make_request(
                    "GET",
                    f"projects/{project_key}/repos/{repo_slug}/pull-requests/{pr_id}/changes",
                    params={"start": start, "limit": page_limit},
                )

                for change in resp.get("values", []):
                    files.append(
                        {
                            "path": change["path"]["toString"],
                            "status": change["type"],
                            "old_path": None,
                        }
                    )
                    if limit is not None and limit > 0 and len(files) >= limit:
                        return files

                if resp.get("isLastPage", True):
                    break
                start = resp.get("start", 0) + len(resp.get("values", []))

        return files


class RepositoryWalker:
    """Walk a local git repository and extract code files."""

    def __init__(self, repo_path: str):
        """Initialise repository walker.

        Args:
            repo_path: Path to cloned repository
        """
        self.repo_path = Path(repo_path)
        self.logger = get_logger()

        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

    def walk_groovy_files(self) -> Generator[Tuple[str, str], None, None]:
        """Walk repository and yield Groovy files with content.

        Yields:
            Tuple of (file_path, file_content)
        """
        yield from self._walk_by_extension([".groovy", ".gradle"])

    def walk_java_files(self) -> Generator[Tuple[str, str], None, None]:
        """Walk repository and yield Java files with content.

        Yields:
            Tuple of (file_path, file_content)
        """
        yield from self._walk_by_extension([".java"])

    def walk_all_code_files(self) -> Generator[Tuple[str, str], None, None]:
        """Walk repository and yield all code files with content.

        Yields:
            Tuple of (file_path, file_content)
        """
        yield from self._walk_by_extension(
            [".groovy", ".gradle", ".java", ".xml", ".properties", ".yml", ".yaml"]
        )

    def walk_by_pattern(self, patterns: List[str]) -> Generator[Tuple[str, str], None, None]:
        """Walk repository for files matching patterns.

        Args:
            patterns: List of file patterns (e.g., ["**/Jenkinsfile", "**/*-mule.xml"])

        Yields:
            Tuple of (file_path, file_content)
        """
        from fnmatch import fnmatch

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.repo_path)

                for pattern in patterns:
                    if fnmatch(str(rel_path), pattern):
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="replace")
                            yield str(rel_path), content
                        except Exception as e:
                            self.logger.warning(f"Failed to read {rel_path}: {e}")
                        break

    def _walk_by_extension(self, extensions: List[str]) -> Generator[Tuple[str, str], None, None]:
        """Walk repository and yield files with specific extensions.

        Args:
            extensions: List of file extensions (e.g., [".groovy", ".java"])

        Yields:
            Tuple of (file_path, file_content)
        """
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                # Skip hidden directories and vendor dirs
                parts = file_path.parts
                if any(
                    part.startswith(".") or part in ["node_modules", "target", "build"]
                    for part in parts
                ):
                    continue

                try:
                    rel_path = file_path.relative_to(self.repo_path)
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    yield str(rel_path), content
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")

    def walk_files(
        self,
        extensions: Optional[List[str]] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, Optional[datetime]], None, None]:
        """Walk repository files with optional extension and date filtering.

        Args:
            extensions: File extensions to include (e.g., ['.java', '.groovy'])
            modified_before: Only include files modified before this date
            modified_after: Only include files modified after this date

        Yields:
            (file_path, content, modification_date) tuples
        """
        if extensions is None:
            extensions = []

        # Normalise extensions to include leading dot
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        if not extensions:
            # If no extensions specified, walk all code files
            yield from self.walk_all_code_files_dated(
                modified_before=modified_before, modified_after=modified_after
            )
        else:
            # Walk files with specific extensions
            yield from self._walk_by_extension_dated(
                extensions=extensions,
                modified_before=modified_before,
                modified_after=modified_after,
            )

    def list_files_by_extension(self, extension: str) -> List[str]:
        """List all files with a specific extension.

        Args:
            extension: File extension (e.g., ".groovy")

        Returns:
            List of relative file paths
        """
        files = []
        for file_path in self.repo_path.rglob(f"*{extension}"):
            if file_path.is_file():
                parts = file_path.parts
                if not any(
                    part.startswith(".") or part in ["node_modules", "target", "build"]
                    for part in parts
                ):
                    rel_path = file_path.relative_to(self.repo_path)
                    files.append(str(rel_path))
        return sorted(files)

    def get_directory_structure(self) -> Dict[str, Any]:
        """Get repository directory structure summary.

        Returns:
            Dictionary with directory statistics
        """
        stats: Dict[str, Any] = {
            "total_files": 0,
            "by_extension": {},
            "by_directory": {},
            "groovy_files": [],
            "java_files": [],
            "gradle_files": [],
            "jenkins_files": [],
        }

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                parts = file_path.parts
                if any(
                    part.startswith(".") or part in ["node_modules", "target", "build"]
                    for part in parts
                ):
                    continue

                stats["total_files"] += 1

                # Track by extension
                ext = file_path.suffix or "no_ext"
                stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

                # Track by directory
                if len(parts) > 1:
                    directory = parts[0]
                    stats["by_directory"][directory] = stats["by_directory"].get(directory, 0) + 1

                # Track specific file types
                rel_path = str(file_path.relative_to(self.repo_path))
                if file_path.suffix == ".groovy":
                    stats["groovy_files"].append(rel_path)
                elif file_path.suffix == ".java":
                    stats["java_files"].append(rel_path)
                elif file_path.name in ["build.gradle", "Jenkinsfile"]:
                    if file_path.name == "build.gradle":
                        stats["gradle_files"].append(rel_path)
                    else:
                        stats["jenkins_files"].append(rel_path)

        return stats

    def _get_file_mod_date(self, file_path: Path) -> datetime:
        """Get file modification date from git history.

        Uses git log to determine when a file was last modified.
        Falls back to filesystem mtime if git is unavailable.

        Args:
            file_path: Path to the file

        Returns:
            datetime object of last modification
        """
        try:
            # Get the last commit date for this file using git
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", "--", str(file_path)],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse git timestamp (format: 2025-01-15 10:30:45 +0000)
                date_str = result.stdout.strip().split()[0]
                return datetime.fromisoformat(date_str)
        except Exception as e:
            self.logger.debug(f"Failed to get git mod date for {file_path}: {e}")

        # Fallback to filesystem mtime
        try:
            mtime = file_path.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        except Exception as e:
            self.logger.warning(f"Failed to get mtime for {file_path}: {e}")
            return datetime.now()

    def walk_groovy_files_dated(
        self,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, datetime], None, None]:
        """Walk repository and yield Groovy files with modification dates.

        Args:
            modified_before: Only yield files modified before this date
            modified_after: Only yield files modified after this date

        Yields:
            Tuple of (file_path, file_content, modification_date)
        """
        yield from self._walk_by_extension_dated(
            [".groovy", ".gradle"],
            modified_before=modified_before,
            modified_after=modified_after,
        )

    def walk_java_files_dated(
        self,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, datetime], None, None]:
        """Walk repository and yield Java files with modification dates.

        Args:
            modified_before: Only yield files modified before this date
            modified_after: Only yield files modified after this date

        Yields:
            Tuple of (file_path, file_content, modification_date)
        """
        yield from self._walk_by_extension_dated(
            [".java"],
            modified_before=modified_before,
            modified_after=modified_after,
        )

    def walk_all_code_files_dated(
        self,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, datetime], None, None]:
        """Walk repository and yield all code files with modification dates.

        Args:
            modified_before: Only yield files modified before this date
            modified_after: Only yield files modified after this date

        Yields:
            Tuple of (file_path, file_content, modification_date)
        """
        yield from self._walk_by_extension_dated(
            [".groovy", ".gradle", ".java", ".xml", ".properties", ".yml", ".yaml"],
            modified_before=modified_before,
            modified_after=modified_after,
        )

    def _walk_by_extension_dated(
        self,
        extensions: List[str],
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
    ) -> Generator[Tuple[str, str, datetime], None, None]:
        """Walk repository and yield files with extensions and modification dates.

        Args:
            extensions: List of file extensions (e.g., [".groovy", ".java"])
            modified_before: Only yield files modified before this date
            modified_after: Only yield files modified after this date

        Yields:
            Tuple of (file_path, file_content, modification_date)
        """
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                # Skip hidden directories and vendor dirs
                parts = file_path.parts
                if any(
                    part.startswith(".") or part in ["node_modules", "target", "build"]
                    for part in parts
                ):
                    continue

                # Get modification date
                mod_date = self._get_file_mod_date(file_path)

                # Apply date filters
                if modified_before and mod_date >= modified_before:
                    continue
                if modified_after and mod_date <= modified_after:
                    continue

                try:
                    rel_path = file_path.relative_to(self.repo_path)
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    yield str(rel_path), content, mod_date
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")

    def compare_versions(
        self,
        v1_date: datetime,
        v2_date: datetime,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare code across two date points for drift analysis.

        Captures snapshots of all code files at two different dates
        and provides drift metrics (added, removed, modified files).

        Args:
            v1_date: First snapshot date (typically older)
            v2_date: Second snapshot date (typically newer)
            extensions: File extensions to compare (defaults to all code files)

        Returns:
            Dictionary with comparison data:
            {
                "v1_date": datetime,
                "v2_date": datetime,
                "v1_files": {file_path: (content, mod_date), ...},
                "v2_files": {file_path: (content, mod_date), ...},
                "summary": {
                    "v1_file_count": N,
                    "v2_file_count": N,
                    "added": [paths],
                    "removed": [paths],
                    "modified": [paths],
                    "unchanged": [paths],
                    "total_added": N,
                    "total_removed": N,
                    "total_modified": N,
                },
            }
        """
        if extensions is None:
            extensions = [".groovy", ".gradle", ".java", ".xml", ".properties", ".yml", ".yaml"]

        self.logger.info(f"Comparing versions: {v1_date} vs {v2_date}")

        # Capture V1 snapshot (files modified before v2_date, on or after v1_date)
        v1_files = {}
        for file_path, content, mod_date in self._walk_by_extension_dated(
            extensions, modified_before=v2_date, modified_after=v1_date
        ):
            v1_files[file_path] = (content, mod_date)

        self.logger.info(f"V1 snapshot: {len(v1_files)} files")

        # Capture V2 snapshot (files modified before now, on or after v2_date)
        v2_files = {}
        for file_path, content, mod_date in self._walk_by_extension_dated(
            extensions, modified_after=v2_date
        ):
            v2_files[file_path] = (content, mod_date)

        self.logger.info(f"V2 snapshot: {len(v2_files)} files")

        # Compute diff
        v1_paths = set(v1_files.keys())
        v2_paths = set(v2_files.keys())

        added = sorted(v2_paths - v1_paths)
        removed = sorted(v1_paths - v2_paths)
        common = v1_paths & v2_paths

        # Find modified files (content changed)
        modified = []
        unchanged = []
        for path in common:
            v1_content = v1_files[path][0]
            v2_content = v2_files[path][0]
            if v1_content != v2_content:
                modified.append(path)
            else:
                unchanged.append(path)

        modified = sorted(modified)
        unchanged = sorted(unchanged)

        return {
            "v1_date": v1_date,
            "v2_date": v2_date,
            "v1_files": v1_files,
            "v2_files": v2_files,
            "summary": {
                "v1_file_count": len(v1_files),
                "v2_file_count": len(v2_files),
                "added": added,
                "removed": removed,
                "modified": modified,
                "unchanged": unchanged,
                "total_added": len(added),
                "total_removed": len(removed),
                "total_modified": len(modified),
                "total_unchanged": len(unchanged),
                "drift_percentage": (
                    round(
                        (len(added) + len(removed) + len(modified)) / max(len(v1_files), 1) * 100, 2
                    )
                    if v1_files
                    else 0.0
                ),
            },
        }
