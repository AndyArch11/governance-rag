"""Unified Git ingestion configuration for multi-platform code repositories.

Supports configuration for:
- Bitbucket (Server/Cloud)
- GitHub (Cloud/Enterprise)
- GitLab (Cloud/Self-hosted)
- Azure DevOps (future)

Configuration can be set via environment variables or .env file.
"""

from pathlib import Path
from typing import Optional

from scripts.ingest.ingest_config import IngestConfig


class GitIngestConfig(IngestConfig):
    """Unified configuration for Git-based code ingestion.

    Extends base IngestConfig with Git provider-specific settings.
    Automatically loads provider-specific config based on GIT_PROVIDER setting.
    """

    def __init__(self):
        """Initialise Git ingestion configuration."""
        super().__init__()

        # ====================================================================
        # Generic Git Settings
        # ====================================================================
        self.git_provider = self.get_str(
            "GIT_PROVIDER", "bitbucket"
        ).lower()  # bitbucket|github|gitlab|azure
        self.git_host = self.get_str("GIT_HOST", "")  # Base URL for self-hosted instances
        # GitHub defaults (available even if provider != github)
        self.github_host = self.get_str("GITHUB_HOST", "https://github.com")
        self.github_token = self.get_str("GITHUB_TOKEN", "")
        self.github_owner = self.get_str("GITHUB_OWNER", "")
        self.github_repo = self.get_str("GITHUB_REPO", "")
        self.github_verify_ssl = self.get_bool("GITHUB_VERIFY_SSL", True)
        self.github_api_url = self.get_str("GITHUB_API_URL", "https://api.github.com")

        # Repository settings (common across providers)
        self.git_branch = self.get_str("GIT_BRANCH", "main")
        self.git_clone_dir = self.get_str("GIT_CLONE_DIR", "~/rag-project/repos")
        # Each provider keeps its own clone directory underneath the base path
        base_clone_path = Path(self.git_clone_dir)
        self.provider_clone_dir = str(base_clone_path / (self.git_provider or "git"))
        self.git_reset_repo = self.get_bool("GIT_RESET_REPO", False)

        # File type filters (shared across all providers)
        self.file_types = self.get_str(
            "GIT_FILE_TYPES",
            "java,groovy,gvy,gsp,gradle,xml,properties,yaml,yml,js,jsx,ts,tsx,cs,ps1,psm1,tf,sql,html,htm,json,py,go,rs,rb,php,c,cpp,h,hpp",
        ).split(",")

        # ====================================================================
        # Code Ingestion Settings (shared)
        # ====================================================================
        self.generate_summaries = self.get_bool("GIT_GENERATE_SUMMARIES", True)
        self.use_llm_summaries = self.get_bool("GIT_USE_LLM_SUMMARIES", False)
        self.enable_dlp = self.get_bool("GIT_ENABLE_DLP", True)
        self.exclude_tests = self.get_bool("GIT_EXCLUDE_TESTS", False)
        self.no_refresh = self.get_bool("GIT_NO_REFRESH", False)

        # ====================================================================
        # Bitbucket Configuration
        # ====================================================================
        if self.git_provider == "bitbucket":
            self.bitbucket_host = self.git_host or self.get_str("BITBUCKET_HOST", "")
            self.bitbucket_username = self.get_str("BITBUCKET_USERNAME", "")
            self.bitbucket_api_username = self.get_str(
                "BITBUCKET_API_USERNAME", self.bitbucket_username
            )
            self.bitbucket_password = self.get_str("BITBUCKET_PASSWORD", "")
            self.bitbucket_is_cloud = self.get_bool("BITBUCKET_IS_CLOUD", False)
            self.bitbucket_verify_ssl = self.get_bool("BITBUCKET_VERIFY_SSL", True)
            self.bitbucket_project = self.get_str("BITBUCKET_PROJECT", "")
            self.bitbucket_repo = self.get_str("BITBUCKET_REPO", "")

        # ====================================================================
        # GitHub Configuration
        # ====================================================================
        elif self.git_provider == "github":
            self.github_host = self.git_host or self.github_host
            self.github_token = self.github_token or self.get_str(
                "GITHUB_TOKEN", ""
            )  # Personal access token
            self.github_owner = self.github_owner or self.get_str(
                "GITHUB_OWNER", ""
            )  # Organisation/user
            self.github_repo = self.github_repo or self.get_str("GITHUB_REPO", "")
            self.github_verify_ssl = self.github_verify_ssl
            # GitHub Enterprise specific
            self.github_api_url = self.github_api_url or self.get_str(
                "GITHUB_API_URL",
                (
                    "https://api.github.com"
                    if self.github_host == "https://github.com"
                    else f"{self.github_host}/api/v3"
                ),
            )

        # ====================================================================
        # GitLab Configuration (future)
        # ====================================================================
        elif self.git_provider == "gitlab":
            self.gitlab_host = self.git_host or self.get_str("GITLAB_HOST", "https://gitlab.com")
            self.gitlab_token = self.get_str("GITLAB_TOKEN", "")  # Private/personal access token
            self.gitlab_group = self.get_str("GITLAB_GROUP", "")  # Group ID or path
            self.gitlab_repo = self.get_str("GITLAB_REPO", "")
            self.gitlab_verify_ssl = self.get_bool("GITLAB_VERIFY_SSL", True)

        # ====================================================================
        # Azure DevOps Configuration (future)
        # ====================================================================
        elif self.git_provider == "azure":
            self.azure_host = self.git_host or self.get_str("AZURE_HOST", "https://dev.azure.com")
            self.azure_organisation = self.get_str("AZURE_ORGANISATION", "")
            self.azure_project = self.get_str("AZURE_PROJECT", "")
            self.azure_repo = self.get_str("AZURE_REPO", "")
            self.azure_token = self.get_str("AZURE_TOKEN", "")  # PAT

    def validate(self) -> bool:
        """Validate provider-specific configuration.

        Returns:
            True if all required settings are present, False otherwise
        """
        if self.git_provider == "bitbucket":
            return bool(self.bitbucket_host and self.bitbucket_username and self.bitbucket_password)
        elif self.git_provider == "github":
            return bool(self.github_token and self.github_owner)
        elif self.git_provider == "gitlab":
            return bool(self.gitlab_token and self.gitlab_group)
        elif self.git_provider == "azure":
            return bool(self.azure_organisation and self.azure_project and self.azure_token)
        return False
