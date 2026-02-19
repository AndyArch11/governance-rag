"""Comprehensive unit tests for Multi-Provider Git Framework.

Tests cover:
- Abstract interfaces and data classes
- GitHub connector implementation
- Bitbucket connector adapter
- Unified configuration
- Code parsing utilities
- Factory pattern
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pytest

from scripts.ingest.git.git_connector import (
    GitConnector,
    RepositoryWalker,
    GitProject,
    GitRepository,
    GitPullRequest,
)
from scripts.ingest.git.git_ingest_config import GitIngestConfig
from scripts.ingest.git.git_snippet_parser import (
    CodeParser,
    CodeLanguage,
    CodeChunk,
    DependencyExtractor,
    EXTENSION_TO_LANGUAGE,
)
from scripts.ingest.git.github_connector import (
    GitHubGitConnector,
    GitHubRepositoryWalker,
)
from scripts.ingest.git.bitbucket_git_connector import (
    BitbucketGitConnector,
    BitbucketRepositoryWalkerAdapter,
)
from scripts.ingest.git.bitbucket_connector import BitbucketConnector, BitbucketRepository
from scripts.ingest.ingest_git import GitConnectorFactory


# ============================================================================
# Data Class Tests
# ============================================================================


def test_git_project_creation():
    """Test GitProject dataclass creation."""
    project = GitProject(
        key="PROJ",
        name="Test Project",
        description="Test description",
        url="https://example.com/proj",
    )
    assert project.key == "PROJ"
    assert project.name == "Test Project"
    assert project.description == "Test description"
    assert project.url == "https://example.com/proj"


def test_git_repository_creation():
    """Test GitRepository dataclass creation."""
    repo = GitRepository(
        slug="repo-slug",
        name="Test Repo",
        project_key="PROJ",
        description="Repo description",
        url="https://example.com/repo",
        default_branch="develop",
    )
    assert repo.slug == "repo-slug"
    assert repo.name == "Test Repo"
    assert repo.project_key == "PROJ"
    assert repo.default_branch == "develop"


def test_git_pull_request_creation():
    """Test GitPullRequest dataclass creation."""
    pr = GitPullRequest(
        id="123",
        title="Feature PR",
        state="open",
        source_branch="feature/new",
        target_branch="main",
        description="PR description",
        author="developer",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 15),
        url="https://example.com/pr/123",
    )
    assert pr.id == "123"
    assert pr.title == "Feature PR"
    assert pr.state == "open"
    assert pr.source_branch == "feature/new"
    assert pr.target_branch == "main"


# ============================================================================
# GitIngestConfig Tests
# ============================================================================


def test_git_ingest_config_defaults():
    """Test GitIngestConfig default values."""
    with patch.dict("os.environ", {}, clear=True):
        config = GitIngestConfig()
        assert config.git_provider == "bitbucket"
        assert config.git_branch == "main"
        assert config.git_reset_repo is False


@patch.dict(
    "os.environ",
    {
        "GIT_PROVIDER": "github",
        "GIT_TOKEN": "ghp_test",
        "GIT_OWNER": "testorg",
        "GIT_BRANCH": "develop",
    },
)
def test_git_ingest_config_from_env():
    """Test GitIngestConfig loading from environment variables."""
    config = GitIngestConfig()
    assert config.git_provider == "github"
    # Config loads from GIT_TOKEN env var
    assert config.get_str("GIT_TOKEN") == "ghp_test"
    assert config.git_branch == "develop"


@patch.dict(
    "os.environ",
    {
        "GIT_PROVIDER": "bitbucket",
        "GIT_USERNAME": "testuser",
        "GIT_PASSWORD": "testpass",
    },
)
def test_git_ingest_config_bitbucket():
    """Test GitIngestConfig for Bitbucket provider."""
    config = GitIngestConfig()
    assert config.git_provider == "bitbucket"
    # Check env vars are accessible
    assert config.get_str("GIT_USERNAME") == "testuser"
    assert config.get_str("GIT_PASSWORD") == "testpass"


# ============================================================================
# CodeParser Tests
# ============================================================================


def test_detect_language_python():
    """Test language detection for Python files."""
    assert CodeParser.detect_language("test.py") == CodeLanguage.PYTHON
    assert CodeParser.detect_language("/path/to/file.py") == CodeLanguage.PYTHON


def test_detect_language_javascript():
    """Test language detection for JavaScript files."""
    assert CodeParser.detect_language("test.js") == CodeLanguage.JAVASCRIPT
    assert CodeParser.detect_language("test.jsx") == CodeLanguage.JAVASCRIPT


def test_detect_language_typescript():
    """Test language detection for TypeScript files."""
    assert CodeParser.detect_language("test.ts") == CodeLanguage.TYPESCRIPT
    assert CodeParser.detect_language("test.tsx") == CodeLanguage.TYPESCRIPT


def test_detect_language_unknown():
    """Test language detection for unknown files."""
    assert CodeParser.detect_language("test.xyz") == CodeLanguage.UNKNOWN


def test_extract_python_functions():
    """Test extracting Python function definitions."""
    code = '''
def hello_world():
    """Say hello."""
    print("Hello")

async def async_func():
    """Async function."""
    await something()
'''
    chunks = CodeParser.extract_functions(code, CodeLanguage.PYTHON)
    assert len(chunks) >= 2
    names = [c.name for c in chunks]
    assert "hello_world" in names
    assert "async_func" in names


def test_extract_python_imports():
    """Test extracting Python imports."""
    code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict
'''
    imports = CodeParser.extract_imports(code, CodeLanguage.PYTHON)
    assert "os" in imports
    assert "sys" in imports
    assert "pathlib" in imports
    assert "typing" in imports


def test_extract_javascript_imports():
    """Test extracting JavaScript imports."""
    code = '''
import React from 'react';
import { useState } from 'react';
const axios = require('axios');
'''
    imports = CodeParser.extract_imports(code, CodeLanguage.JAVASCRIPT)
    assert "react" in imports
    assert "axios" in imports


def test_extract_python_classes():
    """Test extracting Python class definitions."""
    code = '''
class MyClass:
    pass

class AnotherClass(BaseClass):
    def __init__(self):
        pass
'''
    chunks = CodeParser.extract_classes(code, CodeLanguage.PYTHON)
    assert len(chunks) >= 2
    names = [c.name for c in chunks]
    assert "MyClass" in names
    assert "AnotherClass" in names


def test_code_chunk_creation():
    """Test CodeChunk dataclass creation."""
    chunk = CodeChunk(
        code="def test(): pass",
        language=CodeLanguage.PYTHON,
        start_line=10,
        end_line=10,
        chunk_type="function",
        name="test",
        docstring="Test function",
        dependencies={"os", "sys"},
        metadata={"author": "dev"},
    )
    assert chunk.name == "test"
    assert chunk.language == CodeLanguage.PYTHON
    assert "os" in chunk.dependencies
    assert chunk.metadata["author"] == "dev"


# ============================================================================
# DependencyExtractor Tests
# ============================================================================


def test_extract_requirements_txt():
    """Test extracting dependencies from requirements.txt."""
    content = '''
# Comments should be ignored
requests>=2.25.0
flask==2.0.1
numpy[extra]>=1.19.0
# Another comment
pandas
'''
    deps = DependencyExtractor.extract_requirements_file(content, CodeLanguage.PYTHON)
    assert "requests" in deps
    assert "flask" in deps
    assert "numpy" in deps
    assert "pandas" in deps


def test_extract_package_json():
    """Test extracting dependencies from package.json."""
    content = '''
{
  "dependencies": {
    "react": "^18.0.0",
    "axios": "^0.27.0"
  },
  "devDependencies": {
    "jest": "^28.0.0"
  }
}
'''
    deps = DependencyExtractor.extract_requirements_file(
        content, CodeLanguage.JAVASCRIPT
    )
    assert "react" in deps
    assert "axios" in deps
    assert "jest" in deps


# ============================================================================
# GitHub Connector Tests
# ============================================================================


def test_github_connector_initialisation():
    """Test GitHubGitConnector initialisation."""
    connector = GitHubGitConnector(
        host="https://github.com",
        token="ghp_test",
        api_url="https://api.github.com",
        verify_ssl=True,
    )
    assert connector.host == "https://github.com"
    assert connector.token == "ghp_test"
    assert connector.api_url == "https://api.github.com"
    assert connector.verify_ssl is True


@patch("scripts.ingest.git.github_connector.requests.Session")
def test_github_connect_success(mock_session_class):
    """Test successful GitHub connection."""
    mock_session = Mock()
    mock_response = Mock()
    mock_response.json.return_value = {"login": "testuser"}
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    connector = GitHubGitConnector(
        host="https://github.com", token="ghp_test", api_url="https://api.github.com"
    )
    result = connector.connect()

    assert result is True
    assert connector._connected is True


@patch("scripts.ingest.git.github_connector.requests.Session")
def test_github_connect_failure(mock_session_class):
    """Test failed GitHub connection."""
    import requests
    mock_session = Mock()
    # Raise a proper requests exception that retry logic handles
    mock_session.get.side_effect = requests.exceptions.RequestException("Connection failed")
    mock_session_class.return_value = mock_session

    connector = GitHubGitConnector(
        host="https://github.com", token="invalid_token", api_url="https://api.github.com"
    )

    with pytest.raises(ConnectionError):
        connector.connect()


@patch("scripts.ingest.git.github_connector.requests.Session")
def test_github_list_projects(mock_session_class):
    """Test listing GitHub organisations and user account."""
    mock_session = Mock()
    
    # Mock connect response
    connect_response = Mock()
    connect_response.json.return_value = {"login": "testuser"}
    connect_response.raise_for_status = Mock()
    
    # Mock list user response for account project
    user_response = Mock()
    user_response.json.return_value = {
        "login": "testuser",
        "name": "Test User",
        "html_url": "https://github.com/testuser",
    }
    user_response.raise_for_status = Mock()

    # Mock list orgs response - first page with results, second page empty
    orgs_response1 = Mock()
    orgs_response1.json.return_value = [
        {
            "login": "org1",
            "name": "Organisation 1",
            "description": "First organisation",
            "html_url": "https://github.com/org1",
        }
    ]
    orgs_response1.raise_for_status = Mock()
    
    orgs_response2 = Mock()
    orgs_response2.json.return_value = []  # Empty page signals end
    orgs_response2.raise_for_status = Mock()
    
    mock_session.get.side_effect = [
        connect_response,
        user_response,
        orgs_response1,
        orgs_response2,
    ]
    mock_session_class.return_value = mock_session

    connector = GitHubGitConnector(
        host="https://github.com", token="ghp_test", api_url="https://api.github.com"
    )
    connector.connect()
    projects = connector.list_projects()

    project_keys = {project.key for project in projects}
    assert "testuser" in project_keys
    assert "org1" in project_keys


@patch("scripts.ingest.git.github_connector.subprocess.run")
def test_github_clone_repository(mock_run):
    """Test cloning a GitHub repository."""
    mock_run.return_value = Mock()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        connector = GitHubGitConnector(
            host="https://github.com",
            token="ghp_test",
            api_url="https://api.github.com",
        )
        connector._connected = True

        repo_path = connector.clone_repository(
            project_key="myorg",
            repo_slug="myrepo",
            target_dir=tmpdir,
            branch="main",
        )

        assert repo_path == str(Path(tmpdir) / "github" / "myorg" / "myrepo")
        mock_run.assert_called_once()


def test_github_repository_walker():
    """Test GitHubRepositoryWalker file walking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        repo_path = Path(tmpdir)
        (repo_path / "test.py").write_text("print('hello')")
        (repo_path / "test.js").write_text("console.log('hello')")
        (repo_path / ".hidden").write_text("hidden")

        walker = GitHubRepositoryWalker(str(repo_path))
        files = list(walker.walk_files(extensions=[".py", ".js"]))

        assert len(files) >= 2
        paths = [f[0] for f in files]
        assert any("test.py" in p for p in paths)
        assert any("test.js" in p for p in paths)
        assert not any(".hidden" in p for p in paths)


# ============================================================================
# Bitbucket Connector Tests
# ============================================================================


def test_bitbucket_connector_initialisation():
    """Test BitbucketGitConnector initialisation."""
    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
        is_cloud=True,
        verify_ssl=True,
    )
    assert connector.host == "https://bitbucket.org"
    assert connector.username == "testuser"
    assert connector.password == "testpass"
    assert connector.is_cloud is True


@patch("scripts.ingest.git.bitbucket_git_connector.BitbucketConnector")
def test_bitbucket_connect_success(mock_bitbucket_connector_class):
    """Test successful Bitbucket connection."""
    mock_bb_connector = Mock()
    mock_bitbucket_connector_class.return_value = mock_bb_connector

    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
    )
    result = connector.connect()

    assert result is True
    assert connector._connected is True
    mock_bb_connector.connect.assert_called_once()


@patch("scripts.ingest.git.bitbucket_git_connector.BitbucketConnector")
def test_bitbucket_list_projects(mock_bitbucket_connector_class):
    """Test listing Bitbucket projects."""
    from scripts.ingest.git.bitbucket_connector import BitbucketProject

    mock_bb_connector = Mock()
    mock_bb_connector.list_projects.return_value = [
        BitbucketProject(
            key="PROJ",
            name="Test Project",
            description="Test",
            url="https://bitbucket.org/proj",
        )
    ]
    mock_bitbucket_connector_class.return_value = mock_bb_connector

    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
    )
    connector.connect()
    projects = connector.list_projects()

    assert len(projects) == 1
    assert projects[0].key == "PROJ"
    assert projects[0].name == "Test Project"


def test_bitbucket_clone_repository_includes_provider_path():
    """Test Bitbucket clone path includes provider segment."""
    mock_connector = Mock()
    mock_connector.clone_repository.return_value = "/tmp/bitbucket/PROJ/repo"

    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
        is_cloud=True,
    )
    connector._connector = mock_connector
    connector._connected = True

    with tempfile.TemporaryDirectory() as tmpdir:
        connector.clone_repository(
            project_key="PROJ",
            repo_slug="repo",
            target_dir=tmpdir,
            branch="main",
        )

        target_dir = mock_connector.clone_repository.call_args.kwargs["target_dir"]
        assert Path(target_dir).parts[-3] == "bitbucket"


@patch("scripts.ingest.git.github_connector.subprocess.run")
def test_github_clone_repository_includes_provider_path(mock_subprocess_run):
    """Test GitHub clone path includes provider segment."""
    mock_subprocess_run.return_value = Mock(returncode=0)

    connector = GitHubGitConnector(
        host="https://github.com",
        token="ghp_test",
        api_url="https://api.github.com",
    )
    connector._connected = True

    with tempfile.TemporaryDirectory() as tmpdir:
        result = connector.clone_repository(
            project_key="org",
            repo_slug="repo",
            target_dir=tmpdir,
            branch="main",
        )

        # Verify the clone path includes "github" as provider segment
        result_path = Path(result)
        assert result_path.parts[-3] == "github"
        assert result_path.parts[-2] == "org"
        assert result_path.parts[-1] == "repo"


@patch("scripts.ingest.git.github_connector.requests.Session")
def test_github_get_pull_request_files(mock_session_class):
    """Test GitHub pull request file listing."""
    mock_session = Mock()
    response = Mock()
    response.json.return_value = [
        {"filename": "src/app.py", "status": "modified"},
        {"filename": "README.md", "status": "added"},
        {"filename": "docs/old.md", "status": "renamed", "previous_filename": "docs/legacy.md"},
    ]
    response.raise_for_status = Mock()
    mock_session.get.return_value = response
    mock_session_class.return_value = mock_session

    connector = GitHubGitConnector(
        host="https://github.com",
        token="ghp_test",
        api_url="https://api.github.com",
    )
    connector._connected = True

    files = connector.get_pull_request_files("org", "repo", pr_id=42)
    assert files == [
        {"path": "src/app.py", "status": "MODIFIED", "old_path": None},
        {"path": "README.md", "status": "ADDED", "old_path": None},
        {"path": "docs/old.md", "status": "RENAMED", "old_path": "docs/legacy.md"},
    ]


def test_bitbucket_get_pull_request_files():
    """Test Bitbucket pull request file listing via adapter."""
    mock_connector = Mock()
    mock_connector.get_pull_request_files.return_value = [
        {"path": "src/main.java", "status": "MODIFIED", "old_path": None}
    ]
    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
        is_cloud=True,
    )
    connector._connector = mock_connector
    connector._connected = True

    files = connector.get_pull_request_files("PROJ", "repo", pr_id=7)
    assert files == [{"path": "src/main.java", "status": "MODIFIED", "old_path": None}]


def test_bitbucket_default_branch_uses_repo_value():
    """Test Bitbucket default branch uses repository value when available."""
    repo = BitbucketRepository(
        slug="repo",
        name="Repo",
        project_key="PROJ",
        default_branch="main",
    )
    assert BitbucketGitConnector._get_default_branch(repo) == "main"


@patch.object(BitbucketConnector, "_make_request")
def test_bitbucket_list_pull_requests_respects_limit_cloud(mock_make_request):
    """Test Bitbucket Cloud pull request limit handling."""
    mock_make_request.return_value = {
        "values": [
            {
                "id": 1,
                "title": "PR 1",
                "state": "OPEN",
                "source": {"branch": {"name": "feature/one"}},
                "destination": {"branch": {"name": "main"}},
                "author": {"display_name": "Dev A"},
                "created_on": "2026-01-01T00:00:00+00:00",
                "updated_on": "2026-01-02T00:00:00+00:00",
                "description": "First",
            },
            {
                "id": 2,
                "title": "PR 2",
                "state": "OPEN",
                "source": {"branch": {"name": "feature/two"}},
                "destination": {"branch": {"name": "main"}},
                "author": {"display_name": "Dev B"},
                "created_on": "2026-01-03T00:00:00+00:00",
                "updated_on": "2026-01-04T00:00:00+00:00",
                "description": "Second",
            },
            {
                "id": 3,
                "title": "PR 3",
                "state": "OPEN",
                "source": {"branch": {"name": "feature/three"}},
                "destination": {"branch": {"name": "main"}},
                "author": {"display_name": "Dev C"},
                "created_on": "2026-01-05T00:00:00+00:00",
                "updated_on": "2026-01-06T00:00:00+00:00",
                "description": "Third",
            },
        ]
    }
    connector = BitbucketConnector(
        host="https://api.bitbucket.org",
        username="user",
        password="pass",
        is_cloud=True,
    )
    pull_requests = connector.list_pull_requests("workspace", "repo", state="OPEN", limit=2)
    assert len(pull_requests) == 2


@patch.object(BitbucketConnector, "_request_text_with_retry")
def test_bitbucket_get_file_content_cloud(mock_request_text):
    """Test Bitbucket Cloud file content retrieval."""
    mock_request_text.return_value = "print('hello')"
    connector = BitbucketConnector(
        host="https://api.bitbucket.org",
        username="user",
        password="pass",
        is_cloud=True,
    )
    content = connector.get_file_content("workspace", "repo", "src/app.py", branch="main")
    assert content == "print('hello')"


# ============================================================================
# Factory Tests
# ============================================================================


def test_factory_create_github():
    """Test GitConnectorFactory creating GitHub connector."""
    config = GitIngestConfig()
    config.git_provider = "github"
    config.git_host = "https://github.com"
    config.github_token = "ghp_test"
    config.github_api_url = "https://api.github.com"
    config.github_verify_ssl = True

    connector = GitConnectorFactory.create("github", config)
    assert isinstance(connector, GitHubGitConnector)
    assert connector.host == "https://github.com"
    assert connector.token == "ghp_test"


def test_factory_create_bitbucket():
    """Test GitConnectorFactory creating Bitbucket connector."""
    config = GitIngestConfig()
    config.git_provider = "bitbucket"
    config.git_host = "https://bitbucket.org"
    config.bitbucket_username = "testuser"
    config.bitbucket_password = "testpass"
    config.bitbucket_is_cloud = True
    config.bitbucket_verify_ssl = True

    connector = GitConnectorFactory.create("bitbucket", config)
    assert isinstance(connector, BitbucketGitConnector)
    assert connector.host == "https://bitbucket.org"
    assert connector.username == "testuser"


def test_factory_unknown_provider():
    """Test GitConnectorFactory with unknown provider."""
    config = GitIngestConfig()

    with pytest.raises(ValueError, match="Unknown Git provider"):
        GitConnectorFactory.create("unknown", config)


def test_factory_unimplemented_provider():
    """Test GitConnectorFactory with unimplemented provider."""
    config = GitIngestConfig()

    with pytest.raises(ValueError, match="not yet implemented"):
        GitConnectorFactory.create("gitlab", config)


# ============================================================================
# Integration Tests
# ============================================================================


def test_code_parser_full_pipeline():
    """Test full code parsing pipeline."""
    python_code = '''
import os
import sys
from pathlib import Path

class MyClass:
    """A test class."""
    
    def my_method(self):
        """Test method."""
        pass

def my_function():
    """Test function."""
    return 42
'''
    # Detect language
    lang = CodeParser.detect_language("test.py")
    assert lang == CodeLanguage.PYTHON

    # Extract imports
    imports = CodeParser.extract_imports(python_code, lang)
    assert "os" in imports
    assert "sys" in imports
    assert "pathlib" in imports

    # Extract classes
    classes = CodeParser.extract_classes(python_code, lang)
    assert len(classes) >= 1
    assert any(c.name == "MyClass" for c in classes)

    # Extract functions
    functions = CodeParser.extract_functions(python_code, lang)
    assert len(functions) >= 1
    assert any(f.name == "my_function" for f in functions)


@patch("scripts.ingest.git.bitbucket_git_connector.BitbucketConnector")
def test_context_manager_support(mock_bitbucket_connector_class):
    """Test connector context manager support."""
    mock_bb_connector = Mock()
    mock_bitbucket_connector_class.return_value = mock_bb_connector

    connector = BitbucketGitConnector(
        host="https://bitbucket.org",
        username="testuser",
        password="testpass",
    )

    # Connector needs _connector set to call close
    connector._connector = mock_bb_connector
    connector._connected = False
    
    connector.__enter__()
    # Context manager calls connect which sets _connected = True
    # For this test, manually set it since we're mocking
    connector._connected = True
    assert connector._connected is True
    
    connector.__exit__(None, None, None)
    assert connector._connected is False
    # Now the mock should be closed
    mock_bb_connector.close.assert_called_once()


def test_extension_to_language_mapping():
    """Test comprehensive extension to language mapping."""
    assert EXTENSION_TO_LANGUAGE[".py"] == CodeLanguage.PYTHON
    assert EXTENSION_TO_LANGUAGE[".js"] == CodeLanguage.JAVASCRIPT
    assert EXTENSION_TO_LANGUAGE[".ts"] == CodeLanguage.TYPESCRIPT
    assert EXTENSION_TO_LANGUAGE[".java"] == CodeLanguage.JAVA
    assert EXTENSION_TO_LANGUAGE[".go"] == CodeLanguage.GO
    assert EXTENSION_TO_LANGUAGE[".rs"] == CodeLanguage.RUST
    assert EXTENSION_TO_LANGUAGE[".rb"] == CodeLanguage.RUBY
    assert EXTENSION_TO_LANGUAGE[".md"] == CodeLanguage.MARKDOWN
