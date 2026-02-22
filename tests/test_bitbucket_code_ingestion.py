"""Comprehensive unit tests for BitbucketCodeIngestion module.

TODO: Move examples.git.bitbucket_code_ingestion to a better location.

Tests cover core business logic, error handling, and edge cases
with focus on newly refactored helper methods.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from examples.git.bitbucket_code_ingestion import (
    BitbucketCodeIngestion,
    _clone_repository_safe,
    _get_file_iterator,
    _parse_and_aggregate_files,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_connector():
    """Create a mock BitbucketConnector."""
    return Mock()


@pytest.fixture
def mock_parser():
    """Create a mock CodeParser."""
    parser = Mock()
    # Default parse result
    result = Mock()
    result.external_dependencies = {"org.apache:lib:1.0"}
    result.service_type = "service"
    result.errors = []
    result.to_dict.return_value = {"parsed": "data"}
    parser.parse_file.return_value = result
    return parser


@pytest.fixture
def mock_walker():
    """Create a mock RepositoryWalker."""
    walker = Mock()
    walker.get_directory_structure.return_value = {
        "groovy_files": ["Route.groovy"],
        "java_files": ["Main.java"],
        "gradle_files": ["build.gradle"],
        "jenkins_files": [],
    }
    return walker


@pytest.fixture
def temp_repo():
    """Create a temporary repository structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create test files
        (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
        (repo_path / "src" / "main" / "java").mkdir(parents=True)

        groovy_file = repo_path / "src" / "main" / "groovy" / "Route.groovy"
        groovy_file.write_text("""
        from('jms:queue:input')
            .to('service')
            .to('kafka:output')
        """)

        java_file = repo_path / "src" / "main" / "java" / "Main.java"
        java_file.write_text("""
        public class Main {
            public static void main(String[] args) {}
        }
        """)

        gradle_file = repo_path / "build.gradle"
        gradle_file.write_text("""
        dependencies {
            implementation 'org.apache:lib:1.0'
        }
        """)

        yield repo_path


# ============================================================================
# Test _clone_repository_safe
# ============================================================================


class TestCloneRepositorySafe:
    """Tests for _clone_repository_safe helper."""

    def test_clone_success(self, mock_connector):
        """Successful clone returns path and no error."""
        mock_connector.clone_repository.return_value = "/tmp/repo"

        path, error = _clone_repository_safe(mock_connector, "PROJ", "repo-slug")

        assert path == "/tmp/repo"
        assert error is None
        mock_connector.clone_repository.assert_called_once_with(
            "PROJ", "repo-slug", target_dir=None
        )

    def test_clone_with_custom_path(self, mock_connector):
        """Custom clone path is passed through."""
        mock_connector.clone_repository.return_value = "/custom/repo"

        path, error = _clone_repository_safe(
            mock_connector, "PROJ", "repo-slug", clone_path="/custom"
        )

        assert path == "/custom/repo"
        mock_connector.clone_repository.assert_called_once_with(
            "PROJ", "repo-slug", target_dir="/custom"
        )

    def test_clone_failure_returns_error_dict(self, mock_connector):
        """Clone failure returns error dictionary."""
        mock_connector.clone_repository.side_effect = RuntimeError("Network error")

        path, error = _clone_repository_safe(mock_connector, "PROJ", "repo-slug")

        assert path is None
        assert error is not None
        assert error["repository"] == "repo-slug"
        assert error["project_key"] == "PROJ"
        assert "Clone failed" in error["error"]

    def test_clone_permission_error(self, mock_connector):
        """Permission errors are caught and reported."""
        mock_connector.clone_repository.side_effect = PermissionError("Access denied")

        path, error = _clone_repository_safe(mock_connector, "PROJ", "repo-slug")

        assert path is None
        assert "Clone failed" in error["error"]


# ============================================================================
# Test _get_file_iterator
# ============================================================================


class TestGetFileIterator:
    """Tests for _get_file_iterator router."""

    def test_route_all_files(self, mock_walker):
        """Route 'all' to walk_all_code_files."""
        mock_walker.walk_all_code_files.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["all"], dated=False)

        mock_walker.walk_all_code_files.assert_called_once()
        assert iterator is not None

    def test_route_groovy_files(self, mock_walker):
        """Route 'groovy' to walk_groovy_files."""
        mock_walker.walk_groovy_files.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["groovy"], dated=False)

        mock_walker.walk_groovy_files.assert_called_once()

    def test_route_java_files(self, mock_walker):
        """Route 'java' to walk_java_files."""
        mock_walker.walk_java_files.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["java"], dated=False)

        mock_walker.walk_java_files.assert_called_once()

    def test_route_gradle_with_groovy(self, mock_walker):
        """Route 'gradle' to walk_groovy_files (same as groovy)."""
        mock_walker.walk_groovy_files.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["gradle"], dated=False)

        mock_walker.walk_groovy_files.assert_called_once()

    def test_route_custom_patterns(self, mock_walker):
        """Custom patterns route to walk_by_pattern."""
        mock_walker.walk_by_pattern.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["*.custom", "*.xyz"], dated=False)

        mock_walker.walk_by_pattern.assert_called_once()
        call_args = mock_walker.walk_by_pattern.call_args[0][0]
        # Pattern normalisation may create **/*.custom form
        assert any("custom" in str(p) for p in call_args) or "*.custom" in call_args

    def test_route_dated_all_files(self, mock_walker):
        """Dated routing for 'all' files."""
        before = datetime.now()
        after = datetime.now() - timedelta(days=1)
        mock_walker.walk_all_code_files_dated.return_value = iter([])

        iterator = _get_file_iterator(
            mock_walker, ["all"], dated=True, modified_before=before, modified_after=after
        )

        mock_walker.walk_all_code_files_dated.assert_called_once_with(
            modified_before=before, modified_after=after
        )

    def test_route_dated_groovy_files(self, mock_walker):
        """Dated routing for groovy files."""
        before = datetime.now()
        mock_walker.walk_groovy_files_dated.return_value = iter([])

        iterator = _get_file_iterator(mock_walker, ["groovy"], dated=True, modified_before=before)

        mock_walker.walk_groovy_files_dated.assert_called_once()


# ============================================================================
# Test _parse_and_aggregate_files
# ============================================================================


class TestParseAndAggregateFiles:
    """Tests for _parse_and_aggregate_files helper."""

    def test_parse_single_file(self, mock_parser):
        """Parse a single file and aggregate results."""
        files = [("file.groovy", "content")]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert len(parsed) == 1
        assert parsed[0]["file_path"] == "file.groovy"
        assert parsed[0]["repository"] == "repo"
        assert parsed[0]["project_key"] == "PROJ"
        assert "org.apache:lib:1.0" in deps
        assert services["service"] == 1
        assert errors == 0

    def test_parse_multiple_files(self, mock_parser):
        """Parse multiple files and aggregate counts."""
        files = [
            ("file1.groovy", "content1"),
            ("file2.groovy", "content2"),
            ("file3.groovy", "content3"),
        ]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert len(parsed) == 3
        assert services["service"] == 3
        assert errors == 0

    def test_parse_with_dates(self, mock_parser):
        """Parse files with modification dates."""
        now = datetime.now()
        files = [
            ("file.groovy", "content", now),
        ]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=True
        )

        assert len(parsed) == 1
        assert "modified_date" in parsed[0]
        assert isinstance(parsed[0]["modified_date"], str)

    def test_parse_error_handling(self, mock_parser):
        """Errors during parsing are caught and counted."""
        mock_parser.parse_file.side_effect = RuntimeError("Parse failed")

        files = [("file.groovy", "content")]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert len(parsed) == 0
        assert errors == 1

    def test_parse_mixed_success_and_error(self, mock_parser):
        """Mix of successful and failed parses."""
        success_result = Mock()
        success_result.external_dependencies = {"lib:1.0"}
        success_result.service_type = "service"
        success_result.errors = []
        success_result.to_dict.return_value = {"ok": True}

        mock_parser.parse_file.side_effect = [
            success_result,
            RuntimeError("Failed"),
            success_result,
        ]

        files = [
            ("file1.groovy", "content1"),
            ("file2.groovy", "content2"),
            ("file3.groovy", "content3"),
        ]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert len(parsed) == 2
        assert errors == 1
        assert services["service"] == 2

    def test_aggregate_multiple_service_types(self, mock_parser):
        """Aggregate different service types."""
        result1 = Mock()
        result1.external_dependencies = set()
        result1.service_type = "service"
        result1.errors = []
        result1.to_dict.return_value = {"type": "service"}

        result2 = Mock()
        result2.external_dependencies = set()
        result2.service_type = "controller"
        result2.errors = []
        result2.to_dict.return_value = {"type": "controller"}

        mock_parser.parse_file.side_effect = [result1, result2, result1]

        files = [
            ("file1.groovy", "content1"),
            ("file2.groovy", "content2"),
            ("file3.groovy", "content3"),
        ]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert services["service"] == 2
        assert services["controller"] == 1

    def test_aggregate_external_dependencies(self, mock_parser):
        """Aggregate unique external dependencies."""
        result1 = Mock()
        result1.external_dependencies = {"lib1:1.0", "lib2:2.0"}
        result1.service_type = None
        result1.errors = []
        result1.to_dict.return_value = {}

        result2 = Mock()
        result2.external_dependencies = {"lib2:2.0", "lib3:3.0"}
        result2.service_type = None
        result2.errors = []
        result2.to_dict.return_value = {}

        mock_parser.parse_file.side_effect = [result1, result2]

        files = [("file1.groovy", "content1"), ("file2.groovy", "content2")]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        # Should have 3 unique dependencies
        assert len(deps) == 3
        assert "lib1:1.0" in deps
        assert "lib2:2.0" in deps
        assert "lib3:3.0" in deps

    def test_parse_errors_from_parse_result(self, mock_parser):
        """Count parse result errors in total error count."""
        result_with_errors = Mock()
        result_with_errors.external_dependencies = set()
        result_with_errors.service_type = None
        result_with_errors.errors = ["error1", "error2"]
        result_with_errors.to_dict.return_value = {}

        mock_parser.parse_file.return_value = result_with_errors

        files = [("file.groovy", "content")]

        parsed, deps, services, errors = _parse_and_aggregate_files(
            files, "repo", "PROJ", mock_parser, include_dates=False
        )

        assert len(parsed) == 1
        assert errors == 1  # Count incremented due to parse result having errors


# ============================================================================
# Integration Tests for Refactored Methods
# ============================================================================


class TestIngestRepositoryRefactored:
    """Tests for refactored ingest_repository method."""

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    @patch("examples.git.bitbucket_code_ingestion.RepositoryWalker")
    def test_ingest_repository_success(self, mock_walker_class, mock_connector_class, temp_repo):
        """Successful repository ingestion."""
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector
        mock_connector.clone_repository.return_value = str(temp_repo)

        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.get_directory_structure.return_value = {"groovy_files": ["Route.groovy"]}
        mock_walker.walk_all_code_files.return_value = iter(
            [
                ("Route.groovy", "content"),
            ]
        )

        ingestion = BitbucketCodeIngestion(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
        )

        result = ingestion.ingest_repository("PROJ", "repo-slug")

        assert "repository" in result
        assert result["repository"] == "repo-slug"
        assert "project_key" in result
        assert result["project_key"] == "PROJ"
        assert "parsed_files" in result
        assert "summary" in result

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    def test_ingest_repository_clone_failure(self, mock_connector_class):
        """Clone failure returns error dict."""
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector
        mock_connector.clone_repository.side_effect = RuntimeError("Clone failed")

        ingestion = BitbucketCodeIngestion(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
        )

        result = ingestion.ingest_repository("PROJ", "repo-slug")

        assert "error" in result
        assert "Clone failed" in result["error"]


class TestIngestRepositoryDatedRefactored:
    """Tests for refactored ingest_repository_dated method."""

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    @patch("examples.git.bitbucket_code_ingestion.RepositoryWalker")
    def test_ingest_dated_with_dates(self, mock_walker_class, mock_connector_class, temp_repo):
        """Dated ingestion with modification dates."""
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector
        mock_connector.clone_repository.return_value = str(temp_repo)

        now = datetime.now()

        mock_walker = Mock()
        mock_walker_class.return_value = mock_walker
        mock_walker.walk_all_code_files_dated.return_value = iter(
            [
                ("Route.groovy", "content", now),
            ]
        )

        ingestion = BitbucketCodeIngestion(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
        )

        before = datetime.now() + timedelta(days=1)
        result = ingestion.ingest_repository_dated("PROJ", "repo-slug", modified_before=before)

        assert "date_filter" in result
        assert result["date_filter"]["modified_before"] is not None
        assert "parsed_files" in result
        if result["parsed_files"]:
            assert "modified_date" in result["parsed_files"][0]
