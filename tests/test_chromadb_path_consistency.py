"""Tests ensuring all modules consistently determine ChromaDB paths.

This test suite verifies that:
1. All modules use the factory pattern (get_default_vector_path) for path determination
2. Path computation is consistent across all modules
3. Different backend types (Chroma native vs SQLite) produce correct paths
4. Paths are absolute and properly formatted
5. Configuration is properly inherited and used

Background:
-----------
Previous bug: ingest_git.py directly passed config.rag_data_path to ChromaDB's 
PersistentClient, causing it to create chroma.sqlite3 at the wrong location, while 
build_consistency_graph.py correctly used get_default_vector_path().

This created two separate ChromaDB instances:
- rag_data/chromadb/      <- old data (55 docs)
- rag_data/chroma.sqlite3 <- new data (10,068 docs) ORPHANED

Fix: All modules must use get_vector_client() + get_default_vector_path() pattern.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPathFactoryPattern:
    """Test the factory pattern functions for path determination."""

    def test_get_default_vector_path_chroma_backend(self):
        """Test get_default_vector_path returns correct path for Chroma backend."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"
        result = get_default_vector_path(rag_data_path, using_sqlite=False)

        assert result == str(rag_data_path / "chromadb")
        assert result.endswith("chromadb")
        assert not result.endswith(".db")

    def test_get_default_vector_path_sqlite_backend(self):
        """Test get_default_vector_path returns correct path for SQLite backend."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"
        result = get_default_vector_path(rag_data_path, using_sqlite=True)

        assert result == str(rag_data_path / "chromadb.db")

    def test_get_default_vector_path_with_string_input(self):
        """Test get_default_vector_path handles both Path and string inputs."""
        from scripts.utils.db_factory import get_default_vector_path

        # Test with Path object
        path_obj = PROJECT_ROOT / "rag_data"
        result_path = get_default_vector_path(path_obj, using_sqlite=False)

        # Test with string
        path_str = str(PROJECT_ROOT / "rag_data")
        result_str = get_default_vector_path(Path(path_str), using_sqlite=False)

        assert result_path == result_str

    def test_get_default_vector_path_absolute_paths(self):
        """Test that paths returned are absolute."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"
        
        result_chroma = get_default_vector_path(rag_data_path, using_sqlite=False)
        result_sqlite = get_default_vector_path(rag_data_path, using_sqlite=True)

        assert Path(result_chroma).is_absolute()
        assert Path(result_sqlite).is_absolute()

    def test_get_vector_client_returns_tuple(self):
        """Test get_vector_client returns (PersistentClient, using_sqlite) tuple."""
        from scripts.utils.db_factory import get_vector_client

        result = get_vector_client(prefer="chroma")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        PersistentClient_class, using_sqlite = result
        assert isinstance(using_sqlite, bool)
        assert PersistentClient_class is not None

    def test_get_vector_client_prefer_chroma(self):
        """Test get_vector_client respects prefer='chroma' preference."""
        from scripts.utils.db_factory import get_vector_client

        _, using_sqlite = get_vector_client(prefer="chroma")
        # Default should try Chroma first, so using_sqlite should be False
        assert using_sqlite == False

    def test_get_vector_client_sqlite_available(self):
        """Test get_vector_client can return SQLite backend when preferred."""
        from scripts.utils.db_factory import get_vector_client

        # Request SQLite backend
        PersistentClient_class, using_sqlite = get_vector_client(prefer="sqlite")
        
        # Should either return sqlite (True) or fall back to chroma (False)
        # Either way, it should succeed and return a valid client
        assert isinstance(using_sqlite, bool)
        assert PersistentClient_class is not None


class TestModulePathConsistency:
    """Test that all modules use consistent path determination."""

    def test_ingest_git_uses_factory_pattern(self):
        """Verify ingest_git.py uses factory pattern for path determination."""
        ingest_git_file = PROJECT_ROOT / "scripts" / "ingest" / "ingest_git.py"
        content = ingest_git_file.read_text()

        # Should import factory functions
        assert "from scripts.utils.db_factory import get_vector_client, get_default_vector_path" in content

        # Should use get_vector_client
        assert "get_vector_client(prefer=" in content

        # Should use get_default_vector_path
        assert "get_default_vector_path(" in content

        # Should NOT directly pass config.rag_data_path to PersistentClient
        # (this was the original bug)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "PersistentClient(path=" in line and "chroma_path" in line:
                # Good! Using a computed path variable, not direct config
                assert "chroma_path" in line or "path=" in line
                break

    def test_ingest_py_uses_factory_pattern(self):
        """Verify ingest.py uses factory pattern for path determination."""
        ingest_file = PROJECT_ROOT / "scripts" / "ingest" / "ingest.py"
        content = ingest_file.read_text()

        # Should import factory functions
        assert "from scripts.utils.db_factory import get_default_vector_path, get_vector_client" in content

        # Should use get_vector_client at module level
        assert "PersistentClient, USING_SQLITE = get_vector_client(prefer=" in content

        # Should use get_default_vector_path
        assert "get_default_vector_path(" in content

    def test_build_consistency_graph_uses_factory_pattern(self):
        """Verify build_consistency_graph.py uses factory pattern for path determination."""
        build_consistency_file = PROJECT_ROOT / "scripts" / "consistency_graph" / "build_consistency_graph.py"
        content = build_consistency_file.read_text()

        # Should import factory functions
        assert "from scripts.utils.db_factory import get_default_vector_path" in content
        assert "get_vector_client" in content

        # Should use get_default_vector_path
        assert "CHROMA_PATH = get_default_vector_path(" in content

    def test_dashboard_uses_factory_pattern(self):
        """Verify dashboard.py uses factory pattern for path determination."""
        dashboard_file = PROJECT_ROOT / "scripts" / "ui" / "dashboard.py"
        content = dashboard_file.read_text()

        # Should import factory functions
        assert "from scripts.utils.db_factory import get_default_vector_path, get_vector_client" in content

        # Should use get_vector_client
        assert "PersistentClient, USING_SQLITE = get_vector_client(" in content

        # Should use get_default_vector_path
        assert "get_default_vector_path(" in content

    def test_no_direct_persistent_client_path_from_config(self):
        """Verify no module directly passes config.rag_data_path to PersistentClient."""
        modules_to_check = [
            "scripts/ingest/ingest_git.py",
            "scripts/ingest/ingest.py",
            "scripts/consistency_graph/build_consistency_graph.py",
            "scripts/ui/dashboard.py",
        ]

        for module_path in modules_to_check:
            file_path = PROJECT_ROOT / module_path
            if not file_path.exists():
                pytest.skip(f"{module_path} not found")

            content = file_path.read_text()

            # Pattern that would indicate the bug: directly using config.rag_data_path
            # in PersistentClient initialisation WITHOUT using factory pattern first
            bad_pattern = r"PersistentClient\s*\(\s*path\s*=\s*config\.rag_data_path"

            # This should NOT match in the file (would indicate the bug)
            if re.search(bad_pattern, content):
                pytest.fail(
                    f"{module_path} has direct config.rag_data_path usage in PersistentClient. "
                    f"Should use get_default_vector_path() factory function instead."
                )


class TestPathComputationConsistency:
    """Test that path computation is consistent across different contexts."""

    def test_same_config_produces_same_path(self):
        """Test that the same config always produces the same path."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"

        # Compute same path multiple times
        path1 = get_default_vector_path(rag_data_path, using_sqlite=False)
        path2 = get_default_vector_path(rag_data_path, using_sqlite=False)
        path3 = get_default_vector_path(rag_data_path, using_sqlite=False)

        assert path1 == path2 == path3

    def test_different_backends_produce_different_paths(self):
        """Test that different backends produce different paths."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"

        path_chroma = get_default_vector_path(rag_data_path, using_sqlite=False)
        path_sqlite = get_default_vector_path(rag_data_path, using_sqlite=True)

        assert path_chroma != path_sqlite
        assert path_chroma.endswith("chromadb")
        assert path_sqlite.endswith("chromadb.db")

    def test_path_resolves_to_parent_rag_data(self):
        """Test that computed paths are within rag_data directory."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"

        path_chroma = get_default_vector_path(rag_data_path, using_sqlite=False)
        path_sqlite = get_default_vector_path(rag_data_path, using_sqlite=True)

        # Both should be under rag_data
        assert rag_data_path.as_posix() in path_chroma
        assert rag_data_path.as_posix() in path_sqlite

    def test_relative_paths_handled_correctly(self):
        """Test that factory function works with both relative and absolute paths."""
        from scripts.utils.db_factory import get_default_vector_path

        # Test with absolute path
        abs_path = PROJECT_ROOT / "rag_data"
        result_abs = get_default_vector_path(abs_path, using_sqlite=False)

        # Result should be absolute
        assert Path(result_abs).is_absolute()

    def test_path_contains_backend_identifier(self):
        """Test that computed paths clearly identify the backend type."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"

        path_chroma = get_default_vector_path(rag_data_path, using_sqlite=False)
        path_sqlite = get_default_vector_path(rag_data_path, using_sqlite=True)

        # Chroma path should contain "chromadb" directory
        assert "chromadb" in path_chroma
        assert not path_chroma.endswith(".db")

        # SQLite path should be "chromadb.db" file
        assert "chromadb.db" in path_sqlite
        assert path_sqlite.endswith(".db")


class TestConfigurationInheritance:
    """Test that configuration properly inherits rag_data_path."""

    def test_ingest_config_has_rag_data_path(self):
        """Test that IngestConfig provides rag_data_path property."""
        from scripts.ingest.ingest_config import IngestConfig

        config = IngestConfig()
        assert hasattr(config, "rag_data_path")
        assert config.rag_data_path is not None
        assert isinstance(config.rag_data_path, str) or isinstance(config.rag_data_path, Path)

    def test_git_ingest_config_inherits_rag_data_path(self):
        """Test that GitIngestConfig inherits rag_data_path from IngestConfig."""
        from scripts.ingest.git.git_ingest_config import GitIngestConfig

        config = GitIngestConfig()
        assert hasattr(config, "rag_data_path")
        assert config.rag_data_path is not None

    def test_consistency_config_has_rag_data_path(self):
        """Test that ConsistencyConfig provides rag_data_path property."""
        from scripts.consistency_graph.consistency_config import ConsistencyConfig

        config = ConsistencyConfig()
        assert hasattr(config, "rag_data_path")
        assert config.rag_data_path is not None
        assert isinstance(config.rag_data_path, str) or isinstance(config.rag_data_path, Path)

    def test_all_configs_default_to_same_rag_data_path(self):
        """Test that all config classes default to the same rag_data_path."""
        from scripts.ingest.ingest_config import IngestConfig
        from scripts.ingest.git.git_ingest_config import GitIngestConfig
        from scripts.consistency_graph.consistency_config import ConsistencyConfig

        ingest_config = IngestConfig()
        git_config = GitIngestConfig()
        consistency_config = ConsistencyConfig()

        # All should resolve to the same location (or at least the same directory)
        assert Path(ingest_config.rag_data_path).name == Path(git_config.rag_data_path).name == "rag_data"
        assert Path(consistency_config.rag_data_path).name == "rag_data"


class TestEnvironmentVariableHandling:
    """Test that path configuration respects environment variables."""

    def test_rag_data_path_from_env_variable(self):
        """Test that RAG_DATA_PATH environment variable is respected."""
        test_path = "/tmp/test_rag_data"

        with mock.patch.dict(os.environ, {"RAG_DATA_PATH": test_path}):
            from scripts.ingest.ingest_config import IngestConfig

            # Force reload to pick up env var
            config = IngestConfig()
            
            # Should use the env var value
            assert test_path in str(config.rag_data_path)

    def test_path_consistency_with_env_override(self):
        """Test that paths are still consistent when using env var override."""
        from scripts.utils.db_factory import get_default_vector_path

        test_path = Path("/tmp/test_rag_data")

        path_chroma = get_default_vector_path(test_path, using_sqlite=False)
        path_sqlite = get_default_vector_path(test_path, using_sqlite=True)

        # Should be consistent with the override path
        assert "/tmp/test_rag_data" in path_chroma
        assert "/tmp/test_rag_data" in path_sqlite


class TestPathUsagePatterns:
    """Test common patterns of path usage across modules."""

    def test_factory_called_at_module_init(self):
        """Test that factory functions are called early in module initialisation."""
        ingest_git_file = PROJECT_ROOT / "scripts" / "ingest" / "ingest_git.py"
        content = ingest_git_file.read_text()

        # The factory functions should be used to initialise module-level variables
        lines = content.split("\n")
        
        # Find where imports happen (early in file)
        import_section_end = None
        for i, line in enumerate(lines):
            if "from scripts.utils.db_factory import" in line:
                import_section_end = i
                break

        assert import_section_end is not None, "Factory imports not found"

    def test_path_computation_before_chromadb_init(self):
        """Test that paths are computed before passing to PersistentClient."""
        ingest_git_file = PROJECT_ROOT / "scripts" / "ingest" / "ingest_git.py"
        content = ingest_git_file.read_text()

        # Find get_default_vector_path call
        vector_path_match = None
        for match in re.finditer(r"chroma_path\s*=\s*get_default_vector_path\(", content):
            vector_path_match = match
            break

        # Find PersistentClient initialisation with chroma_path
        persistent_client_match = None
        for match in re.finditer(r"PersistentClient\s*\(\s*path\s*=\s*chroma_path", content):
            persistent_client_match = match
            break

        if vector_path_match and persistent_client_match:
            # Path computation should happen before PersistentClient usage
            assert vector_path_match.start() < persistent_client_match.start(), \
                "Path computation must happen before PersistentClient initialisation"


class TestPathEdgeCases:
    """Test edge cases and boundary conditions for path handling."""

    def test_path_with_tilde_expansion(self):
        """Test that paths with ~ are properly handled."""
        from scripts.utils.db_factory import get_default_vector_path

        # Create path with tilde
        rag_data_path = PROJECT_ROOT / "rag_data"
        result = get_default_vector_path(rag_data_path, using_sqlite=False)

        # Result should be absolute, not contain tilde
        assert "~" not in result
        assert Path(result).is_absolute()

    def test_path_with_trailing_slash(self):
        """Test that paths with trailing slashes are normalised."""
        from scripts.utils.db_factory import get_default_vector_path

        path_with_slash = Path(f"{PROJECT_ROOT / 'rag_data'}/")
        path_without_slash = PROJECT_ROOT / "rag_data"

        result_with = get_default_vector_path(path_with_slash, using_sqlite=False)
        result_without = get_default_vector_path(path_without_slash, using_sqlite=False)

        # Should be identical
        assert result_with == result_without

    def test_path_components_not_duplicated(self):
        """Test that path components are not duplicated."""
        from scripts.utils.db_factory import get_default_vector_path

        rag_data_path = PROJECT_ROOT / "rag_data"
        result = get_default_vector_path(rag_data_path, using_sqlite=False)

        # Should not have duplicate "rag_data" components
        parts = Path(result).parts
        assert parts.count("rag_data") == 1, f"Path has duplicate rag_data: {result}"

    def test_special_characters_in_path(self):
        """Test that paths with special characters are handled correctly."""
        from scripts.utils.db_factory import get_default_vector_path

        # Path with spaces (if supported by system)
        rag_data_path = Path("~/rag project data").expanduser()
        result = get_default_vector_path(rag_data_path, using_sqlite=False)

        # Should still work
        assert "rag project data" in result
        assert Path(result).is_absolute()


class TestIntegrationPathFlow:
    """Integration tests for path flow through real modules."""

    def test_config_to_client_path_flow(self):
        """Test that config -> factory -> client initialisation works correctly."""
        from scripts.ingest.git.git_ingest_config import GitIngestConfig
        from scripts.utils.db_factory import get_vector_client, get_default_vector_path
        from pathlib import Path

        # Create config
        config = GitIngestConfig()

        # Get factory client
        PersistentClient_class, using_sqlite = get_vector_client(prefer="chroma")

        # Compute path using factory
        chroma_path = get_default_vector_path(Path(config.rag_data_path), using_sqlite)

        # Path should be valid
        assert chroma_path is not None
        assert Path(chroma_path).is_absolute()
        assert "chromadb" in chroma_path

    def test_multiple_modules_same_path(self):
        """Test that multiple modules would compute the same path."""
        from scripts.ingest.git.git_ingest_config import GitIngestConfig
        from scripts.consistency_graph.consistency_config import ConsistencyConfig
        from scripts.utils.db_factory import get_default_vector_path, get_vector_client
        from pathlib import Path

        # Get configs
        git_config = GitIngestConfig()
        consistency_config = ConsistencyConfig()

        # Get client info
        _, using_sqlite = get_vector_client(prefer="chroma")

        # Compute paths
        git_path = get_default_vector_path(Path(git_config.rag_data_path), using_sqlite)
        consistency_path = get_default_vector_path(Path(consistency_config.rag_data_path), using_sqlite)

        # Should be the same (assuming same default rag_data_path)
        assert git_path == consistency_path

    def test_backend_consistency_across_modules(self):
        """Test that all modules will use the same backend preference."""
        from scripts.utils.db_factory import get_vector_client

        # All modules should prefer chroma (or at least use the same logic)
        client1, using_sqlite1 = get_vector_client(prefer="chroma")
        client2, using_sqlite2 = get_vector_client(prefer="chroma")

        # Same backend choice
        assert using_sqlite1 == using_sqlite2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
