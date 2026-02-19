"""Pytest configuration and fixtures for RAG test suite."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to Python path so scripts module can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock dash and plotly modules BEFORE any test imports them
# This prevents import errors when running with coverage
dash_mock = MagicMock()
dash_mock.html = MagicMock()
dash_mock.dcc = MagicMock()
dash_mock.callback = MagicMock()
dash_mock.Input = MagicMock()
dash_mock.Output = MagicMock()
dash_mock.State = MagicMock()

sys.modules['dash'] = dash_mock
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

# Import academic model fixtures
pytest_plugins = ["tests.conftest_academic_models"]

# Stub langchain_ollama if not installed to allow tests to import modules
if "langchain_ollama" not in sys.modules:
    dummy_module = types.ModuleType("langchain_ollama")

    class _DummyLLM:  # noqa: D401
        def __init__(self, model=None):
            self.model = model

        def invoke(self, prompt):
            return ""

    class _DummyEmbeddings:  # noqa: D401
        def __init__(self, model=None, num_ctx=None, keep_alive=None, **kwargs):
            self.model = model
            self.num_ctx = num_ctx
            self.keep_alive = keep_alive

        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            # Delegate to embed_documents for compatibility with production code
            return self.embed_documents([text])[0]

    dummy_module.OllamaLLM = _DummyLLM
    dummy_module.OllamaEmbeddings = _DummyEmbeddings
    sys.modules["langchain_ollama"] = dummy_module


@pytest.fixture(autouse=True)
def cleanup_cache_singletons():
    """Clean up cache singletons after each test to prevent ResourceWarnings.
    
    This ensures that any CacheDB instances created during a test are properly
    closed after the test completes, preventing unclosed database connection
    warnings in test teardown.
    """
    yield
    
    # After each test, clear the cache singleton registry
    try:
        from scripts.utils.db_factory import _cache_instances, _cache_lock
        
        # Close all instances in the registry
        with _cache_lock:
            instances_to_close = list(_cache_instances.values())
            _cache_instances.clear()
        
        # Close instances outside the lock
        for instance in instances_to_close:
            try:
                instance.close()
            except Exception:
                pass
    except Exception:
        pass
