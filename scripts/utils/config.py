"""Base configuration class with common environment variable loading patterns.

Provides utilities for:
- Environment variable loading with type conversion
- Singleton pattern
- Path expansion and validation
- Dotenv integration
- Common configuration properties shared across modules
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv

T = TypeVar("T", bound="BaseConfig")


class BaseConfig:
    """Base configuration class with environment variable utilities.

    Provides common functionality for loading configuration from environment
    variables with type conversion, default values, and path handling.

    Subclasses should define their configuration in __init__ using the
    helper methods provided (get_str, get_int, get_bool, get_path, etc.).

    Example:
        class MyConfig(BaseConfig):
            def __init__(self):
                super().__init__()
                self.my_setting = self.get_str("MY_SETTING", "default_value")
                self.my_count = self.get_int("MY_COUNT", 10)
                self.enabled = self.get_bool("MY_ENABLED", True)
    """

    _instances: Dict[Type, Any] = {}
    _overrides: Dict[str, Any] = {}

    def __init__(self, load_env: bool = True, overrides: Optional[Dict[str, Any]] = None):
        """Initialise base configuration.

        Args:
            load_env: Whether to load .env file (default: True)
            overrides: Optional override values (highest priority)
        """
        if load_env:
            # Load .env without overriding existing environment variables
            load_dotenv(override=False)
        if overrides:
            self.set_overrides(overrides)

    @classmethod
    def get_singleton(cls: Type[T]) -> T:
        """Get or create singleton instance of config class.

        Returns:
            Singleton instance of the config class

        Example:
            config = MyConfig.get_singleton()
        """
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

    @classmethod
    def set_overrides(cls, overrides: Dict[str, Any]) -> None:
        """Set high-priority override values (e.g., from CLI).

        Args:
            overrides: Mapping of env var names to values.
        """
        # Preserve keys even if value is falsy (e.g., 0, "", False)
        for key, value in overrides.items():
            cls._overrides[key] = value

    @classmethod
    def clear_overrides(cls) -> None:
        """Clear override values (useful between runs/tests)."""
        cls._overrides.clear()

    @classmethod
    def _get_override(cls, var_name: str) -> Any:
        if var_name in cls._overrides:
            return cls._overrides[var_name]
        return None

    # Environment variable loading utilities

    @classmethod
    def get_str(cls, var_name: str, default: str = "") -> str:
        """Get string value from environment variable.

        Args:
            var_name: Environment variable name
            default: Default value if not set

        Returns:
            String value from environment or default
        """
        override = cls._get_override(var_name)
        if override is not None:
            return str(override)
        return os.getenv(var_name, default)

    @classmethod
    def get_int(cls, var_name: str, default: int = 0) -> int:
        """Get integer value from environment variable.

        Args:
            var_name: Environment variable name
            default: Default value if not set or invalid

        Returns:
            Integer value from environment or default
        """
        override = cls._get_override(var_name)
        if override is not None:
            try:
                return int(override)
            except (TypeError, ValueError):
                return default
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @classmethod
    def get_float(cls, var_name: str, default: float = 0.0) -> float:
        """Get float value from environment variable.

        Args:
            var_name: Environment variable name
            default: Default value if not set or invalid

        Returns:
            Float value from environment or default
        """
        override = cls._get_override(var_name)
        if override is not None:
            try:
                return float(override)
            except (TypeError, ValueError):
                return default
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @classmethod
    def get_bool(cls, var_name: str, default: bool = False) -> bool:
        """Get boolean value from environment variable.

        Treats "true", "yes", "1", "on" (case-insensitive) as True.
        Treats "false", "no", "0", "off" (case-insensitive) as False.

        Args:
            var_name: Environment variable name
            default: Default value if not set

        Returns:
            Boolean value from environment or default
        """
        override = cls._get_override(var_name)
        if override is not None:
            if isinstance(override, bool):
                return override
            return str(override).lower() in ("true", "yes", "1", "on")
        value = os.getenv(var_name)
        if value is None:
            return default
        return value.lower() in ("true", "yes", "1", "on")

    @classmethod
    def get_path(cls, var_name: str, default: str, expand_user: bool = True) -> str:
        """Get path value from environment variable.

        Args:
            var_name: Environment variable name
            default: Default path value
            expand_user: Whether to expand ~ to user home directory

        Returns:
            Path string, potentially expanded
        """
        override = cls._get_override(var_name)
        value = str(override) if override is not None else os.getenv(var_name, default)
        if expand_user:
            return os.path.expanduser(value)
        return value

    @classmethod
    def get_list(
        cls, var_name: str, default: Optional[List[str]] = None, separator: str = ","
    ) -> List[str]:
        """Get list value from environment variable (comma-separated by default).

        Args:
            var_name: Environment variable name
            default: Default list if not set
            separator: String separator for list items

        Returns:
            List of strings from environment or default
        """
        if default is None:
            default = []

        override = cls._get_override(var_name)
        if override is not None:
            if isinstance(override, list):
                return override
            value = str(override)
        else:
            value = os.getenv(var_name)
        if value is None or value.strip() == "":
            return default

        return [item.strip() for item in value.split(separator) if item.strip()]

    def ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure exists

        Returns:
            The path object (for chaining)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Common configuration properties used across modules

    @property
    def rag_data_path(self) -> str:
        """Get RAG data path (ChromaDB storage location).

        Common across all modules. Override in subclass if needed.
        """
        if not hasattr(self, "_rag_data_path"):
            self._rag_data_path = self.get_path(
                "RAG_DATA_PATH",
                "~/rag-project/rag_data",
            )
        return self._rag_data_path

    @rag_data_path.setter
    def rag_data_path(self, value: str) -> None:
        """Set RAG data path."""
        self._rag_data_path = value

    @property
    def chunk_collection_name(self) -> str:
        """Get chunk collection name in ChromaDB.

        Common across all modules.
        """
        if not hasattr(self, "_chunk_collection_name"):
            self._chunk_collection_name = self.get_str(
                "CHUNK_COLLECTION_NAME",
                "governance_docs_chunks",
            )
        return self._chunk_collection_name

    @chunk_collection_name.setter
    def chunk_collection_name(self, value: str) -> None:
        """Set chunk collection name."""
        self._chunk_collection_name = value

    @property
    def doc_collection_name(self) -> str:
        """Get document collection name in ChromaDB.

        Common across all modules.
        """
        if not hasattr(self, "_doc_collection_name"):
            self._doc_collection_name = self.get_str(
                "DOC_COLLECTION_NAME",
                "governance_docs_documents",
            )
        return self._doc_collection_name

    @doc_collection_name.setter
    def doc_collection_name(self, value: str) -> None:
        """Set document collection name."""
        self._doc_collection_name = value

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path.

        Common across all modules. Override in subclass for custom location.
        """
        if not hasattr(self, "_logs_dir"):
            # Default to project root logs directory
            project_root = Path(__file__).parent.parent.parent
            self._logs_dir = self.ensure_dir(project_root / "logs")
        return self._logs_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary of all public attributes (non-methods, non-private),
            with JSON-safe value conversions (e.g., sets to lists), and
            includes common configuration properties (e.g., collection names).
        """
        result: Dict[str, Any] = {}
        skip_keys = {"logger", "args", "version_lock"}
        for key, value in self.__dict__.items():
            if key in skip_keys or key.startswith("_") or callable(value):
                continue

            # Convert non-JSON-serialisable types
            if isinstance(value, set):
                result[key] = list(value)
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            elif hasattr(value, "__dict__") and not isinstance(value, type):
                # Convert complex objects to string representation
                result[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                # Fallback: ensure any unsupported type becomes a string
                result[key] = str(value)
            else:
                result[key] = value

        # Include common properties explicitly (backed by private attrs)
        try:
            result["rag_data_path"] = str(self.rag_data_path)
        except Exception:
            pass
        try:
            result["chunk_collection_name"] = self.chunk_collection_name
        except Exception:
            pass
        try:
            result["doc_collection_name"] = self.doc_collection_name
        except Exception:
            pass

        return result

    def __repr__(self) -> str:
        """String representation of configuration."""
        items = ", ".join(f"{k}={repr(v)}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({items})"
