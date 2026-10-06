"""Embedding model configuration and verified model capabilities.

The active model must be declared with ``EMBEDDING_MODEL_NAME``. Keeping its
vector dimension and usable context in one place prevents ingestion and
retrieval from producing incompatible embeddings.
"""

from dataclasses import dataclass

from scripts.utils.config import BaseConfig


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Verified characteristics required by the embedding pipeline."""

    name: str
    dimensions: int
    max_tokens: int


SUPPORTED_EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "mxbai-embed-large": EmbeddingModelSpec(
        name="mxbai-embed-large",
        dimensions=1024,
        max_tokens=512,
    ),
    "nomic-embed-text": EmbeddingModelSpec(
        name="nomic-embed-text",
        dimensions=768,
        max_tokens=2048,
    ),
}

DEFAULT_EMBEDDING_MODEL_NAME = "mxbai-embed-large"


class EmbeddingConfig(BaseConfig):
    """Load and validate embedding model configuration.

    Environment Variables:
        EMBEDDING_MODEL_NAME: Supported Ollama embedding model name. Defaults
            to ``mxbai-embed-large``.
    """

    def __init__(self) -> None:
        """Initialise the configured model and its verified capabilities."""
        super().__init__()
        self.model_name = self.get_str("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)
        self.model_spec = get_embedding_model_spec(self.model_name)


def get_embedding_model_spec(model_name: str) -> EmbeddingModelSpec:
    """Return verified capabilities for a supported embedding model.

    Args:
        model_name: Ollama embedding model name, without an implicit tag.

    Raises:
        ValueError: When the configured model has no verified capabilities.
    """
    try:
        return SUPPORTED_EMBEDDING_MODELS[model_name]
    except KeyError as exc:
        supported_models = ", ".join(sorted(SUPPORTED_EMBEDDING_MODELS))
        raise ValueError(
            f"Unsupported EMBEDDING_MODEL_NAME '{model_name}'. "
            f"Supported models: {supported_models}."
        ) from exc


_CONFIG = EmbeddingConfig()
EMBEDDING_MODEL_NAME = _CONFIG.model_spec.name
EXPECTED_EMBEDDING_DIM = _CONFIG.model_spec.dimensions
EMBEDDING_MODEL_MAX_TOKEN_LIMIT = _CONFIG.model_spec.max_tokens
