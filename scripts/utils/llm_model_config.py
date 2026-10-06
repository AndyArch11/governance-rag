"""Validated Ollama LLM model selection for application roles.

Model names are centralised here so ingestion, RAG generation, and consistency
validation cannot silently use different or unavailable defaults.
"""

from dataclasses import dataclass

from scripts.utils.config import BaseConfig


@dataclass(frozen=True)
class LLMModelSpec:
    """Supported Ollama model for application generation and validation."""

    name: str
    parameter_scale: str
    max_context_tokens: int


SUPPORTED_LLM_MODELS: dict[str, LLMModelSpec] = {
    "qwen3.6:27b": LLMModelSpec("qwen3.6:27b", "27B", 262144),
    "gemma4:26b": LLMModelSpec("gemma4:26b", "26B", 262144),
    "deepseek-r1:32b": LLMModelSpec("deepseek-r1:32b", "32B", 131072),
    "qwen2.5:14b-instruct-q4_K_M": LLMModelSpec("qwen2.5:14b-instruct-q4_K_M", "14B", 32768),
    "qwen2.5:7b-instruct": LLMModelSpec("qwen2.5:7b-instruct", "7B", 32768),
    "llama3.1:8b": LLMModelSpec("llama3.1:8b", "8B", 131072),
    "gemma3:27b": LLMModelSpec("gemma3:27b", "27B", 131072),
    "mistral": LLMModelSpec("mistral", "7B", 32768),
    "neural-chat:7b": LLMModelSpec("neural-chat:7b", "7B", 32768),
}

DEFAULT_LLM_MODEL_NAME = "qwen3.6:27b"


def get_llm_model_spec(model_name: str) -> LLMModelSpec:
    """Return the specification for a supported Ollama LLM.

    Args:
        model_name: Exact Ollama model name selected for an application role.

    Raises:
        ValueError: When the configured model is not in the supported registry.
    """
    try:
        return SUPPORTED_LLM_MODELS[model_name]
    except KeyError as exc:
        supported_models = ", ".join(sorted(SUPPORTED_LLM_MODELS))
        raise ValueError(
            f"Unsupported LLM model '{model_name}'. Supported models: {supported_models}."
        ) from exc


def get_configured_llm_model(
    config: BaseConfig,
    environment_variable: str,
    default_model_name: str = DEFAULT_LLM_MODEL_NAME,
) -> str:
    """Load and validate an LLM model selected for a named application role.

    Args:
        config: Configuration source providing environment lookup.
        environment_variable: Environment variable selecting the model.
        default_model_name: Validated model used when no override is present.

    Returns:
        Canonical model name suitable for ``OllamaLLM``.
    """
    configured_model_name = config.get_str(environment_variable, default_model_name)
    return get_llm_model_spec(configured_model_name).name


def get_configured_context_window(
    config: BaseConfig,
    model_name: str,
    environment_variable: str,
    default_tokens: int = 8192,
) -> int:
    """Return a bounded Ollama context-window request for a selected model.

    Args:
        config: Configuration source providing environment lookup.
        model_name: Validated Ollama model name for the application role.
        environment_variable: Environment variable selecting requested tokens.
        default_tokens: Conservative context-window request when unset.

    Returns:
        Context tokens clamped to the model's verified maximum.
    """
    requested_tokens = config.get_int(environment_variable, default_tokens)
    return min(max(1024, requested_tokens), get_llm_model_spec(model_name).max_context_tokens)
