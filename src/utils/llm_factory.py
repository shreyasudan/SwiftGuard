"""LLM Factory for provider-agnostic model initialization."""

import os
from typing import Optional, Any
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"


class ModelType(Enum):
    """Model performance tiers."""
    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


# Model configurations for each provider
MODEL_CONFIGS = {
    LLMProvider.OPENAI: {
        ModelType.FAST: "gpt-4o-mini",
        ModelType.BALANCED: "gpt-4o",
        ModelType.POWERFUL: "gpt-4o"
    },
    LLMProvider.TOGETHER: {
        ModelType.FAST: "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        ModelType.BALANCED: "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        ModelType.POWERFUL: "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    },
    LLMProvider.ANTHROPIC: {
        ModelType.FAST: "claude-3-5-haiku-20241022",
        ModelType.BALANCED: "claude-3-5-sonnet-20241022",
        ModelType.POWERFUL: "claude-3-5-sonnet-20241022"
    }
}


def get_llm(
    provider: Optional[str] = None,
    model_type: str = "fast",
    temperature: float = 0,
    max_tokens: int = 500,
    **kwargs
) -> Any:
    """Factory function to get appropriate LLM based on provider.

    Args:
        provider: LLM provider name. Defaults to environment variable LLM_PROVIDER.
        model_type: Type of model - "fast", "balanced", or "powerful"
        temperature: Model temperature for randomness
        max_tokens: Maximum tokens in response
        **kwargs: Additional provider-specific arguments

    Returns:
        LLM instance from the appropriate provider

    Raises:
        ValueError: If provider is not supported
        ImportError: If provider package is not installed
    """
    # Get provider from environment if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Convert string to enum
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        supported = ", ".join([p.value for p in LLMProvider])
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

    # Convert model type to enum
    try:
        model_type_enum = ModelType(model_type)
    except ValueError:
        model_type_enum = ModelType.FAST

    # Get model name for this provider and type
    model_name = MODEL_CONFIGS[provider_enum][model_type_enum]

    # Initialize based on provider
    if provider_enum == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires langchain-openai. "
                "Install with: pip install langchain-openai"
            )

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )

    elif provider_enum == LLMProvider.TOGETHER:
        try:
            from langchain_together import Together
        except ImportError:
            raise ImportError(
                "Together provider requires langchain-together. "
                "Install with: pip install langchain-together"
            )

        api_key = kwargs.get("api_key") or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        return Together(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )

    elif provider_enum == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires langchain-anthropic. "
                "Install with: pip install langchain-anthropic"
            )

        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        )


def get_embedding_model(provider: Optional[str] = None, **kwargs) -> Any:
    """Get embedding model based on provider.

    Args:
        provider: Provider name. Defaults to EMBEDDING_PROVIDER env var or "openai"
        **kwargs: Additional provider-specific arguments

    Returns:
        Embedding model instance
    """
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require langchain-openai. "
                "Install with: pip install langchain-openai"
            )

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return OpenAIEmbeddings(
            model=kwargs.get("model", "text-embedding-3-small"),
            api_key=api_key
        )

    elif provider == "together":
        try:
            from langchain_together import TogetherEmbeddings
        except ImportError:
            raise ImportError(
                "Together embeddings require langchain-together. "
                "Install with: pip install langchain-together"
            )

        api_key = kwargs.get("api_key") or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        return TogetherEmbeddings(
            model=kwargs.get("model", "togethercomputer/m2-bert-80M-8k-retrieval"),
            api_key=api_key
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_provider_info() -> dict:
    """Get information about the current provider configuration.

    Returns:
        Dictionary with provider information
    """
    provider = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")

    return {
        "llm_provider": provider,
        "embedding_provider": embedding_provider,
        "models": MODEL_CONFIGS.get(LLMProvider(provider), {}),
        "api_key_set": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "together": bool(os.getenv("TOGETHER_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY"))
        }
    }