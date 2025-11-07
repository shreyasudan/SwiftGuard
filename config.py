"""Configuration settings for AgenticGuard with provider support."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgenticConfig:
    """Configuration class for AgenticGuard settings."""

    # Provider Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Model Configurations by Provider
    MODELS = {
        "openai": {
            "fast": "gpt-4o-mini",
            "balanced": "gpt-4o",
            "powerful": "gpt-4o",
            "embedding": "text-embedding-3-small"
        },
        "together": {
            "fast": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "balanced": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "powerful": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "embedding": "togethercomputer/m2-bert-80M-8k-retrieval"
        },
        "anthropic": {
            "fast": "claude-3-5-haiku-20241022",
            "balanced": "claude-3-5-sonnet-20241022",
            "powerful": "claude-3-5-sonnet-20241022"
        }
    }

    # Performance Settings
    CLASSIC_MODE_TIMEOUT = 0.5  # seconds
    PRECISION_MODE_TIMEOUT = 2.0  # seconds

    # Agent Settings
    DETECTOR_MAX_TOKENS = 500
    ANALYZER_MAX_TOKENS = 1000
    RESPONDER_MAX_TOKENS = 500

    # Confidence Thresholds
    BLOCK_THRESHOLD = 0.8
    FLAG_THRESHOLD = 0.7

    # Cache Settings (for future implementation)
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate the configuration and return status."""
        issues = []
        warnings = []

        # Check LLM provider
        if cls.LLM_PROVIDER not in cls.MODELS:
            issues.append(f"Invalid LLM_PROVIDER: {cls.LLM_PROVIDER}")

        # Check API keys based on provider
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is required for OpenAI provider")
        elif cls.LLM_PROVIDER == "together" and not cls.TOGETHER_API_KEY:
            issues.append("TOGETHER_API_KEY is required for Together provider")
        elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY is required for Anthropic provider")

        # Check embedding provider
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is required for OpenAI embeddings")
        elif cls.EMBEDDING_PROVIDER == "together" and not cls.TOGETHER_API_KEY:
            issues.append("TOGETHER_API_KEY is required for Together embeddings")

        # Warnings
        if cls.ENABLE_CACHE:
            warnings.append("Cache is enabled but not yet implemented")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "provider": cls.LLM_PROVIDER,
            "embedding_provider": cls.EMBEDDING_PROVIDER
        }

    @classmethod
    def get_model(cls, model_type: str) -> str:
        """Get model name for the current provider and type."""
        provider_models = cls.MODELS.get(cls.LLM_PROVIDER, {})
        return provider_models.get(model_type, "")

    @classmethod
    def display_config(cls) -> None:
        """Display current configuration."""
        print("=== AgenticGuard Configuration ===")
        print(f"LLM Provider: {cls.LLM_PROVIDER}")
        print(f"Embedding Provider: {cls.EMBEDDING_PROVIDER}")
        print(f"Models:")
        for model_type in ["fast", "balanced", "powerful"]:
            print(f"  {model_type}: {cls.get_model(model_type)}")
        print(f"Cache Enabled: {cls.ENABLE_CACHE}")

        # Validate
        validation = cls.validate_config()
        if not validation["valid"]:
            print("\n⚠️  Configuration Issues:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
        if validation["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")


# Quick test
if __name__ == "__main__":
    AgenticConfig.display_config()