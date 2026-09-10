from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── LLM Provider API Keys ──────────────────────────────────────────
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    gemini_api_key: Optional[str] = Field(None, description="Google Gemini API key")
    google_api_key: Optional[str] = Field(None, description="Google API key (fallback for Gemini)")
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")

    # ── Data Acquisition API Keys ──────────────────────────────────────
    firecrawl_api_key: Optional[str] = Field(None, description="Firecrawl web scraping API key")
    tavily_api_key: Optional[str] = Field(None, description="Tavily AI search API key")

    # ── Kaggle Credentials ─────────────────────────────────────────────
    kaggle_username: Optional[str] = Field(None, description="Kaggle username")
    kaggle_key: Optional[str] = Field(None, description="Kaggle API key")

    # ── Vector Store Keys ──────────────────────────────────────────────
    pinecone_api_key: Optional[str] = Field(None, description="Pinecone cloud vector DB API key")

    # ── Observability (Langfuse) ───────────────────────────────────────
    langfuse_public_key: Optional[str] = Field(None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(None, description="Langfuse secret key")
    langfuse_host: str = Field("https://us.cloud.langfuse.com", description="Langfuse host URL")

    # ── Application Configuration ──────────────────────────────────────
    openai_model: str = Field("gpt-4o-mini", description="Default OpenAI model")
    vector_store_path: str = Field("./data/vector_store", description="Local vector store path")
    max_upload_size_mb: int = Field(20, ge=1, le=100, description="Maximum size for one uploaded file")
    max_document_chars: int = Field(50_000, ge=1_000, le=500_000, description="Maximum text retained per source document")
    max_sources_per_run: int = Field(20, ge=1, le=100, description="Maximum sources processed in one research run")
    scraping_delay_min: int = Field(1, description="Minimum delay between scraping requests (seconds)")
    scraping_delay_max: int = Field(3, description="Maximum delay between scraping requests (seconds)")
    # Providers and model availability differ by account. Configure verified
    # fallbacks explicitly rather than spending retries on stale defaults.
    llm_fallback_models: str = Field("", description="Comma-separated list of verified fallback models")
    litellm_drop_params: bool = Field(True, description="Drop unsupported provider kwargs during fallbacks")
    # Comma-separated list of allowed CORS origins. Override in staging/prod via env.
    cors_origins: str = Field(
        "http://localhost:5173",
        description="Comma-separated allowed CORS origins (e.g. https://app.example.com,http://localhost:5173)"
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS_ORIGINS into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Convenience Properties ─────────────────────────────────────────

    @property
    def gemini_key(self) -> Optional[str]:
        """Resolve Gemini key with GOOGLE_API_KEY fallback."""
        return self.gemini_api_key or self.google_api_key

    @property
    def langfuse_enabled(self) -> bool:
        """Check if Langfuse observability is configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def kaggle_configured(self) -> bool:
        """Check if Kaggle API credentials are available."""
        return bool(self.kaggle_username and self.kaggle_key)

    def get_provider_key(self, provider: str, user_key: Optional[str] = None) -> Optional[str]:
        """
        Resolve API key for a given LLM provider.
        User-supplied BYO key takes priority over environment configuration.
        """
        if user_key and user_key.strip():
            return user_key.strip()

        provider_clean = provider.lower().strip()
        key_map = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_key,
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
        }
        return key_map.get(provider_clean)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached singleton accessor for application settings.
    Call this anywhere instead of os.getenv() for typed, validated configuration.
    """
    return Settings()
