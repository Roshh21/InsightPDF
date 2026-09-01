"""
Application configuration.

All configuration is sourced from environment variables (see .env.example).
Nothing secret is hard-coded, and nothing here is ever sent to the frontend
verbatim -- the public/settings API exposes only non-secret derived flags.

LLM configuration is a THREE-SLOT ordered chain (Primary -> Secondary ->
Tertiary), each independently pointed at any supported provider. The
defaults use only providers with a genuinely free hosted tier (Groq,
Google Gemini, OpenRouter's free-tagged models) so the application works
end-to-end without paying for LLM usage. No provider is hard-coded as
mandatory: a slot whose API key is missing is simply skipped at runtime
(see app/llm/gateway.py) rather than causing a startup failure. Optional
paid providers (Anthropic, OpenAI direct) remain available if you want to
point a slot at them instead -- see PRIMARY_LLM_PROVIDER and friends below.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]  # backend/
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

LLMProviderId = Literal["groq", "gemini", "openrouter", "anthropic", "openai", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -----------------------------------------------------------
    APP_NAME: str = "InsightPDF"
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: str = "http://localhost:3000"
    LOG_LEVEL: str = "INFO"

    # --- Database --------------------------------------------------------
    # Postgres is the intended production database. SQLite works for a
    # zero-setup local trial (see README) but Postgres is recommended.
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://insightpdf:insightpdf@localhost:5432/insightpdf"
    )

    # --- File storage ----------------------------------------------------
    UPLOAD_DIR: str = str(UPLOAD_DIR)
    MAX_UPLOAD_MB: int = 50

    # --- Vector store ------------------------------------------------------
    VECTOR_STORE: Literal["chroma", "qdrant"] = "chroma"
    CHROMA_DIR: str = str(CHROMA_DIR)
    QDRANT_URL: Optional[str] = None  # if unset, Qdrant runs embedded/local at QDRANT_PATH
    QDRANT_PATH: str = str(DATA_DIR / "qdrant")
    QDRANT_API_KEY: Optional[str] = None

    # --- Embeddings (always free/local -- never a paid API) -----------------
    EMBEDDING_PROVIDER: Literal["huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM Gateway: ordered free-provider-first failover chain ------------
    # Each slot names a provider id; ModelGateway resolves it to a concrete
    # implementation and skips the slot entirely if its API key is missing,
    # so leaving a slot's key blank never prevents the app from starting.
    PRIMARY_LLM_PROVIDER: LLMProviderId = "groq"
    PRIMARY_LLM_MODEL_FAST: str = "llama-3.1-8b-instant"
    PRIMARY_LLM_MODEL_STRONG: str = "llama-3.3-70b-versatile"

    SECONDARY_LLM_PROVIDER: LLMProviderId = "gemini"
    SECONDARY_LLM_MODEL_FAST: str = "gemini-3.1-flash-lite"
    SECONDARY_LLM_MODEL_STRONG: str = "gemini-3.7-flash"

    TERTIARY_LLM_PROVIDER: LLMProviderId = "openrouter"
    TERTIARY_LLM_MODEL_FAST: str = "meta-llama/llama-4-scout:free"
    TERTIARY_LLM_MODEL_STRONG: str = "openai/gpt-oss-120b:free"

    # Free-tier API keys. Get these from:
    #   Groq       -> https://console.groq.com/keys           (no card required)
    #   Gemini     -> https://aistudio.google.com/apikey        (no card required)
    #   OpenRouter -> https://openrouter.ai/keys                (no card required for :free models)
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    # Optional, for OpenRouter's request-attribution headers (not required).
    OPENROUTER_SITE_URL: Optional[str] = "http://localhost:3000"
    OPENROUTER_APP_NAME: str = "InsightPDF"

    # Optional PAID providers -- never required. Only used if you explicitly
    # point PRIMARY/SECONDARY/TERTIARY_LLM_PROVIDER at "anthropic" or "openai".
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None  # override for a self-hosted OpenAI-compatible endpoint

    LLM_REQUEST_TIMEOUT_S: int = 60
    # How long (seconds) to stop retrying a provider after it returns a
    # rate-limit/quota error, so the gateway doesn't waste requests hammering
    # a known-exhausted free tier before its window resets.
    LLM_PROVIDER_COOLDOWN_S: int = 90

    # --- Web research tool ---------------------------------------------
    TAVILY_API_KEY: Optional[str] = None

    # --- Retrieval -------------------------------------------------------
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_MAX_REFINE_ATTEMPTS: int = 2
    RETRIEVAL_MIN_RELEVANCE: float = 0.35

    # --- Evaluation --------------------------------------------------------
    EVALUATION_ENABLED: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
