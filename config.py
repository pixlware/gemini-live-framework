"""
Centralized configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = "0.1.0"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Gemini Live Framework"
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"

    # Google Cloud / Vertex AI
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = ""
    GEMINI_LOCATION: str = "us-central1"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_LIVE_MODEL: str = "gemini-live-2.5-flash-native-audio"
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    CLOUD_LOGGING_ENABLED: bool = False
    GCS_BUCKET_NAME: str = ""

    # Audio / DFN
    DFN_THREAD_LIMIT: int = 0

    # Telemetry
    TELEMETRY_MODE: str = "disabled"

    class Config:
        env_file = ".env"


# Create singleton instance
settings = Settings()
