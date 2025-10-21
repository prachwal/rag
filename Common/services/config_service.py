"""
Configuration service for reading application settings from .env files.

This module provides a centralized way to manage application configuration
with proper validation, type safety, and error handling.
"""

import os
import secrets
from typing import Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pydantic.types import PositiveInt


class AppSettings(BaseSettings):
    """Application settings with validation and type safety."""

    # Application settings
    app_name: str = Field(default="RAG Application", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: PositiveInt = Field(default=8000, alias="PORT")

    # Database settings
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    neon_db_url: Optional[str] = Field(default=None, alias="NEON_DB_URL")

    # API settings
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    api_timeout: PositiveInt = Field(default=30, alias="API_TIMEOUT")

    # YouTube API settings
    youtube_api_key: Optional[str] = Field(default=None, alias="YOUTUBE_API_KEY")
    youtube_api_timeout: PositiveInt = Field(default=30, alias="YOUTUBE_API_TIMEOUT")

    # HuggingFace API settings
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")
    huggingface_api_timeout: PositiveInt = Field(default=30, alias="HUGGINGFACE_API_TIMEOUT")

    # Embedding model settings
    embedding_model: str = Field(default="Voicelab/sbert-large-cased-pl", alias="EMBEDDING_MODEL")
    
    # Model memory and performance settings
    max_model_memory_gb: int = Field(default=4, alias="MAX_MODEL_MEMORY_GB")
    enable_model_quantization: bool = Field(default=False, alias="ENABLE_MODEL_QUANTIZATION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    
    # Logging settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")

    # Security settings
    secret_key: str = Field(alias="SECRET_KEY", min_length=32, default_factory=lambda: secrets.token_hex(32))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if v and not (v.startswith("postgresql://") or v.startswith("sqlite://")):
            raise ValueError("Database URL must start with postgresql:// or sqlite://")
        return v

    @field_validator("neon_db_url")
    @classmethod
    def validate_neon_db_url(cls, v):
        """Validate Neon database URL format."""
        if v and not v.startswith("postgresql://"):
            raise ValueError("Neon database URL must start with postgresql://")
        return v


class ConfigService:
    """Service for managing application configuration."""

    _instance: Optional['ConfigService'] = None
    _settings: Optional[AppSettings] = None

    def __new__(cls) -> 'ConfigService':
        """Singleton pattern to ensure single configuration instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the configuration service."""
        if self._settings is None:
            self._load_configuration()

    def _load_configuration(self) -> None:
        """Load configuration from .env file and environment variables."""
        # Load .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
        else:
            print("Warning: .env file not found. Using environment variables only.")

        try:
            # Create settings with current environment
            self._settings = AppSettings()
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    @property
    def settings(self) -> AppSettings:
        """Get application settings."""
        if self._settings is None:
            raise RuntimeError("Configuration not loaded")
        return self._settings

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting by key."""
        return getattr(self.settings, key, default)

    def is_debug_mode(self) -> bool:
        """Check if application is running in debug mode."""
        return self.settings.debug

    def get_database_url(self) -> Optional[str]:
        """Get database URL."""
        return self.settings.database_url

    def get_neon_db_url(self) -> Optional[str]:
        """Get Neon database URL."""
        return self.settings.neon_db_url

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            "api_key": self.settings.api_key,
            "timeout": self.settings.api_timeout,
        }

    def get_youtube_config(self) -> Dict[str, Any]:
        """Get YouTube API configuration."""
        return {
            "api_key": self.settings.youtube_api_key,
            "timeout": self.settings.youtube_api_timeout,
        }
    def get_huggingface_config(self) -> Dict[str, Any]:
        """Get HuggingFace API configuration."""
        return {
            "token": self.settings.huggingface_token,
            "timeout": self.settings.huggingface_api_timeout,
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration."""
        return {
            "model": self.settings.embedding_model,
        }
    def reload_config(self) -> None:
        """Reload configuration from .env file and environment variables."""
        self._load_configuration()

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        global _config_service_instance
        _config_service_instance = None
        # Also reset the lazy wrapper
        global config_service
        config_service = _ConfigServiceLazy()


# Global configuration instance - lazy initialization
class _ConfigServiceLazy:
    """Lazy initialization wrapper for ConfigService."""

    def __init__(self):
        self._instance = None
        self._instance_error = None

    def __call__(self):
        if self._instance_error:
            raise self._instance_error
        if self._instance is None:
            try:
                self._instance = ConfigService()
            except Exception as e:
                self._instance_error = e
                raise
        return self._instance

    def __getattr__(self, name):
        return getattr(self(), name)

# Global configuration instance
config_service = _ConfigServiceLazy()


def get_settings() -> AppSettings:
    """Get application settings - convenience function."""
    return config_service.settings


def get_config() -> ConfigService:
    """Get configuration service instance."""
    return config_service()