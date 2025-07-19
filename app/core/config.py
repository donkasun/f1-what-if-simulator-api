"""
Configuration management for F1 What-If Simulator API.
"""

from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    
    # CORS configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    
    # External API configuration
    openf1_api_url: str = Field(
        default="https://api.openf1.org",
        description="OpenF1 API base URL"
    )
    openf1_api_timeout: int = Field(
        default=30,
        description="OpenF1 API timeout in seconds"
    )
    
    # Model configuration
    model_path: str = Field(
        default="app/models/lap_time_predictor.joblib",
        description="Path to the ML model file"
    )
    
    # Cache configuration
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Maximum cache size"
    )
    
    # Security configuration
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings() 