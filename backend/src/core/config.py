"""Core configuration management."""

from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AGI Platform"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/agi_platform"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Model Settings
    HF_TOKEN: str = ""
    MODEL_CACHE_DIR: str = "./model_cache"
    
    # Security Settings
    MAX_PROMPT_LENGTH: int = 10000
    RATE_LIMIT_PER_HOUR: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
