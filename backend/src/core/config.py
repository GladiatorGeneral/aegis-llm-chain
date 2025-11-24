"""Core configuration management."""

import os
import secrets
from typing import List
from pydantic_settings import BaseSettings
from pydantic import validator
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with security validation."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AGI Platform"
    ENVIRONMENT: str = "development"  # development | staging | production
    
    # Security - NO DEFAULTS FOR PRODUCTION!
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database - NO DEFAULT CREDENTIALS!
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Model Settings
    HF_TOKEN: str = ""
    MODEL_CACHE_DIR: str = "./model_cache"
    
    # Security Settings
    MAX_PROMPT_LENGTH: int = 10000
    RATE_LIMIT_PER_HOUR: int = 100
    MAX_FILE_UPLOAD_SIZE_MB: int = 10
    FORCE_HTTPS: bool = False
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v, values):
        """Validate that SECRET_KEY is secure in production."""
        environment = values.get('ENVIRONMENT', 'development')
        
        # Check for weak/default secret keys
        weak_secrets = [
            'your-secret-key-here-change-in-production',
            'dev-secret-key-change-in-production',
            'change-me',
            'secret',
            'password',
            ''
        ]
        
        if environment == 'production' and (v in weak_secrets or len(v) < 32):
            raise ValueError(
                "❌ SECURITY ERROR: SECRET_KEY must be a strong random key in production! "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        
        if v in weak_secrets and environment != 'production':
            logger.warning("⚠️  Using default SECRET_KEY - ONLY ACCEPTABLE IN DEVELOPMENT!")
        
        return v
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v, values):
        """Validate database URL doesn't contain weak credentials."""
        environment = values.get('ENVIRONMENT', 'development')
        
        if environment == 'production' and ('password' in v.lower() or 'user:pass' in v.lower()):
            raise ValueError(
                "❌ SECURITY ERROR: DATABASE_URL contains weak credentials! "
                "Use strong passwords in production."
            )
        
        return v
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'

def _get_settings() -> Settings:
    """Initialize settings with security checks."""
    # Generate temporary secret for development if not set
    dev_secret = os.getenv('SECRET_KEY') or secrets.token_urlsafe(32)
    dev_db_url = os.getenv('DATABASE_URL') or "postgresql://postgres:postgres@localhost:5432/agi_platform"
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'development' and not os.getenv('SECRET_KEY'):
        logger.warning("No SECRET_KEY in environment - generating temporary key for development")
        logger.warning("Copy .env.template to .env and set proper values!")
    
    try:
        return Settings(
            SECRET_KEY=dev_secret,
            DATABASE_URL=dev_db_url,
            ENVIRONMENT=environment
        )
    except Exception as e:
        logger.error(f"Failed to initialize settings: {str(e)}")
        raise

# Initialize settings
settings = _get_settings()

# Log security status
if settings.ENVIRONMENT == 'production':
    logger.info("🔒 Running in PRODUCTION mode with secure configuration")
else:
    logger.warning("⚠️  Running in DEVELOPMENT mode - not for production use!")
