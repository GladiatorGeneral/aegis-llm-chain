from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application
    app_name: str = 'AGI Platform'
    debug: bool = True
    api_v1_prefix: str = '/api/v1'
    
    # Security
    secret_key: str = 'your-secret-key-change-in-production-use-strong-random-key'
    algorithm: str = 'HS256'
    access_token_expire_minutes: int = 30
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        'http://localhost:3000',
        'http://localhost:8000',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:8000',
    ]
    ALLOWED_METHODS: List[str] = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH']
    ALLOWED_HEADERS: List[str] = ['*']
    ALLOW_CREDENTIALS: bool = True
    
    # Performance & Scaling
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300
    MAX_REQUEST_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # API Keys (Optional - load from environment)
    HF_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    
    # Database (if needed)
    DATABASE_URL: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    
    class Config:
        env_file = '.env'
        case_sensitive = False
        extra = 'allow'  # Allow extra fields from .env

settings = Settings()
