"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta

from core.auth import create_access_token, verify_password
from core.config import settings

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class User(BaseModel):
    """User model."""
    username: str
    email: str
    disabled: Optional[bool] = None

@router.post("/token", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    """Login endpoint to get access token.
    
    SECURITY WARNING: This is a development-only placeholder!
    
    Production deployment MUST implement:
    1. Database-backed user authentication
    2. Password hashing with bcrypt/argon2
    3. Rate limiting to prevent brute force attacks
    4. Account lockout after failed attempts
    5. Multi-factor authentication (MFA)
    6. Audit logging for authentication attempts
    7. Secure session management
    
    DO NOT use hardcoded credentials in production!
    """
    # DEVELOPMENT ONLY: Remove this placeholder before production deployment
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Authentication not configured. Please implement proper user authentication before deploying.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # TODO: Replace with proper authentication:
    # 1. Query user from database
    # 2. Verify hashed password
    # 3. Check account status (active/disabled)
    # 4. Log authentication attempt
    # 5. Create and return secure JWT token

@router.get("/me", response_model=User)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """Get current user information."""
    # TODO: Implement actual user retrieval
    return User(username="demo", email="demo@example.com", disabled=False)
