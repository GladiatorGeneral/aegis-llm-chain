"""FastAPI application entry point for AGI Platform."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.v1 import cognitive, models, workflows, auth

app = FastAPI(
    title="AGI Platform API",
    description="Universal AI platform with security-first architecture",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(cognitive.router, prefix="/api/v1/cognitive", tags=["cognitive"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AGI Platform API", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
