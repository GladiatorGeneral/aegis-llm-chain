"""Security utilities and validation."""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel

class SecurityViolation(Exception):
    """Exception raised when a security violation is detected."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class SecurityConfig(BaseModel):
    """Security configuration model."""
    max_prompt_length: int = 10000
    blocked_patterns: List[str] = [
        r"(\bpassword\b|\bsecret\b|\bapi[_-]?key\b)",
        r"(\bssh-rsa\s+AAAAB3NzaC1yc2|ecdsa-sha2-)",
        r"(\bBEGIN\s+(RSA|EC|DSA)\s+PRIVATE KEY\b)"
    ]
    rate_limits: Dict[str, int] = {
        "generation": 100,  # requests per hour
        "analysis": 200,
        "workflow": 50
    }

class SecurityLayer:
    """Security layer for input/output validation and filtering."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in config.blocked_patterns
        ]
    
    async def validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate input for security concerns."""
        if len(text) > self.config.max_prompt_length:
            return False, f"Input exceeds maximum length of {self.config.max_prompt_length}"
        
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, "Blocked pattern detected in input"
        
        return True, None
    
    async def filter_output(self, text: str) -> str:
        """Filter output for sensitive information."""
        # Implement PII redaction, content filtering, etc.
        filtered_text = text
        # Add filtering logic here
        return filtered_text
    
    async def sanitize_output(self, text: str) -> str:
        """Sanitize output to remove PII and sensitive data."""
        # PII patterns for redaction
        pii_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
        ]
        
        sanitized = text
        for pattern, replacement in pii_patterns:
            sanitized = re.sub(pattern, replacement, sanitized)
        
        return sanitized
    
    async def sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent injection attacks."""
        # Remove potentially dangerous characters
        sanitized = text.replace("<", "&lt;").replace(">", "&gt;")
        return sanitized

# Global security layer instance
security_layer = SecurityLayer(SecurityConfig())
