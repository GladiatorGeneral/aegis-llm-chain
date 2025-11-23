"""Validation utilities."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError
import re

class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class Validator:
    """Generic validator for various data types."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_model_id(model_id: str) -> ValidationResult:
        """Validate model ID format."""
        errors = []
        
        if not model_id:
            errors.append("Model ID cannot be empty")
        elif len(model_id) > 100:
            errors.append("Model ID too long (max 100 characters)")
        elif not re.match(r'^[a-zA-Z0-9._-]+$', model_id):
            errors.append("Model ID contains invalid characters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
    
    @staticmethod
    def validate_workflow_definition(workflow: Dict[str, Any]) -> ValidationResult:
        """Validate workflow definition."""
        errors = []
        warnings = []
        
        # Check required fields
        if "name" not in workflow:
            errors.append("Workflow name is required")
        
        if "steps" not in workflow:
            errors.append("Workflow steps are required")
        elif not isinstance(workflow["steps"], list):
            errors.append("Workflow steps must be a list")
        elif len(workflow["steps"]) == 0:
            errors.append("Workflow must have at least one step")
        elif len(workflow["steps"]) > 50:
            warnings.append("Workflow has many steps, consider breaking into smaller workflows")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_parameters(
        parameters: Dict[str, Any],
        schema: Dict[str, type]
    ) -> ValidationResult:
        """Validate parameters against a schema."""
        errors = []
        
        for param_name, param_type in schema.items():
            if param_name not in parameters:
                errors.append(f"Missing required parameter: {param_name}")
            elif not isinstance(parameters[param_name], param_type):
                errors.append(
                    f"Parameter {param_name} must be of type {param_type.__name__}"
                )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

# Global validator instance
validator = Validator()
