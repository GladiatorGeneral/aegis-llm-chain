"""Pydantic types for workflows."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class TaskType(str, Enum):
    """Available task types for workflow steps."""
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    TRANSFORMATION = "transformation"

class StepDefinition(BaseModel):
    """Workflow step definition."""
    step_id: str = Field(..., description="Unique identifier for the step")
    task_type: TaskType = Field(..., description="Type of task to perform")
    model_id: Optional[str] = Field(None, description="Model to use for this step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    dependencies: List[str] = Field(default_factory=list, description="IDs of steps this depends on")
    timeout_seconds: int = Field(300, description="Maximum execution time")
    retry_count: int = Field(0, description="Number of retries on failure")

class WorkflowDefinition(BaseModel):
    """Complete workflow definition."""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    steps: List[StepDefinition] = Field(..., description="List of workflow steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    max_execution_time: int = Field(3600, description="Maximum total execution time")

class ExecutionContext(BaseModel):
    """Context for workflow execution."""
    workflow_id: str
    user_id: str
    environment: str = "production"
    variables: Dict[str, Any] = Field(default_factory=dict)
    security_level: str = "standard"
