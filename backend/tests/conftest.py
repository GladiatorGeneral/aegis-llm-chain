"""Test configuration and fixtures."""

import pytest
import asyncio
from typing import Generator

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_security_layer():
    """Mock security layer for testing."""
    from core.security import SecurityLayer, SecurityConfig
    return SecurityLayer(SecurityConfig())

@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    from models.registry import ModelRegistry
    return ModelRegistry()

@pytest.fixture
def mock_workflow_orchestrator():
    """Mock workflow orchestrator for testing."""
    from workflows.orchestrator import WorkflowOrchestrator
    return WorkflowOrchestrator()
