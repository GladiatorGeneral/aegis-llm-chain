"""
Distributed inference module based on NVRAR concepts
Hierarchical all-reduce communication for multi-node scaling
"""

from .base import (
    ParallelismStrategy,
    CommunicationBackend,
    DistributedConfig,
    DistributedInferenceEngine
)
from .nvrar import NVRAREngine, distributed_engine

__all__ = [
    'ParallelismStrategy',
    'CommunicationBackend',
    'DistributedConfig',
    'DistributedInferenceEngine',
    'NVRAREngine',
    'distributed_engine'
]
