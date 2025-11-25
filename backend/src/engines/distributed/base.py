from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ParallelismStrategy(str, Enum):
    TENSOR_PARALLELISM = "tensor_parallelism"
    PIPELINE_PARALLELISM = "pipeline_parallelism" 
    EXPERT_PARALLELISM = "expert_parallelism"
    HYBRID_PARALLELISM = "hybrid_parallelism"

class CommunicationBackend(str, Enum):
    NCCL = "nccl"
    NVSHMEM = "nvshmem"
    GLOO = "gloo"
    MPI = "mpi"

class DistributedConfig(BaseModel):
    """Configuration for distributed inference"""
    model_config = ConfigDict(use_enum_values=True, extra="forbid", protected_namespaces=())
    
    parallelism_strategy: ParallelismStrategy = ParallelismStrategy.TENSOR_PARALLELISM
    communication_backend: CommunicationBackend = CommunicationBackend.NCCL
    num_nodes: int = 1
    gpus_per_node: int = 8
    model_sharding: Dict[str, Any] = {}
    all_reduce_algorithm: str = "auto"  # "nccl", "nvrar", "ring"       
    hierarchical_all_reduce: bool = True
    intra_node_backend: CommunicationBackend = CommunicationBackend.NCCL
    inter_node_backend: CommunicationBackend = CommunicationBackend.NVSHMEM
    
class DistributedInferenceEngine(ABC):
    """Abstract base for distributed inference engines"""
    
    @abstractmethod
    async def initialize_distributed(self, config: DistributedConfig):
        """Initialize distributed communication backend"""
        pass
    
    @abstractmethod
    async def all_reduce(self, tensor: Any, operation: str = "sum") -> Any:
        """Perform all-reduce operation with optimized algorithm"""
        pass
    
    @abstractmethod
    async def model_parallel_forward(self, inputs: Any, model_shards: List[Any]) -> Any:
        """Execute forward pass across model shards"""
        pass
    
    @abstractmethod
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication performance statistics"""
        pass
