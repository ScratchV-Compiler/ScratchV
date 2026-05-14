"""Memory module — re-exports from scratchv_dag.

Provides L1 cache simulation and cache-aware memory allocation.
The implementation lives in the standalone ``scratchv_dag`` package;
this module serves as a compatibility shim.
"""
# flake8: noqa
from scratchv_dag import (               # noqa: F401
    L1Cache,
    CacheConfig,
    CacheStats,
    MemoryAllocator,
    AllocationPolicy,
    MemoryRegion,
    AllocStats,
)

__all__ = [
    "L1Cache", "CacheConfig", "CacheStats",
    "MemoryAllocator", "AllocationPolicy", "MemoryRegion", "AllocStats",
]
