"""Memory module: cache simulation and memory allocation."""
from scratchv.memory.cache import L1Cache, CacheConfig, CacheStats
from scratchv.memory.allocator import (
    MemoryAllocator,
    AllocationPolicy,
    MemoryRegion,
    AllocStats,
)

__all__ = [
    "L1Cache", "CacheConfig", "CacheStats",
    "MemoryAllocator", "AllocationPolicy", "MemoryRegion", "AllocStats",
]
