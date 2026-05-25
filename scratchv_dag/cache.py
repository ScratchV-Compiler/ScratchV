"""
L1 cache simulator for edge NPU.

Models a 4 MB L1 data cache with configurable line size, set-associativity,
write policy, and LRU replacement.  Tracks hit/miss rates, evictions, and
access latency cycles.

This is a *functional* simulator: it tracks which addresses hit or miss
but does not store actual data values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# CacheConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheConfig:
    """Configuration parameters for the L1 cache.

    Defaults::
        total_size     4 MB  (typical edge-NPU L1)
        line_size      64 B
        associativity  8-way
        write_back     True
        hit_latency    2 cycles
        miss_latency   20 cycles (penalty to go to L2 / DRAM)
    """

    total_size: int = 4 * 1024 * 1024
    """Total cache capacity in bytes."""

    line_size: int = 64
    """Cache line width in bytes (must be a power of two)."""

    associativity: int = 8
    """Set-associativity (1 = direct-mapped)."""

    write_back: bool = True
    """True = write-back (+ write-allocate); False = write-through."""

    write_allocate: bool = True
    """Allocate a cache line on write miss (typical for write-back)."""

    hit_latency: int = 2
    """Latency in cycles for a cache hit."""

    miss_latency: int = 20
    """Additional latency in cycles for a cache miss."""

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def num_lines(self) -> int:
        """Total number of cache lines."""
        return self.total_size // self.line_size

    @property
    def num_sets(self) -> int:
        """Number of sets in the cache."""
        return self.num_lines // self.associativity

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        assert self.total_size > 0, "total_size must be positive"
        assert self.total_size % self.line_size == 0, \
            "total_size must be a multiple of line_size"
        assert self.line_size > 0 and (self.line_size & (self.line_size - 1)) == 0, \
            "line_size must be a positive power of two"
        assert self.associativity > 0, "associativity must be positive"
        assert self.num_sets > 0, "total_size too small for given config"


# ═══════════════════════════════════════════════════════════════════════════════
# CacheStats
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheStats:
    """Performance counters collected by the cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    write_backs: int = 0
    total_cycles: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of accesses that hit in the cache."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Fraction of accesses that missed."""
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Average latency per access in cycles."""
        total = self.hits + self.misses
        return self.total_cycles / total if total > 0 else 0.0

    def reset(self) -> None:
        """Zero all counters."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.write_backs = 0
        self.total_cycles = 0
        self.bytes_read = 0
        self.bytes_written = 0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.2%}, evictions={self.evictions}, "
            f"write_backs={self.write_backs}, "
            f"avg_latency={self.avg_latency:.1f}cy)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CacheLine
# ═══════════════════════════════════════════════════════════════════════════════

class CacheLine:
    """A single cache line with tag, validity, dirtiness, and LRU timestamp."""

    __slots__ = ("tag", "valid", "dirty", "last_access")

    def __init__(self) -> None:
        self.tag: int = 0
        self.valid: bool = False
        self.dirty: bool = False
        self.last_access: int = 0

    def __repr__(self) -> str:
        return (
            f"Line(tag=0x{self.tag:x}, valid={self.valid}, "
            f"dirty={self.dirty}, lru={self.last_access})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# L1Cache
# ═══════════════════════════════════════════════════════════════════════════════

class L1Cache:
    """Set-associative L1 data cache simulator.

    Typical usage::

        cache = L1Cache()
        latency = cache.read(0x1000, 4)   # read  4 bytes from address
        latency = cache.write(0x1000, 4)  # write 4 bytes to address
        print(cache.stats)                # inspect counters
    """

    __slots__ = (
        "config", "stats",
        "_sets", "_clock",
        "_mask_offset", "_mask_index", "_tag_shift",
    )

    def __init__(self, config: CacheConfig = None) -> None:
        self.config = config if config is not None else CacheConfig()
        self.stats = CacheStats()
        self._clock = 0

        # Build the cache as a 2-D list:  sets × ways
        self._sets: List[List[CacheLine]] = [
            [CacheLine() for _ in range(self.config.associativity)]
            for _ in range(self.config.num_sets)
        ]

        # Precompute address-decomposition masks
        self._mask_offset = int(math.log2(self.config.line_size))
        self._mask_index  = int(math.log2(self.config.num_sets))
        self._tag_shift   = self._mask_offset + self._mask_index

    # ── Public API ─────────────────────────────────────────────────────────

    def read(self, addr: int, size: int = 4) -> int:
        """Read *size* bytes starting at *addr*.

        Returns the total latency in cycles.
        """
        latency = 0
        first = addr // self.config.line_size
        last  = (addr + size - 1) // self.config.line_size

        for line_addr in range(first, last + 1):
            block_addr = line_addr * self.config.line_size
            latency += self._access_line(block_addr, is_write=False)

        if size > self.config.line_size:
            latency += self.config.miss_latency  # cross-line penalty

        self.stats.total_cycles += latency
        self.stats.bytes_read += size
        return latency

    def write(self, addr: int, size: int = 4) -> int:
        """Write *size* bytes starting at *addr*.

        Returns the total latency in cycles.
        """
        latency = 0
        first = addr // self.config.line_size
        last  = (addr + size - 1) // self.config.line_size

        for line_addr in range(first, last + 1):
            block_addr = line_addr * self.config.line_size
            latency += self._access_line(block_addr, is_write=True)

        if size > self.config.line_size:
            latency += self.config.miss_latency

        self.stats.total_cycles += latency
        self.stats.bytes_written += size
        return latency

    def flush(self) -> int:
        """Write back all dirty lines and invalidate.  Returns total cycles."""
        cycles = 0
        for line_set in self._sets:
            for line in line_set:
                if line.valid and line.dirty:
                    cycles += self.config.miss_latency
                    self.stats.write_backs += 1
                    line.dirty = False
        self.stats.total_cycles += cycles
        return cycles

    def reset(self) -> None:
        """Clear the entire cache and zero all statistics."""
        for line_set in self._sets:
            for line in line_set:
                line.valid = False
                line.dirty = False
                line.tag = 0
                line.last_access = 0
        self.stats.reset()
        self._clock = 0

    # ── Internals ──────────────────────────────────────────────────────────

    def _addr_to_set_tag(self, addr: int) -> (int, int):
        """Decompose a byte address into ``(set_index, tag)``."""
        set_idx = (addr >> self._mask_offset) & (self.config.num_sets - 1)
        tag     = addr >> self._tag_shift
        return set_idx, tag

    def _access_line(self, block_addr: int, is_write: bool) -> int:
        """Access the cache line covering *block_addr*.  Returns latency."""
        self._clock += 1
        set_idx, tag = self._addr_to_set_tag(block_addr)
        line_set = self._sets[set_idx]

        # ── Probe for a hit ────────────────────────────────────────────────
        for line in line_set:
            if line.valid and line.tag == tag:
                self.stats.hits += 1
                line.last_access = self._clock
                if is_write and self.config.write_back:
                    line.dirty = True
                return self.config.hit_latency

        # ── Miss ───────────────────────────────────────────────────────────
        self.stats.misses += 1

        if not self.config.write_allocate and is_write:
            return self.config.miss_latency  # write-no-allocate

        # Find victim (LRU within the set)
        victim = line_set[0]
        for line in line_set[1:]:
            if not line.valid:
                victim = line
                break
            if line.last_access < victim.last_access:
                victim = line

        # Evict
        if victim.valid and victim.dirty:
            self.stats.write_backs += 1
            self.stats.evictions += 1

        # Fill
        victim.tag = tag
        victim.valid = True
        victim.dirty = is_write and self.config.write_back
        victim.last_access = self._clock

        return self.config.hit_latency + self.config.miss_latency

    # ── Debug ──────────────────────────────────────────────────────────────

    def dump(self) -> str:
        """Return a human-readable dump of cache configuration and state."""
        cfg = self.config
        lines = [
            f"L1 Cache ({cfg.total_size >> 20} MB, "
            f"{cfg.line_size} B lines, {cfg.associativity}-way):",
            f"  Sets: {cfg.num_sets}, Lines: {cfg.num_lines}",
            f"  Stats: {self.stats}",
        ]
        shown = 0
        for set_idx, line_set in enumerate(self._sets):
            valid = [ln for ln in line_set if ln.valid]
            if valid and shown < 8:
                lines.append(f"  Set {set_idx}: {valid}")
                shown += 1
        return "\n".join(lines)
