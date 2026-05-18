"""
L1 cache simulator for edge NPU.

Models a 4 MB L1 cache with configurable line size, associativity,
and replacement policy. Tracks hits, misses, evictions and bandwidth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class CacheConfig:
    """Configuration for the L1 cache."""
    total_size: int = 4 * 1024 * 1024   # 4 MB
    line_size: int = 64                  # bytes per cache line
    associativity: int = 8               # N-way set associative
    write_back: bool = True              # True = write-back, False = write-through
    write_allocate: bool = True          # allocate on write miss
    hit_latency: int = 2                 # cycles (typical L1)
    miss_latency: int = 20               # cycles (penalty to go to L2/DRAM)

    @property
    def num_lines(self) -> int:
        return self.total_size // self.line_size

    @property
    def num_sets(self) -> int:
        return self.num_lines // self.associativity

    def __post_init__(self):
        assert self.total_size > 0 and self.total_size % self.line_size == 0
        assert self.line_size > 0 and (self.line_size & (self.line_size - 1)) == 0
        assert self.associativity > 0
        assert self.num_sets > 0


@dataclass(slots=True)
class CacheStats:
    """Cache performance counters."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    write_backs: int = 0
    total_cycles: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        total = self.hits + self.misses
        return self.total_cycles / total if total > 0 else 0.0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.write_backs = 0
        self.total_cycles = 0
        self.bytes_read = 0
        self.bytes_written = 0

    def __repr__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.2%}, evictions={self.evictions}, "
                f"write_backs={self.write_backs}, "
                f"avg_latency={self.avg_latency:.1f}cy)")


# ═══════════════════════════════════════════════════════════
# Cache line
# ═══════════════════════════════════════════════════════════

@dataclass(slots=True)
class CacheLine:
    """A single cache line."""
    tag: int = 0
    valid: bool = False
    dirty: bool = False
    last_access: int = 0  # for LRU

    def __repr__(self) -> str:
        return (f"Line(tag=0x{self.tag:x}, valid={self.valid}, "
                f"dirty={self.dirty}, lru={self.last_access})")


# ═══════════════════════════════════════════════════════════
# L1Cache
# ═══════════════════════════════════════════════════════════

class L1Cache:
    """Set-associative L1 cache simulator.

    Usage:
        cache = L1Cache()
        cache.read(0x1000, 4)   # read 4 bytes from addr 0x1000
        cache.write(0x1000, 4)  # write 4 bytes to addr 0x1000
        print(cache.stats)
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self._clock = 0

        # Build cache: list of sets, each set has N lines
        self._sets: list[list[CacheLine]] = [
            [CacheLine() for _ in range(self.config.associativity)]
            for _ in range(self.config.num_sets)
        ]

        self._mask_offset = int(math.log2(self.config.line_size))
        self._mask_index = int(math.log2(self.config.num_sets))
        self._tag_shift = self._mask_offset + self._mask_index

    # ── Public API ─────────────────────────────────────

    def read(self, addr: int, size: int = 4) -> int:
        """Read `size` bytes from `addr`. Returns total latency."""
        latency = 0
        start_line = addr // self.config.line_size
        end_line = (addr + size - 1) // self.config.line_size

        for line_addr in range(start_line, end_line + 1):
            block_addr = line_addr * self.config.line_size
            latency += self._access_line(block_addr, is_write=False)

        if size > self.config.line_size:
            latency += self.config.miss_latency  # cross-line penalty

        self.stats.total_cycles += latency
        self.stats.bytes_read += size
        return latency

    def write(self, addr: int, size: int = 4) -> int:
        """Write `size` bytes to `addr`. Returns total latency."""
        latency = 0
        start_line = addr // self.config.line_size
        end_line = (addr + size - 1) // self.config.line_size

        for line_addr in range(start_line, end_line + 1):
            block_addr = line_addr * self.config.line_size
            latency += self._access_line(block_addr, is_write=True)

        if size > self.config.line_size:
            latency += self.config.miss_latency

        self.stats.total_cycles += latency
        self.stats.bytes_written += size
        return latency

    def flush(self) -> int:
        """Flush all dirty lines. Returns total cycles."""
        cycles = 0
        for set_idx in range(self.config.num_sets):
            for line in self._sets[set_idx]:
                if line.valid and line.dirty:
                    cycles += self.config.miss_latency
                    self.stats.write_backs += 1
                    line.dirty = False
        self.stats.total_cycles += cycles
        return cycles

    def reset(self) -> None:
        """Reset cache state and stats."""
        for set_idx in range(self.config.num_sets):
            for line in self._sets[set_idx]:
                line.valid = False
                line.dirty = False
                line.tag = 0
                line.last_access = 0
        self.stats.reset()
        self._clock = 0

    # ── Internals ──────────────────────────────────────

    def _addr_to_set_tag(self, addr: int) -> tuple[int, int]:
        """Extract (set_index, tag) from an address."""
        set_idx = (addr >> self._mask_offset) & (self.config.num_sets - 1)
        tag = addr >> self._tag_shift
        return set_idx, tag

    def _access_line(self, block_addr: int, is_write: bool) -> int:
        """Access a single cache line. Returns latency."""
        self._clock += 1
        set_idx, tag = self._addr_to_set_tag(block_addr)
        line_set = self._sets[set_idx]

        # Look for a hit
        for line in line_set:
            if line.valid and line.tag == tag:
                # Cache hit
                self.stats.hits += 1
                line.last_access = self._clock
                if is_write and self.config.write_back:
                    line.dirty = True
                return self.config.hit_latency

        # Cache miss
        self.stats.misses += 1

        if not self.config.write_allocate and is_write:
            # Write-no-allocate: skip cache, go to next level
            return self.config.miss_latency

        # Find an eviction candidate (LRU)
        victim = self._find_lru(line_set)
        assert victim is not None

        # Write back if dirty
        if victim.valid and victim.dirty:
            self.stats.write_backs += 1
            self.stats.evictions += 1

        # Fill the line
        victim.tag = tag
        victim.valid = True
        victim.dirty = is_write and self.config.write_back
        victim.last_access = self._clock

        return self.config.hit_latency + self.config.miss_latency

    def _find_lru(self, line_set: list[CacheLine]) -> CacheLine:
        """Find the least-recently-used line in a set."""
        lru_line = line_set[0]
        lru_time = lru_line.last_access
        for line in line_set[1:]:
            if not line.valid:
                return line  # empty slot
            if line.last_access < lru_time:
                lru_time = line.last_access
                lru_line = line
        return lru_line

    # ── Debug ──────────────────────────────────────────

    def dump(self) -> str:
        lines = [
            f"L1 Cache ({self.config.total_size >> 20} MB, "
            f"{self.config.line_size}B lines, "
            f"{self.config.associativity}-way):",
            f"  Sets: {self.config.num_sets}, Lines: {self.config.num_lines}",
            f"  Stats: {self.stats}",
        ]
        # Print first few non-empty sets
        shown = 0
        for set_idx in range(self.config.num_sets):
            valid_lines = [l for l in self._sets[set_idx] if l.valid]
            if valid_lines and shown < 8:
                lines.append(f"  Set {set_idx}: {valid_lines}")
                shown += 1
        return "\n".join(lines)
