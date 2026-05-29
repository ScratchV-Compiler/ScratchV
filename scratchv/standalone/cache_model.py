#!/usr/bin/env python3
"""Set-associative cache model with LRU replacement for RV32IM emulators.

Models L1 I-cache and D-cache with configurable parameters.
Designed to be attached to the benchmark.py emulator for realistic
cache performance simulation.

Usage:
    from cache_model import CacheSim, CACHE_CONFIGS

    icache = CacheSim(name="I$", sets=64, ways=2, block_size=32)
    dcache = CacheSim(name="D$", sets=128, ways=4, block_size=32)

    # For each instruction fetch:
    icache.access(addr=pc, is_read=True)
    # For each data access:
    dcache.access(addr=mem_addr, is_read=True)  # or is_read=False for stores

    # Print results:
    icache.print_stats()
    dcache.print_stats()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class CacheStats:
    """Statistics for a single cache."""
    name: str = ""
    hits: int = 0
    misses: int = 0
    # Breakdown
    compulsory_misses: int = 0
    conflict_misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.total if self.total > 0 else 0.0

    @property
    def mpki(self) -> float:
        """Misses per 1000 accesses."""
        return self.misses / (self.total / 1000) if self.total > 0 else 0.0


class CacheLine:
    """A single cache line with tag, valid bit, and LRU tracking."""
    __slots__ = ("tag", "valid", "lru")

    def __init__(self):
        self.tag: int = 0
        self.valid: bool = False
        self.lru: int = 0  # timestamp of last access


class CacheSim:
    """Set-associative cache with LRU replacement.

    Models a single level of cache with configurable sets, ways, and block size.
    Uses write-allocate + write-back policy for stores.

    Address decomposition (32-bit):
        | tag (remaining bits) | index (log2 sets) | offset (log2 block_size) |

    Args:
        name: Cache name ("I$" or "D$").
        sets: Number of sets (power of 2).
        ways: Associativity (number of ways per set).
        block_size: Block size in bytes (power of 2).
        write_allocate: If True, allocate on store miss (default True).
    """

    def __init__(
        self,
        name: str = "cache",
        sets: int = 64,
        ways: int = 2,
        block_size: int = 32,
        write_allocate: bool = True,
    ):
        self.name = name
        self.sets = sets
        self.ways = ways
        self.block_size = block_size
        self.write_allocate = write_allocate

        # Derived parameters
        self._offset_bits = block_size.bit_length() - 1
        self._index_bits = sets.bit_length() - 1
        self._index_mask = sets - 1

        # Storage: list of sets, each with `ways` CacheLine objects
        self._lines: list[list[CacheLine]] = [
            [CacheLine() for _ in range(ways)]
            for _ in range(sets)
        ]

        # Global LRU counter (increments on each access)
        self._lru_counter: int = 0

        # Statistics
        self.stats = CacheStats(name=name)

        # Track which addresses we've seen (for compulsory miss detection)
        # Compulsory: first access to a block
        self._seen_blocks: set[int] = set()

    @property
    def total_size_bytes(self) -> int:
        """Total cache size in bytes."""
        return self.sets * self.ways * self.block_size

    def _decompose(self, addr: int) -> tuple[int, int, int]:
        """Decompose address into (tag, index, offset)."""
        offset = addr & (self.block_size - 1)
        index = (addr >> self._offset_bits) & self._index_mask
        tag = addr >> (self._offset_bits + self._index_bits)
        return tag, index, offset

    def access(self, addr: int, is_read: bool = True) -> bool:
        """Access the cache at the given address.

        Returns True on hit, False on miss.
        """
        tag, index, offset = self._decompose(addr)
        lines = self._lines[index]
        self._lru_counter += 1

        # Check for hit
        for way_idx in range(self.ways):
            line = lines[way_idx]
            if line.valid and line.tag == tag:
                # Hit!
                line.lru = self._lru_counter
                self.stats.hits += 1
                return True

        # Miss — find victim line
        self.stats.misses += 1

        # Classify miss type
        block_addr = addr >> self._offset_bits  # block-aligned address
        if block_addr not in self._seen_blocks:
            self.stats.compulsory_misses += 1
            self._seen_blocks.add(block_addr)
        else:
            self.stats.conflict_misses += 1

        # Find LRU victim
        victim_way = 0
        min_lru = lines[0].lru
        for way_idx in range(1, self.ways):
            if lines[way_idx].lru < min_lru:
                min_lru = lines[way_idx].lru
                victim_way = way_idx

        # Evict + allocate
        lines[victim_way].tag = tag
        lines[victim_way].valid = True
        lines[victim_way].lru = self._lru_counter

        return False

    def ifetch(self, pc: int) -> bool:
        """Alias for instruction fetch."""
        return self.access(pc, is_read=True)

    def load(self, addr: int) -> bool:
        """Alias for data load."""
        return self.access(addr, is_read=True)

    def store(self, addr: int) -> bool:
        """Data store access."""
        tag, index, offset = self._decompose(addr)
        lines = self._lines[index]

        # Check for hit (write hit)
        for way_idx in range(self.ways):
            line = lines[way_idx]
            if line.valid and line.tag == tag:
                line.lru = self._lru_counter
                self.stats.hits += 1
                return True

        # Store miss
        self.stats.misses += 1
        block_addr = addr >> self._offset_bits
        if block_addr not in self._seen_blocks:
            self.stats.compulsory_misses += 1
            self._seen_blocks.add(block_addr)
        else:
            self.stats.conflict_misses += 1

        if self.write_allocate:
            # Allocate on write miss
            victim_way = 0
            min_lru = lines[0].lru
            for way_idx in range(1, self.ways):
                if lines[way_idx].lru < min_lru:
                    min_lru = lines[way_idx].lru
                    victim_way = way_idx
            lines[victim_way].tag = tag
            lines[victim_way].valid = True
            lines[victim_way].lru = self._lru_counter

        return False

    def print_stats(self, prefix: str = "  ") -> str:
        """Format statistics as a human-readable string."""
        s = self.stats
        lines = []
        lines.append(f"{prefix}{self.name}: {self.sets} sets × {self.ways} ways × "
                     f"{self.block_size} B = {self.total_size_bytes:,} B "
                     f"({self.total_size_bytes/1024:.1f} KB)")
        lines.append(f"{prefix}  Accesses: {s.total:>12,}")
        lines.append(f"{prefix}  Hits:     {s.hits:>12,}  ({s.hit_rate*100:6.2f}%)")
        lines.append(f"{prefix}  Misses:   {s.misses:>12,}  ({s.miss_rate*100:6.2f}%)")
        lines.append(f"{prefix}  MPKI:     {s.mpki:>12.2f}")
        if s.misses > 0:
            compulsory_pct = s.compulsory_misses / s.misses * 100
            conflict_pct = s.conflict_misses / s.misses * 100
            lines.append(f"{prefix}  Compulsory misses: {s.compulsory_misses:>8,} "
                         f"({compulsory_pct:.1f}% of misses)")
            lines.append(f"{prefix}  Conflict misses:   {s.conflict_misses:>8,} "
                         f"({conflict_pct:.1f}% of misses)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export stats as a dictionary."""
        s = self.stats
        return {
            "name": s.name,
            "config": f"{self.sets}x{self.ways}x{self.block_size}",
            "size_bytes": self.total_size_bytes,
            "size_kb": self.total_size_bytes / 1024,
            "accesses": s.total,
            "hits": s.hits,
            "misses": s.misses,
            "hit_rate_pct": round(s.hit_rate * 100, 3),
            "miss_rate_pct": round(s.miss_rate * 100, 3),
            "mpki": round(s.mpki, 2),
            "compulsory_misses": s.compulsory_misses,
            "conflict_misses": s.conflict_misses,
        }


# ── Pre-configured cache configurations ──────────────────────────────────────

CACHE_CONFIGS = {
    # Tiny embedded (e.g., SiFive E20/E21)
    "tiny": {"sets": 16, "ways": 2, "block_size": 16},
    # Small embedded (common RV32IM MCU)
    "small": {"sets": 64, "ways": 2, "block_size": 32},
    # Medium (typical RISC-V application processor)
    "medium": {"sets": 128, "ways": 4, "block_size": 32},
    # Large (RISC-V Linux-capable)
    "large": {"sets": 256, "ways": 8, "block_size": 64},
    # For CNN inference — larger D$ to hold filter windows
    "cnn_small": {"sets": 64, "ways": 2, "block_size": 32},
    "cnn_medium": {"sets": 128, "ways": 4, "block_size": 32},
    "cnn_large": {"sets": 256, "ways": 4, "block_size": 64},
}

# Default L1 cache configs for different scenarios
DEFAULT_ICACHE_CONFIGS = {
    "embedded": {"sets": 64, "ways": 2, "block_size": 32},     # 4 KB
    "microcontroller": {"sets": 16, "ways": 2, "block_size": 16},  # 0.5 KB
    "application": {"sets": 128, "ways": 4, "block_size": 64},     # 32 KB
}

DEFAULT_DCACHE_CONFIGS = {
    "embedded": {"sets": 128, "ways": 4, "block_size": 32},    # 16 KB
    "microcontroller": {"sets": 32, "ways": 2, "block_size": 16},  # 1 KB
    "application": {"sets": 256, "ways": 8, "block_size": 64},     # 128 KB
}


def create_cache_pair(
    level: str = "embedded",
    suffix: str = "",
) -> tuple[CacheSim, CacheSim]:
    """Create a matching I$/D$ pair from pre-configured profiles."""
    ic = DEFAULT_ICACHE_CONFIGS.get(level, DEFAULT_ICACHE_CONFIGS["embedded"])
    dc = DEFAULT_DCACHE_CONFIGS.get(level, DEFAULT_DCACHE_CONFIGS["embedded"])
    ic_name = f"I${suffix}" if suffix else "I$"
    dc_name = f"D${suffix}" if suffix else "D$"
    return (
        CacheSim(name=ic_name, **ic),
        CacheSim(name=dc_name, **dc),
    )
