"""
Cache-aware memory allocator for edge NPU.

Implements three allocation strategies:

* **Buddy** (default) — power-of-two block splitting and coalescing.
  Fast and low-fragmentation for typical NPU tensor sizes.
* **First-fit** — simple bump-pointer allocation with freed-region reuse.

All allocations are aligned to the L1 cache line size (64 B) by default
to avoid false sharing.  A **scratchpad** region (first 25 % of the pool)
models on-chip SRAM for explicit DMA / tile transfers.

The allocator is *address-based* — it manages offsets into a fixed-size
pool and does not interact with actual OS memory mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# AllocationPolicy
# ═══════════════════════════════════════════════════════════════════════════════

class AllocationPolicy(Enum):
    """Strategy used by the memory allocator."""
    FIRST_FIT = "first_fit"
    """Simple bump-pointer allocation through the general region."""
    BUDDY     = "buddy"
    """Buddy-system: power-of-two blocks, split, and coalesce."""


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryRegion
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryRegion:
    """A contiguous range of memory within the pool.

    Attributes:
        name:      Human-readable label.
        base:      Base offset (bytes from pool start).
        size:      Size in bytes.
        used:      Whether this region is currently allocated.
        alignment: Required alignment constraint.
    """

    name: str
    base: int
    size: int
    used: bool = False
    alignment: int = 4

    @property
    def end(self) -> int:
        """Exclusive end offset."""
        return self.base + self.size

    def __repr__(self) -> str:
        status = "used" if self.used else "free"
        return (
            f"Region({self.name}: 0x{self.base:x}-0x{self.end:x}, "
            f"{self.size} B, {status}, align={self.alignment})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AllocStats
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AllocStats:
    """Allocation statistics tracked by the allocator."""
    total_allocated: int = 0
    total_freed: int = 0
    num_allocs: int = 0
    num_frees: int = 0
    largest_free_block: int = 0
    fragmentation_pct: float = 0.0
    cache_misses_avoided: int = 0

    def __repr__(self) -> str:
        active = self.num_allocs - self.num_frees
        return (
            f"AllocStats(allocated={self.total_allocated}, "
            f"freed={self.total_freed}, active={active}, "
            f"largest_free={self.largest_free_block}, "
            f"frag={self.fragmentation_pct:.1f}%)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryAllocator
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryAllocator:
    """Cache-aware memory allocator with buddy system and scratchpad region.

    The 4 MB pool is split::

        [ scratchpad  (25 %) ] [ general-purpose  (75 %) ]
         ↑ uncached / DMA       ↑ cached, buddy-managed

    All allocations are aligned to *cache_line* (default 64 B) to
    avoid L1 cache line bouncing between NPU tiles.

    Usage::

        alloc = MemoryAllocator(pool_size=4*1024*1024)
        a = alloc.alloc(4096)                       # aligned to 64 B
        b = alloc.alloc(256, alignment=4096)         # page-aligned
        s = alloc.scratchpad_alloc(1024)             # from scratchpad
        alloc.free(a)
    """

    __slots__ = (
        "pool_size", "cache_line", "policy", "stats",
        "scratchpad", "_regions", "_freed_regions",
        "_next_id", "_scratchpad_cursor", "_general_cursor",
        "_buddy_free", "_buddy_allocated",
    )

    def __init__(
        self,
        pool_size: int = 4 * 1024 * 1024,
        cache_line: int = 64,
        scratchpad_ratio: float = 0.25,
        policy: AllocationPolicy = AllocationPolicy.BUDDY,
    ) -> None:
        self.pool_size = pool_size
        self.cache_line = cache_line
        self.policy = policy
        self.stats = AllocStats()

        # Split the pool.
        scratch_size = int(pool_size * scratchpad_ratio)
        scratch_size = self._align_up(scratch_size, cache_line)
        gen_size = pool_size - scratch_size

        self.scratchpad = MemoryRegion("scratchpad", 0, scratch_size)
        self._regions: List[MemoryRegion] = [
            MemoryRegion("general", scratch_size, gen_size),
        ]
        self._freed_regions: List[MemoryRegion] = []
        self._next_id = 0

        # Cursors
        self._scratchpad_cursor = 0
        self._general_cursor = self._regions[0].base

        # Buddy free lists:  block_size → [base_addr, …]
        self._buddy_free: Dict[int, List[int]] = {}
        # Allocated:  base_addr → block_size
        self._buddy_allocated: Dict[int, int] = {}

        if policy == AllocationPolicy.BUDDY:
            self._init_buddy(gen_size)

    # ── Public API ─────────────────────────────────────────────────────────

    def alloc(
        self,
        size: int,
        alignment: int = 0,
        prefer_scratchpad: bool = False,
    ) -> int:
        """Allocate *size* bytes.

        Args:
            size:            Requested size in bytes.
            alignment:       Required alignment (0 → *cache_line* default).
            prefer_scratchpad: If True, try the scratchpad region first.

        Returns:
            Base offset from pool start, or **-1** on failure.
        """
        alignment = alignment or self.cache_line
        size = self._align_up(size, alignment)

        # Try scratchpad first if requested.
        if prefer_scratchpad:
            aligned = self._align_up(self._scratchpad_cursor, alignment)
            if aligned + size <= self.scratchpad.end:
                self._scratchpad_cursor = aligned + size
                self._update_stats(size, alignment)
                return aligned
            # fall through to general pool

        # General pool.
        if self.policy == AllocationPolicy.BUDDY:
            addr = self._buddy_alloc(size)
        else:
            aligned = self._align_up(self._general_cursor, alignment)
            if aligned + size <= self._regions[0].end:
                self._general_cursor = aligned + size
                addr = aligned
            else:
                addr = -1

        if addr >= 0:
            self._update_stats(size, alignment)
        return addr

    def free(self, addr: int) -> bool:
        """Release a previously allocated block.

        Returns ``True`` if the address was recognised and freed.
        """
        # Scratchpad frees are a no-op (no individual tracking).
        if self._addr_in_region(addr, self.scratchpad):
            return True

        if self.policy == AllocationPolicy.BUDDY:
            return self._buddy_free_block(addr)

        # First-fit: linear scan for a matching used region.
        for region in self._regions:
            if region.base == addr and region.used:
                region.used = False
                self._freed_regions.append(region)
                self.stats.total_freed += region.size
                self.stats.num_frees += 1
                self._coalesce()
                return True
        return False

    def scratchpad_alloc(self, size: int, alignment: int = 64) -> int:
        """Shorthand for allocating from the scratchpad (uncached SRAM)."""
        return self.alloc(size, alignment, prefer_scratchpad=True)

    def is_in_scratchpad(self, addr: int) -> bool:
        """Check whether *addr* falls within the scratchpad region."""
        return self._addr_in_region(addr, self.scratchpad)

    def get_region_info(self, addr: int) -> Optional[MemoryRegion]:
        """Return the region metadata for *addr*, or ``None``."""
        if self._addr_in_region(addr, self.scratchpad):
            return self.scratchpad
        for r in self._regions:
            if self._addr_in_region(addr, r):
                return r
        for r in self._freed_regions:
            if self._addr_in_region(addr, r):
                return r
        return None

    def reset(self) -> None:
        """Reset all state — all memory becomes free again."""
        self._scratchpad_cursor = 0
        gen = self._regions[0]
        gen_size = self.pool_size - self.scratchpad.size
        self._regions = [MemoryRegion("general", self.scratchpad.size, gen_size)]
        self._freed_regions.clear()
        self._general_cursor = self._regions[0].base
        self.stats = AllocStats()
        self._buddy_free.clear()
        self._buddy_allocated.clear()
        if self.policy == AllocationPolicy.BUDDY:
            self._init_buddy(gen_size)

    # ── Buddy system ───────────────────────────────────────────────────────

    def _init_buddy(self, total_size: int) -> None:
        """Seed the buddy free lists from a contiguous region."""
        self._buddy_free.clear()
        self._buddy_allocated.clear()
        base = self._regions[0].base

        max_pow2 = 1 << (total_size.bit_length() - 1)
        self._buddy_free[max_pow2] = [base]

        remainder = total_size - max_pow2
        if remainder > 0:
            pow2 = 1 << (remainder.bit_length() - 1)
            self._buddy_free[pow2] = [base + max_pow2]

    def _buddy_alloc(self, size: int) -> int:
        """Allocate a power-of-two block via the buddy system."""
        block_size = 1 << (max(size, self.cache_line).bit_length() - 1)
        if block_size < size:
            block_size <<= 1

        # Find the smallest available block ≥ block_size.
        candidates = sorted(s for s in self._buddy_free if self._buddy_free[s])
        for s in candidates:
            if s >= block_size:
                addr = self._buddy_free[s].pop(0)
                # Split until we reach the target size.
                while s > block_size:
                    s >>= 1
                    buddy = addr + s
                    self._buddy_free.setdefault(s, []).append(buddy)
                self._buddy_allocated[addr] = block_size
                return addr
        return -1

    def _buddy_free_block(self, addr: int) -> bool:
        """Free a buddy block and coalesce with its buddy if possible."""
        block_size = self._buddy_allocated.pop(addr, None)
        if block_size is None:
            return False

        self._buddy_free.setdefault(block_size, []).append(addr)

        # Coalesce upward.
        while True:
            fl = self._buddy_free[block_size]
            buddy = addr ^ block_size
            if buddy in fl:
                fl.remove(buddy)
                addr = min(addr, buddy)
                block_size <<= 1
                self._buddy_free.setdefault(block_size, []).append(addr)
                self.stats.total_freed += block_size // 2
            else:
                break

        self.stats.total_freed += block_size
        self.stats.num_frees += 1
        return True

    # ── Coalescing (first-fit only) ────────────────────────────────────────

    def _coalesce(self) -> None:
        """Merge adjacent free regions."""
        free = sorted(
            (r for r in self._freed_regions if not r.used),
            key=lambda r: r.base,
        )
        self._freed_regions = [r for r in self._freed_regions if r.used]

        merged: List[MemoryRegion] = []
        for r in free:
            if merged and merged[-1].end == r.base:
                prev = merged[-1]
                merged[-1] = MemoryRegion(prev.name, prev.base,
                                          prev.size + r.size)
            else:
                merged.append(r)
        self._freed_regions.extend(merged)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _update_stats(self, size: int, alignment: int) -> None:
        self.stats.total_allocated += size
        self.stats.num_allocs += 1
        if alignment >= self.cache_line:
            self.stats.cache_misses_avoided += 1

    @staticmethod
    def _align_up(addr: int, alignment: int = 4) -> int:
        """Round *addr* up to the next multiple of *alignment*."""
        if alignment <= 0:
            alignment = 4
        mask = alignment - 1
        return (addr + mask) & ~mask

    @staticmethod
    def _addr_in_region(addr: int, region: MemoryRegion) -> bool:
        """True iff *addr* is in [region.base, region.base + region.size)."""
        return region.base <= addr < region.base + region.size

    # ── Debug ──────────────────────────────────────────────────────────────

    def dump(self) -> str:
        """Return a multi-line dump of allocator state."""
        lines = [
            f"MemoryAllocator ({self.pool_size >> 20} MB pool, "
            f"policy={self.policy.value}, "
            f"cache_line={self.cache_line} B):",
            f"  Scratchpad: {self.scratchpad}",
            f"  General cursor: 0x{self._general_cursor:x}",
            f"  Regions ({len(self._regions)}):",
        ]
        for r in self._regions:
            lines.append(f"    {r}")
        freed = self._freed_regions
        if freed:
            lines.append(f"  Freed regions ({len(freed)}):")
            for r in freed[:8]:
                lines.append(f"    {r}")
            if len(freed) > 8:
                lines.append(f"    … (+{len(freed) - 8})")
        if self.policy == AllocationPolicy.BUDDY:
            lines.append("  Buddy free lists:")
            for size, addrs in sorted(self._buddy_free.items()):
                if addrs:
                    lines.append(f"    {size} B: {len(addrs)} blocks")
        lines.append(f"  Stats: {self.stats}")
        return "\n".join(lines)
