"""
Cache-aware memory allocator for edge NPU.

Implements:
- Buddy allocator for configurable memory pool sizes
- Cache-line-aligned allocation for L1 cache-friendly access patterns
- Scratchpad region for explicit DMA/tile memory
- Allocation tracking and statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AllocationPolicy(Enum):
    """Memory allocation strategy."""
    FIRST_FIT  = "first_fit"
    BEST_FIT   = "best_fit"
    BUDDY      = "buddy"


@dataclass
class MemoryRegion:
    """A contiguous memory region."""
    name: str
    base: int          # base address (byte offset from pool start)
    size: int          # size in bytes
    used: bool = False
    alignment: int = 4 # required alignment

    @property
    def end(self) -> int:
        return self.base + self.size

    def __repr__(self) -> str:
        status = "used" if self.used else "free"
        return (f"Region({self.name}: 0x{self.base:x}-0x{self.end:x}, "
                f"{self.size}B, {status}, align={self.alignment})")


@dataclass
class AllocStats:
    """Allocation statistics."""
    total_allocated: int = 0
    total_freed: int = 0
    num_allocs: int = 0
    num_frees: int = 0
    largest_free_block: int = 0
    fragmentation_pct: float = 0.0
    cache_misses_avoided: int = 0  # from aligned allocations

    def __repr__(self) -> str:
        return (f"AllocStats(allocated={self.total_allocated}, "
                f"freed={self.total_freed}, "
                f"active={self.num_allocs - self.num_frees}, "
                f"largest_free={self.largest_free_block}, "
                f"frag={self.fragmentation_pct:.1f}%)")


# ═══════════════════════════════════════════════════════════
# MemoryAllocator
# ═══════════════════════════════════════════════════════════

class MemoryAllocator:
    """Cache-aware memory allocator with buddy system and alignment support.

    The pool is divided into a scratchpad region (fast, explicit DMA) and
    a general-purpose region (cached). All allocations are cache-line-aligned
    by default (64B) to avoid L1 cache line ping-pong.
    """

    def __init__(
        self,
        pool_size: int = 4 * 1024 * 1024,          # 4 MB total
        cache_line: int = 64,                       # L1 cache line size
        scratchpad_ratio: float = 0.25,             # 25% for scratchpad
        policy: AllocationPolicy = AllocationPolicy.BUDDY,
    ):
        self.pool_size = pool_size
        self.cache_line = cache_line
        self.policy = policy
        self.stats = AllocStats()

        # Split pool: scratchpad (high-speed, uncached) + general (cached)
        scratch_size = int(pool_size * scratchpad_ratio)
        # Align scratchpad size to cache line
        scratch_size = self._align_up(scratch_size, cache_line)
        gen_size = pool_size - scratch_size

        self.scratchpad = MemoryRegion("scratchpad", 0, scratch_size)
        self._regions: list[MemoryRegion] = [
            MemoryRegion("general", scratch_size, gen_size),
        ]
        self._freed_regions: list[MemoryRegion] = []
        self._next_id = 0

        # Scratchpad cursor (next free address)
        self._scratchpad_cursor = 0

        # For buddy: power-of-two free lists
        self._buddy_free: dict[int, list[int]] = {}  # size -> list of base addrs
        self._buddy_allocated: dict[int, int] = {}    # id -> base addr

        # Populate buddy free list
        if policy == AllocationPolicy.BUDDY:
            self._init_buddy(gen_size)

    # ── Public API ─────────────────────────────────────

    def alloc(self, size: int, alignment: int = 0,
              prefer_scratchpad: bool = False) -> int:
        """Allocate `size` bytes. Returns base address (offset from pool start).

        Args:
            size: Requested size in bytes.
            alignment: Required alignment (0 = use cache_line default).
            prefer_scratchpad: If True, try scratchpad region first.

        Returns:
            Base offset, or -1 if allocation fails.
        """
        alignment = alignment or self.cache_line
        size = self._align_up(size, alignment)

        if prefer_scratchpad:
            aligned_base = self._align_up(self._scratchpad_cursor, alignment)
            if aligned_base + size <= self.scratchpad.size:
                self._scratchpad_cursor = aligned_base + size
                return aligned_base
            # fall through to general pool

        if self.policy == AllocationPolicy.BUDDY:
            addr = self._buddy_alloc(size)
        else:
            gen = self._regions[0]
            cursor = getattr(self, "_general_cursor", gen.base)
            aligned_base = self._align_up(cursor, alignment)
            if aligned_base + size <= gen.end:
                self._general_cursor = aligned_base + size
                addr = aligned_base
            else:
                addr = -1

        if addr >= 0:
            self.stats.total_allocated += size
            self.stats.num_allocs += 1
            # If aligned to cache line, we avoided a potential false-sharing miss
            if alignment >= self.cache_line:
                self.stats.cache_misses_avoided += 1

        return addr

    def free(self, addr: int) -> bool:
        """Free a previously allocated block.

        Returns True if the address was freed successfully.
        """
        # Check scratchpad
        if self._addr_in_region(addr, self.scratchpad):
            return True  # scratchpad doesn't track individual frees

        if self.policy == AllocationPolicy.BUDDY:
            return self._buddy_free_block(addr)

        # Linear scan for first-fit segments
        for i, region in enumerate(self._regions):
            if region.base == addr and region.used:
                region.used = False
                self._freed_regions.append(region)
                self.stats.total_freed += region.size
                self.stats.num_frees += 1
                # Coalesce adjacent free regions
                self._coalesce()
                return True
        return False

    def scratchpad_alloc(self, size: int, alignment: int = 64) -> int:
        """Allocate from the scratchpad (uncached, fast SRAM)."""
        return self.alloc(size, alignment, prefer_scratchpad=True)

    def get_region_info(self, addr: int) -> Optional[MemoryRegion]:
        """Get info about which region an address belongs to."""
        if self._addr_in_region(addr, self.scratchpad):
            return self.scratchpad
        for region in self._regions:
            if self._addr_in_region(addr, region):
                return region
        return None

    def is_in_scratchpad(self, addr: int) -> bool:
        return self._addr_in_region(addr, self.scratchpad)

    def reset(self) -> None:
        """Reset all allocations."""
        self._scratchpad_cursor = 0
        gen_size = self.pool_size - self.scratchpad.size
        self._regions = [MemoryRegion("general", self.scratchpad.size, gen_size)]
        self._freed_regions.clear()
        self.stats = AllocStats()
        self._next_id = 0
        if self.policy == AllocationPolicy.BUDDY:
            self._init_buddy(gen_size)

    # ── Buddy allocator ────────────────────────────────

    def _init_buddy(self, total_size: int) -> None:
        self._buddy_free.clear()
        self._buddy_allocated.clear()
        # Find the largest power of two <= total_size
        max_pow2 = 1 << (total_size.bit_length() - 1)
        base = self._regions[0].base
        self._buddy_free[max_pow2] = [base]
        # Add remaining chunk as a smaller block
        remainder = total_size - max_pow2
        if remainder > 0:
            pow2 = 1 << (remainder.bit_length() - 1)
            self._buddy_free[pow2] = [base + max_pow2]

    def _buddy_alloc(self, size: int) -> int:
        """Allocate using buddy system.

        Rounds up size to the next power of two, finds a free block
        of that size, splitting larger blocks as needed.
        """
        block_size = 1 << (max(size, self.cache_line).bit_length() - 1)
        if block_size < size:
            block_size <<= 1

        # Find an available block of suitable size
        available_sizes = sorted(s for s in self._buddy_free if self._buddy_free[s])
        if not available_sizes:
            return -1

        # Find smallest available size >= block_size
        chosen_size = None
        for s in available_sizes:
            if s >= block_size:
                chosen_size = s
                break

        if chosen_size is None:
            return -1

        # Split until we get the target size
        free_list = self._buddy_free[chosen_size]
        addr = free_list.pop(0)

        while chosen_size > block_size:
            chosen_size >>= 1
            buddy_addr = addr + chosen_size
            self._buddy_free.setdefault(chosen_size, []).append(buddy_addr)

        self._buddy_allocated[addr] = block_size
        return addr

    def _buddy_free_block(self, addr: int) -> bool:
        """Free a buddy-allocated block, coalescing with its buddy."""
        block_size = self._buddy_allocated.pop(addr, None)
        if block_size is None:
            return False

        self._buddy_free.setdefault(block_size, []).append(addr)

        # Coalesce: repeatedly merge with buddy if both are free
        while True:
            free_list = self._buddy_free[block_size]
            buddy_addr = addr ^ block_size  # XOR to find buddy
            if buddy_addr in free_list:
                free_list.remove(buddy_addr)
                addr = min(addr, buddy_addr)
                block_size <<= 1
                self._buddy_free.setdefault(block_size, []).append(addr)
                self.stats.total_freed += block_size // 2
            else:
                break

        self.stats.total_freed += block_size
        self.stats.num_frees += 1
        return True

    # ── First-fit / Best-fit helpers ───────────────────

    def _alloc_from_region(self, region: MemoryRegion,
                           size: int, alignment: int) -> int:
        """Allocate from a region using first-fit.

        Never mutates the input region's base/size; tracks the cursor
        in a caller-owned variable.
        """
        # Allocate from the general region by managing a cursor
        cursor = getattr(self, "_general_cursor", region.base)
        aligned_base = self._align_up(cursor, alignment)
        if aligned_base + size <= region.end:
            self._general_cursor = aligned_base + size
            return aligned_base
        return -1

    def _coalesce(self) -> None:
        """Coalesce adjacent free regions."""
        free_regs = sorted(
            [r for r in self._freed_regions if not r.used],
            key=lambda r: r.base,
        )
        self._freed_regions = [r for r in self._freed_regions if r.used]

        merged = []
        for r in free_regs:
            if merged and merged[-1].end == r.base:
                merged[-1] = MemoryRegion(
                    merged[-1].name, merged[-1].base,
                    merged[-1].size + r.size,
                )
            else:
                merged.append(r)
        self._freed_regions.extend(merged)

    # ── Utilities ──────────────────────────────────────

    @staticmethod
    def _align_up(addr: int, alignment: int) -> int:
        if alignment <= 0:
            alignment = 4
        mask = alignment - 1
        return (addr + mask) & ~mask

    @staticmethod
    def _addr_in_region(addr: int, region: MemoryRegion) -> bool:
        return region.base <= addr < region.base + region.size

    # ── Debug ──────────────────────────────────────────

    def dump(self) -> str:
        lines = [
            f"Memory Allocator ({self.pool_size >> 20} MB pool, "
            f"policy={self.policy.value}, "
            f"cache_line={self.cache_line}B):",
            f"  Scratchpad: {self.scratchpad} ({self.align_up(0, 0)})",
            f"  Regions ({len(self._regions)}):",
        ]
        for r in self._regions:
            lines.append(f"    {r}")
        if self._freed_regions:
            lines.append(f"  Freed regions ({len(self._freed_regions)}):")
            for r in self._freed_regions[:8]:
                lines.append(f"    {r}")
            if len(self._freed_regions) > 8:
                lines.append(f"    ... (+{len(self._freed_regions)-8})")
        if self.policy == AllocationPolicy.BUDDY:
            lines.append(f"  Buddy free lists:")
            for size, addrs in sorted(self._buddy_free.items()):
                if addrs:
                    lines.append(f"    {size}B: {len(addrs)} blocks")
        lines.append(f"  Stats: {self.stats}")
        return "\n".join(lines)

    def align_up(self, addr: int, alignment: int = 0) -> int:
        return self._align_up(addr, alignment or self.cache_line)
