"""Small synchronous worker team for the external bare-metal RV64 runtime."""
import os


UNSET = object()


def validate_cores(cores):
    if type(cores) is not int or cores not in (1, 2, 4):
        raise ValueError("cores (legacy harts) must be an integer in {1, 2, 4}")
    return cores


def select_cores(cores=UNSET, *, harts=UNSET, allow_auto=False):
    """Validate either spelling, rejecting conflicting explicit requests."""
    def check(value):
        return value if allow_auto and type(value) is str and value == "auto" else validate_cores(value)
    if cores is not UNSET:
        cores = check(cores)
    if harts is not UNSET:
        harts = check(harts)
    if cores is not UNSET and harts is not UNSET and cores != harts:
        raise ValueError("cores and legacy harts disagree")
    return cores if cores is not UNSET else harts if harts is not UNSET else 1


def read_cores(record, *, default=1):
    """Read current/legacy metadata; UNSET requires an explicit count."""
    cores, harts = record.get("cores", UNSET), record.get("harts", UNSET)
    if cores is UNSET and harts is UNSET:
        if default is UNSET:
            raise ValueError("Missing cores (or legacy harts) field")
        return validate_cores(default)
    return select_cores(cores, harts=harts)


def resolve_cores(cores):
    """Resolve an explicit auto request once, using process CPU availability.

    Prefer Linux affinity over machine-wide counts. This is a startup choice,
    not a load monitor or cgroup bandwidth-quota policy.
    """
    if type(cores) is not str or cores != "auto":
        return validate_cores(cores)
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        available = 0
    if not available:
        try:
            available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        except (OSError, NotImplementedError):
            available = 1
    return 4 if available >= 4 else 2 if available >= 2 else 1


def validate_harts(harts):
    """Compatibility alias for validate_cores."""
    return validate_cores(harts)


def resolve_harts(harts):
    """Compatibility alias for resolve_cores."""
    return resolve_cores(harts)



# C and preprocessed assembly both accept the old compiler definition.
CORE_DEFINITIONS = r"""
#if defined(SCRATCHV_CORES) && defined(SCRATCHV_HARTS) && SCRATCHV_CORES != SCRATCHV_HARTS
#error "SCRATCHV_CORES and legacy SCRATCHV_HARTS disagree"
#endif
#ifndef SCRATCHV_CORES
#ifdef SCRATCHV_HARTS
#define SCRATCHV_CORES SCRATCHV_HARTS
#else
#define SCRATCHV_CORES 1
#endif
#endif
#ifndef SCRATCHV_HARTS
#define SCRATCHV_HARTS SCRATCHV_CORES
#endif
"""

STARTUP_ASSEMBLY = CORE_DEFINITIONS + r"""
.section .data,"aw"
.balign 4
sv_boot_ready:
    .word 0
.section .text.start,"ax"
.global sv_boot
sv_boot:
    csrr a0, mhartid
    li t0, SCRATCHV_CORES
    bgeu a0, t0, .Lpark
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __stack_top
    slli t0, a0, 20
    sub sp, sp, t0
    csrw mie, zero
    csrci mstatus, 8
    la t0, sv_trap
    csrw mtvec, t0
    li t0, 0x6000
    csrs mstatus, t0
    csrw fcsr, zero
    # MSIP wakes WFI with MSIE set even though global MIE stays clear.
    li t0, 8
    csrw mie, t0
    bnez a0, .Lwait_boot
    la t0, __bss_start
    la t1, __bss_end
.Lzero:
    bgeu t0, t1, .Lrelease_boot
    sd zero, 0(t0)
    addi t0, t0, 8
    j .Lzero
.Lrelease_boot:
    la t0, sv_boot_ready
    li t1, 1
    fence rw,w
    sw t1, 0(t0)
    call sv_main
    j .Lpark
.Lwait_boot:
    la t0, sv_boot_ready
.Lwait_ready:
    lw t1, 0(t0)
    beqz t1, .Lwait_ready
    fence r,rw
    call sv_worker
.Lpark:
    csrw mie, zero
.Lpark_wait:
    wfi
    j .Lpark_wait
.balign 4
sv_trap:
    csrr a0, mcause
    call sv_trap_report
    j .Lpark
"""


# Included after the external runtime's types and its one-owner UART frame().
WORKER_SOURCE = CORE_DEFINITIONS + r'''
typedef void (*sv_task_fn)(void *, usize, usize);
#if SCRATCHV_CORES > 1
static usize sv_generation, sv_done[SCRATCHV_CORES];
static unsigned sv_workers_ready, sv_failure;
static usize sv_count;
static sv_task_fn sv_function;
static void *sv_context;

/* QEMU virt CLINT/ACLINT MSWI: one 32-bit MSIP register per core.
 * Publish RAM state before the MMIO doorbell. Notifications can coalesce:
 * generation/ready/done/failure, rather than the interrupt, own the state. */
static void sv_notify(usize core) {
    __asm__ volatile("fence rw,ow" ::: "memory");
    ((volatile u32 *)0x02000000UL)[core]=1;
}
static void sv_clear_doorbell(usize core) {
    ((volatile u32 *)0x02000000UL)[core]=0;
    __asm__ volatile("fence iorw,iorw" ::: "memory");
}
static void sv_park_worker(void) {
    /* A pending doorbell must not turn permanent WFI into a busy loop. */
    __asm__ volatile("csrw mie, zero" ::: "memory");
    for (;;) __asm__ volatile("wfi" ::: "memory");
}
static void sv_check_workers(void) {
    unsigned failure = __atomic_load_n(&sv_failure, __ATOMIC_ACQUIRE);
    if (failure) frame(failure, 0);
}
static void sv_execute(usize core, usize count, sv_task_fn fn, void *context) {
    usize width=count/SCRATCHV_CORES, remainder=count%SCRATCHV_CORES;
    usize begin=core*width+(core<remainder ? core : remainder);
    usize end=begin+width+(core<remainder);
    if (begin != end) fn(context, begin, end);
}
void sv_worker(usize core) {
    usize seen=0;
    __atomic_fetch_add(&sv_workers_ready, 1, __ATOMIC_RELEASE);
    sv_notify(0);
    for (;;) {
        /* Clear before checking state: a publication after the check leaves
         * MSIP pending and wakes WFI. Clearing after the check could lose it. */
        sv_clear_doorbell(core);
        if (__atomic_load_n(&sv_failure, __ATOMIC_ACQUIRE)) sv_park_worker();
        usize generation=__atomic_load_n(&sv_generation, __ATOMIC_ACQUIRE);
        if (generation == seen) {
            __asm__ volatile("wfi" ::: "memory");
            continue;
        }
        sv_execute(core, sv_count, sv_function, sv_context);
        __atomic_store_n(&sv_done[core], generation, __ATOMIC_RELEASE);
        sv_notify(0);
        seen=generation;
    }
}
void sv_parallel_for(usize count, sv_task_fn fn, void *context) {
    for (;;) {
        sv_clear_doorbell(0);
        sv_check_workers();
        if (__atomic_load_n(&sv_workers_ready, __ATOMIC_ACQUIRE) == SCRATCHV_CORES-1)
            break;
        __asm__ volatile("wfi" ::: "memory");
    }
    usize generation=__atomic_load_n(&sv_generation, __ATOMIC_RELAXED)+1;
    sv_count=count; sv_function=fn; sv_context=context;
    __atomic_store_n(&sv_generation, generation, __ATOMIC_RELEASE);
    for (usize core=1; core<SCRATCHV_CORES; ++core) sv_notify(core);
    sv_execute(0, count, fn, context);
    for (usize core=1; core<SCRATCHV_CORES; ++core) {
        for (;;) {
            sv_clear_doorbell(0);
            sv_check_workers();
            if (__atomic_load_n(&sv_done[core], __ATOMIC_ACQUIRE) == generation)
                break;
            __asm__ volatile("wfi" ::: "memory");
        }
    }
    sv_check_workers();
}
void sv_trap_report(u64 cause) {
    usize core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    unsigned status=0x80000000U|(unsigned)cause;
    if (!core) frame(status, 0);
    unsigned empty=0;
    __atomic_compare_exchange_n(&sv_failure, &empty, status, 0,
                                __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    sv_notify(0);
    sv_park_worker();
}
#else
static void sv_check_workers(void) {}
void sv_parallel_for(usize count, sv_task_fn fn, void *context) {
    if (count) fn(context, 0, count);
}
void sv_trap_report(u64 cause) { frame(0x80000000U|(u32)cause, 0); }
#endif
'''
