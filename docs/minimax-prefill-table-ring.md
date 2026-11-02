# Prefill MoE table ring

Status: default on in `strix-minimax-rank` only (the app sets
STRIX_MM_PF_TABLE_RING=1 when unset and STRIX_MM_BASELINE is not 1);
`strix-minimax-serve` and the library keep it off because serve is not qualified.
Qualification (rank app, both ranks): prefill state, final rows and tokens
identical at 1K and 8K, in-process A/B/A and fresh-process B A B A; 1K TTFT
-0.5 %, 8K not slower. The ring is process-wide: one engine per process driven
by a single host thread.

STRIX_MM_PF_TABLE_RING=0 retains the table-lifetime stream sync; =1
stages the same table bytes through four pinned host slots. The environment is
captured once at Engine startup. Engine::set_prefill_table_ring(int) and
moe::opt::set_prefill_table_ring(int) accept -1 to restore that captured default,
0 for off and 1 for on. Other values throw.

Each slot is allocated lazily with hipHostMalloc for the table prefix of
prefill_moe_workspace_bound(max(rows, 4096)). Larger calls grow a slot after its
previous copies complete. Groups, pair/token tables and either item layout keep
their existing device offsets and copy sizes. A DisableTiming event is recorded
immediately after the copies, before the kernels. Only reuse of that slot waits
on its event. The enabled successful enqueue performs no stream synchronization.
Upload/record errors drain the stream before unwinding; a failed drain aborts.

The module ring and its setters/cleanup require one host thread per process.
Engine::~Impl calls cleanup_prefill_table_ring before HIP teardown. Cleanup waits
for outstanding table copies, then releases all pinned slots and events. It
aborts on a HIP cleanup error rather than freeing a potentially live source.
Standalone callers must also call cleanup before HIP runtime teardown.

The engine's pf_ws has no users besides enqueue_prefill_moe. All calls, including
interleaved chunks, use the default stream. Its order places each next table
upload after the preceding call's kernels, even when layouts differ. The ring
protects host source lifetime; it does not grant concurrent access to a device
workspace. Standalone callers must preserve that stream-order contract.

The planner, copied table values, kernels, launch order, FP operations and split
reductions are unchanged. Only host staging and waiting change. Decode paths are
unchanged. Bitwise identity is the required result, pending node qualification.

STRIX_MM_PF_TABLE_STATS=1 or STRIX_MM_PROFILE=1 prints the per-call count, rows,
host milliseconds from the route-sync return through enqueue_prefill_moe's final
launch/return, and cumulative milliseconds. This includes host planning, table
copies, any slot wait, launch submission and existing profiling instrumentation.
It is a host interval, not a measurement of GPU idle time. Printing occurs after
the measured interval and can perturb the next call's scheduling.

On the node, run the benchmark with --test --tests test_prefill_table_ring.
It compares every output byte with the ring off for two and six consecutive
calls on a nonblocking stream, with no intervening stream sync. Six calls wrap
all four slots. Calls rotate routes, vary weights and row counts, and immediately
overwrite/destroy host routes after enqueue. Cases cover T9/T16/T100/T4096 and
crafted row-tile boundaries through 513, all four exact cores and v4 mtfit off/on.
Both arms use poisoned storage; mismatches throw and cause a nonzero exit.
Repeat with real model weights on both ranks.

--ab pftab:PROMPT.u32:BLOCK runs A B A B A B A prefills. A disables the ring; B
enables it. State (K/V/index-key) and final-row FNV checks are mandatory, using
the existing failure exit on a mismatch. The captured default is restored after
the A/B run. Measure with interleaving and verify generated tokens on both ranks.

Local validation: extracted host ring/upload code compiled as C++20 with MSVC
/W4 /WX. A deferred-copy HIP mock passed two/six-call sequences for both item
layouts, slot growth, source overwrite, injected copy/record errors and cleanup.
These checks do not establish HIP compilation, GPU exactness or throughput.
