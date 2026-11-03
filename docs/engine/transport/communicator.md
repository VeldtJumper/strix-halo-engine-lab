# Two-rank communicator

Status: first layer of the distributed runtime. `include/strix/dist/communicator.hpp`, `src/dist_communicator.cpp`
(library `strix_dist`), the probe `strix-dist-tcp-probe` and the two-rank fixtures in
`tests/hip_ds41_formats_tests.hip`.

## Pieces

- **`Transport`**: moves whole messages between the two ranks, in order. Backends:
  - `memory_pair()`: both ranks in one process (the emulated two-node setup: two host threads, one GPU, one
    stream per rank);
  - `tcp_listen` / `tcp_connect`: one TCP connection with `TCP_NODELAY`, bounded waits, for bring-up and
    debugging between hosts; a persistent sender thread does the send half of an exchange;
  - `usb4_host(devices, weights)`: the host path of USB4 streams. The message is split over the listed streams
    (in the same order on both ranks) at 64-byte boundaries, in proportion to integer weights (equal by default),
    so the trailer travels in the last part; each stream has a persistent thread that writes its part and then
    reads the peer's part;
  - `usb4_host_auto(options, &topology)`: the same transport over every live stream, found and paired at startup
    (see *Automatic USB4 pairing*). This is the entry point for runs; the explicit list is for bring-up.
  Each stream has a persistent writer and a persistent reader, so a stream's part is written and read at the
  same time: messages of any size go through (the first version wrote a whole part before reading, and two
  ranks doing that deadlocked once a part exceeded what the two rings buffer, about 8 MB: a 42 MB message on
  four streams). Messages shorter than `min_part_bytes` per stream use fewer streams; the striping order takes
  the first stream of every cable, then the second, so a short message still spreads over the cables.
- **Failures**: every transfer waits in `poll()` slices with a progress deadline (`io_timeout`, 120 s by
  default, `$STRIX_DIST_USB4_TIMEOUT_MS`) and an abort flag. The first failing stream (ENODEV after an unplug,
  ETIMEDOUT from the driver, the peer's CLOSE, the deadline) is reported as the cause, the other transfers stop,
  every stream is closed (the peer reads CLOSE on its live streams and fails at once instead of waiting) and the
  transport refuses later calls. `strix-dist-transport-tests` checks this over Unix socket pairs: 42 MB
  exchanges, a peer that dies mid-exchange (reported in ~50 ms), a silent peer (the deadline).
  `DirectTransport` declares the device-buffer path (registered windows, submit, repost, completion by sequence).
- **`Transport::exchange(out, in)`**: both ranks send and receive at once; backends use persistent threads, never a
  thread per call.
- **Message format**: the payload, then a 64-byte trailer (`magic, epoch, plan, sequence, request, bytes, rank,
  tag, layer, crc32`). The trailer is the last thing on the wire, so its sequence number marks the payload before it
  as complete; the receiver checks every field and the CRC-32 of the payload. Sizes are never sent: both ranks
  derive them from shared data (rows, routes, plan).
- **`Communicator::exchange(stream, local, remote, bytes, tag)`**: copies the device buffer to pinned memory after
  its producer, waits for the copy on the CPU, sends it with its trailer while receiving the peer's message,
  checks the peer's trailer and CRC, and enqueues the copy of the payload into the device buffer `remote`. Both
  ranks call the same collectives in the same order.
- **`start(stream, local, bytes, tag)` / `finish(stream, remote)`**: the same exchange in two halves. `start`
  enqueues the copy out and returns; the communicator's progress thread waits for the copy on the CPU, then
  sends and receives. Work enqueued between the two calls runs on the GPU while the message travels. `finish`
  waits for the progress thread, checks the peer's message and enqueues the copy in. `exchange` is `start`
  followed by `finish`.

## Synchronization contract

Bytes of a device buffer leave only after the work producing them is complete as seen by the CPU, and a receive
buffer is reused only after the work reading it is complete as seen by the CPU. No backend may rely on implicit
device fences: work submitted through HIP user queues publishes none to other DMA engines. The host-staged
backends satisfy this with the stream synchronization at the start of `exchange` and the synchronous copy back;
a device-buffer backend must check producer and consumer completion on the CPU before `submit` and `repost`.

## Automatic USB4 pairing

Cabling is not assumed: any number of cables, straight or crossed, 20 or 40 Gb/s links, and stream device
numbers that change between boots. `usb4_host_auto` builds the pairing at startup:

1. **Candidates.** The stream entries of the layout file (`$STRIX_DIST_USB4_LAYOUT`, default
   `/run/usb4stream-layout.json`): keys `<service>/<stream>` with a `device`. A stream is opened only if the
   service is present under `/sys/bus/thunderbolt/devices` (a peer is attached on that port on this boot; a stream configured on a domain
   whose controller is down is never opened). Link rate and lanes (both directions) and the peer's identity
   come from the XDomain's sysfs entries. sysfs decides: the layout is written at one moment and can be stale both
   ways (streams of a link that went down still listed as openable, or live streams listed as unattached by a
   layout written during boot). `Usb4AutoOptions::exclude` / `$STRIX_DIST_USB4_EXCLUDE` (device files or keys)
   keeps streams out explicitly. Streams are opened non-blocking, so a stream whose link is down fails at once
   instead of waiting for it (the explicit-list `usb4_host` opens the same way, then switches to blocking I/O).
   The plan reads after the handshake wait at most 20 s (poll), so a peer or link lost during pairing fails it
   instead of blocking it.
2. **Handshake on every opened stream.** Messages are 64-byte units: hello (protocol version, rank, process
   nonce, the sender's stream index, device number and open generation, host name, CRC-32C) and acknowledgement
   (the same, plus the peer's nonce, stream index and generation, which proves both directions). With the stream
   module in use, frames written before the peer has opened its end can hold back later frames of that
   direction, so rank 0 never writes first:
   - rank 1 sends a hello every 250 ms; without an answer within 1.5 s of opening it closes the stream, waits
     0.5 s and opens it again (fresh rings, next generation);
   - rank 0 answers each new hello (nonce and generation) with its hello and an acknowledgement in one write,
     and reopens at once when it reads the peer's CLOSE (its own CLOSE then finds rank 1's end closed);
   - rank 1 acknowledges rank 0's hello.
   Every unit written is consumed by the other side inside the handshake, so the streams are clean afterwards.
   Units that are not valid handshake units (left over from an earlier session) are skipped and counted. After
   the first stream pairs, the others have a grace period (5 s by default).
3. **Rejection with reasons.** Streams that do not complete are closed and listed in the receipt (no answer,
   one direction only, reopen count, last error). Streams answered by a different peer process, or by a process
   with the same rank, fail the pairing: two jobs or two rank-0 processes are on the link.
4. **Order and plan.** Accepted streams are ordered by rank 0's stream index. Rank 0 completes a stream only on
   rank 1's acknowledgement, so its set is a subset of rank 1's; rank 0 sends its set on its first stream,
   rank 1 finds it, keeps exactly that set (closing any extra stream) and echoes it.
5. **Per-cable capacity.** The streams of one cable (one XDomain) are measured together with a short exchange
   in both directions (16 MiB by default, a 64-byte barrier on the cable before each of five repetitions, the
   median after a warm-up, contents checked). The measured
   rate in Mb/s, shared equally among the cable's streams, becomes the integer striping weight. Rank 0's
   measurement decides and travels in a second plan. With `probe_bytes = 0` the weights come from the link rate.
6. **Receipt.** `Usb4Topology::receipt()` lists, per stream, the local key and device, domain, port, cable label,
   link rate, the peer's device and index, the cable's measured rate and the weight, and every candidate not
   used with its reason. The two-node fixtures print it at startup.

Fewer live cables or streams simply give a smaller plan. TCP is never chosen automatically; it stays an explicit
debugging transport.

What the stream service could expose to make this simpler (none of it is required):
- the peer's stream identity per stream (host UUID, service and stream name, device number) once the rings are
  up, which would make the hello round confirmatory only;
- a per-stream state (rings started, peer open, peer closed) readable without opening the device;
- the XDomain link state and lanes in the layout file, refreshed on hot-plug, with a generation counter;
- an rx polling mode (`rx_napi`/`spin_us`) as a per-open option, so a job can choose it without module
  parameters.

## GPU-direct exchanges (`gpu_direct.hpp`)

With streams in the driver's gpu_direct mode, an exchange goes from a device buffer to the peer's device buffer
without host copies: `Communicator(group, control, usb4_gpu_direct(...), max_bytes)`, with the same `exchange` /
`start` / `finish` calls. `control` is a host-path transport (the pairing, the setup messages, and messages larger
than the GPU-direct capacity).

- **Streams.** The layout's streams whose configfs `gpu_direct` reads 1; this code never writes that attribute.
  Host-path pairing skips them. A gpu_direct stream pairs with the stream of the same name at the other end of
  the same cable; the cable map comes from the host-path streams of that cable, so every cable keeps one
  host-mode stream. The endpoint identity reported by the driver (QUERY) must name the peer host of that cable.
- **Arenas.** Per stream, a TX and an RX window of `ring_frames` x 4 KB in device memory, exported as DMA-BUF and
  imported once; `depth` slots (4) per direction, `floor((ring_frames - depth) / depth)` frames each (1 MB with
  1024 frames). The terminal frame of every operation is built and checked by the kernel (patch 0016): epoch,
  sequence, payload size, frame count, slot.
- **One exchange** (sequence k + 1, slot k % depth, on `floor((bytes + 64) / min_part)` streams, at least one):
  1. `start`: a scatter kernel copies the caller's buffer into the TX slots and writes a 64-byte header after
     the last part: magic, sequence, plan, request, size, collective tag, layer and a checksum of every 4-byte
     word, sum of `(w ^ (i * 0x9E3779B1)) * (2i + 1)`. Then an event is recorded. The RX slot's previous
     operation is retired once the CPU has seen its consumer copy complete.
  2. Progress thread: PREPARE + SUBMIT of RX k, a CPU wait for the producer event (DMA-BUF reservations do not
     see HIP work), PREPARE + SUBMIT of TX k, then EVENT polls until RX and TX are complete. It then writes
     k + 1 into the completion flag (device signal memory).
  3. `finish`: the stream waits for the flag (`hipStreamWaitValue32`). A gather kernel checks the flag's failure
     value, copies the RX slots into the caller's buffer and checks the received header against this rank's
     identity and the checksum of the received words. Then the consumer event is recorded. Neither call waits on
     the CPU for the GPU.
- **Order and identity.** Both ranks post and submit in sequence order. A TX that reaches the peer before its RX
  is posted waits on end-to-end flow control. If the ranks disagree on the order or the size, the kernel's
  terminal check fails (`GpuQueueTerminalMismatch`). If they disagree on the plan, the request, the collective
  or the layer, the receiving GPU refuses the exchange. A checksum mismatch is refused too. No mismatch can
  deliver shifted or foreign data.
- **Failures.** Every exchange has a deadline (30 s). On a host-side error (deadline, refused terminal, driver
  error), the progress thread stops every queue of the pair. The peer's operations on those streams then
  fail, or reach their deadline, so both ranks stop. It then writes the failure value into the flag. A gather that sees the
  failure value, or refuses a header, records the reason in host-mapped memory and traps. The stream's queue
  faults, so nothing enqueued after it consumes the buffer, and later HIP calls on the stream fail.
  `failure()` (and every later `start` / `finish`, which throw it) gives the GPU's reason, the peer's tag and
  layer, and the host-side cause. The destructor always drains the progress thread.
- **Opening.** The stream device is opened non-blocking. EAGAIN / EINTR are retried for at most the setup
  timeout, and the configfs state is checked between attempts.
- **Any size.** Bytes that are not a whole 4-byte word travel zero-padded in the last word; only the real bytes
  are written. Parts are `ceil(wire / used)` rounded up to 64 B and checked against the slot before any launch.
  Messages above the capacity are exchanged in capacity-sized chunks.
- **Completions.** The scatter's last block stores the sequence into host-mapped memory after a system-scope fence.
  The progress thread polls that word (no HIP call while the caller launches work), then confirms with the
  producer event once before SUBMIT. A job ends when every RX has landed; TX completions are consumed later
  (`STRIX_DIST_GPU_LAZY_TX=0` waits for them), at the latest before that TX slot is reused or at drain. The
  gather loads the completion flag once per block.
- **Epochs.** The wall clock in ns, above both ranks' clocks, never random: the driver refuses an epoch lower than
  a device's last one.
- **Size router.** `STRIX_DIST_ROUTE` unset (or `table`) is the measured default:
  - with host streams (a mixed layout such as data0 host + data1 gpu_direct): up to 64 KB on the host path, two
    rings up to 1 MB, and above that the mixed route with 35 % of the bytes on the host path
    (`65536:host,1048576:gpu:2,inf:mix:RINGS:35`);
  - without host streams (every stream gpu_direct): one ring up to 32 KB, two rings up to 256 KB, every ring above.

  Other values:
  - `auto` measures the host path and the device exchange on 1..rings rings at a grid of sizes when the Communicator
    is built, at idle, and rank 0's table is applied by both ranks. On the hardware it chose worse than the table.
  - `gpu`, `host`, or an explicit table `UP_TO:host|gpu[:RINGS],...`.

  The host path is never chosen when the transport is only a TCP control channel. A requested ring count is raised
  to the fewest rings whose slots hold the message.
- **Mixed route.** A table entry `UP_TO:mix:RINGS:HOST_PERCENT` sends the head of a message (HOST_PERCENT % of
  the bytes, a multiple of 64 B, at most the host bound) over the host path and the rest over RINGS rings, both
  at once. The host part carries its own trailer and CRC; each part lands at its own offset of the receive buffer,
  so the delivered bytes, and every reduction that reads them, are those of either path alone. This uses the
  host streams and the GPU-direct rings of a mixed layout together for large messages. `set_routes()` replaces the
  table between exchanges (both ranks, same table); the probe sweeps host shares with `STRIX_PROBE_MIX`.
- **Receive posted ahead.** Each exchange on the rings posts the next exchange's receive before its own TX
  (`STRIX_DIST_GPU_PREPOST`, default on; 0 off). The peer can only send exchange k+1 after receiving exchange k, so it
  always finds its receive descriptors posted.
  - Why: on these controllers a TX that starts while the peer's GPU receive ring is empty can wait about 100-200 ms
    for E2E credits. Back to back, that happened in 9-15 of 3000 exchanges at every size from 20 KB to 1 MB and on a
    single ring; with the receive posted ahead, in none of 18 000 (p50 +2 µs). Host-mode rings keep their receive ring
    full and never showed it. The credit mechanism is inferred from these measurements, not proven.
  - The next exchange's geometry comes from `DeviceExchange::expect_next()`: the communicator gives the next chunk of a
    chunked message exactly, and otherwise a message of the same size again.
  - A wrong guess is seen the same way by both ranks, which make the same calls; both then complete one internal
    exchange of the guessed geometry (scratch buffers, a control identity) that posts the real one's receive.
  - `Communicator::quiesce()` consumes a receive posted ahead the same way. Call it on both ranks at the same point
    before staying idle for long (e.g. at the end of a request in a server): a posted receive counts against the
    driver's DMA deadline (30 s).
- **Blocking wait.** After an exchange of at most 64 KB (`STRIX_DIST_GPU_BLOCK_WAIT=N`, 0 off, 1 every size), the
  next exchange's caller waits for that exchange's gather instead of spinning. With polled completion this took
  decode from 17.8 to 20.2 tok/s. Applied to prefill's large chunks it halved throughput, hence the size limit.
- **Fallback.** `usb4_gpu_direct_streams()` exchanges both ranks' attached gpu_direct stream counts before any
  setup. When either rank has none, the apps continue on the host path (`usb4gpu` keeps its host streams;
  `usb4gpu4` opens `usb4_host_auto`), and both ranks make the same choice. A failure later in the setup throws
  with its cause (fail closed).
- **Every stream in gpu_direct mode** (`usb4_gpu_direct_all`). A TCP connection over Ethernet carries only the
  rendezvous: inventory, epoch, pool identity, session parameters, barriers, the ring order. Then:
  - **HELLO.** An in-band HELLO on every ring gives the cable map, since a ring is point to point. Crossed cables and
    stream numbering that differs between the hosts are accepted; a ring whose HELLO endpoints do not match its
    QUERY identity both ways is refused.
  - **Ring order.** The rings are ranked by measured 64 B round trip, and ring 0 carries the one-ring messages.
  - **Ring limit.** `max_streams` keeps the first rings of that agreed order.
  - App transports: `usb4gpu4:HOST:PORT` (rank, batch) and `usb4-gpu4:HOST:PORT` (prefill), where rank 0 listens on
    PORT.
- **Recovery.** Not automatic. A failed exchange poisons the session, and both ranks must restart it. The
  driver's RECOVER with a fresh epoch, coordinated over the control transport, is the intended path and is not
  implemented.
- **Test without hardware.** `strix-hip-dist-gpu-direct-tests` runs two ranks on one GPU against a fake of the
  queue ioctls. It checks exchanges of 12 B to the full capacity and chunked larger messages through the
  Communicator, byte-exact. Failure cases run in child processes: a size mismatch (terminal check), a collective
  mismatch (GPU refusal naming the peer's tag), and a silent peer (deadline). In each case both ranks stop with
  the cause.
  The fake pairs operations in physical (FIFO) order. Further cases: sizes at every slot edge and odd sizes with
  guard bytes; the bootstrap with crossed, renumbered and foreign rings followed by byte-exact exchanges; dropping
  rings after the bootstrap; the router's calibration on two threads.

## Expert parallelism of the deepseek41 MoE layer

Each rank runs the router and the shared expert on the same inputs (replicated, identical bits), computes the
routed experts it owns (ids below or above the split, `restrict_routes`), forms the sum of its half
(`enqueue_half_sum`), exchanges it, and forms `(shared + low) + high` (`enqueue_canonical_sum`). This is the
single-node combine operation by operation, so both ranks produce the single-node output bitwise. The raw slot
vectors can be exchanged instead (`TagKind::raw_slots`) when the experts are not split by id.

Overlap: a rank enqueues the router, its routed experts and its half sum, starts the exchange, enqueues the shared
expert (replicated, independent of the exchange) and then finishes the exchange and forms the canonical sum. The
shared expert's GPU time hides the transfer. The order of operations inside every sum is unchanged, so the bits are
the same with and without overlap.

## Measurements, GPU-direct on 2 rings (data1 of both cables) vs the host path on 2 streams (data0)

Probe `strix-dist-gpu-probe`: GPU producer, exchange, stream synchronize; every received word is checked, and all
exchanges were byte-exact. The probe synchronises the two ranks before every timed exchange (`STRIX_PROBE_SYNC=0`
restores the former free-running loop) and reports both ranks' percentiles and the pair maximum, so the 1 MiB figures
no longer flip between runs: host path, 1 MiB, pair maximum p50 300-318 µs (26.4-27.9 Gb/s per direction, three
fresh-process runs) with the rendezvous, against a bimodal 168-521 µs spread without it.

| p50 µs | 64 B | 20 KB | 160 KB | 21 MB |
|---|---|---|---|---|
| host path, 2 streams | 26 | 36 | 146 | 15 074 |
| GPU-direct, 2 rings | 33 | 53 | 95 | 7 491 |
| GPU-direct, 2 ms idle before each | 110 | 565 | 753 | 7 901 |
| host path, 2 ms idle before each | 599 | 350 | 263 | 15 590 |

- With the gather loading the completion flag in every thread, it ran at ~1.2-4 GB/s (131 µs at 160 KB). With one
  load per block it takes 10.9 µs.
- Reads of the RX arena itself run at full speed, before and after DMA (5-8 µs for copy or checksum kernels up to
  1 MB).

deepseek41 runner, 40 layers, two nodes. Tokens, batch requests and logits were identical to the host path in
every run.

| workload | host path | GPU-direct, every size | size router (auto) |
|---|---|---|---|
| decode, tok/s | 20.15 | 13.7-15.8 | 20.18 |
| batch N=8, ms/step | 126.0 | 122.7-123.5 | — |
| prefill 8K row split, tok/s | 566.5 | 568-583 | 556 |

With every size on GPU-direct, decode loses because small exchanges follow idle gaps and the GPU rings complete
through interrupts (no polling).

## Measurements, four streams on two cables (kernel with receive polling)

- Round trip of a 20 KB message: 32 us p50 / 58 us p99 striped over 4 streams, 57 / 68 us on one stream; 120 KB:
  77 / 82 us. Default: every stream for every message (`min_part_bytes` 0).
- 21 MB exchange: 4.7 ms; 42 MB: 9.3 ms (about 4.7 GB/s per direction). Each cable carries 17-19 Gb/s per
  direction whatever its link rate (20 or 40 Gb/s): the host path is the limit.
- A rank killed during 42 MB exchanges: the other rank fails about 2 s later with the cause ("the peer closed the
  stream"), and nothing hangs.
- Two-node decode of the deepseek41 runner (40 layers, tensor-parallel attention, 81 exchanges per token): with the
  shared expert overlapped with the half-sum exchange, 17.70 tok/s (56.5 ms/token), against 16.94 tok/s
  (59.0 ms/token) in the serial order. Same tokens on both ranks and in every variant.

## Measurements (node1 <-> node2, one USB4 cable, two streams, host path, interrupt-driven module)

- Expert parallelism by halves over USB4 between the two nodes: the output of layer 20 equals the single-node
  output bitwise on both ranks (1 and 6 rows, 20-30 inputs each), and the two ranks' outputs agree.
- Back-to-back exchange of one layer's half sums, staging included: 71-151 us (1 row, 20 KB) and 142-198 us
  (6 rows, 120 KB) per exchange; the link alone (`strix-dist-tcp-probe usb4`): 42-51 us and 133-143 us round trip.
- After the receiver has been idle for 1-3 ms (the compute between two layers' exchanges), the first message is
  late: round trip p50 75-360 us at 20 KB and p99 1-3 ms, with blocking or polled reads alike. Clock-aligned
  timelines of the layer runs show single receives delayed by 1-2 ms. This is in the receive interrupt path of the
  loaded stream module (no receive polling); the per-layer exchange needs a polled receive path before expert
  parallelism pays.
- `strix-dist-tcp-probe` over the 1 GbE LAN between the two nodes: round trip 56 us at 64 B, 410 us at 20 KB,
  0.118 GB/s one way. TCP over this link is for bring-up only.

## MiniMax grouped async start

MiniMax grouped prefill enables `start_async` on its own communicator by default,
without setting the global async environment switch. Each grouped call requires
every member to have at least `STRIX_MM_PF_ASYNC_MIN_ROWS` rows (default 2048;
0 = unconditional); rejected groups use ordinary start/finish with the same
scheduling as `STRIX_MM_PF_ASYNC_START=0`. Admitted calls publish each member's
receive destination at start and guards it with a startup-allocated last-reader
event. The worker protects pinned receive staging before reuse. A source-free
event guards the next producer; `STRIX_MM_PF_LATE_SRC_FENCE=0` instead waits
immediately after successful start. `STRIX_MM_PF_ASYNC_START=0` restores ordinary
start/finish. Unsupported routes fall back without recording caller events.
Wire payloads, sequence/tags, framing and CRC are unchanged. Pipeline and ordinary
host reuse timing have separate counters; pipeline worker time overlaps other work.
`cancel_pending` poisons the communicator and joins host submission workers; callers
must then drain GPU work before releasing their events or buffers.

Two-node TP2 measurements have identical token IDs in every arm and show
balanced 4096+3844 lone-prefill gains of -2.3 / -2.5 %. The neutral 4096+443
regime is gated off by default. Full prompt-output/state/logit and complete-workload
lifetime qualification remain pending. See the [engine contract and lifetime/qualification procedure](../../minimax-grouped-prefill.md).
