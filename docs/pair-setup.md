# Set up the DeepSeek pair with `ds41-pair`

[English](pair-setup.md) · [Español](pair-setup.es.md) · [Two nodes over USB4](pair.md) · [Main README](../README.md)

`scripts/pair/ds41-pair` takes two Strix Halo machines from a fresh checkout to a running, verified DeepSeek V4.1 Flash
pair: preflight, USB4 driver, build, weights, optional expert preparation, launch and an exact smoke test. Run the
same commands on both nodes. Every subcommand is idempotent: re-run it after a failure or an interruption and it
continues where it stopped. It needs Python 3.10+ and only the standard library.

The script never uses root by itself. Steps that need root (installing the USB4 modules, the udev rule and the
stream service) are printed for you to run; `driver --yes-root` runs them with `sudo`, showing each command first.

## The whole path

On **both** nodes (node A becomes rank 0 and serves HTTP, node B rank 1):

```sh
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models   # on node B: --rank 1 --peer user@node-a
scripts/pair/ds41-pair check
scripts/pair/ds41-pair driver          # builds the modules, then prints the root steps
scripts/pair/ds41-pair driver-verify
scripts/pair/ds41-pair build
scripts/pair/ds41-pair fetch           # node B can use: fetch --from-peer
scripts/pair/ds41-pair prepare         # optional, once
```

Then, on node A:

```sh
scripts/pair/ds41-pair run --both      # starts rank 1 on node B over SSH, then rank 0 here
scripts/pair/ds41-pair smoke
```

`smoke` ends with `PASS smoke ... token ids equal the published reference` and one speed line. The server speaks the
OpenAI and Anthropic APIs at `http://127.0.0.1:8080` on node A ([serving](serving.md)).

## Before you start

- Two Strix Halo machines (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`) with 128 GB each, Linux, and ROCm with a HIP
  compiler ([requirements](getting-started/README.md#what-you-need)).
- Two USB4 cables directly between the machines (one also works, at about half the link bandwidth). Plug them in
  with both machines on.
- About **390 GB** free for the model files on each node, plus about 77 GB in `~/.cache` if you use `prepare`.
- SSH from each node to the other with a key (`ssh -o BatchMode=yes user@node-b true` must work). The pair
  itself does not use the network; SSH is only for `check`, `fetch --from-peer`, `run --both`, `stop --both` and
  `driver-verify --link-test`.
- Keep the IOMMU on. The USB4 driver relies on it to contain the controllers; `check` fails when it is disabled on
  the kernel command line.

## Config file

`init` writes `~/.config/strix-pair/ds41.conf` (or the file in `--config FILE` / `$STRIX_PAIR_CONFIG`). It is
shell syntax, one `KEY=value` per line, and you can edit it. Each node has its own file; only `RANK` and `PEER`
differ.

| Key | Default | Meaning |
|---|---|---|
| `RANK` | (required) | `0` serves HTTP, `1` is the other half |
| `PEER` | | SSH destination of the other node, for example `user@node-b` |
| `MODEL_DIR` | (required) | the production files, in the layout below |
| `PROFILE` | `default` | `default` (the exact production configuration), `fast` or `quality`; the same on both ranks |
| `BUILD_DIR` | `build-ds41` in the checkout | engine build directory |
| `DEPS_DIR` | `~/.local/share/strix-pair` | trellis library, driver and kernel sources, built modules, manifests |
| `PEER_REPO`, `PEER_MODEL_DIR` | same paths as here | the checkout and model directory on the peer |
| `ROCM_PATH` | `hipconfig -R`, else `/opt/rocm` | ROCm root |
| `HTTP_BIND`, `HTTP_PORT` | `127.0.0.1`, `8080` | rank 0 HTTP address; see [exposing the servers](serving.md#exposing-the-servers) before binding another address |
| `API_KEY_FILE` | | file with the HTTP API key |
| `CONTEXT` | server default | context length |
| `EXTRA_ARGS` | | more `strix-deepseek41-serve` options, the same on both ranks |

`ds41-pair config` prints the effective values.

## Subcommands

### `init`

`init --rank 0|1 --model-dir DIR [--peer USER@HOST] [--profile P] [--build-dir DIR] [--deps-dir DIR]
[--http-bind ADDR] [--http-port N] [--api-key-file F] [--context N] [--peer-repo DIR] [--peer-model-dir DIR]
[--rocm DIR]` writes or updates the config. Options you do not give keep their current value.

### `check`

Read-only preflight of this node, one `PASS`, `WARN`, `FAIL` or `INFO` line per item; the exit status is 1 when
anything fails. `--no-peer` skips the SSH part.

| Item | Checks |
|---|---|
| `cpu`, `gpu`, `kfd` | a Ryzen AI Max CPU, a `gfx1151` GPU in the KFD topology, `/dev/kfd` read/write for you (else join the `render` and `video` groups) |
| `memory` | a 128 GB node; the GPU can map about 117 GiB of system memory (GTT, `ttm pages_limit`); `kho=off`; warns when the firmware reserves a large VRAM carve-out, or when other processes hold the GPU |
| `kernel`, `iommu` | the running kernel; the IOMMU is on and the USB4 controllers are translated (`DMA` / `DMA-FQ`); `iommu=pt` warns, `amd_iommu=off` / `iommu=off` fails. It never suggests disabling the IOMMU |
| `usb4`, `link` | `thunderbolt_stream` loaded, `thunderbolt_net` not loaded, each cable's speed and lanes from sysfs: Gen3 x2 (20 Gb/s per lane, 40 Gb/s per direction) passes; Gen2 or one lane warns with "re-seat that cable with both nodes on" |
| `streams` | four `/dev/tbstream*` devices you can read and write, and the stream layout file |
| `disk`, `models` | free space against what `fetch` still has to transfer; the production files by name and size |
| `rocm`, `tools` | a HIP compiler, CMake 3.24+, Ninja, Git, Python 3.10+, the OpenSSL headers (faster hashing), the kernel build tools for `driver` |
| `build`, `peer` | the engine binary; the peer is reachable over SSH, has the other rank, and runs the same engine binary (SHA-256) |

### `driver` and `driver-verify`

The pair uses the [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver): patched `thunderbolt`
and `thunderbolt_stream` modules for one pinned upstream kernel (Linux 7.3-rc5), plus a boot service that creates two
streams per cable. `driver` (no root) picks the route for this node:

- **The stream module of the pinned series is already loaded** (a kernel built from the series): nothing to build.
  It only prints the memory settings below if they are missing.
- **The running kernel is the series' version but lacks the module:** it checks out the driver repository and the
  upstream tree at their pinned commits (a shallow Git fetch, once), applies the series, configures the tree with
  the running kernel's configuration plus the driver fragment, builds `thunderbolt`, `thunderbolt_stream` and
  `thunderbolt_net` against the running kernel's `Module.symvers` (linux-headers; without it the whole tree is
  built once, or force that with `--full`), checks each module's version magic and stages them in
  `DEPS_DIR/usb4-modules/<kernel>/` with their SHA-256 sums and the `thunderbolt_stream` srcversion.
- **Any other kernel, or USB4 built in:** loadable modules cannot match it. `driver` says so and stops;
  `driver --kernel` then builds the whole patched kernel as a new kernel next to the current one (LOCALVERSION
  under `-strix-gpu-bosgame-`, checked with the driver's `check_kconfig.py`; Debian packages with `bindeb-pkg`
  where `dpkg-deb` exists), about 30-60 min of compiling. `--localversion` sets its suffix.

Then it prints the root steps for this node: install the modules or the kernel package, a udev rule giving your
group read/write access to `/dev/tbstream*`, a modprobe rule that keeps `thunderbolt_net` off (it would take one ring
pair per controller from the streams), `usb4stream-describe` and `setup-usb4stream-lab` from the driver repository
and the stream service, and, where missing, the two memory settings of the driver's node setup: `options ttm
pages_limit=30720000` (lets the GPU map 117 GiB of system memory; the pair does not fit the default half of RAM) and
`kho=off` on the kernel command line. The new modules or kernel take effect at the next boot; nothing is reloaded
under a running system. Boot a new kernel once with `grub-reboot` before making it the default, keep the current
kernel as the fallback entry, and never disable the IOMMU. With Secure Boot on, sign the modules with an enrolled
key first (`driver` warns). `--dry-run` shows the route and the plan without compiling.

`--yes-root` runs those steps with `sudo`, printing each one before it runs and stopping at the first failure.
`--rebuild` rebuilds the modules; `--jobs N` sets the parallelism.

`driver-verify` (read-only) checks on this node: both modules loaded (the stream module of the pinned series, and the
staged build when `driver` built one), no `thunderbolt_net`, both cables up at Gen3 x2, and the four streams of the layout file attached and read/write.
`--link-test` also pairs the four streams with the peer the way the engine does and exchanges checked payloads
(it starts the other side over SSH and uses `strix-dist-host-exchange-bench` from the build). Run it with no engine
running on either node.

### `build`

Checks out the [strix-trellis](https://github.com/hyukudan/strix-trellis) library at a pinned commit, builds it for
`gfx1151` with the dense R1 TP API, then configures and builds the engine with the production DeepSeek options
(`STRIX_FAMILY_DEEPSEEK41=ON`, `STRIX_DS41_TRELLIS=ON`, `STRIX_DS41_TRELLIS_API=dense_r1`, the library and its
headers; OpenSSL hashing when the headers are present). The targets are `strix-deepseek41-serve`,
`strix-deepseek41-rank`, `strix-tokenize`, `strix-ds41-prepare-load` and `strix-dist-host-exchange-bench`.
It writes `BUILD_DIR/ds41-pair-build.json` with the engine and library commits and every binary's SHA-256.
Build the same commit on both nodes; `check` compares the binaries. `--jobs N`, `--clean`.

### `fetch`

Downloads the production files from the three mirrors at the revisions pinned in
`scripts/bench/lib/fetch_models.py` and checks every file against the mirror's `SHA256SUMS` at that revision:

| Mirror | Files |
|---|---|
| [deepseek-v4.1-flash-gguf-mirror](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror) | `DeepSeek-V4.1-Flash-1017-Q2.gguf` (365.7 GB), `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` (8.0 GB) |
| [DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6) | `c-u6.gguf` (7.6 GB), `c-u6.json`, `aux/vocab/*.bin`, `aux/draft-keep-stats.jsonl` |
| [DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) | `ours-K4/` (3.6 GB), `ours-K5/` (4.5 GB), `mixB.txt` (production: ours-K4 plus the ours-K5 head) |

The result is the documented layout ([reproduce](reproduce.md#models-and-provenance)); `vocab/` is filled from the
verified `aux/vocab/`:

```text
/path/to/models/
  DeepSeek-V4.1-Flash-1017-Q2.gguf
  DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
  c-u6.gguf  c-u6.json
  ours-K4/  ours-K5/  mixB.txt
  aux/  vocab/
  SHA256SUMS.<mirror>  LICENSE*  README*
```

Downloads go to `NAME.part` and resume after an interruption (re-run `fetch`). A file is only renamed into place
after its SHA-256 matches; a mismatching file is removed and fetched again. Hashes of verified files are
remembered by size and modification time in `MODEL_DIR/.ds41-pair-verified.json`, so a re-run does not re-read
390 GB. For a gated or rate-limited download, export `HF_TOKEN` (it is sent only to the mirror and never printed).

| Option | Effect |
|---|---|
| `--dry-run` | the plan: each missing, partial or mismatching file with its size, the bytes left and the free space; transfers nothing |
| `--from-peer` | copy each file from the peer's model directory with `rsync` over SSH instead of downloading it twice, then verify it here |
| `--verify` | re-hash every file even when the remembered hash is still valid |
| `--verify-only` | hash the files already present against the manifests; never transfer |
| `--only GLOB` | limit to matching files (repeatable), for example `--only 'ours-K*'` |

`--from-peer` uses whatever address `PEER` resolves to. The USB4 links carry the engine's streams, not IP
(`thunderbolt_net` is blacklisted), so the copy normally goes over your LAN.

### `prepare` (optional)

Runs `strix-ds41-prepare-load` once for this rank with the production dense identity: a rank-local pack of the
routed experts in the TP `dup` layout that the server uses, about 72 GiB in `~/.cache/strix/ds41-prepared`. The
server finds and validates it at every start and loads its experts about 20 s faster; without it, the original
loader runs with one notice line. Details: [one-time expert preparation](models/deepseek-v4.1/prepared-load.md).
Prepare on each node, after `fetch` and `build`; prepare again after changing the model files (remove the old pack).
`--dry-run` prints the command and the space needed; `--force` prepares even when a pack exists.

### `run`, `stop`, `status`, `logs`, `unit`

`run` starts this node's rank in the background through `scripts/pair/ds41-pair-launch` and waits for its `READY`
line (model load and warm-up take one to a few minutes; the two ranks wait up to a minute for each other). `--both`
first starts the peer's rank over SSH. `--foreground` runs it in the terminal; `--no-wait` returns at once;
`--wait S` sets the wait. Logs go to `~/.local/state/strix-pair/rank.log`.

`ds41-pair-launch` is the plain production command with your config:

```sh
STRIX_DS41_PACKS=/path/to/models \
  BUILD_DIR/strix-deepseek41-serve /path/to/models/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --rank R --transport usb4auto --profile default --http-bind 127.0.0.1 --http-port 8080
```

The server resolves the overlay, the trellis packs, the drafter, TP attention and device prefill itself
([production defaults](models/deepseek-v4.1/production-defaults.md)). Extra arguments go to the server, for example
`scripts/pair/ds41-pair-launch --print-effective-config`.

`stop [--both]` sends SIGTERM (the server finishes running requests; stopping rank 0 also stops rank 1). `status`
shows the process, its READY state and `/health`. `logs [-n N]` prints the end of the log.

`unit` writes a systemd user unit `strix-ds41-pair.service` for this rank (it runs `ds41-pair-launch`, restarts on
failure after 30 s) and prints `systemctl --user` commands; `--install` also enables and starts it. To start it at
boot without logging in, run `sudo loginctl enable-linger USER` once. Use either the unit or `run`, not both.

### `smoke`

On rank 0 (on rank 1 it runs on the peer over SSH). Waits for `/health`, sends one fixed prompt with greedy decoding
and `return_token_ids`, and compares the 32 token IDs with the published reference in
`scripts/pair/ds41-smoke-reference.json`. Then one 128-token request prints a speed line: time to first token and
decode tokens per second of one stream (DSpark is on, so the rate depends on the text). The reference is for the
`default` profile; with `fast` the IDs are printed without a verdict. A mismatch means the two nodes differ in build,
profile or files: run `check` on both and `fetch --verify-only`.

## When something fails

| Symptom | What to do |
|---|---|
| `link ... trained at Gen2` or `single lane` | re-seat that cable with both nodes on, then `driver-verify` |
| `no USB4 link up` | cables plugged with both nodes on; `driver-verify`; the stream service ran (`systemctl status usb4stream-pair`) |
| `driver`: series is for another kernel | `driver --kernel` (or the driver repository `BUILD.md` by hand), boot the new kernel once, `driver-verify` |
| `memory ... GTT limit` | install the `ttm.conf` step from `driver`, reboot |
| `streams ... not read/write for you` | install the udev rule from `driver`, log in again |
| `run`: no READY, POISONED | `ds41-pair logs`; both ranks must run the same build and profile; start the other rank within a minute |
| `fetch`: SHA-256 mismatch | re-run `fetch` (the bad copy was removed) |
| `smoke`: token IDs differ | same commit built on both nodes (`check`), `PROFILE=default`, `fetch --verify-only` on both |
