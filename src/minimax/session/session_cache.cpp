#include "strix/minimax/session_cache.hpp"
#include "strix/minimax/model_assets.hpp"
#include "strix/minimax/asset_input.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
extern char **environ;
#endif
namespace strix::minimax::session {
namespace {
void require(bool ok, const std::string &why) {
  if (!ok)
    throw std::runtime_error("MiniMax session: " + why);
}
// Nonblocking acquisition avoids crossed distributed writers waiting forever
// while each holds its local lock. Contention is a collective budget refusal.
class DirectoryLock {
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
public:
  explicit DirectoryLock(const std::filesystem::path& dir) {
    require(!dir.empty(), "session_budget: empty disk directory");
    std::filesystem::create_directories(dir);
    const auto path = dir / ".session.lock";
#ifdef _WIN32
    handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    OVERLAPPED offset{};
    if (handle_ != INVALID_HANDLE_VALUE &&
        LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                   0, 1, 0, &offset)) return;
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ >= 0 && ::flock(fd_, LOCK_EX | LOCK_NB) == 0) return;
    if (fd_ >= 0) ::close(fd_);
#endif
    throw std::runtime_error("MiniMax session: session_budget: disk directory locked/unavailable");
  }
  ~DirectoryLock() {
#ifdef _WIN32
    OVERLAPPED offset{};
    UnlockFileEx(handle_, 0, 1, 0, &offset);
    CloseHandle(handle_);
#else
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
#endif
  }
  DirectoryLock(const DirectoryLock&) = delete;
  DirectoryLock& operator=(const DirectoryLock&) = delete;
};
struct Writer {
  Bytes b;
  void u64(std::uint64_t x) {
    for (int i = 0; i < 8; ++i)
      b.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
  }
  void raw(std::span<const std::uint8_t> x) {
    b.insert(b.end(), x.begin(), x.end());
  }
  void text(const std::string &x) {
    u64(x.size());
    raw({reinterpret_cast<const std::uint8_t *>(x.data()), x.size()});
  }
};
struct Reader {
  std::span<const std::uint8_t> b;
  std::size_t p = 0;
  std::span<const std::uint8_t> raw(std::uint64_t n) {
    require(n <= b.size() - p, "truncated/bounds");
    auto v = b.subspan(p, static_cast<std::size_t>(n));
    p += static_cast<std::size_t>(n);
    return v;
  }
  std::uint64_t u64() {
    auto v = raw(8);
    std::uint64_t x = 0;
    for (unsigned i = 0; i < 8; ++i)
      x |= std::uint64_t(v[i]) << (8 * i);
    return x;
  }
  std::string text() {
    auto n = u64();
    require(n <= 1048576, "string limit");
    auto v = raw(n);
    return {reinterpret_cast<const char *>(v.data()), v.size()};
  }
  bool bit() {
    auto x = u64();
    require(x <= 1, "noncanonical bool");
    return x != 0;
  }
  std::uint32_t u32() {
    auto x = u64();
    require(x <= UINT32_MAX, "u32 overflow");
    return static_cast<std::uint32_t>(x);
  }
  void end() { require(p == b.size(), "trailing bytes"); }
};
std::string strict_file_sha(const std::filesystem::path &path) {
#if defined(__linux__)
  struct stat before{}, after{};
  require(::stat(path.c_str(), &before) == 0 && S_ISREG(before.st_mode),
          "stat content asset");
#endif
  const auto digest = model_content_identity_hex(sha256_file_content(path));
#if defined(__linux__)
  require(::stat(path.c_str(), &after) == 0 && before.st_dev == after.st_dev &&
              before.st_ino == after.st_ino &&
              before.st_size == after.st_size &&
              before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
              before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
              before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
              before.st_ctim.tv_nsec == after.st_ctim.tv_nsec,
          "asset identity changed during SHA-256 lookup/hash");
#endif
  return digest;
}
Bytes string_bytes(const std::string &s) { return {s.begin(), s.end()}; }
std::string random_generation() {
  std::random_device r;
  Writer w;
  for (int i = 0; i < 8; ++i)
    w.u64(r());
  w.u64(std::chrono::steady_clock::now().time_since_epoch().count());
  return sha(w.b);
}
void frontier(Writer &w, const Frontier &f) {
  w.u64(f.consumed);
  w.u64(f.emitted);
  w.u64(f.accepted);
  w.u64(f.drafter);
  w.u64(f.control);
  w.u64(f.pending);
  w.u64(f.pending_valid);
  w.u64(f.drafter_ready);
  w.text(f.sampling);
  w.text(f.lineage);
}
void valid(const Checkpoint &c) {
  validate_id(c.id);
  validate_id(c.generation);
  require((c.world == 1 || c.world == 2) && c.rank < c.world, "TP geometry");
  require(c.frontier.consumed == c.history.size() &&
              c.frontier.accepted == c.frontier.consumed &&
              c.frontier.drafter <= c.frontier.consumed &&
              c.frontier.sampling == "greedy" && !c.frontier.lineage.empty(),
          "unsupported frontier/sampling");
  require(c.sections.size() <= 512, "section count");
  require(c.frontier.consumed <= SIZE_MAX / 76800 && c.frontier.pending_valid &&
              c.frontier.pending < 200064 &&
              c.frontier.emitted <= c.frontier.consumed + 1 &&
              c.frontier.control <= 1,
          "committed frontier bounds");
  require(std::all_of(c.history.begin(), c.history.end(),
                      [](auto t) { return t < 200064; }),
          "token vocabulary");
  require(c.sections.size() == (c.frontier.drafter_ready ? 185U : 181U),
          "complete section set required");
  for (unsigned l = 0; l < 60; ++l)
    for (unsigned k = 0; k < 3; ++k) {
      const auto &s = c.sections[l * 3 + k];
      require(s.name == "layer" + std::to_string(l) +
                            (k == 0   ? ".K"
                             : k == 1 ? ".V"
                                      : ".index") &&
                  s.replicated == (k == 2) &&
                  s.bytes.size() == c.frontier.consumed * (k == 2 ? 256 : 512),
              "cache-row layout");
    }
  require(
      c.sections[180].name == "pending.logits" && !c.sections[180].replicated &&
          c.sections[180].bytes.size() == (c.rank == 0 ? 100096U : 99968U) * 4,
      "pending FP32 logits shard");
  if (c.frontier.drafter_ready) {
    require(c.frontier.drafter == c.frontier.consumed,
            "incomplete drafter frontier");
    const char *names[] = {"draft.K", "draft.V", "draft.positions",
                           "draft.continuation"};
    const std::uint64_t sizes[] = {5ULL * 4096 * 512 * 2, 5ULL * 4096 * 512 * 2,
                                   4096 * 8, 32};
    for (unsigned i = 0; i < 4; ++i)
      require(c.sections[181 + i].name == names[i] &&
                  c.sections[181 + i].replicated &&
                  c.sections[181 + i].bytes.size() == sizes[i],
              "drafter section layout");
    auto word = [](const Bytes &b, std::size_t p) {
      std::uint64_t x = 0;
      for (unsigned j = 0; j < 8; ++j)
        x |= std::uint64_t(b[p + j]) << (8 * j);
      return x;
    };
    const auto &positions = c.sections[183].bytes;
    for (std::uint64_t p = 0; p < 4096; ++p) {
      auto v = word(positions, static_cast<std::size_t>(p * 8));
      require(v == UINT64_MAX || (v < c.frontier.consumed && v % 4096 == p),
              "draft ring-position map");
    }
    for (auto p = c.frontier.consumed > 4095 ? c.frontier.consumed - 4095 : 0;
         p < c.frontier.consumed; ++p)
      require(word(positions, static_cast<std::size_t>((p % 4096) * 8)) == p,
              "incomplete draft ring");
    const auto &meta = c.sections[184].bytes;
    require(word(meta, 0) == 1 && word(meta, 24) <= word(meta, 16) &&
                word(meta, 16) <= word(meta, 8),
            "draft continuation counters/policy");
  } else
    require(c.frontier.drafter == 0, "unexpected drafter frontier");
  std::set<std::string> names;
  for (const auto &s : c.sections)
    require(!s.name.empty() && names.insert(s.name).second,
            "duplicate/empty section");
}
std::filesystem::path manifest(const std::filesystem::path &dir,
                               const std::string &id) {
  validate_id(id);
  return dir / (id + ".group");
}
std::string file_name(const Checkpoint &c) {
  return c.id + "." + c.generation + ".r" + std::to_string(c.rank) + ".session";
}
struct Receipt {
  std::string generation, compat, shared, digest;
  std::uint32_t rank, world;
  std::uint64_t bytes;
};
Bytes receipt(const Receipt &r) {
  Writer w;
  w.text("strix-session-group-v1");
  w.text(r.generation);
  w.text(r.compat);
  w.text(r.shared);
  w.text(r.digest);
  w.u64(r.rank);
  w.u64(r.world);
  w.u64(r.bytes);
  return w.b;
}
Receipt parse_receipt(std::span<const std::uint8_t> b) {
  Reader r{b};
  require(r.text() == "strix-session-group-v1", "group schema");
  Receipt o;
  o.generation = r.text();
  o.compat = r.text();
  o.shared = r.text();
  o.digest = r.text();
  o.rank = r.u32();
  o.world = r.u32();
  o.bytes = r.u64();
  r.end();
  return o;
}
std::string shared(const Checkpoint &c) {
  Writer w;
  w.text(c.id);
  w.text(c.compat);
  frontier(w, c.frontier);
  w.u64(c.history.size());
  for (auto t : c.history)
    w.u64(t);
  for (const auto &s : c.sections)
    if (s.replicated) {
      w.text(s.name);
      w.text(sha(s.bytes));
    }
  return sha(w.b);
}
} // namespace
// Same digest as sha256_bytes; the CPU's SHA extensions when present (state payloads are hundreds of MB, hashed
// several times per save and resume).
std::string sha(std::span<const std::uint8_t> b) {
  return model_content_identity_hex(sha256_bytes_fast(std::as_bytes(b)));
}
void validate_id(const std::string &s) {
  require(!s.empty() && s.size() <= 128 &&
              std::all_of(s.begin(), s.end(),
                          [](char c) {
                            return (c >= 'a' && c <= 'z') ||
                                   (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '_' ||
                                   c == '-';
                          }),
          "unsafe id/generation");
}
std::string compat_record(std::map<std::string, std::string> fields) {
  Writer w;
  w.text("strix-minimax-session-schema-v1");
  for (const auto &[k, v] : fields) {
    w.text(k);
    w.text(v);
  }
  return std::string(w.b.begin(), w.b.end());
}
// Operational storage, diagnostics and scheduler leases do not change model arithmetic.
// Keep unknown numerical switches conservative: only exclude named operations/path components.
bool compatibility_environment_variable(const std::string &k) {
  if (k.starts_with("STRIX_SESSION_"))
    return false;
  const bool tracked = k.starts_with("STRIX_") || k.starts_with("HIP_") ||
                       k.starts_with("HSA_") || k.starts_with("ROCR_") ||
                       k.starts_with("AMD_");
  if (!tracked || k == "STRIX_HASH_CACHE" ||
      k == "STRIX_MM_STATE_FNV")
    return false;
  std::istringstream parts(k);
  std::string part;
  while (std::getline(parts, part, '_'))
    if (part == "DIR" || part == "DIRS" || part == "DIRECTORY" ||
        part == "DIRECTORIES" || part == "PATH" || part == "PATHS" ||
        part == "LOG" || part == "LOGS" || part == "TRACE" ||
        part == "TRACES" || part == "MEMTRACE" || part == "PROFILE")
      return false;
  return true;
}
namespace {
std::string finish_compatibility(std::map<std::string, std::string> fields,
    const std::filesystem::path& executable, const LoaderSnapshot& snapshot,
    std::chrono::steady_clock::time_point hash_start, FileHashCacheStats stats_before) {
  fields["executable"] = strict_file_sha(executable);
#ifdef _WIN32
  char **environment = _environ;
#else
  char **environment = environ;
#endif
  for (char **e = environment; e && *e; ++e) {
    std::string s = *e;
    auto p = s.find('=');
    if (p == std::string::npos)
      continue;
    auto k = s.substr(0, p);
    if (compatibility_environment_variable(k))
      fields["env:" + k] = s.substr(p + 1);
  }
#ifndef _WIN32
  // Kernel/runtime libraries loaded in executable mappings are part of build
  // identity.
  std::ifstream maps("/proc/self/maps");
  require(bool(maps), "cannot enumerate loaded libraries");
  std::string line;
  std::set<std::string> libraries;
  while (std::getline(maps, line)) {
    std::istringstream row(line);
    std::string address, permissions, offset, device, inode, path;
    row >> address >> permissions >> offset >> device >> inode;
    std::getline(row, path);
    const auto start = path.find_first_not_of(' ');
    if (start == std::string::npos ||
        permissions.find('x') == std::string::npos)
      continue;
    path.erase(0, start);
    if (path.starts_with('/') && path.find(".so") != std::string::npos)
      libraries.insert(path);
  }
  for (const auto &library : libraries)
    fields["library:" + library] = strict_file_sha(library);
#endif
  fields["layout"] = "60x(K[n][256],V[n][256],index[n][128]):bf16-LE;KV-"
                     "sharded;index-replicated;DSpark5x4096x512:bf16";
  verify_loader_snapshot(snapshot);
  const auto stats = file_hash_cache_stats();
  std::fprintf(stderr, "session hashing seconds %.6f cache_hits %llu cache_misses %llu\n",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - hash_start).count(),
      static_cast<unsigned long long>(stats.hits - stats_before.hits),
      static_cast<unsigned long long>(stats.misses - stats_before.misses));
  return compat_record(std::move(fields));
}
} // namespace
std::string compatibility(std::map<std::string, std::string> fields,
                          const std::filesystem::path &model,
                          const std::filesystem::path &executable, AssetLoader loader) {
  [[maybe_unused]] const auto hash_start = std::chrono::steady_clock::now();
  [[maybe_unused]] const auto stats_before = file_hash_cache_stats();
  LoaderSnapshot snapshot;
  std::set<std::filesystem::path> consumed;
  if (loader == AssetLoader::Pack) {
    consumed.insert(model / "model.safetensors.index.json");
    for (const auto& path : assets::pack_files(model)) consumed.insert(path);
  } else {
    for (const auto& path : assets::draft_files(model)) consumed.insert(path);
  }
#ifdef __linux__
  for (const auto& path : consumed) {
    assets::Input input(path, &snapshot);
  }
  return compatibility(std::move(fields), snapshot, executable);
#else
  // CPU-only prewarming fallback; live sessions require loader fd identities.
  std::map<std::string, std::string> roots;
  for (const auto& path : consumed) roots[path.generic_string()] = strict_file_sha(path);
  fields["model-root"] = sha(string_bytes(compat_record(std::move(roots))));
  return finish_compatibility(std::move(fields), executable, {}, hash_start, stats_before);
#endif
}
std::string compatibility(std::map<std::string, std::string> fields,
                          const LoaderSnapshot& snapshot,
                          const std::filesystem::path& executable) {
  const auto hash_start = std::chrono::steady_clock::now();
  const auto stats_before = file_hash_cache_stats();
  require(!snapshot.empty(), "empty loader snapshot");
  std::map<std::string, std::string> roots;
  for (const auto& file : snapshot) {
    const auto digest = model_content_identity_hex(sha256_file_content(file.path, file.identity));
    const auto reference = file.path.generic_string();
    const auto [it, inserted] = roots.emplace(reference, digest);
    if (!inserted && it->second != digest)
      throw std::runtime_error("model files changed during startup");
  }
  fields["model-root"] = sha(string_bytes(compat_record(std::move(roots))));
  return finish_compatibility(std::move(fields), executable, snapshot, hash_start, stats_before);
}

PrewarmReceipt prewarm(const std::filesystem::path &model,
                      const std::filesystem::path &draft,
                      const std::filesystem::path &draft_exl3,
                      const std::filesystem::path &executable,
                      const std::vector<std::string> &extra_files) {
  const auto pass = [&] {
    for (const auto &path : extra_files) (void)strict_file_sha(path);
    (void)compatibility({}, model, executable);
    if (!draft.empty()) (void)compatibility({}, draft, executable, AssetLoader::Draft);
    if (!draft_exl3.empty()) (void)compatibility({}, draft_exl3, executable);
  };
  const auto start = std::chrono::steady_clock::now();
  pass();
  const auto before = file_hash_cache_stats();
  const auto second = std::chrono::steady_clock::now();
  pass();
  const auto after = file_hash_cache_stats();
  require(after.misses == before.misses && after.hits > before.hits,
          "prewarm second pass is not all hits");
  const auto end = std::chrono::steady_clock::now();
  return {std::chrono::duration<double>(end-start).count(),
          std::chrono::duration<double>(end-second).count(), after.hits-before.hits};
}
std::vector<Section> split_state(std::span<const std::uint8_t> b,
                                 std::uint64_t n) {
  require(n <= SIZE_MAX / 76800 && b.size() == n * 76800, "cache-row size");
  std::vector<Section> out;
  std::size_t offset = 0;
  for (unsigned l = 0; l < 60; ++l) {
    for (unsigned k = 0; k < 3; ++k) {
      auto size = static_cast<std::size_t>(n) * (k == 2 ? 256 : 512);
      auto v = b.subspan(offset, size);
      out.push_back({"layer" + std::to_string(l) +
                         (k == 0   ? ".K"
                          : k == 1 ? ".V"
                                   : ".index"),
                     k == 2, Bytes(v.begin(), v.end())});
      offset += size;
    }
  }
  return out;
}
Bytes join_state(const std::vector<Section> &sections) {
  Bytes b;
  for (const auto &s : sections)
    if (s.name.starts_with("layer"))
      b.insert(b.end(), s.bytes.begin(), s.bytes.end());
  return b;
}
Bytes encode(const Checkpoint &c) {
  valid(c);
  Writer w;
  w.text("strix-session-v1");
  w.text("minimax-conversation-v1");
  w.text(c.id);
  w.text(c.generation);
  w.text(c.compat);
  w.text(sha(string_bytes(c.compat)));
  w.u64(c.rank);
  w.u64(c.world);
  frontier(w, c.frontier);
  w.u64(c.history.size());
  for (auto t : c.history)
    w.u64(t);
  w.u64(c.sections.size());
  std::uint64_t offset = 0;
  for (const auto &s : c.sections) {
    w.text(s.name);
    w.u64(c.rank);
    w.u64(s.replicated);
    w.u64(offset);
    w.u64(s.bytes.size());
    w.text(sha(s.bytes));
    offset += s.bytes.size();
  }
  for (const auto &s : c.sections)
    w.raw(s.bytes);
  w.text(sha(w.b));
  return w.b;
}
Checkpoint decode(std::span<const std::uint8_t> b, const std::string &compat,
                  std::uint32_t rank, std::uint32_t world,
                  std::uint64_t max_bytes) {
  require(b.size() <= max_bytes && b.size() >= 72, "envelope size");
  auto body = b.first(b.size() - 72);
  Reader trailer{b.last(72)};
  require(trailer.text() == sha(body), "session_corrupt: manifest root SHA-256");
  Reader r{body};
  require(r.text() == "strix-session-v1" &&
              r.text() == "minimax-conversation-v1",
          "schema");
  Checkpoint c;
  c.id = r.text();
  c.generation = r.text();
  c.compat = r.text();
  require(c.compat == compat && r.text() == sha(string_bytes(compat)),
          "session_incompatible: compat mismatch");
  c.rank = r.u32();
  c.world = r.u32();
  require(c.rank == rank && c.world == world, "TP geometry mismatch");
  auto &f = c.frontier;
  f.consumed = r.u64();
  f.emitted = r.u64();
  f.accepted = r.u64();
  f.drafter = r.u64();
  f.control = r.u64();
  f.pending = r.u32();
  f.pending_valid = r.bit();
  f.drafter_ready = r.bit();
  f.sampling = r.text();
  f.lineage = r.text();
  auto n = r.u64();
  require(n <= body.size() / 8, "history bounds");
  for (std::uint64_t i = 0; i < n; ++i)
    c.history.push_back(r.u32());
  auto count = r.u64();
  require(count <= 512, "section count");
  struct Desc {
    std::uint64_t n;
    std::string hash;
  };
  std::vector<Desc> desc;
  std::uint64_t offset = 0;
  for (std::uint64_t i = 0; i < count; ++i) {
    Section s;
    s.name = r.text();
    require(r.u64() == rank, "section rank");
    s.replicated = r.bit();
    require(r.u64() == offset, "section offset");
    auto size = r.u64();
    require(size <= body.size() - offset, "section size");
    offset += size;
    desc.push_back({size, r.text()});
    c.sections.push_back(std::move(s));
  }
  for (std::size_t i = 0; i < desc.size(); ++i) {
    auto v = r.raw(desc[i].n);
    require(sha(v) == desc[i].hash, "session_corrupt: section SHA-256");
    c.sections[i].bytes.assign(v.begin(), v.end());
  }
  r.end();
  valid(c);
  return c;
}
std::string state_digest(const Checkpoint &c) {
  auto canonical = c;
  canonical.id = "canonical";
  canonical.generation = "canonical";
  return sha(encode(canonical));
}
Bytes read(const std::filesystem::path &p, std::uint64_t max_bytes) {
  auto n = std::filesystem::file_size(p);
  require(n <= max_bytes && n <= std::numeric_limits<std::size_t>::max(),
          "file admission");
  Bytes b(static_cast<std::size_t>(n));
  std::ifstream f(p, std::ios::binary);
  require(bool(f), "open " + p.string());
  if (n)
    f.read(reinterpret_cast<char *>(b.data()), static_cast<std::streamsize>(n));
  require(bool(f) && f.peek() == std::char_traits<char>::eof(),
          "read/truncated/changed file");
  return b;
}
void atomic_write(const std::filesystem::path &p,
                  std::span<const std::uint8_t> b) {
  std::filesystem::create_directories(p.parent_path());
  auto tmp = p;
  tmp += ".tmp-" + random_generation();
#ifdef _WIN32
  auto h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
  require(h != INVALID_HANDLE_VALUE, "create temporary");
  bool ok = true;
  std::size_t pos = 0;
  while (pos < b.size()) {
    DWORD n = 0;
    auto size =
        static_cast<DWORD>(std::min<std::size_t>(b.size() - pos, 1U << 30));
    if (!WriteFile(h, b.data() + pos, size, &n, nullptr) || !n) {
      ok = false;
      break;
    }
    pos += n;
  }
  ok = ok && FlushFileBuffers(h);
  CloseHandle(h);
  require(ok, "write/fsync");
  require(MoveFileExW(tmp.c_str(), p.c_str(),
                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH),
          "rename");
#else
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  require(fd >= 0, "create temporary");
  std::size_t pos = 0;
  bool ok = true;
  while (pos < b.size()) {
    auto n = ::write(fd, b.data() + pos,
                     std::min<std::size_t>(b.size() - pos, 1U << 30));
    if (n <= 0) {
      ok = false;
      break;
    }
    pos += static_cast<std::size_t>(n);
  }
  ok = ok && ::fsync(fd) == 0;
  int closed = ::close(fd);
  require(ok && closed == 0, "write/fsync");
  require(::rename(tmp.c_str(), p.c_str()) == 0, "rename");
  int dir = ::open(p.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
  require(dir >= 0, "directory open");
  ok = ::fsync(dir) == 0;
  closed = ::close(dir);
  require(ok && closed == 0, "directory fsync");
#endif
}
void agree(bool success, const Exchange &x, const std::string &phase) {
  Writer w;
  w.text(phase);
  w.u64(success);
  auto peer = x(w.b);
  Reader r{peer};
  require(r.text() == phase, "collective phase mismatch");
  bool other = r.bit();
  r.end();
  require(success && other, "collective refused: " + phase);
}
Cache::Cache(std::string compat, std::uint32_t rank, std::uint32_t world,
             std::uint64_t budget, std::filesystem::path dir, Exchange x,
             std::uint64_t max_parked, std::uint64_t disk_budget)
    : compat_(std::move(compat)), rank_(rank), world_(world),
      max_parked_(max_parked), disk_budget_(disk_budget), budget_(budget),
      directory_(std::move(dir)), exchange_(std::move(x)) {
  require((world == 1 || world == 2) && rank < world && bool(exchange_),
          "cache geometry/exchange");
}
void Cache::identify(Checkpoint &c) {
  c.rank = rank_;
  c.world = world_;
  c.compat = compat_;
  auto local = string_bytes(random_generation());
  auto peer = exchange_(local);
  c.generation = rank_ == 0 ? std::string(local.begin(), local.end())
                            : std::string(peer.begin(), peer.end());
  bool ok = true;
  try {
    valid(c);
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "capture-frontier-validated");
}
Stats Cache::stats() const {
  Stats s;
  s.entries = entries_.size();
  s.evictions = evictions_;
  for (const auto &[id, e] : entries_) {
    (void)id;
    s.bytes += e.bytes.size();
  }
  std::error_code ec;
  if (!directory_.empty() && std::filesystem::exists(directory_, ec))
    for (const auto& e : std::filesystem::directory_iterator(directory_))
      if (e.is_regular_file() && e.path().extension() == ".session") s.disk_bytes += e.file_size();
  return s;
}
std::uint64_t available_memory() {
#ifdef _WIN32
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof(s);
  require(GlobalMemoryStatusEx(&s), "memory availability unavailable");
  return s.ullAvailPhys;
#else
  std::ifstream f("/proc/meminfo");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream row(line);
    std::string key, unit;
    std::uint64_t kb;
    if (row >> key >> kb >> unit && key == "MemAvailable:" && unit == "kB" &&
        kb <= UINT64_MAX / 1024)
      return kb * 1024;
  }
  throw std::runtime_error("MiniMax session: MemAvailable unavailable");
#endif
}
bool memory_admitted(std::uint64_t parked, std::uint64_t payload,
                     std::uint64_t budget, std::uint64_t available) {
  // Six payloads cover input, split rows, codec growth/copy, verification and
  // output. Keep 256 MiB physical headroom.
  constexpr std::uint64_t reserve = 256ULL << 20;
  if (parked > budget || payload > (budget - parked) / 6 ||
      available <= reserve)
    return false;
  return payload <= (available - reserve) / 6;
}
void Cache::prepare_capture(std::uint64_t payload) {
  const auto available = available_memory();
  require(memory_admitted(0, payload, budget_, available),
          "session_budget: single capture peak exceeds budget/MemAvailable");
  while (!memory_admitted(stats().bytes, payload, budget_, available)) {
    require(!entries_.empty(), "capture admission");
    auto victim = std::min_element(entries_.begin(), entries_.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.second.touched < b.second.touched;
                                   });
    entries_.erase(victim);
    ++evictions_;
  }
}
// A full cache must still restore its own entries: evict other parked sessions (LRU) until the restore peak fits,
// never the one being resumed.
void Cache::prepare_resume(const std::string &id, std::uint64_t payload) {
  const auto available = available_memory();
  // Refuse before evicting anything when even the target alone cannot be restored.
  const auto target = entries_.find(id);
  require(target != entries_.end() && memory_admitted(target->second.bytes.size(), payload, budget_, available),
          "host budget/capture peak/MemAvailable admission refused");
  while (!memory_admitted(stats().bytes, payload, budget_, available)) {
    auto victim = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->first != id && (victim == entries_.end() || it->second.touched < victim->second.touched)) victim = it;
    require(victim != entries_.end(), "host budget/capture peak/MemAvailable admission refused");
    entries_.erase(victim);
    ++evictions_;
  }
}
void Cache::admit(std::uint64_t payload) const {
  require(memory_admitted(stats().bytes, payload, budget_, available_memory()),
          "host budget/capture peak/MemAvailable admission refused");
}
void Cache::capture(Checkpoint c) {
  std::uint64_t bytes = 1048576;
  for (const auto &s : c.sections)
    bytes += s.bytes.size();
  bool ok = true;
  try {
    prepare_capture(bytes);
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "session_budget: capture-admission");
  identify(c);
  Bytes b;
  bool encoded = true;
  try {
    b = encode(c);
  } catch (...) {
    encoded = false;
  }
  agree(encoded, exchange_, "capture-encoded");
  agree(b.size() <= budget_, exchange_, "ram-admission");
  auto own = receipt({c.generation, sha(string_bytes(compat_)), shared(c),
                      sha(b), rank_, world_, b.size()});
  auto other = parse_receipt(world_ == 1 ? own : exchange_(own));
  require(other.generation == c.generation &&
              other.compat == sha(string_bytes(compat_)) &&
              other.shared == shared(c) && other.world == world_ &&
              other.rank == (world_ == 1 ? rank_ : 1 - rank_),
          "RAM peer frontier/replicated mismatch");
  auto old = entries_.find(c.id);
  if (old != entries_.end())
    entries_.erase(old);
  require(max_parked_ > 0, "session_budget: max parked zero");
  while (stats().bytes > budget_ - b.size() || entries_.size() >= max_parked_) {
    auto victim = std::min_element(entries_.begin(), entries_.end(),
                                   [](const auto &a, const auto &v) {
                                     return a.second.touched < v.second.touched;
                                   });
    entries_.erase(victim);
    ++evictions_;
  }
  entries_[c.id] = {std::move(b), ++clock_};
}
void Cache::checkpoint(Checkpoint c) {
  identify(c);
  const auto start = std::chrono::steady_clock::now();
  double encode_seconds = 0, write_seconds = 0;
  Bytes b;
  bool ok = true;
  try {
    std::uint64_t bytes = 1048576;
    for (const auto &s : c.sections)
      bytes += s.bytes.size();
    prepare_capture(bytes);
    b = encode(c);
    encode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    require(b.size() <= budget_ && b.size() <= disk_budget_, "session_budget: disk admission");
  } catch (...) { ok = false; }
  agree(ok, exchange_, "session_budget: disk-encode-admission");
  std::unique_ptr<DirectoryLock> disk_lock;
  try { disk_lock = std::make_unique<DirectoryLock>(directory_); }
  catch (...) { ok = false; }
  agree(ok, exchange_, "session_budget: disk-lock");
  // Keep the lock through admission, eviction, payload write and manifest
  // publication. Existing generations remain charged during replacement.
  // Rank 0 selects the disk LRU victim; both ranks delete the same id. Orphan
  // generations are pruned too, so interrupted publication cannot bypass the cap.
  for (;;) {
    bool fits = true;
    std::string victim;
    try {
      fits = stats().disk_bytes <= disk_budget_ - b.size();
      if (!fits && rank_ == 0) {
        auto oldest = std::filesystem::file_time_type::max();
        for (const auto& e : std::filesystem::directory_iterator(directory_)) {
          if (e.path().extension() != ".session") continue;
          const auto name = e.path().filename().string();
          const auto dot = name.find('.');
          if (dot == std::string::npos) continue;
          if (e.last_write_time() < oldest) { oldest = e.last_write_time(); victim = name.substr(0, dot); }
        }
      }
    } catch (...) { ok = false; }
    agree(ok, exchange_, "session_budget: disk-scan");
    const auto peer_fit = exchange_(Bytes{static_cast<std::uint8_t>(fits)});
    if (fits && !peer_fit.empty() && peer_fit[0]) break;
    if (rank_ == 0 && victim.empty()) {
      // The peer may have more orphan bytes. Select its oldest id if local fits.
      victim = "";
    }
    std::string local_victim = victim;
    try {
    if (rank_ != 0 && !fits) {
      auto oldest = std::filesystem::file_time_type::max();
      for (const auto& e : std::filesystem::directory_iterator(directory_))
        if (e.path().extension() == ".session" && e.last_write_time() < oldest) {
          oldest = e.last_write_time(); local_victim = e.path().filename().string().substr(0, e.path().filename().string().find('.'));
        }
    }
    } catch (...) { ok = false; }
    agree(ok, exchange_, "session_budget: peer-disk-scan");
    auto peer_victim = exchange_(string_bytes(local_victim));
    if (rank_ == 0 && victim.empty()) victim.assign(peer_victim.begin(), peer_victim.end());
    peer_victim = exchange_(string_bytes(victim));
    if (rank_ != 0) victim.assign(peer_victim.begin(), peer_victim.end());
    require(!victim.empty(), "session_budget: no disk LRU victim");
    erase_locked(victim); ++evictions_;
  }
  ok = true;
  try {
    const auto write_start = std::chrono::steady_clock::now();
    atomic_write(directory_ / file_name(c), b);
    write_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - write_start)
                        .count();
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "rank-durable");
  Receipt local{c.generation, sha(string_bytes(compat_)),
                shared(c),    sha(b),
                rank_,        world_,
                b.size()};
  auto own = receipt(local);
  auto peer = world_ == 1 ? own : exchange_(own);
  auto other = parse_receipt(peer);
  require(other.generation == local.generation &&
              other.compat == local.compat && other.shared == local.shared &&
              other.world == world_ &&
              other.rank == (world_ == 1 ? rank_ : 1 - rank_),
          "rank generation/frontier/replicated mismatch");
  Writer w;
  w.text("strix-session-group-v1");
  w.text(c.id);
  w.u64(world_);
  w.u64(own.size());
  w.raw(rank_ == 0 ? own : peer);
  if (world_ == 2) {
    w.u64(peer.size());
    w.raw(rank_ == 0 ? peer : own);
  }
  w.text(sha(w.b));
  ok = true;
  try {
    const auto manifest_start = std::chrono::steady_clock::now();
    atomic_write(manifest(directory_, c.id), w.b);
    write_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - manifest_start).count();
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "manifest-published");
  std::fprintf(
      stderr,
      "rank %u session checkpoint bytes %zu encode_s %.6f write_fsync_s %.6f "
      "total_s %.6f\n",
      rank_, b.size(), encode_seconds, write_seconds,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
}
Checkpoint Cache::resume(const std::string &id, bool disk) {
  double read_seconds = 0;
  const auto start = std::chrono::steady_clock::now();
  validate_id(id);
  Checkpoint c;
  bool ok = true;
  std::string failure;
  Bytes group;
  std::unique_ptr<DirectoryLock> disk_lock;
  try {
    if (disk) disk_lock = std::make_unique<DirectoryLock>(directory_);
    if (!disk) {
      auto it = entries_.find(id);
      require(it != entries_.end(), "session_not_found: RAM miss");
      prepare_resume(id, it->second.bytes.size());
      it = entries_.find(id);
      c = decode(it->second.bytes, compat_, rank_, world_, budget_);
      it->second.touched = ++clock_;
    } else {
      require(on_disk(id), "session_not_found: disk miss");
      const auto manifest_start = std::chrono::steady_clock::now();
      group = read(manifest(directory_, id), 1U << 20);
      read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - manifest_start).count();
      require(group.size() >= 72, "group truncated");
      Reader tail{std::span<const std::uint8_t>(group).last(72)};
      auto body = std::span<const std::uint8_t>(group).first(group.size() - 72);
      require(tail.text() == sha(body), "session_corrupt: group root");
      Reader r{body};
      require(r.text() == "strix-session-group-v1" && r.text() == id &&
                  r.u64() == world_,
              "group identity");
      std::vector<Receipt> rr;
      for (unsigned i = 0; i < world_; ++i)
        rr.push_back(parse_receipt(r.raw(r.u64())));
      r.end();
      for (unsigned i = 0; i < world_; ++i)
        require(rr[i].rank == i && rr[i].world == world_ &&
                    rr[i].generation == rr[0].generation &&
                    rr[i].compat == sha(string_bytes(compat_)) &&
                    rr[i].shared == rr[0].shared,
                "session_incompatible: group rank mismatch");
      Checkpoint key;
      key.id = id;
      key.generation = rr[rank_].generation;
      key.rank = rank_;
      validate_id(key.generation);
      const auto path = directory_ / file_name(key);
      const auto actual = std::filesystem::file_size(path);
      require(actual == rr[rank_].bytes, "session_corrupt: rank file size disagrees with receipt");
      admit(actual);
      // Bound read by the admitted size too: growth after stat cannot allocate more.
      const auto read_start = std::chrono::steady_clock::now();
      auto bytes = read(path, actual);
      read_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - read_start).count();
      require(bytes.size() == rr[rank_].bytes && sha(bytes) == rr[rank_].digest,
              "session_corrupt: rank digest");
      c = decode(bytes, compat_, rank_, world_, std::uint64_t{1} << 40);
      require(c.id == id && c.generation == key.generation &&
                  shared(c) == rr[rank_].shared,
              "rank identity");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rank %u session restore refused: %s\n", rank_,
                 e.what());
    failure = e.what();
    if (failure.find("session_not_found") == std::string::npos && failure.find("session_incompatible") == std::string::npos &&
        failure.find("session_corrupt") == std::string::npos && failure.find("session_budget") == std::string::npos) {
      failure = (failure.find("budget") != std::string::npos || failure.find("admission") != std::string::npos ? "session_budget: " : "session_corrupt: ") + failure;
    }
    ok = false;
  }
  const auto peer_failure = exchange_(string_bytes(failure));
  if (!ok || !peer_failure.empty())
    throw std::runtime_error(!ok ? failure : std::string(peer_failure.begin(), peer_failure.end()));
  auto own = receipt({c.generation, sha(string_bytes(compat_)), shared(c),
                      disk ? sha(group) : state_digest(c), rank_, world_, 0});
  auto other = parse_receipt(world_ == 1 ? own : exchange_(own));
  require(other.generation == c.generation && other.shared == shared(c) &&
              other.compat == sha(string_bytes(compat_)) &&
              (!disk || other.digest == sha(group)),
          "restore peer mismatch");
  if (disk) {
    std::error_code touch_error;
    Checkpoint key; key.id = id; key.generation = c.generation; key.rank = rank_;
    std::filesystem::last_write_time(directory_ / file_name(key), std::filesystem::file_time_type::clock::now(), touch_error);
  }
  std::fprintf(
      stderr, "rank %u session %s read_verify_agree_s %.6f read_io_s %.6f\n", rank_,
      disk ? "restore" : "resume",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count(), read_seconds);
  return c;
}
bool Cache::on_disk(const std::string& id) const {
  std::error_code error;
  return !directory_.empty() && std::filesystem::exists(manifest(directory_, id), error);
}
std::vector<Summary> Cache::describe(bool include_disk) const {
  // Listing must not deserialize all payloads: disk capacity can greatly exceed
  // the RAM budget. Read a bounded header only; resume validates the full file.
  auto header = [](std::span<const std::uint8_t> bytes, std::uint64_t total) {
    Reader r{bytes};
    require(r.text() == "strix-session-v1" && r.text() == "minimax-conversation-v1", "session_corrupt: list schema");
    Summary out; out.id = r.text(); (void)r.text(); (void)r.text(); (void)r.text();
    (void)r.u32(); (void)r.u32();
    out.frontier.consumed = r.u64(); out.frontier.emitted = r.u64();
    out.frontier.accepted = r.u64(); out.frontier.drafter = r.u64();
    out.frontier.control = r.u64(); out.frontier.pending = r.u32();
    out.bytes = total; return out;
  };
  std::vector<Summary> result;
  for (const auto& id : list()) {
    auto it = entries_.find(id);
    if (it != entries_.end()) result.push_back(header(it->second.bytes, it->second.bytes.size()));
    else if (include_disk) {
      auto group = read(manifest(directory_, id), 1U << 20);
      require(group.size() >= 72, "session_corrupt: list group truncated");
      auto body = std::span<const std::uint8_t>(group).first(group.size() - 72);
      Reader trailer{std::span<const std::uint8_t>(group).last(72)};
      require(trailer.text() == sha(body), "session_corrupt: list group checksum");
      Reader r{body};
      require(r.text() == "strix-session-group-v1" && r.text() == id && r.u64() == world_, "session_corrupt: list group identity");
      Receipt receipt_{};
      for (unsigned n = 0; n < world_; ++n) { auto rr = parse_receipt(r.raw(r.u64())); if (n == rank_) receipt_ = rr; }
      validate_id(receipt_.generation);
      Checkpoint key; key.id = id; key.generation = receipt_.generation; key.rank = rank_;
      auto path = directory_ / file_name(key);
      const auto size = std::filesystem::file_size(path);
      Bytes prefix(static_cast<std::size_t>(std::min<std::uint64_t>(size, 4U << 20)));
      std::ifstream stream(path, std::ios::binary);
      stream.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
      require(bool(stream), "session_corrupt: list header read");
      result.push_back(header(prefix, size));
    }
  }
  return result;
}
void Cache::erase(const std::string &id) {
  std::unique_ptr<DirectoryLock> disk_lock;
  bool ok = true;
  try { if (!directory_.empty()) disk_lock = std::make_unique<DirectoryLock>(directory_); }
  catch (...) { ok = false; }
  agree(ok, exchange_, "session_budget: delete-lock");
  erase_locked(id);
}
void Cache::erase_locked(const std::string &id) {
  validate_id(id);
  bool ok = true;
  try {
    entries_.erase(id); // rank payloads are immutable; remove only this
                        // session's local files
    // RAM-only caches have no directory; never resolve paths against the working directory.
    if (!directory_.empty()) std::filesystem::remove(manifest(directory_, id));
    if (!directory_.empty() && std::filesystem::exists(directory_))
      for (const auto &e : std::filesystem::directory_iterator(directory_))
        if (e.is_regular_file()) {
          auto name = e.path().filename().string();
          if (name.starts_with(id + ".") &&
              name.ends_with(".r" + std::to_string(rank_) + ".session"))
            std::filesystem::remove(e.path());
        }
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "delete");
}
std::vector<std::string> Cache::list() const {
  std::set<std::string> ids;
  for (const auto &[id, e] : entries_) {
    (void)e;
    ids.insert(id);
  }
  if (std::filesystem::exists(directory_))
    for (const auto &e : std::filesystem::directory_iterator(directory_))
      if (e.path().extension() == ".group")
        ids.insert(e.path().stem().string());
  return {ids.begin(), ids.end()};
}
} // namespace strix::minimax::session
