#include "minimax/session/runtime.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strix::minimax::session {
namespace {
using Clock = std::chrono::steady_clock;
void need(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(std::string("MiniMax session: ") + s);
}
template <class T> Bytes bytes(const std::vector<T> &v) {
  const auto *p = reinterpret_cast<const std::uint8_t *>(v.data());
  return Bytes(p, p + v.size() * sizeof(T));
}
template <class T> std::vector<T> values(const Bytes &b) {
  need(b.size() % sizeof(T) == 0, "typed section size");
  std::vector<T> v(b.size() / sizeof(T));
  if (!b.empty())
    std::memcpy(v.data(), b.data(), b.size());
  return v;
}
std::vector<std::uint32_t> tokens(const std::string &path) {
  return values<std::uint32_t>(read(path, 16ULL << 20));
}
} // namespace
Runtime::Runtime(engine::Engine &e, dspark::DeviceDrafter *d,
                 const engine::Options &eo, const std::string &exe,
                 RunOptions o, const std::string &draft_dir,
                 const std::string &draft_exl3, const LoaderSnapshot& draft_snapshot)
    : eng_(e), dd_(o.draft ? d : nullptr), options_(std::move(o)) {
  need(options_.budget > 0 && eo.slots >= 2,
       "explicit budget and at least two slots required");
  need(!options_.draft || dd_, "DSpark requested without drafter");
  exchange_ = [this](std::span<const std::uint8_t> b) {
    std::uint64_t n = b.size(), peer = 0;
    eng_.exchange_bytes(&n, &peer, 8);
    need(peer <= 1048576 && n <= 1048576, "control message limit");
    Bytes out(static_cast<std::size_t>(std::max(n, peer))), in(out.size());
    std::copy(b.begin(), b.end(), out.begin());
    for (std::size_t p = 0; p < out.size(); p += 65536)
      eng_.exchange_bytes(out.data() + p, in.data() + p,
                          std::min<std::size_t>(65536, out.size() - p));
    in.resize(static_cast<std::size_t>(peer));
    return in;
  };
  std::map<std::string, std::string> fields{
      {"world", "2"},
      {"heads", "KV:2/rank,Q:32/rank;vocab:100096+99968"},
      {"capacity", std::to_string(eo.capacity)},
      {"max_rows", std::to_string(eo.max_rows)},
      {"slots", std::to_string(eo.slots)},
      {"producer", "single-row-suffix;fixed-spec-width-1-v1"},
      {"draft", dd_ ? "fixed:1" : "off"}};
  // Capture architecture + loaded runtime version through the drafter's
  // HIP-free query.
  fields["gpu-runtime"] = dspark::DeviceDrafter::runtime_identity();
  (void)draft_dir;
  auto snapshot = e.loader_snapshot();
  const auto exl3_snapshot = e.draft_loader_snapshot();
  if (dd_) {
    need(!draft_snapshot.empty(), "missing drafter loader snapshot");
    fields["draft-assets"] = compatibility({}, draft_snapshot, exe);
    if (!draft_exl3.empty())
      fields["draft-exl3-assets"] = compatibility({}, exl3_snapshot, exe);
  }
  auto compat = compatibility(std::move(fields), snapshot, exe);
  snapshot.insert(snapshot.end(), exl3_snapshot.begin(), exl3_snapshot.end());
  if (dd_) snapshot.insert(snapshot.end(), draft_snapshot.begin(), draft_snapshot.end());
  verify_loader_snapshot(snapshot);
  // Construction stays local: startup agrees success before any cache operation
  // can enter the peer channel. Do not add constructor collectives here.
  cache_ =
      std::make_unique<Cache>(std::move(compat), e.rank(), 2, options_.budget,
                              options_.disk_enabled ? options_.directory : std::string{}, exchange_,
                              options_.max_parked, options_.disk_budget);
  // Taps are scoped to named requests; construction leaves ordinary serving untouched.
}
bool Runtime::begin_request(const std::string& id, std::span<const std::uint32_t> prompt) {
  validate_id(id);
  // Only an exact extension of the consumed + emitted-pending history is a hit.
  const auto exists = cache_->in_ram(id) || (options_.disk_enabled && cache_->on_disk(id));
  const auto peer_exists = eng_.peer(exists ? 1U : 0U);
  if (exists && peer_exists) {
    restore_auto(id);
    const auto prefix = history_.size() + (frontier_.control ? 1U : 0U);
    if (prompt.size() >= prefix && std::equal(history_.begin(), history_.end(), prompt.begin()) &&
        (!frontier_.control || prompt[history_.size()] == frontier_.pending)) {
      suffix(prompt.subspan(prefix));
      return true;
    }
  }
  start(prompt);
  return false;
}
void Runtime::start(std::span<const std::uint32_t> prompt) {
  need(!prompt.empty() && std::all_of(prompt.begin(), prompt.end(),
                                       [](auto t) { return t < 200064; }),
       "prompt tokens");
  need(prompt.size() < eng_.capacity(), "context_length_exceeded: prompt does not fit the context");
  eng_.set_taps(dd_ != nullptr);
  if (dd_)
    dd_->reset_context(slot_);
  if (dd_)
    eng_.set_prefill_tap_sink(
        [this](const float *taps, std::size_t n, std::uint32_t s,
               std::uint64_t p) { dd_->append(s, taps, n, p, nullptr); });
  auto t = Clock::now();
  auto next = eng_.prefill(slot_, prompt, 0);
  eng_.set_prefill_tap_sink({});
  history_.assign(prompt.begin(), prompt.end());
  frontier_ = {};
  frontier_.consumed = frontier_.accepted = prompt.size();
  frontier_.pending = next;
  frontier_.drafter_ready = dd_ != nullptr;
  frontier_.drafter = dd_ ? prompt.size() : 0;
  logits_ = eng_.last_logits(1);
  windows_ = verified_ = accepted_ = 0;
  active_ = true;
  std::fprintf(
      stderr, "rank %u session cold_prefill_P_s %.6f tokens %zu\n", eng_.rank(),
      std::chrono::duration<double>(Clock::now() - t).count(), prompt.size());
}
std::vector<std::uint32_t> Runtime::step(std::span<const engine::Row> rows, std::span<const std::uint32_t> stops) {
  auto g = eng_.step(rows);
  auto all = eng_.last_logits(rows.size());
  auto consumed = std::size_t{1};
  if (rows.size() > 1 && rows[1].token == g[0] && std::find(stops.begin(), stops.end(), g[0]) == stops.end())
    ++consumed;
  if (dd_) {
    dd_->append(slot_, eng_.last_taps(), consumed, frontier_.consumed, nullptr);
    dspark::DeviceDrafter::device_synchronize();
  }
  for (std::size_t i = 0; i < consumed; ++i)
    history_.push_back(rows[i].token);
  frontier_.consumed += consumed;
  frontier_.accepted = frontier_.consumed;
  frontier_.drafter = dd_ ? frontier_.consumed : 0;
  frontier_.pending = g[consumed - 1];
  logits_.assign(all.begin() + static_cast<std::ptrdiff_t>(
                                   (consumed - 1) * eng_.vocab_range().second),
                 all.begin() + static_cast<std::ptrdiff_t>(
                                   consumed * eng_.vocab_range().second));
  g.resize(consumed);
  return g;
}
void Runtime::suffix(std::span<const std::uint32_t> input) {
  const auto begin = Clock::now();
  need(active_, "no active session");
  need(std::all_of(input.begin(), input.end(),
                   [](auto token) { return token < 200064; }),
       "suffix token");
  need(frontier_.consumed < eng_.capacity() &&
           input.size() < eng_.capacity() - frontier_.consumed,
       "context_length_exceeded: session history plus the new turn does not fit the context");
  // Consume the pending token only when it was emitted; after a suffix the
  // pending prediction is un-emitted.
  if (frontier_.control) {
    engine::Row row{slot_, frontier_.pending, frontier_.consumed};
    (void)step(std::span(&row, 1));
  }
  for (auto token : input) {
    need(token < 200064, "suffix token");
    engine::Row row{slot_, token, frontier_.consumed};
    (void)step(std::span(&row, 1));
  }
  frontier_.control = 0;
  std::fprintf(stderr, "rank %u session suffix_input_s %.6f tokens %zu\n",
               eng_.rank(),
               std::chrono::duration<double>(Clock::now() - begin).count(),
               input.size());
}
std::uint32_t Runtime::draft() {
  dd_->forward_block(slot_, eng_.embed_row(frontier_.pending),
                     eng_.embed_row(dspark::noise_token), frontier_.consumed,
                     nullptr);
  const auto range = eng_.vocab_range();
  auto base = eng_.head_logits(dd_->y(), dspark::block);
  auto mine = dd_->markov_step_host(frontier_.pending, base, range.first,
                                    range.first + range.second, nullptr);
  auto other = eng_.peer2(std::bit_cast<std::uint32_t>(mine.value), mine.id);
  return dspark::combine(mine, {std::bit_cast<float>(other[0]), other[1]}).id;
}
std::uint64_t Runtime::generation_room() const {
  // generate() keeps a DSpark block plus one row of headroom past the last generated token.
  const auto used = frontier_.consumed + dspark::block + 1;
  return active_ && used < eng_.capacity() ? eng_.capacity() - used : 0;
}
std::vector<std::uint32_t> Runtime::generate(unsigned count,
                                             const std::string &dump,
    std::span<const std::uint32_t> stops,
    const std::function<bool(std::span<const std::uint32_t>)>& publish) {
  const auto begin = Clock::now();
  need(active_ && count > 0, "generation bounds");
  need(count <= generation_room(), "context_length_exceeded: no room left in the context to generate");
  std::ofstream ids, lg;
  if (!dump.empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(dump).parent_path());
    ids.open(dump + ".u32", std::ios::binary);
    lg.open(dump + ".f32", std::ios::binary);
    need(bool(ids) && bool(lg), "dump open");
  }
  unsigned emitted = 0;
  bool stopped = false;
  auto publish_chunk = [&](std::span<const std::uint32_t> chunk) {
    const bool local = eng_.rank() != 0 || !publish || publish(chunk);
    const auto peer = eng_.peer(local ? 1U : 0U);
    if (!local || !peer) stopped = true;
  };
  std::vector<std::uint32_t> result;
  auto emit = [&](std::uint32_t token, std::span<const float> row) {
    if (ids.is_open()) {
      ids.write(reinterpret_cast<const char *>(&token), 4);
      lg.write(reinterpret_cast<const char *>(row.data()),
               static_cast<std::streamsize>(row.size() * 4));
      need(bool(ids) && bool(lg), "dump write");
    }
    result.push_back(token);
    ++emitted;
    ++frontier_.emitted;
    frontier_.control = 1;
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) stopped = true;
  };
  if (!frontier_.control) {
    emit(frontier_.pending, logits_);
    if (publish || !stops.empty()) publish_chunk(std::span(result).last(1));
  }
  while (emitted < count && !stopped) {
    engine::Row rows[2] = {{slot_, frontier_.pending, frontier_.consumed},
                           {slot_, 0, frontier_.consumed + 1}};
    unsigned n = 1;
    if (dd_ && count - emitted >= 2) {
      rows[1].token = draft();
      n = 2;
      ++windows_;
      ++verified_;
    }
    auto g = step(std::span(rows, n), stops);
    auto all = eng_.last_logits(n);
    if (n == 2)
      accepted_ += g.size() - 1;
    for (std::size_t i = 0; i < g.size(); ++i)
      emit(g[i], std::span(all).subspan(i * eng_.vocab_range().second,
                                        eng_.vocab_range().second));
    if (publish || !stops.empty()) publish_chunk(std::span(result).last(g.size()));
  }
  std::fprintf(
      stderr,
      "rank %u session continuation windows %llu verified %llu accepted %llu\n",
      eng_.rank(), static_cast<unsigned long long>(windows_),
      static_cast<unsigned long long>(verified_),
      static_cast<unsigned long long>(accepted_));
  std::fprintf(
      stderr, "rank %u session generation_s %.6f tokens %u\n", eng_.rank(),
      std::chrono::duration<double>(Clock::now() - begin).count(), emitted);
  return result;
}
Checkpoint Runtime::capture(const std::string &id) {
  need(active_ && !eng_.prefill_open(), "capture requires committed boundary");
  auto size = frontier_.consumed * 76800 + eng_.vocab_range().second * 4 +
              (dd_ ? 5ULL * 4096 * 512 * 4 + 4096 * 8 + 32 : 0) + 1048576;
  bool ok = true;
  try {
    cache_->prepare_capture(size);
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "session_budget: runtime-capture-admission");
  Checkpoint c;
  c.id = id;
  c.generation = "unpublished";
  c.frontier = frontier_;
  c.history = history_;
  c.rank = eng_.rank();
  c.world = 2;
  Bytes raw(static_cast<std::size_t>(frontier_.consumed * 76800));
  eng_.export_rows(slot_, 0, frontier_.consumed, raw.data());
  c.sections = split_state(raw, frontier_.consumed);
  c.sections.push_back({"pending.logits", false, bytes(logits_)});
  if (dd_) {
    dd_->flush(slot_, nullptr);  // rows staged by the serve step join the context before it is exported
    auto s = dd_->export_context(slot_);
    c.sections.push_back({"draft.K", true, bytes(s.k)});
    c.sections.push_back({"draft.V", true, bytes(s.v)});
    c.sections.push_back({"draft.positions", true, bytes(s.positions)});
    c.sections.push_back(
        {"draft.continuation", true,
         bytes(std::vector<std::uint64_t>{1, windows_, verified_, accepted_})});
  }
  return c;
}
void Runtime::checkpoint(const std::string &id, bool disk) {
  agree(!disk || options_.disk_enabled, exchange_, "session_budget: disk tier disabled");
  auto t = Clock::now();
  auto c = capture(id);
  auto capture_s = std::chrono::duration<double>(Clock::now() - t).count();
  if (disk)
    cache_->checkpoint(std::move(c));
  else
    cache_->capture(std::move(c));
  std::fprintf(stderr,
               "rank %u session capture_s %.6f total_checkpoint_s %.6f\n",
               eng_.rank(), capture_s,
               std::chrono::duration<double>(Clock::now() - t).count());
}
void Runtime::park(const std::string &id) {
  checkpoint(id, false);
  active_ = false;
  std::vector<std::uint32_t>().swap(history_);
  std::vector<float>().swap(logits_);
}
void Runtime::erase(const std::string& id) {
  const auto local = cache_->in_ram(id) || cache_->on_disk(id);
  const auto peer = eng_.peer(local ? 1U : 0U);
  need(local || peer, "session_not_found: unknown session");
  cache_->erase(id);
}
void Runtime::restore_auto(const std::string& id) {
  const auto local = (cache_->in_ram(id) ? 1U : 0U) | (options_.disk_enabled && cache_->on_disk(id) ? 2U : 0U);
  const auto both = local & eng_.peer(local);
  need(both != 0, "session_not_found: no shared checkpoint");
  restore(id, (both & 1U) == 0);
}
void Runtime::restore(const std::string &id, bool disk) {
  auto begin = Clock::now();
  agree(!disk || options_.disk_enabled, exchange_, "session_budget: disk tier disabled");
  eng_.set_taps(dd_ != nullptr);
  auto c = cache_->resume(id, disk);
  auto copied = Clock::now();
  bool ok = true;
  auto staging = 1U - slot_;
  try {
    need(c.frontier.consumed <= eng_.capacity() &&
             c.frontier.drafter_ready == (dd_ != nullptr),
         "restore capacity/draft");
    auto raw = join_state(c.sections);
    eng_.import_rows(staging, 0, c.frontier.consumed, raw.data());
    if (dd_) {
      dspark::ContextSnapshot s;
      s.k = values<std::uint16_t>(c.sections[181].bytes);
      s.v = values<std::uint16_t>(c.sections[182].bytes);
      s.positions = values<std::int64_t>(c.sections[183].bytes);
      auto counters = values<std::uint64_t>(c.sections[184].bytes);
      need(counters[0] == 1, "draft continuation policy");
      dd_->import_context(staging, s, c.frontier.consumed);
    }
  } catch (...) {
    ok = false;
  }
  agree(ok, exchange_, "import-unpublished-slot");
  slot_ = staging;
  active_ = true;
  frontier_ = c.frontier;
  history_ = std::move(c.history);
  logits_ = values<float>(c.sections[180].bytes);
  if (dd_) {
    auto v = values<std::uint64_t>(c.sections[184].bytes);
    windows_ = v[1];
    verified_ = v[2];
    accepted_ = v[3];
  }
  std::fprintf(
      stderr,
      "rank %u session restore_read_verify_s %.6f copy_publish_s %.6f\n",
      eng_.rank(), std::chrono::duration<double>(copied - begin).count(),
      std::chrono::duration<double>(Clock::now() - copied).count());
}
std::string Runtime::digest() {
  auto c = capture("canonical");
  c.compat = "canonical-state";
  return state_digest(c);
}
bool parse_option(RunOptions &o, const std::string &a,
                  const std::function<std::string()> &next) {
  if (a == "--baseline") {}
  else if (a == "--session-cache" || a == "--session-disk" || a == "--session-ram-bytes" || a == "--session-max-parked" || a == "--session-dir" || a == "--session-disk-bytes") { (void)next(); }
  else if (a == "--session-host-budget")
    o.budget = std::stoull(next());
  else if (a == "--session-cache-dir")
    o.directory = next();
  else if (a == "--session-id")
    o.id = next();
  else if (a == "--session-save") {
    o.save = next();
    o.directory = o.save;
  } else if (a == "--session-load") {
    o.load = next();
    o.directory = o.load;
  } else if (a == "--session-prompt")
    o.prompt = next();
  else if (a == "--session-suffix")
    o.suffix = next();
  else if (a == "--session-control-output")
    o.control_output = next();
  else if (a == "--session-output")
    o.output = next();
  else if (a == "--session-generate")
    o.generate = static_cast<unsigned>(std::stoul(next()));
  else if (a == "--session-suffix-generate")
    o.suffix_generate = static_cast<unsigned>(std::stoul(next()));
  else if (a == "--session-dspark")
    o.draft = true;
  else
    return false;
  return true;
}
void run_cli(Runtime &r, const RunOptions &o) {
  if (!o.load.empty())
    r.restore(o.id, true);
  else {
    r.start(tokens(o.prompt));
    r.generate(o.generate);
  }
  if (!o.save.empty())
    r.checkpoint(o.id, true);
  if (!o.suffix.empty()) {
    r.suffix(tokens(o.suffix));
    r.generate(o.suffix_generate, o.output);
  }
  if (!o.output.empty()) {
    auto d = r.digest();
    atomic_write(o.output + ".sha256", Bytes(d.begin(), d.end()));
  }
  if (!o.control_output.empty()) {
    need(o.load.empty() && !o.prompt.empty() && !o.suffix.empty(),
         "never-saved control needs prompt and suffix");
    r.start(tokens(o.prompt));
    r.generate(o.generate); // This control session is never captured before
                            // continuation.
    r.suffix(tokens(o.suffix));
    r.generate(o.suffix_generate, o.control_output);
    auto d = r.digest();
    atomic_write(o.control_output + ".sha256", Bytes(d.begin(), d.end()));
  }
}
} // namespace strix::minimax::session
