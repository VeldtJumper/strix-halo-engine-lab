#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "strix/minimax/session_cache.hpp"
#include "strix/minimax/session_startup.hpp"
#include "strix/minimax/model_assets.hpp"
#include "strix/model_content_hash.hpp"
#ifndef _WIN32
#include "minimax/engine/mm_common.hpp"
#endif
#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <exception>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <sys/file.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#endif
using namespace strix::minimax::session;
void check(bool x, const char *why) {
  if (!x)
    throw std::runtime_error(why);
}
template <class F> void refuses(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  check(failed, "expected refusal");
}
Checkpoint sample(unsigned rank = 0, unsigned world = 1) {
  Checkpoint c;
  c.id = "A";
  c.generation = "test";
  c.compat = compat_record({{"mode", "exact"}});
  c.rank = rank;
  c.world = world;
  c.history = {3, 4, 5};
  c.frontier.consumed = c.frontier.accepted = 3;
  c.frontier.emitted = 1;
  c.frontier.pending = 6;
  c.sections = split_state(Bytes(3 * 76800, 0), 3);
  c.sections[0].bytes[0] = static_cast<std::uint8_t>(rank);
  c.sections.push_back(
      {"pending.logits", false, Bytes((rank == 0 ? 100096 : 99968) * 4, 0)});
  return c;
}
// The same shape with `positions` state rows: large enough that the 1 MiB capture margin is small next to the payload.
Checkpoint sized_sample(const std::string &id, std::uint64_t positions) {
  auto c = sample();
  c.id = id;
  c.history.assign(positions, 3);
  c.frontier.consumed = c.frontier.accepted = positions;
  c.sections = split_state(Bytes(positions * 76800, 0), positions);
  c.sections.push_back({"pending.logits", false, Bytes(100096 * 4, 0)});
  return c;
}
struct Pair {
  std::barrier<> sync{2};
  std::array<Bytes, 2> messages;
  Exchange at(unsigned rank) {
    return [this, rank](std::span<const std::uint8_t> b) {
      messages[rank].assign(b.begin(), b.end());
      sync.arrive_and_wait();
      auto peer = messages[1 - rank];
      sync.arrive_and_wait();
      return peer;
    };
  }
};
template <class F> void pair_run(F f) {
  std::mutex watchdog_mutex;
  std::condition_variable watchdog_cv;
  bool done = false;
  std::thread watchdog([&] {
    std::unique_lock<std::mutex> lock(watchdog_mutex);
    if (!watchdog_cv.wait_for(lock, std::chrono::seconds(10), [&] { return done; })) {
      std::cerr << "FAIL paired harness timeout (never a refusal)\n";
      std::abort();
    }
  });
  std::array<std::exception_ptr, 2> errors;
  std::thread a([&] {
    try {
      f(0);
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  std::thread b([&] {
    try {
      f(1);
    } catch (...) {
      errors[1] = std::current_exception();
    }
  });
  a.join();
  b.join();
  { std::lock_guard<std::mutex> lock(watchdog_mutex); done = true; }
  watchdog_cv.notify_one();
  watchdog.join();
  for (auto &e : errors)
    if (e)
      std::rethrow_exception(e);
}
void test_pair_startup() {
  for (bool explicit_on : {false, true}) {
    for (unsigned refusing_rank : {0U, 1U}) {
      for (bool fail_factory : {false, true}) {
        std::barrier sync(2);
        std::array<StartupVote, 2> messages;
        std::array<std::string, 2> logs, failures;
        std::array<unsigned, 2> calls{};
        pair_run([&](unsigned rank) {
          auto exchange = [&](const StartupVote& local) {
            messages[rank] = local; sync.arrive_and_wait();
            auto peer = messages[1 - rank]; sync.arrive_and_wait(); return peer;
          };
          try {
            auto rt = admit_runtime_pair(
                {fail_factory || rank != refusing_rank, explicit_on && rank == 0, rank == refusing_rank ? "resource refusal" : ""},
                rank, exchange, [&]() -> std::unique_ptr<int> {
                  ++calls[rank];
                  if (fail_factory && rank == refusing_rank) throw std::runtime_error("loader snapshot refused");
                  return std::make_unique<int>(1);
                }, [&](const std::string& line) { logs[rank] = line; });
            check(!rt, "both ranks disable sessions on refusal");
          } catch (const std::exception& e) { failures[rank] = e.what(); }
        });
        check(logs[0] == logs[1] && !logs[0].empty(), "rank-agreed refusal reason");
        check(failures[0] == failures[1], "both ranks exit with same error");
        check(explicit_on == !failures[0].empty(), "only explicit startup refusal exits");
        check(calls[0] == (fail_factory ? 1U : 0U) && calls[1] == calls[0], "shared construction admission");
      }
    }
  }
  {
    std::barrier sync(2);
    std::array<StartupVote, 2> messages;
    pair_run([&](unsigned rank) {
      auto exchange = [&](const StartupVote& local) {
        messages[rank] = local; sync.arrive_and_wait();
        auto peer = messages[1 - rank]; sync.arrive_and_wait(); return peer;
      };
      auto rt = admit_runtime_pair({true, false, ""}, rank, exchange,
          [] { return std::make_unique<int>(7); },
          [](const std::string& line) { check(line == "session cache: on (runtime live)", "both ranks publish runtime"); });
      check(rt && *rt == 7, "successful rank-agreed construction");
      rt = admit_runtime_pair({false, false, rank == 0 ? "first refusal" : "second refusal"}, rank, exchange,
          [] { return std::make_unique<int>(7); },
          [](const std::string& line) { check(line == "session cache: off (first refusal)", "deterministic competing reasons"); });
      check(!rt, "both refused without construction");
    });
  }
  std::barrier sync(2);
  std::array<StartupVote, 2> messages;
  pair_run([&](unsigned rank) {
    auto exchange = [&](const StartupVote& local) {
      messages[rank] = local; sync.arrive_and_wait();
      auto peer = messages[1 - rank]; sync.arrive_and_wait(); return peer;
    };
    auto rt = admit_runtime_pair({rank == 0, true, rank == 1 ? "baseline" : ""}, rank, exchange,
        [] { throw std::runtime_error("baseline must not construct"); return std::unique_ptr<int>{}; },
        [](const std::string& line) { check(line == "session cache: off (baseline)", "peer baseline wins"); });
    check(!rt, "baseline disables both ranks");
  });
}
int main() {
  try {
    test_pair_startup();
    std::string startup_log;
    auto log = [&](const std::string &line) { startup_log = line; };
    auto fail_start = []() -> std::unique_ptr<int> { throw std::runtime_error("injected failure"); };
    check(!start_runtime(fail_start, false, log), "automatic failure stayed enabled");
    check(startup_log == "session cache: off (injected failure)", "automatic failure reason");
    refuses([&] { start_runtime(fail_start, true, log); });
    check(startup_log == "session cache: initialization failed (injected failure)", "explicit failure diagnostic");
    check(*start_runtime([] { return std::make_unique<int>(1); }, true, log) == 1,
          "runtime startup failed");
    check(startup_log == "session cache: on (runtime live)", "missing live receipt");
    auto dir = std::filesystem::current_path() / "minimax-session-cpu-fixture";
    check(!std::filesystem::exists(dir),
          "fixture directory already exists; move it before testing");
    std::filesystem::create_directories(dir);
    check(
        sha(Bytes{'a', 'b', 'c'}) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA vector");
    auto c = sample();
    auto bytes = encode(c);
    auto restored = decode(bytes, c.compat, 0, 1, bytes.size());
    check(encode(restored) == bytes, "round trip");
    refuses([&] { decode(bytes, "wrong", 0, 1, bytes.size()); });
    refuses([&] { decode(bytes, c.compat, 1, 2, bytes.size()); });
    refuses([&] { decode(bytes, c.compat, 0, 1, bytes.size() - 1); });
    auto section_bad = bytes;
    section_bad[section_bad.size() - 80] ^= 1;
    auto root = sha(std::span(section_bad).first(section_bad.size() - 72));
    std::copy(root.begin(), root.end(), section_bad.end() - 64);
    refuses([&] { decode(section_bad, c.compat, 0, 1, section_bad.size()); });
    auto corrupt = bytes;
    corrupt[corrupt.size() - 80] ^= 1;
    refuses([&] { decode(corrupt, c.compat, 0, 1, corrupt.size()); });
    corrupt = bytes;
    corrupt.resize(40);
    refuses([&] { decode(corrupt, c.compat, 0, 1, bytes.size()); });
    refuses([&] { validate_id("../escape"); });
    // Preserve signed zero / NaN payload bits as opaque bytes; no float
    // conversion in codec.
    std::copy_n(Bytes{0, 0, 0, 128, 1, 0, 192, 127}.begin(), 8,
                c.sections[180].bytes.begin());
    check(decode(encode(c), c.compat, 0, 1, 1000000).sections[180].bytes ==
              c.sections[180].bytes,
          "FP32 bits");
    auto d = c;
    d.generation = "different";
    check(state_digest(c) == state_digest(d), "canonical generation");
    Bytes raw(5 * 76800, 0xA5);
    auto sections = split_state(raw, 5);
    check(join_state(sections) == raw, "cache row split/join");
    refuses([&] { split_state(std::span(raw).first(raw.size() - 1), 5); });
    check(!memory_admitted(0, UINT64_MAX, UINT64_MAX, UINT64_MAX),
          "overflow admission");
    check(!memory_admitted(100, 100, 699, UINT64_MAX), "capture peak budget");
    check(!memory_admitted(0, 100, 1000, 256ULL << 20), "available headroom");
    check(memory_admitted(100, 100, 700, UINT64_MAX), "capture peak boundary");
    auto incomplete = c;
    incomplete.sections.pop_back();
    refuses([&] { encode(incomplete); });
    auto draft = c;
    draft.frontier.drafter_ready = true;
    draft.frontier.drafter = 3;
    draft.sections.push_back(
        {"draft.K", true, Bytes(5ULL * 4096 * 512 * 2, 0)});
    draft.sections.push_back(
        {"draft.V", true, Bytes(5ULL * 4096 * 512 * 2, 0)});
    Bytes positions(4096 * 8, 255);
    for (unsigned i = 0; i < 3; ++i) {
      std::fill_n(positions.begin() + i * 8, 8, std::uint8_t{0});
      positions[i * 8] = static_cast<std::uint8_t>(i);
    }
    draft.sections.push_back({"draft.positions", true, std::move(positions)});
    Bytes continuation(32, 0);
    continuation[0] = 1;
    draft.sections.push_back(
        {"draft.continuation", true, std::move(continuation)});
    auto draft_bytes = encode(draft);
    check(encode(decode(draft_bytes, draft.compat, 0, 1, draft_bytes.size())) ==
              draft_bytes,
          "DSpark ring round trip");
    draft.sections[183].bytes[0] = 9;
    refuses([&] { encode(draft); });
    Exchange self = [](std::span<const std::uint8_t> b) {
      return Bytes(b.begin(), b.end());
    };
    Cache cache(c.compat, 0, 1, 16ULL << 20, dir / "single", self);
    cache.capture(c);
    auto b = c;
    b.id = "B";
    cache.capture(b);
    cache.resume("A", false);
    auto e = c;
    e.id = "C";
    cache.capture(e);
    cache.resume("B", false);
    check(cache.stats().entries == 3, "RAM entries");
    cache.checkpoint(c);
    check(state_digest(cache.resume("A", true)) == state_digest(c),
          "disk restart");
    std::uint64_t payload = 1048576;
    for (const auto &sec : c.sections)
      payload += sec.bytes.size();
    Cache lru(c.compat, 0, 1, 6 * payload + encode(c).size() * 3 / 2,
              dir / "lru", self);
    lru.capture(c);
    lru.capture(b);
    lru.resume("A", false);
    lru.capture(e);
    refuses([&] { lru.resume("B", false); });
    check(lru.stats().evictions == 1, "peak-aware LRU");
    {
      // Saturated cache: the second capture fits (budget >= 6P + E), but resuming the first with both parked needs
      // 8E > budget; only the other entry may go (budget >= 7E).
      const auto big_a = sized_sample("A", 100), big_b = sized_sample("B", 100);
      std::uint64_t big_payload = 1048576;
      for (const auto &sec : big_a.sections)
        big_payload += sec.bytes.size();
      const std::uint64_t entry = encode(big_a).size();
      const std::uint64_t budget = std::max(6 * big_payload + entry, 7 * entry) + 4096;
      check(budget < 8 * entry, "saturated-resume fixture is tight enough");
      Cache full(big_a.compat, 0, 1, budget, dir / "full", self);
      full.capture(big_a);
      full.capture(big_b);
      check(full.in_ram("A") && full.in_ram("B") && full.stats().evictions == 0, "both entries parked");
      check(state_digest(full.resume("A", false)) == state_digest(big_a) && !full.in_ram("B") &&
                full.stats().evictions == 1, "full cache resumes by evicting another entry, never the target");
    }
    Cache tiny(c.compat, 0, 1, 1, dir / "tiny", self);
    refuses([&] { tiny.capture(c); });
    check(tiny.stats().bytes == 0, "oversized admission");
    Pair pair;
    std::array<std::unique_ptr<Cache>, 2> ranks;
    for (unsigned r = 0; r < 2; ++r)
      ranks[r] = std::make_unique<Cache>(c.compat, r, 2, 32ULL << 20,
                                         dir / ("rank" + std::to_string(r)),
                                         pair.at(r));
    pair_run([&](unsigned r) { ranks[r]->checkpoint(sample(r, 2)); });
    pair_run([&](unsigned r) {
      auto got = ranks[r]->resume("A", true);
      check(got.sections[0].bytes[0] == r, "rank shard");
    });
    auto both_refuse = [&](auto restore) {
      std::array<bool, 2> refused{};
      pair_run([&](unsigned r) {
        try { restore(r); } catch (const std::exception &) { refused[r] = true; }
      });
      check(refused[0] && refused[1], "explicit refusal required on BOTH ranks");
    };
    // Exercise truncation, enlarged payload and section hash on either rank.
    for (unsigned victim = 0; victim < 2; ++victim) {
      for (unsigned kind = 0; kind < 3; ++kind) {
        pair_run([&](unsigned r) { ranks[r]->checkpoint(sample(r, 2)); });
        for (const auto &path : std::filesystem::directory_iterator(dir / ("rank" + std::to_string(victim)))) {
          if (path.path().extension() != ".session") continue;
          auto original = read(path.path(), 10000000);
          auto damaged = original;
          if (kind == 0) damaged.resize(40);
          else if (kind == 1) damaged.resize(damaged.size() + 1048576, 0);
          else {
            damaged[damaged.size() - 80] ^= 1;
            auto digest = sha(std::span(damaged).first(damaged.size() - 72));
            std::copy(digest.begin(), digest.end(), damaged.end() - 64);
            // Repair outer receipts/root so section validation is reached.
            auto old_hash = sha(original), new_hash = sha(damaged);
            for (unsigned r = 0; r < 2; ++r) {
              auto group_path = dir / ("rank" + std::to_string(r)) / "A.group";
              auto group = read(group_path, 100000);
              auto pos = std::search(group.begin(), group.end(), old_hash.begin(), old_hash.end());
              if (pos == group.end()) continue; // older orphan payload
              std::copy(new_hash.begin(), new_hash.end(), pos);
              auto hash = sha(std::span(group).first(group.size() - 72));
              std::copy(hash.begin(), hash.end(), group.end() - 64);
              atomic_write(group_path, group);
            }
          }
          atomic_write(path.path(), damaged);
        }
        both_refuse([&](unsigned r) { ranks[r]->resume("A", true); });
      }
    }
    // A locally corrupted durable rank refuses both readers before publication.
    for (const auto &path : std::filesystem::directory_iterator(dir / "rank1"))
      if (path.path().extension() == ".session") {
        auto damaged = read(path.path(), 10000000);
        damaged[damaged.size() / 2] ^= 1;
        atomic_write(path.path(), damaged);
      }
    pair_run(
        [&](unsigned r) { refuses([&] { ranks[r]->resume("A", true); }); });
    pair_run([&](unsigned r) { ranks[r]->checkpoint(sample(r, 2)); });
    // Each rank validates locally; either rank's missing file refuses the group
    // on BOTH ranks.
    for (const auto &path : std::filesystem::directory_iterator(dir / "rank1"))
      if (path.path().extension() == ".session")
        std::filesystem::remove(path.path());
    pair_run(
        [&](unsigned r) { refuses([&] { ranks[r]->resume("A", true); }); });
    // Differing replicated bytes prevent group publication.
    pair_run([&](unsigned r) {
      auto bad = sample(r, 2);
      bad.id = "bad";
      bad.sections[2].bytes[0] = static_cast<std::uint8_t>(r);
      refuses([&] { ranks[r]->checkpoint(bad); });
    });
    check(!std::filesystem::exists(dir / "rank0/bad.group"),
          "invalid manifest published");
    cache.erase("A");
    refuses([&] { cache.resume("A", true); });
    // Different valid generations across nodes must not restore as one group.
    pair_run([&](unsigned r) { ranks[r]->checkpoint(sample(r, 2)); });
    auto old_group = read(dir / "rank1/A.group", 100000);
    pair_run([&](unsigned r) { ranks[r]->checkpoint(sample(r, 2)); });
    atomic_write(dir / "rank1/A.group", old_group);
    pair_run(
        [&](unsigned r) { refuses([&] { ranks[r]->resume("A", true); }); });
    std::filesystem::create_directories(dir / "model");
    Bytes weights(4096, 1);
    atomic_write(dir / "model/weights", weights);
    const std::string initial_index = R"({"weight_map":{"tensor":"weights"}})";
    atomic_write(dir / "model/model.safetensors.index.json", Bytes(initial_index.begin(), initial_index.end()));
    atomic_write(dir / "exe", Bytes{1, 2, 3});
    auto stable = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    // Unconsumed entries must not invalidate a restore.
    std::filesystem::create_directories(dir / "model/sub/.cache");
    for (const auto &name : {"README.md", "sub/unindexed.safetensors", "unindexed.safetensors", ".gitattributes", "sub/.metadata", "sub/.cache/asset"}) {
      for (const auto value : {1, 2}) {
        atomic_write(dir / "model" / name, Bytes{static_cast<std::uint8_t>(value)});
        check(stable == compatibility({{"flag", "a"}}, dir / "model", dir / "exe"),
              "unconsumed file changed compatibility");
      }
    }
#ifndef _WIN32
    for (const auto* name : {".unconsumed-link", "sub/unconsumed-link"}) {
      std::filesystem::create_symlink(dir / "absent-target", dir / "model" / name);
      check(stable == compatibility({{"flag", "a"}}, dir / "model", dir / "exe"), "unconsumed symlink blocked init");
    }
#endif
    {
      Cache writer(stable, 0, 1, 32ULL << 20, dir / "metadata", self);
      auto value = sample(0, 1);
      value.compat = stable;
      writer.checkpoint(value);
      atomic_write(dir / "model/.gitattributes", Bytes{3});
      Cache reader(compatibility({{"flag", "a"}}, dir / "model", dir / "exe"),
                   0, 1, 32ULL << 20, dir / "metadata", self);
      check(reader.resume("A", true).compat == stable, "metadata restore refused");
    }
    check(stable != compatibility({{"flag", "b"}}, dir / "model", dir / "exe"),
          "numerics negative");
    auto test_compat_restore = [&](const std::string &changed) {
      for (unsigned victim = 0; victim < 2; ++victim) {
      Pair compat_pair;
      std::array<std::unique_ptr<Cache>, 2> writers, readers;
      for (unsigned r = 0; r < 2; ++r) {
        auto path = dir / ("compat" + std::to_string(r));
        writers[r] = std::make_unique<Cache>(stable, r, 2, 32ULL << 20, path, compat_pair.at(r));
        readers[r] = std::make_unique<Cache>(r == victim ? changed : stable, r, 2, 32ULL << 20, path, compat_pair.at(r));
      }
      pair_run([&](unsigned r) { auto value = sample(r, 2); value.compat = stable; writers[r]->checkpoint(value); });
      both_refuse([&](unsigned r) { readers[r]->resume("A", true); });
      }
    };
    // Loader-consumed shards are assets regardless of name, extension or depth.
    auto shard = [](float value, const std::string& name = "tensor") {
      const std::string header = "{\"" + name + R"(":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
      const auto length = static_cast<std::uint64_t>(header.size());
      Bytes data(8 + header.size() + sizeof(value));
      std::memcpy(data.data(), &length, 8);
      std::memcpy(data.data() + 8, header.data(), header.size());
      std::memcpy(data.data() + 8 + header.size(), &value, sizeof(value));
      return data;
    };
    for (const auto& name : {"visible.safetensors", ".weights.safetensors", ".cache/weights.safetensors", "sub/.cache/opaque-shard"}) {
      const auto path = dir / "model" / name;
      std::filesystem::create_directories(path.parent_path());
      atomic_write(path, shard(1));
      const std::string index = std::string(R"({"weight_map":{"tensor":")") + name + R"("}})";
      atomic_write(dir / "model/model.safetensors.index.json", Bytes(index.begin(), index.end()));
      const auto resolved = strix::minimax::assets::pack_files(dir / "model");
      check(resolved.size() == 1 && resolved[0] == path, "pack inventory resolution");
      stable = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
#ifndef _WIN32
      { strix::minimax::engine::detail::PackReader pack((dir / "model").string());
        check(pack.floats("tensor").at(0) == 1, "hidden shard production load before mutation"); }
#endif
      atomic_write(path, shard(2));
#ifndef _WIN32
      { strix::minimax::engine::detail::PackReader pack((dir / "model").string());
        check(pack.floats("tensor").at(0) == 2, "hidden shard production load after mutation"); }
#endif
      const auto changed = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
      check(changed != stable, "consumed hidden shard mutation escaped compatibility");
      test_compat_restore(changed);
    }
#ifndef _WIN32
    {
      // Bind to what the production loader opened, in both startup orderings.
      const auto model = dir / "retarget-model";
      std::filesystem::create_directories(model);
      atomic_write(model / "one", shard(1));
      atomic_write(model / "two", shard(2));
      const std::string index = R"({"weight_map":{"tensor":"weights"}})";
      atomic_write(model / "model.safetensors.index.json", Bytes(index.begin(), index.end()));
      const auto retarget = [&](const char* target) {
        const auto temporary = model / "next-link";
        std::filesystem::create_symlink(target, temporary);
        std::filesystem::rename(temporary, model / "weights");
      };
      retarget("one");
      const auto original = compatibility({}, model, dir / "exe");
      std::string original_retained;
      { strix::minimax::engine::detail::PackReader first(model.string());
        check(compatibility({}, first.loader_snapshot(), dir / "exe") == original,
              "preload compatibility differs from stable loader snapshot");
        original_retained = compatibility({}, strix::LoaderSnapshot{first.loader_snapshot()[1]}, dir / "exe"); }
      auto saved = sample(); saved.compat = original;
      const auto checkpoint = encode(saved);
      retarget("two");
      strix::minimax::engine::detail::PackReader pack(model.string());
      check(pack.floats("tensor").at(0) == 2, "retarget load two");
      const auto snapshot = pack.loader_snapshot();
      check(snapshot.size() == 3 && snapshot[0].identity.ino != 0,
            "index, data and header fd identities recorded");
      const auto loaded_two = compatibility({}, snapshot, dir / "exe"); // warm two
      check(loaded_two != original, "loaded bytes must select a different key");
      refuses([&] { decode(checkpoint, loaded_two, 0, 1, checkpoint.size()); });
      const auto warm_before = strix::file_hash_cache_stats();
      check(compatibility({}, snapshot, dir / "exe") == loaded_two, "loader snapshot warm key");
      check(strix::file_hash_cache_stats().misses == warm_before.misses, "snapshot warm misses");
      retarget("one");
      check(pack.floats("tensor").at(0) == 2, "retained fd still loads two");
      // All-hit lookup still refuses when the separately closed header fd's path retargeted.
      refuses([&] { compatibility({}, snapshot, dir / "exe"); });
      auto factory = [&]() -> std::unique_ptr<int> {
        compatibility({}, snapshot, dir / "exe");
        return std::make_unique<int>(1);
      };
      check(!start_runtime(factory, false, log), "automatic retarget startup enabled");
      check(startup_log == "session cache: off (model files changed during startup)", "retarget diagnostic");
      refuses([&] { start_runtime(factory, true, log); });
      // A retained data fd can safely use an exact cached digest after retargeting.
      strix::LoaderSnapshot retained{snapshot[1]};
      const auto retained_key = compatibility({}, retained, dir / "exe");
      check(retained_key != original_retained, "retained fd digest aliased original checkpoint");
      std::cout << "PASS retarget: loaded=2 disk=1 old_key="
                << sha(Bytes(original.begin(), original.end())) << " loaded_key="
                << sha(Bytes(loaded_two.begin(), loaded_two.end())) << " closed_path=refused\n";
      // A cold cache must reject opening one to hash a snapshot of two.
      const char* hash_setting = std::getenv("STRIX_HASH_CACHE");
      const bool had_hash_setting = hash_setting != nullptr;
      const std::string previous_hash_setting = hash_setting ? hash_setting : "";
      setenv("STRIX_HASH_CACHE", "0", 1);
      refuses([&] { compatibility({}, retained, dir / "exe"); });
      if (had_hash_setting) setenv("STRIX_HASH_CACHE", previous_hash_setting.c_str(), 1);
      else unsetenv("STRIX_HASH_CACHE");
      // Compatibility-before-load: refuse to reuse a snapshot when bytes change afterward.
      retarget("two");
      atomic_write(model / "two", shard(3));
      refuses([&] { compatibility({}, snapshot, dir / "exe"); });
      // In-place changes must also be caught by the final fstat, even on a retained fd.
      const auto live_path = model / "one";
      const int fd = ::open(live_path.c_str(), O_RDONLY | O_CLOEXEC);
      check(fd >= 0, "open stable loader fd");
      strix::LoaderSnapshot live{strix::loaded_file(live_path, fd, true)};
      (void)compatibility({}, live, dir / "exe");
      { std::fstream out(live_path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(-4, std::ios::end); float value = 9; out.write(reinterpret_cast<const char*>(&value), 4); }
      refuses([&] { strix::verify_loader_snapshot(live); });
      ::close(fd);
    }
    {
      // A directory symlink followed by '..' must resolve as open() does.
      const auto model = dir / "collision-model";
      const auto external = dir / "external";
      std::filesystem::create_directories(model);
      std::filesystem::create_directories(external / "sub");
      std::filesystem::create_directory_symlink(external / "sub", model / "link");
      atomic_write(model / "weights", shard(1, "local"));
      atomic_write(external / "weights", shard(1, "external"));
      const std::string index = R"({"weight_map":{"external":"link/../weights","local":"weights","alias":"./weights"}})";
      atomic_write(model / "model.safetensors.index.json", Bytes(index.begin(), index.end()));
      const auto before_collision = compatibility({}, model, dir / "exe");
      std::map<std::string, std::string> expected_roots;
      for (const auto& paths : std::vector<std::vector<std::filesystem::path>>{
               {model / "model.safetensors.index.json"},
               {model / "weights", model / "./weights"},
               {model / "link/../weights"}}) {
        for (const auto& path : paths)
          expected_roots.emplace(std::filesystem::absolute(path).generic_string(), sha(read(path, 100000)));
      }
      const auto expected_record = compat_record(std::move(expected_roots));
      const auto expected_root = sha(Bytes(expected_record.begin(), expected_record.end()));
      check(before_collision.find(expected_root) != std::string::npos,
            "inventory must retain both references to the same real shard");
      { strix::minimax::engine::detail::PackReader pack(model.string());
        check(pack.floats("external").at(0) == 1, "directory symlink parent load"); }
      atomic_write(external / "weights", shard(2, "external"));
      { strix::minimax::engine::detail::PackReader pack(model.string());
        check(pack.floats("external").at(0) == 2, "directory symlink parent mutation load"); }
      check(before_collision != compatibility({}, model, dir / "exe"),
            "directory symlink parent collision omitted consumed shard");
      const auto external_changed = compatibility({}, model, dir / "exe");
      atomic_write(model / "weights", shard(3, "local"));
      check(external_changed != compatibility({}, model, dir / "exe"),
            "aliased local shard omitted from inventory");
    }
    {
      const auto target = dir / "symlink-target";
      atomic_write(target, shard(1));
      std::filesystem::create_symlink(target, dir / "model/linked.safetensors");
      const std::string index = R"({"weight_map":{"tensor":"linked.safetensors"}})";
      atomic_write(dir / "model/model.safetensors.index.json", Bytes(index.begin(), index.end()));
      stable = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
      { strix::minimax::engine::detail::PackReader pack((dir / "model").string());
        check(pack.floats("tensor").at(0) == 1, "symlink shard load"); }
      atomic_write(target, shard(2));
      { strix::minimax::engine::detail::PackReader pack((dir / "model").string());
        check(pack.floats("tensor").at(0) == 2, "symlink target load after change"); }
      const auto changed = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
      check(changed != stable, "symlink target mutation escaped compatibility");
      test_compat_restore(changed);
    }
#endif
    // DraftFile consumes all root .safetensors, including dot-prefixed ones,
    // even without a pack index. Compatibility uses that same resolver.
    std::filesystem::create_directories(dir / "draft");
    atomic_write(dir / "draft/.draft.safetensors", shard(1));
    const auto draft_files = strix::minimax::assets::draft_files(dir / "draft");
    check(draft_files.size() == 1 && draft_files[0] == dir / "draft/.draft.safetensors", "draft hidden file resolution");
    const auto draft_before = compatibility({}, dir / "draft", dir / "exe", AssetLoader::Draft);
    // DraftFile never reads a pack index, even if one is present and invalid.
    atomic_write(dir / "draft/model.safetensors.index.json", Bytes{1, 2, 3});
    check(draft_before == compatibility({}, dir / "draft", dir / "exe", AssetLoader::Draft), "unconsumed draft index affected key");
    stable = compatibility({{"draft", draft_before}}, dir / "model", dir / "exe");
    atomic_write(dir / "draft/.draft.safetensors", shard(2));
    const auto draft_after = compatibility({}, dir / "draft", dir / "exe", AssetLoader::Draft);
    check(draft_before != draft_after, "consumed hidden draft mutation escaped compatibility");
    test_compat_restore(compatibility({{"draft", draft_after}}, dir / "model", dir / "exe"));
    stable = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    test_compat_restore(compatibility({{"flag", "b"}}, dir / "model", dir / "exe"));
    atomic_write(dir / "exe", Bytes{1, 2, 4});
    test_compat_restore(compatibility({{"flag", "a"}}, dir / "model", dir / "exe"));
    check(stable != compatibility({{"flag", "a"}}, dir / "model", dir / "exe"),
          "exe negative");
    // Real environment enumeration is tested on Windows as well as Linux.
    auto set_env = [](const char *key, const std::string &value) {
#ifdef _WIN32
      check(_putenv_s(key, value.c_str()) == 0, "set environment");
#else
      check(setenv(key, value.c_str(), 1) == 0, "set environment");
#endif
    };
    for (const char *key : {"STRIX_SESSION_POOL_IDENTITY_TEST", "STRIX_HASH_CACHE", "STRIX_SESSION_RAM_BYTES", "STRIX_SESSION_DISK_BYTES", "STRIX_SESSION_CACHE", "STRIX_MM_OUTPUT_DIR", "STRIX_MM_LOG", "STRIX_MM_LOGS", "STRIX_MM_TRACE_PATH", "STRIX_MM_TRACE", "STRIX_MM_CACHE_DIRECTORY", "HIP_TRACE_API", "AMD_LOG_LEVEL", "HSA_LOG_PATH"}) {
      const char *prior = std::getenv(key);
      std::string saved = prior ? prior : "";
      bool present = prior != nullptr;
      auto unchanged = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
      set_env(key, (dir / "operational-test").string());
      auto changed = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
      check(unchanged == changed, "operational metadata changed compatibility");
      Pair operational_pair;
      std::array<std::unique_ptr<Cache>, 2> writers, readers;
      for (unsigned r = 0; r < 2; ++r) {
        auto path = dir / ("operational" + std::to_string(r));
        writers[r] = std::make_unique<Cache>(unchanged, r, 2, 32ULL << 20, path, operational_pair.at(r));
        readers[r] = std::make_unique<Cache>(changed, r, 2, 32ULL << 20, path, operational_pair.at(r));
      }
      pair_run([&](unsigned r) { auto value = sample(r, 2); value.compat = unchanged; writers[r]->checkpoint(value); });
      pair_run([&](unsigned r) { check(readers[r]->resume("A", true).compat == unchanged, "operational restore refused"); });
      set_env(key, saved);
#ifndef _WIN32
      if (!present) unsetenv(key);
#else
      (void)present;
#endif
    }
    set_env("STRIX_MM_TEST_NUMERICS", "one");
    auto numeric = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    set_env("STRIX_MM_TEST_NUMERICS", "two");
    auto numerical_change = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    check(numeric != numerical_change, "numerical environment ignored");
    auto prior_stable = stable;
    stable = numeric;
    test_compat_restore(numerical_change);
    stable = prior_stable;
    atomic_write(dir / "model/model.safetensors.index.json", Bytes(initial_index.begin(), initial_index.end()));
    auto before = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    weights[2048] = 2;
    atomic_write(dir / "model/weights", weights);
    auto model_changed = compatibility({{"flag", "a"}}, dir / "model", dir / "exe");
    check(before != model_changed, "full model middle hash");
    stable = before;
    test_compat_restore(model_changed);
    check(before != compatibility({{"flag", "b"}}, dir / "model",
                                  dir / "model/weights"),
          "numerics flag");
#ifdef __linux__
    // Persistent cache, first fill then verify every production compatibility lookup hit.
    set_env("STRIX_HASH_CACHE", (dir / "hash-cache").string());
    const auto warm = prewarm(dir / "model", dir / "draft", {}, dir / "exe", {(dir / "exe").string()});
    check(warm.hits > 0 && warm.wall_seconds >= warm.second_seconds, "prewarm receipt");
    const auto cold = compatibility({}, dir / "model", dir / "exe");
    const auto stats_before = strix::file_hash_cache_stats();
    check(cold == compatibility({}, dir / "model", dir / "exe"), "prewarm key changed");
    const auto stats_after = strix::file_hash_cache_stats();
    check(stats_after.misses == stats_before.misses && stats_after.hits > stats_before.hits,
          "prewarm second pass not all hits");
    std::cout << "PASS prewarm second pass all hits=" << warm.hits << " misses=0 key=" << sha(Bytes(cold.begin(), cold.end())) << '\n';
#endif
    {
      stable = compatibility({}, dir / "model", dir / "exe");
      const auto whitespace_index = initial_index + " ";
      atomic_write(dir / "model/model.safetensors.index.json", Bytes(whitespace_index.begin(), whitespace_index.end()));
      const auto changed = compatibility({}, dir / "model", dir / "exe");
      check(changed != stable, "consumed pack index omitted");
      test_compat_restore(changed);
    }
    // Count and disk caps are independent of byte-only RAM admission.
    Cache count_lru(c.compat, 0, 1, 32ULL << 20, dir / "count-lru", self, 1);
    auto ca = sample(); count_lru.capture(ca);
    ca.id = "B"; count_lru.capture(ca);
    check(count_lru.stats().entries == 1 && !count_lru.in_ram("A") && count_lru.in_ram("B"), "max parked count cap");
    check(count_lru.describe().front().id == "B", "RAM list metadata");
    Cache disk_lru(c.compat, 0, 1, 32ULL << 20, dir / "disk-lru", self, 8, 1400000);
    ca.id = "A"; disk_lru.checkpoint(ca);
    for (const auto& file : std::filesystem::directory_iterator(dir / "disk-lru"))
      if (file.path().extension() == ".session") std::filesystem::last_write_time(file.path(), std::filesystem::file_time_type::clock::now() - std::chrono::hours(24));
    ca.id = "B"; disk_lru.checkpoint(ca);
    ca.id = "C"; disk_lru.checkpoint(ca);
    check(!disk_lru.on_disk("A") && disk_lru.on_disk("B") && disk_lru.on_disk("C"), "disk LRU eviction");
    check(disk_lru.stats().disk_bytes <= 1400000, "aggregate disk cap");
    check(disk_lru.describe().size() == 2, "restart disk list metadata");
    Cache tiny_disk(c.compat, 0, 1, 32ULL << 20, dir / "tiny-disk", self, 8, 1);
    refuses([&] { tiny_disk.checkpoint(sample()); });
    check(!tiny_disk.on_disk("A"), "oversized disk checkpoint unpublished");
    const auto replacement_dir = dir / "replacement-peak";
    Exchange peak_exchange = [&](std::span<const std::uint8_t> phase_bytes) {
      std::string phase(phase_bytes.begin(), phase_bytes.end());
      if (phase.find("rank-durable") != std::string::npos) {
        std::uint64_t peak_bytes = 0;
        for (const auto& file : std::filesystem::directory_iterator(replacement_dir))
          if (file.path().extension() == ".session") peak_bytes += file.file_size();
        check(peak_bytes <= 900000, "replacement durable peak within quota");
      }
      return Bytes(phase_bytes.begin(), phase_bytes.end());
    };
    Cache replacement(c.compat, 0, 1, 32ULL << 20, replacement_dir, peak_exchange, 8, 900000);
    replacement.checkpoint(sample());
    replacement.checkpoint(sample());
    check(replacement.on_disk("A") && replacement.stats().evictions == 1, "replacement peak evicts old generation before writing");
    // Independent Cache handles must not admit, publish or evict while another
    // writer holds the directory transaction across a collective exchange.
    auto lock_dir = dir / "lock-test";
    std::mutex lock_mutex;
    std::condition_variable lock_cv;
    bool held = false, release = false;
    Exchange held_exchange = [&](std::span<const std::uint8_t> phase_bytes) {
      std::string phase(phase_bytes.begin(), phase_bytes.end());
      if (phase.find("disk-scan") != std::string::npos) {
        std::unique_lock<std::mutex> guard(lock_mutex);
        held = true; lock_cv.notify_one();
        if (!lock_cv.wait_for(guard, std::chrono::seconds(5), [&] { return release; }))
          throw std::runtime_error("lock fixture timeout");
      }
      return Bytes(phase_bytes.begin(), phase_bytes.end());
    };
    Cache writer(c.compat, 0, 1, 32ULL << 20, lock_dir, held_exchange, 8, 1400000);
    Cache contender(c.compat, 0, 1, 32ULL << 20, lock_dir, self, 8, 1400000);
    std::exception_ptr writer_error;
    std::thread writing([&] { try { writer.checkpoint(sample()); } catch (...) { writer_error = std::current_exception(); } });
    bool checkpoint_refused = false, delete_refused = false;
    {
      std::unique_lock<std::mutex> guard(lock_mutex);
      check(lock_cv.wait_for(guard, std::chrono::seconds(5), [&] { return held; }), "writer acquired directory lock");
    }
    try { contender.checkpoint(sample()); } catch (...) { checkpoint_refused = true; }
    try { contender.erase("A"); } catch (...) { delete_refused = true; }
    { std::lock_guard<std::mutex> guard(lock_mutex); release = true; }
    lock_cv.notify_one(); writing.join();
    if (writer_error) std::rethrow_exception(writer_error);
    check(checkpoint_refused && delete_refused, "lock covers quota/publication/eviction/delete");
    contender.checkpoint(sample()); // replacement charges both generations at peak
    check(contender.stats().disk_bytes <= 1400000, "replacement peak cap");
#ifndef _WIN32
    // A separate process holds flock: library writers must honor the same lock.
    int ready[2], done_pipe[2];
    check(::pipe(ready) == 0 && ::pipe(done_pipe) == 0, "lock pipes");
    auto child = ::fork();
    check(child >= 0, "lock fork");
    if (child == 0) {
      ::close(ready[0]); ::close(done_pipe[1]);
      int fd = ::open((lock_dir / ".session.lock").c_str(), O_RDWR);
      if (fd < 0 || ::flock(fd, LOCK_EX) != 0) ::_exit(2);
      char signal = 'x';
      if (::write(ready[1], &signal, 1) != 1 || ::read(done_pipe[0], &signal, 1) != 1) ::_exit(3);
      ::close(fd); ::_exit(0);
    }
    ::close(ready[1]); ::close(done_pipe[0]);
    char signal = 0;
    check(::read(ready[0], &signal, 1) == 1, "child lock ready");
    checkpoint_refused = delete_refused = false;
    try { contender.checkpoint(sample()); } catch (...) { checkpoint_refused = true; }
    try { contender.erase("A"); } catch (...) { delete_refused = true; }
    check(::write(done_pipe[1], &signal, 1) == 1, "child unlock");
    int child_status = 0;
    check(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0, "child lock success");
    ::close(ready[0]); ::close(done_pipe[1]);
    check(checkpoint_refused && delete_refused, "interprocess flock honored");
    contender.checkpoint(sample());
#endif
    // Remove only the fixture created above in this invocation.
    std::filesystem::remove_all(dir);
    std::cout << "MiniMax session CPU tests PASS: exe numerics full-model "
                 "both-rank-refusals operational-restore corrupt-section truncated enlarged missing-rank generation DSpark-map "
                 "peak LRU\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
