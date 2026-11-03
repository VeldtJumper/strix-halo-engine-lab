#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace strix {
// Persisted window-cost tables shared by the adaptive speculation controllers.
//
// A table holds measured costs (robust median ms plus its sample count) under short entry names chosen by the
// engine (for example "r1", "r4"). It is controller input only: a cost steers which window or draft width runs, a
// choice the controller already makes from measured times. Whether that choice can change output bits is the
// engine's property, not this library's: an engine seeds only choices its own exactness guard admits.
//
// The feature is off unless STRIX_COST_TABLE_DIR names a directory; a directory inside a source checkout (any
// ancestor holding .git) is refused. One file per key under it, named <engine>-<hash of the key>.costs.
// Key: engine, binary identity, device identity, model identity, policy identity and context bucket. The file
// repeats every key field and carries a format version and a checksum: a missing, unreadable, oversized, corrupt,
// other-version or mismatched file is ignored (the controller re-times from scratch, as without the feature) and
// the next store replaces it. A valid table is a warm prior, except a rotating fraction of its entries
// (recheck_fraction, default 15 %, at least one) that is re-timed cold as a drift check; a re-timed entry that
// differs from its cached value by more than drift_tolerance (default 15 %) withdraws the whole loaded table for
// this process, and the store bumps the table's epoch and writes only what was measured.
// Stores take a per-key lock file (flock / LockFileEx), re-read the current file under the lock and merge into it
// (never into this process's older snapshot): a writer whose loaded epoch is older than the file's cannot restore
// entries another writer withdrew. The new file is written beside the old one and moved over it atomically
// (rename / MoveFileExW with REPLACE_EXISTING | WRITE_THROUGH). Stores never throw.
struct CostTableKey {
    std::string engine; // [A-Za-z0-9_.-]+, also the file name prefix
    std::string binary; // executable content hash or build id; empty disables the cache
    std::string device; // device identity string
    std::string model;  // model (and drafter) identity
    std::string policy; // policy identity (effective configuration and environment)
    std::uint32_t context_bucket=0;
};
struct CostTableEntry {double ms=0.0;std::uint64_t samples=0;};
struct CostTableOptions {double recheck_fraction=0.15,drift_tolerance=0.15;};
enum class CostTableLoad {not_loaded,missing,loaded,corrupt,version,mismatch};
[[nodiscard]] const char* cost_table_load_name(CostTableLoad) noexcept;

// 0 below 8192 positions, then one bucket per doubling (1: 8K..16K, 2: 16K..32K, ...).
[[nodiscard]] std::uint32_t cost_table_context_bucket(std::uint64_t position) noexcept;
// STRIX_COST_TABLE_DIR, or nullopt (feature off) when unset, empty or inside a source checkout.
[[nodiscard]] std::optional<std::filesystem::path> cost_table_directory_from_env();
[[nodiscard]] bool cost_table_directory_admitted(const std::filesystem::path&);
// STRIX_COST_TABLE_RECHECK (fraction in [0, 1]) and STRIX_COST_TABLE_DRIFT (> 0); invalid values keep defaults.
[[nodiscard]] CostTableOptions cost_table_options_from_env();
// SHA-256 hex of the running executable (computed once per process), "" when it cannot be read.
[[nodiscard]] std::string cost_table_executable_identity();
// SHA-256 hex of `configuration` plus the sorted STRIX_/HIP_/HSA_/AMD_/ROCR_/GPU_ environment, without the
// cost-table variables themselves, the per-window diagnostic log (STRIX_CYCLE_LOG) and the GPU-turn wrapper ids.
[[nodiscard]] std::string cost_table_policy_identity(std::string_view configuration);
[[nodiscard]] std::string cost_table_hash_hex(std::string_view bytes);

class CostTableCache {
public:
    static constexpr std::uint32_t format_version=1;
    CostTableCache(std::filesystem::path directory,CostTableKey key,CostTableOptions options={});
    // Reads the table once; any failure leaves an empty prior. Repeated calls return the first status.
    CostTableLoad load();
    [[nodiscard]] CostTableLoad load_status() const noexcept {return status_;}
    // The prior for `name`: measured in this process, else a loaded entry outside the drift check (and only while
    // no drift was seen). nullopt means "time it".
    [[nodiscard]] std::optional<CostTableEntry> warm(std::string_view name) const;
    [[nodiscard]] bool rechecking(std::string_view name) const;
    [[nodiscard]] std::vector<std::string> recheck_entries() const {return {recheck_.begin(),recheck_.end()};}
    // A measured cost (finite ms > 0, samples > 0; anything else is ignored).
    void record(std::string_view name,CostTableEntry measured);
    [[nodiscard]] bool drifted() const noexcept {return drifted_;}
    [[nodiscard]] bool dirty() const noexcept {return dirty_;}
    [[nodiscard]] std::size_t loaded_entries() const noexcept {return loaded_.size();}
    [[nodiscard]] std::uint64_t generation() const noexcept {return generation_;}
    [[nodiscard]] std::uint64_t epoch() const noexcept {return epoch_;}
    // Atomic write of the merged table; false on any failure (the previous file stays).
    bool store();
    [[nodiscard]] const std::filesystem::path& file() const noexcept {return file_;}
    [[nodiscard]] const CostTableKey& key() const noexcept {return key_;}
    using Entries=std::map<std::string,CostTableEntry,std::less<>>;
    // File body for a table, and its strict parse against `key` (exposed for tests).
    [[nodiscard]] static std::string serialize(const CostTableKey&,std::uint64_t generation,std::uint64_t epoch,const Entries&);
    struct Parsed {CostTableLoad status=CostTableLoad::corrupt;Entries entries;std::uint64_t generation=0,epoch=0;};
    [[nodiscard]] static Parsed parse_body(std::string_view body,const CostTableKey&);
    // Reads at most max_file_bytes (+1 to detect an oversized file); never throws.
    [[nodiscard]] static Parsed read_file(const std::filesystem::path&,const CostTableKey&);
    static constexpr std::size_t max_file_bytes=1U<<20U;
    [[nodiscard]] static bool valid_name(std::string_view) noexcept;
private:
    std::filesystem::path directory_,file_;
    CostTableKey key_;
    CostTableOptions options_;
    CostTableLoad status_=CostTableLoad::not_loaded;
    Entries loaded_,measured_;
    std::set<std::string,std::less<>> recheck_;
    std::uint64_t generation_=0,epoch_=0;
    bool drifted_=false,dirty_=false;
};
} // namespace strix
