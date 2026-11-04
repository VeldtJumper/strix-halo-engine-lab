#pragma once
// The per-block K/V exchange of the two-node schedule as the host reference driver sees it, and an in-process pair
// (two ranks in one process, the mock transport of the CPU tests). The device driver uses the engine's Communicator
// (start_async / finish, chunked host pipeline, CRC-32C trailers) with the same contract:
//   start(tx): the transfer progresses on its own (a progress thread), without finish() being called;
//   tx stays unchanged until the transfer has read it (finish() returned, or the device's source-free event);
//   finish(rx): waits for the peer's message, checks it, writes it to rx. One exchange in flight at a time.
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strix::h3::dist {

// CRC-32C (Castagnoli, reflected 0x82F63B78), software table; the same 32 bits as the engine's trailer CRC.
std::uint32_t crc32c_sw(const void* data, std::size_t n, std::uint32_t crc = 0);

class KvExchange {
public:
    virtual ~KvExchange() = default;
    virtual void start(const std::byte* tx, std::size_t bytes, std::uint32_t layer) = 0;
    virtual void finish(std::byte* rx) = 0;
    // Fatal teardown: the peer's pending finish() fails at once.
    virtual void abort(const std::string& why) = 0;
};

struct KvPairStats {
    std::uint64_t exchanges = 0, bytes = 0;
    double finish_wait_s = 0.0;  // inside finish(), waiting for the peer
};

// Two endpoints in one process. Each endpoint has a persistent progress thread that copies the started message
// onto the "wire" in `chunks` pieces, sleeping `delay / chunks` before each one (a slow link that reads tx during
// the flight, as the transport threads do), and checks the CRC of what it read against the CRC taken at start(): a
// tx changed during the flight fails the exchange. The receiver checks layer, sequence, size and CRC.
class MemoryKvPair {
public:
    explicit MemoryKvPair(std::chrono::microseconds delay = std::chrono::microseconds(0), unsigned chunks = 4,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(60000));
    ~MemoryKvPair();
    MemoryKvPair(const MemoryKvPair&) = delete;
    MemoryKvPair& operator=(const MemoryKvPair&) = delete;
    KvExchange& endpoint(unsigned rank);
    // Test hook: the peer's message for `rank` has fully arrived (without finish() having been called).
    bool arrived(unsigned rank) const;
    KvPairStats stats(unsigned rank) const;
    // Test hook: flip one payload byte of the message `sender` sends for `layer` (after its CRC was taken: the
    // receiver must reject it).
    void corrupt(unsigned sender, std::uint32_t layer);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::h3::dist
