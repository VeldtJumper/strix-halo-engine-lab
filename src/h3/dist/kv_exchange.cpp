// In-process K/V exchange pair (see kv_exchange.hpp).
#include "h3/dist/kv_exchange.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <stdexcept>

namespace strix::h3::dist {

namespace {
const std::array<std::uint32_t, 256>& crc_table() {
    static const std::array<std::uint32_t, 256> t = [] {
        std::array<std::uint32_t, 256> a{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1U) ? (c >> 1) ^ 0x82F63B78U : c >> 1;
            a[i] = c;
        }
        return a;
    }();
    return t;
}
}  // namespace

std::uint32_t crc32c_sw(const void* data, std::size_t n, std::uint32_t crc) {
    const auto& t = crc_table();
    const auto* p = static_cast<const unsigned char*>(data);
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = t[(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    return ~crc;
}

struct Message {
    std::uint64_t sequence = 0;
    std::uint32_t layer = 0;
    std::uint32_t crc = 0;  // the sender's CRC of its message at start()
    std::vector<std::byte> data;
};

struct MemoryKvPair::Impl {
    struct Endpoint final : KvExchange {
        Impl* pair = nullptr;
        unsigned rank = 0;
        // the job of this endpoint's progress thread (guarded by pair->mu)
        bool job = false, sending = false;
        const std::byte* tx = nullptr;
        std::size_t bytes = 0;
        std::uint32_t layer = 0, crc_at_start = 0;
        std::uint64_t sequence = 0;  // of the exchange in flight (counts starts)
        bool started = false;
        std::string send_error;
        std::deque<Message> inbox;   // complete messages from the peer, in order
        std::int64_t corrupt_layer = -1;  // test hook
        KvPairStats stats;
        std::thread progress;

        void start(const std::byte* data, std::size_t n, std::uint32_t lay) override {
            std::unique_lock lk(pair->mu);
            if (!pair->failure.empty()) throw std::runtime_error("kv pair: stopped: " + pair->failure);
            if (started) throw std::logic_error("kv pair: start() while an exchange is in flight");
            if (!data || !n) throw std::invalid_argument("kv pair: empty message");
            started = true;
            tx = data; bytes = n; layer = lay;
            crc_at_start = crc32c_sw(data, n);
            job = true;
            send_error.clear();
            pair->cv.notify_all();
        }
        void finish(std::byte* rx) override {
            const auto t0 = std::chrono::steady_clock::now();
            std::unique_lock lk(pair->mu);
            if (!started) throw std::logic_error("kv pair: finish() without start()");
            const auto deadline = t0 + pair->timeout;
            const auto ready = [&] { return !pair->failure.empty() || (!job && !sending && !inbox.empty()); };
            if (!pair->cv.wait_until(lk, deadline, ready)) {
                pair->fail_locked("finish timeout (rank " + std::to_string(rank) + ", layer " + std::to_string(layer) + ")");
                quiesce(lk);
                throw std::runtime_error("kv pair: " + pair->failure);
            }
            if (!pair->failure.empty()) {
                quiesce(lk);
                throw std::runtime_error("kv pair: stopped: " + pair->failure);
            }
            started = false;
            if (!send_error.empty()) {
                pair->fail_locked(send_error);
                throw std::runtime_error("kv pair: " + send_error);
            }
            Message m = std::move(inbox.front());
            inbox.pop_front();
            const std::uint64_t seq = sequence++;
            std::string bad;
            if (m.sequence != seq) bad = "sequence mismatch";
            else if (m.layer != layer) bad = "layer mismatch";
            else if (m.data.size() != bytes) bad = "size mismatch";
            else if (crc32c_sw(m.data.data(), m.data.size()) != m.crc) bad = "payload CRC mismatch";
            if (!bad.empty()) {
                bad += " (rank " + std::to_string(rank) + ", layer " + std::to_string(layer) + ")";
                pair->fail_locked(bad);
                throw std::runtime_error("kv pair: " + bad);
            }
            lk.unlock();
            std::memcpy(rx, m.data.data(), m.data.size());
            ++stats.exchanges;
            stats.bytes += m.data.size();
            stats.finish_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        // Returns once this endpoint's progress thread no longer reads tx (the caller may then free it).
        void abort(const std::string& why) override {
            std::unique_lock lk(pair->mu);
            pair->fail_locked(why);
            quiesce(lk);
        }
        void quiesce(std::unique_lock<std::mutex>& lk) { pair->cv.wait(lk, [&] { return !job && !sending; }); }
        void run() {
            std::unique_lock lk(pair->mu);
            std::uint64_t sent = 0;
            for (;;) {
                pair->cv.wait(lk, [&] { return pair->stopping || job; });
                if (pair->stopping) return;
                job = false;
                if (!pair->failure.empty()) {  // stopped: never touch tx again
                    pair->cv.notify_all();
                    continue;
                }
                sending = true;
                const std::byte* src = tx;
                const std::size_t n = bytes;
                const std::uint32_t lay = layer, crc0 = crc_at_start;
                lk.unlock();
                Message m;
                m.sequence = sent++;
                m.layer = lay;
                m.crc = crc0;
                m.data.resize(n);
                const unsigned nchunks = std::max(1U, pair->chunks);
                const std::size_t piece = (n + nchunks - 1) / nchunks;
                for (std::size_t at = 0; at < n; at += piece) {
                    if (pair->delay.count()) std::this_thread::sleep_for(pair->delay / nchunks);
                    std::memcpy(m.data.data() + at, src + at, std::min(piece, n - at));
                }
                const bool changed = crc32c_sw(m.data.data(), n) != crc0;
                lk.lock();
                if (corrupt_layer >= 0 && static_cast<std::uint32_t>(corrupt_layer) == lay && n) m.data[n / 2] ^= std::byte{0x20};
                if (changed) send_error = "tx buffer changed while the transfer was reading it (rank " + std::to_string(rank) + ")";
                Endpoint& peer = pair->ep[1U - rank];
                peer.inbox.push_back(std::move(m));
                sending = false;
                pair->cv.notify_all();
            }
        }
    };

    std::mutex mu;
    std::condition_variable cv;
    std::chrono::microseconds delay;
    unsigned chunks;
    std::chrono::milliseconds timeout;
    bool stopping = false;
    std::string failure;
    Endpoint ep[2];

    void fail_locked(const std::string& why) {
        if (failure.empty()) failure = why;
        cv.notify_all();
    }
};

MemoryKvPair::MemoryKvPair(std::chrono::microseconds delay, unsigned chunks, std::chrono::milliseconds timeout)
    : impl_(std::make_unique<Impl>()) {
    impl_->delay = delay;
    impl_->chunks = chunks;
    impl_->timeout = timeout;
    for (unsigned r = 0; r < 2; ++r) {
        impl_->ep[r].pair = impl_.get();
        impl_->ep[r].rank = r;
    }
    for (unsigned r = 0; r < 2; ++r) impl_->ep[r].progress = std::thread([this, r] { impl_->ep[r].run(); });
}

MemoryKvPair::~MemoryKvPair() {
    {
        std::lock_guard lk(impl_->mu);
        impl_->stopping = true;
        impl_->cv.notify_all();
    }
    for (auto& e : impl_->ep)
        if (e.progress.joinable()) e.progress.join();
}

KvExchange& MemoryKvPair::endpoint(unsigned rank) { return impl_->ep[rank & 1U]; }

bool MemoryKvPair::arrived(unsigned rank) const {
    std::lock_guard lk(impl_->mu);
    return !impl_->ep[rank & 1U].inbox.empty();
}

void MemoryKvPair::corrupt(unsigned sender, std::uint32_t layer) {
    std::lock_guard lk(impl_->mu);
    impl_->ep[sender & 1U].corrupt_layer = layer;
}

KvPairStats MemoryKvPair::stats(unsigned rank) const {
    std::lock_guard lk(impl_->mu);
    return impl_->ep[rank & 1U].stats;
}

}  // namespace strix::h3::dist
