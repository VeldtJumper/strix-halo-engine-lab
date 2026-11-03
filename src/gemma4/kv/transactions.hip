#include <hip/hip_runtime.h>
#include "storage.hpp"
#include "gemma4/kernels/internal.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
void fork_checked(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("gemma4 stream fork: ") + hipGetErrorString(e));
}
__device__ void publish_rows(float* k, float* v, std::uint64_t* tags, const float* ok, const float* ov,
                             std::uint64_t start, unsigned rows, unsigned slots, unsigned width, bool global) {
    // One block per producer: rows published in position order even if a prompt spans a ring.
    for (unsigned r = 0; r < rows; ++r) {
        const auto p = start + r, slot = global ? p : p % slots;
        for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
            k[slot * width + i] = ok[std::size_t(r) * width + i];
            v[slot * width + i] = ov[std::size_t(r) * width + i];
        }
        __syncthreads();
        if (threadIdx.x == 0)
            tags[slot] = p;
        __syncthreads();
    }
}
__global__ void publish_kernel(float* k, float* v, std::uint64_t* tags, const float* ok, const float* ov,
                               std::uint64_t start, unsigned rows, unsigned slots, unsigned width, bool global) {
    publish_rows(k, v, tags, ok, ov, start, rows, slots, width, global);
}
__global__ void publish_batch_kernel(const detail::Publication* descriptors, std::uint64_t start, unsigned rows) {
    const auto d = descriptors[blockIdx.x];
    if (d.width)
        publish_rows(d.k, d.v, d.tags, d.overlay_k, d.overlay_v, start, rows, d.slots, d.width, d.global);
}
__global__ void store_kernel(const float* k, const float* v, const KvWrite* dst, unsigned width) {
    const auto r = blockIdx.x;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        dst[r].k[std::size_t(dst[r].row) * width + i] = k[std::size_t(r) * width + i];
        dst[r].v[std::size_t(dst[r].row) * width + i] = v[std::size_t(r) * width + i];
    }
}
void stream_checked(hipError_t e) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("gemma4 device KV stream: ") + hipGetErrorString(e));
}
// Copy between host and device on the stream's queue (or synchronously on the null stream when it has none).
void stream_copy(Queue q, void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind) {
    if (!bytes) return;
    if (!q) { stream_checked(hipMemcpy(dst, src, bytes, kind)); return; }
    stream_checked(hipMemcpyAsync(dst, src, bytes, kind, static_cast<hipStream_t>(q)));
    stream_checked(hipStreamSynchronize(static_cast<hipStream_t>(q)));
}
} // namespace
DeviceStream::Impl::~Impl() {
    if (pinned_publication)
        (void)hipHostFree(pinned_publication);
}
void store_overlay(const float* k, const float* v, const KvWrite* dst, unsigned rows, unsigned width, Queue q) {
    hipLaunchKernelGGL(store_kernel, dim3(rows), dim3(128), 0, static_cast<hipStream_t>(q), k, v, dst, width);
    launch_checked();
}
DeviceStream::Impl::Txn::Txn(std::shared_ptr<Impl> s)
    : stream(std::move(s)), start(stream->frontier), overlay(std::move(stream->spare_overlay)) {
    overlay.resize(stream->cache.size());
}
DeviceStream::Impl::Txn::~Txn() {
    if (stream->pooled && stream->spare_overlay.empty())
        stream->spare_overlay = std::move(overlay);
    if (!published)
        stream->busy = false;
}
void DeviceStream::Impl::Txn::reserve_rows(std::size_t count) {
    if (rows || reserved || published || failed || !count || count > stream->admitted - start)
        throw std::runtime_error("invalid device overlay reservation");
    // Allocate before the first pass. Appending must never reallocate live overlay bytes.
    for (std::size_t l = 0; l < overlay.size(); ++l) {
        const auto bytes = count * stream->cache[l].width * sizeof(float);
        if (bytes > overlay[l].k.size()) { overlay[l].k.resize(bytes); overlay[l].v.resize(bytes); }
    }
    reserved = count;
}
void DeviceStream::Impl::Txn::commit(std::size_t count) {
    if (published || failed || !evaluated || count > rows || stream->poisoned || stream->frontier != start)
        throw std::runtime_error("invalid/stale device KV commit");
    try {
        if (count && stream->publish_batch) {
            std::vector<detail::Publication> descriptors(stream->cache.size());
            for (std::size_t l = 0; l < stream->cache.size(); ++l) {
                auto& c = stream->cache[l];
                descriptors[l] = {c.k.as<float>(), c.v.as<float>(), c.tags.as<std::uint64_t>(),
                                  overlay[l].k.as<float>(), overlay[l].v.as<float>(), c.slots, c.width, c.global};
            }
            const auto bytes = descriptors.size() * sizeof(detail::Publication);
            if (stream->queue && stream->pinned_publication) {
                // Pinned staging: the copy is ordered on the stream's queue before the kernel that reads it.
                std::copy(descriptors.begin(), descriptors.end(), static_cast<detail::Publication*>(stream->pinned_publication));
                stream_checked(hipMemcpyAsync(stream->publication.data(), stream->pinned_publication, bytes,
                                              hipMemcpyHostToDevice, static_cast<hipStream_t>(stream->queue)));
            } else
                stream->publication.upload(descriptors.data(), bytes);
            hipLaunchKernelGGL(publish_batch_kernel, dim3(unsigned(descriptors.size())), dim3(128), 0,
                               static_cast<hipStream_t>(stream->queue),
                               stream->publication.as<detail::Publication>(), start, unsigned(count));
            launch_checked();
        } else if (count)
            for (std::size_t l = 0; l < stream->cache.size(); ++l) {
                auto& c = stream->cache[l];
                if (!c.width)
                    continue;
                hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(128), 0, static_cast<hipStream_t>(stream->queue),
                                   c.k.as<float>(), c.v.as<float>(),
                                   c.tags.as<std::uint64_t>(), overlay[l].k.as<float>(), overlay[l].v.as<float>(),
                                   start, unsigned(count), c.slots, c.width, c.global);
                launch_checked();
            }
        synchronize(stream->queue);
    } catch (...) {
        // Hardware failure can leave partial writes; never offer that stream for continued execution.
        stream->poisoned = true;
        failed = true;
        throw;
    }
    stream->frontier += count;
    published = true;
    stream->busy = false;
}
KvRead DeviceStream::Impl::Txn::read(unsigned producer, std::uint64_t position) const {
    const auto& c = stream->cache.at(producer);
    return {c.k.as<float>(),
            c.v.as<float>(),
            c.tags.as<std::uint64_t>(),
            overlay[producer].k.as<float>(),
            overlay[producer].v.as<float>(),
            start,
            position,
            c.width,
            c.slots,
            unsigned(rows),
            stream->config.window,
            c.global};
}
DeviceStream::DeviceStream(const ModelConfig& config, unsigned admitted) : impl_(std::make_shared<Impl>()) {
    (void)cache_bytes(config, admitted);
    require_device();
    impl_->config = config;
    impl_->admitted = admitted;
    static const bool pooled = [] {
        const auto* p = std::getenv("STRIX_G4_KV_POOL");
        return !p || std::string_view(p) != "0";
    }();
    impl_->pooled = pooled;
    static const bool publish_batch = [] {
        const auto* p = std::getenv("STRIX_G4_KV_PUBLISH");
        return !p || std::string_view(p) != "0";
    }();
    impl_->publish_batch = publish_batch;
    if (publish_batch)
        impl_->publication.resize(std::size_t(config.layers) * sizeof(detail::Publication));
    // Read per stream (cheap, once per request), so an in-process A/B can switch it between requests.
    const bool own_queue = [] {
        const auto* p = std::getenv("STRIX_G4_KV_QUEUE");
        return !p || std::string_view(p) != "0";
    }();
    impl_->writer_queue = own_queue;
    if (own_queue) {
        if (publish_batch)
            stream_checked(hipHostMalloc(&impl_->pinned_publication, impl_->publication.size(), hipHostMallocDefault));
    }
    impl_->cache.resize(config.layers);
    for (unsigned l = 0; l < config.layers; ++l)
        if (config.layer[l].producer == l) {
            auto& c = impl_->cache[l];
            c.width = config.kv_heads * config.layer[l].head_dim;
            c.global = config.layer[l].attention == AttentionKind::global;
            c.slots = c.global ? admitted : config.window;
            c.k.resize(std::size_t(c.slots) * c.width * sizeof(float));
            c.v.resize(c.k.size());
            c.tags.resize(std::size_t(c.slots) * sizeof(std::uint64_t));
            std::vector<std::uint64_t> tags(c.slots, UINT64_MAX);
            stream_copy(impl_->queue, c.tags.data(), tags.data(), c.tags.size(), hipMemcpyHostToDevice);
        }
}
DeviceStream::~DeviceStream() = default;
std::unique_ptr<KvTxn> DeviceStream::begin() {
    if (impl_->busy || impl_->poisoned)
        throw std::runtime_error("device stream busy or failed");
    auto tx = std::make_unique<Impl::Txn>(impl_);
    impl_->busy = true;
    return tx;
}
std::uint64_t DeviceStream::frontier() const {
    return impl_->frontier;
}
KvView DeviceStream::kv_view(std::uint32_t layer, const KvTxn* transaction) const {
    const auto& c = impl_->cache.at(layer);
    if (!c.width || impl_->poisoned)
        throw std::runtime_error("layer has no device KV");
    KvView v{c.k.as<float>(), c.v.as<float>(), c.tags.as<std::uint64_t>(), c.width, c.slots, c.global};
    v.frontier = impl_->frontier;
    if (transaction) {
        const auto* tx = dynamic_cast<const Impl::Txn*>(transaction);
        if (!tx || tx->stream != impl_ || tx->published)
            throw std::invalid_argument("transaction does not belong to this stream");
        const auto r = tx->read(layer, tx->start);
        v.overlay_k = r.overlay_k; v.overlay_v = r.overlay_v; v.frontier = r.frontier; v.candidate_rows = r.candidate_rows;
    }
    return v;
}
std::unique_ptr<DeviceStream> DeviceStream::fork(std::uint32_t admitted) const {
    if (impl_->busy || impl_->poisoned || admitted < impl_->frontier)
        throw std::runtime_error("device stream cannot be forked");
    auto out = std::make_unique<DeviceStream>(impl_->config, admitted);
    // With a writer queue every publication of this stream already completed on it (commit synchronizes that queue
    // and the stream is not busy), so the copy runs there without waiting for other queues; the fork inherits it.
    // Without one, publication ran on the null stream; the copy starts after all device work.
    const auto q = static_cast<hipStream_t>(impl_->queue);
    if (!q)
        fork_checked(hipDeviceSynchronize());
    out->impl_->queue = impl_->queue;
    const auto copy = [&](void* dst, const void* src, std::size_t bytes) {
        fork_checked(q ? hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, q)
                       : hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
    };
    for (std::size_t l = 0; l < impl_->cache.size(); ++l) {
        const auto& from = impl_->cache[l];
        auto& to = out->impl_->cache[l];
        if (!from.width)
            continue;
        const auto slots = from.global ? impl_->frontier : std::uint64_t(from.slots);
        const auto bytes = slots * from.width * sizeof(float);
        copy(to.k.data(), from.k.data(), bytes);
        copy(to.v.data(), from.v.data(), bytes);
        copy(to.tags.data(), from.tags.data(), slots * sizeof(std::uint64_t));
    }
    if (q)
        fork_checked(hipStreamSynchronize(q));
    out->impl_->frontier = impl_->frontier;
    return out;
}
std::string DeviceStream::digest() const {
    if (impl_->poisoned)
        throw std::runtime_error("failed device stream has no canonical state");
    // STRIX_G4_DIGEST_BATCH (default on, read per call): one copy per cache array; =0 restores the reads of three
    // values per position. Both hash the same bytes in the same order.
    const auto* batch_switch = std::getenv("STRIX_G4_DIGEST_BATCH");
    const bool batched = !batch_switch || std::string_view(batch_switch) != "0";
    strix::Sha256Fast h;
    h.update({reinterpret_cast<const std::byte*>(&impl_->frontier), sizeof(impl_->frontier)});
    for (const auto& c : impl_->cache)
        if (c.width) {
            const auto first =
                c.global || impl_->frontier < impl_->config.window ? 0 : impl_->frontier - impl_->config.window;
            if (!batched) {
                std::vector<float> k(c.width), v(c.width);
                for (auto p = first; p < impl_->frontier; ++p) {
                    const auto slot = c.global ? p : p % c.slots;
                    std::uint64_t tag = UINT64_MAX;
                    c.tags.download(&tag, sizeof(tag), std::size_t(slot) * sizeof(tag));
                    if (tag != p)
                        throw std::runtime_error("committed device KV tag missing");
                    c.k.download(k.data(), k.size() * sizeof(float), std::size_t(slot) * k.size() * sizeof(float));
                    c.v.download(v.data(), v.size() * sizeof(float), std::size_t(slot) * v.size() * sizeof(float));
                    h.update({reinterpret_cast<const std::byte*>(&tag), sizeof(tag)});
                    h.update({reinterpret_cast<const std::byte*>(k.data()), k.size() * sizeof(float)});
                    h.update({reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(float)});
                }
                continue;
            }
            // One copy per array of the slots the committed range occupies (the whole sliding ring, global slots
            // [0, frontier)) on the stream's writer queue (null stream without one), one synchronization, then the
            // same per-position hash input as position-by-position reads.
            const auto q = static_cast<hipStream_t>(impl_->queue);
            const std::size_t used = c.global ? std::size_t(impl_->frontier) : std::size_t(c.slots);
            std::vector<std::uint64_t> tags(used);
            std::vector<float> ks(used * c.width), vs(used * c.width);
            if (used) {
                stream_checked(hipMemcpyAsync(tags.data(), c.tags.data(), used * sizeof(std::uint64_t), hipMemcpyDeviceToHost, q));
                stream_checked(hipMemcpyAsync(ks.data(), c.k.data(), ks.size() * sizeof(float), hipMemcpyDeviceToHost, q));
                stream_checked(hipMemcpyAsync(vs.data(), c.v.data(), vs.size() * sizeof(float), hipMemcpyDeviceToHost, q));
                stream_checked(hipStreamSynchronize(q));
            }
            for (auto p = first; p < impl_->frontier; ++p) {
                const auto slot = std::size_t(c.global ? p : p % c.slots);
                const std::uint64_t tag = tags[slot];
                if (tag != p)
                    throw std::runtime_error("committed device KV tag missing");
                h.update({reinterpret_cast<const std::byte*>(&tag), sizeof(tag)});
                h.update({reinterpret_cast<const std::byte*>(ks.data() + slot * c.width), c.width * sizeof(float)});
                h.update({reinterpret_cast<const std::byte*>(vs.data() + slot * c.width), c.width * sizeof(float)});
            }
        }
    return strix::model_content_identity_hex(h.finalize());
}
} // namespace strix::gemma4::gpu
