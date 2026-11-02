#include <hip/hip_runtime.h>
#include "storage.hpp"
#include "gemma4/kernels/internal.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <cstdlib>
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
} // namespace
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
            stream->publication.upload(descriptors.data(), descriptors.size() * sizeof(detail::Publication));
            hipLaunchKernelGGL(publish_batch_kernel, dim3(unsigned(descriptors.size())), dim3(128), 0, nullptr,
                               stream->publication.as<detail::Publication>(), start, unsigned(count));
            launch_checked();
        } else if (count)
            for (std::size_t l = 0; l < stream->cache.size(); ++l) {
                auto& c = stream->cache[l];
                if (!c.width)
                    continue;
                hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(128), 0, nullptr, c.k.as<float>(), c.v.as<float>(),
                                   c.tags.as<std::uint64_t>(), overlay[l].k.as<float>(), overlay[l].v.as<float>(),
                                   start, unsigned(count), c.slots, c.width, c.global);
                launch_checked();
            }
        synchronize();
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
            c.tags.upload(tags.data(), c.tags.size());
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
    // Publication runs on the target queue; the copy starts after all of it.
    fork_checked(hipDeviceSynchronize());
    for (std::size_t l = 0; l < impl_->cache.size(); ++l) {
        const auto& from = impl_->cache[l];
        auto& to = out->impl_->cache[l];
        if (!from.width)
            continue;
        const auto slots = from.global ? impl_->frontier : std::uint64_t(from.slots);
        const auto bytes = slots * from.width * sizeof(float);
        fork_checked(hipMemcpy(to.k.data(), from.k.data(), bytes, hipMemcpyDeviceToDevice));
        fork_checked(hipMemcpy(to.v.data(), from.v.data(), bytes, hipMemcpyDeviceToDevice));
        fork_checked(hipMemcpy(to.tags.data(), from.tags.data(), slots * sizeof(std::uint64_t), hipMemcpyDeviceToDevice));
    }
    out->impl_->frontier = impl_->frontier;
    return out;
}
std::string DeviceStream::digest() const {
    if (impl_->poisoned)
        throw std::runtime_error("failed device stream has no canonical state");
    strix::Sha256Fast h;
    h.update({reinterpret_cast<const std::byte*>(&impl_->frontier), sizeof(impl_->frontier)});
    for (const auto& c : impl_->cache)
        if (c.width) {
            const auto first =
                c.global || impl_->frontier < impl_->config.window ? 0 : impl_->frontier - impl_->config.window;
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
        }
    return strix::model_content_identity_hex(h.finalize());
}
} // namespace strix::gemma4::gpu
