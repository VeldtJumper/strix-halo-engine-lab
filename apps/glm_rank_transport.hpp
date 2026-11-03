#pragma once
// TP2 transport and Communicator setup shared by strix-glm5-rank and strix-glm5-serve.
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::glm::rank_transport {

// GPU-direct exchange opened by open_transport("usb4gpu"), consumed by make_communicator.
inline std::unique_ptr<strix::dist::DeviceExchange>& gpu_direct() {
    static std::unique_ptr<strix::dist::DeviceExchange> direct;
    return direct;
}

// The MiniMax rank app's transport setup (usb4-auto host path, usb4gpu GPU-direct when both ranks agree, tcp).
inline std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    const std::chrono::seconds limit(180);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoul(rest)), limit);
    if (kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return strix::dist::tcp_connect(rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), limit);
    }
    if (kind == "usb4-auto" || kind == "usb4gpu") {
        strix::dist::Usb4AutoOptions o;
        o.rank = rank;
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        receipt = topology.receipt();
        if (kind == "usb4gpu") {
            std::string why_direct, direct_receipt;
            if (strix::dist::usb4_gpu_direct_streams(o, *t, &why_direct) == 0U) {
                receipt += "; gpu-direct unavailable (" + why_direct + "): host path only";
                return t;
            }
            gpu_direct() = strix::dist::usb4_gpu_direct(o, topology, *t, {}, &direct_receipt);
            receipt += "; usb4 " + direct_receipt;
        }
        return t;
    }
    throw std::invalid_argument("unknown transport " + spec);
}

// The engine's Communicator: one exchange carries up to max(8, exchange_rows) rows of 4096 floats.
inline std::unique_ptr<strix::dist::Communicator> make_communicator(std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport,
                                                                    unsigned exchange_rows) {
    strix::dist::Group group;
    group.rank = rank;
    group.world = 2;
    group.plan = 0x474c4d35ULL;  // "GLM5"
    const std::size_t max_bytes = std::size_t{std::max(8U, exchange_rows)} * 4096 * sizeof(float) + 4096;
    auto& direct = gpu_direct();
    return direct ? std::make_unique<strix::dist::Communicator>(group, std::move(transport), std::move(direct), max_bytes)
                  : std::make_unique<strix::dist::Communicator>(group, std::move(transport), max_bytes);
}

}  // namespace strix::glm::rank_transport
