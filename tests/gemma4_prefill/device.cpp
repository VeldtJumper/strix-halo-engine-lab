#include "gemma4/prefill/device.hpp"
#include "gemma4/prefill/reference.hpp"
#include "gemma4/kernels/ops.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <limits>
#include <numeric>
#include <random>

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
namespace gpu = strix::gemma4::gpu;

namespace {
void conversion_tests() {
    std::vector<float> values;
    for (unsigned h = 0; h < 0x7bffU; ++h) {
        const float a = fp16_value(std::uint16_t(h)), b = fp16_value(std::uint16_t(h + 1));
        const float midpoint = float((double(a) + double(b)) * 0.5);
        for (float v : {std::nextafter(midpoint, a), midpoint, std::nextafter(midpoint, b)}) {
            values.push_back(v); values.push_back(-v);
        }
    }
    gpu::Buffer weights(values.size() * sizeof(float)), input(512 * sizeof(float)), output(values.size() * sizeof(float));
    gpu::Buffer scratch(512 * 16 * sizeof(std::uint16_t)), failure(sizeof(unsigned)), maximum(sizeof(unsigned));
    weights.upload(values.data(), weights.size());
    const float one = 1.0F; input.upload(&one, sizeof(one)); failure.zero(); maximum.zero();
    LinearView w{weights.as<std::uint8_t>(), unsigned(values.size()), 1, sizeof(float), WeightFormat::f32};
    pf::linear(w, input.as<float>(), 1, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
    gpu::synchronize();
    unsigned flag = 0; failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
    std::vector<float> actual(values.size()); output.download(actual.data(), output.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint16_t h = 0; CHECK(pf::half_bits(values[i], h));
        std::array<std::uint16_t, 16> a{}, b{}; a[0] = 0x3c00U; b[0] = h;
        CHECK(bit_cast<unsigned>(actual[i]) == bit_cast<unsigned>(pf::instruction(a, b, 0.0F)));
    }
    weights.upload(&one, sizeof(one)); w.outputs = 1;
    for (std::size_t first = 0; first < values.size(); first += 512) {
        const auto count = unsigned(std::min<std::size_t>(512, values.size() - first));
        input.upload(values.data() + first, std::size_t(count) * sizeof(float));
        pf::linear(w, input.as<float>(), count, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
        gpu::synchronize();
        std::vector<std::uint16_t> converted(std::size_t(count) * 16);
        scratch.download(converted.data(), converted.size() * sizeof(std::uint16_t));
        for (unsigned r = 0; r < count; ++r) {
            std::uint16_t h = 0; CHECK(pf::half_bits(values[first + r], h)); CHECK(converted[std::size_t(r) * 16] == h);
            for (unsigned k = 1; k < 16; ++k) CHECK(converted[std::size_t(r) * 16 + k] == 0);
        }
    }
    failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
    for (float bad : {65520.0F, -65520.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        input.upload(&bad, sizeof(bad)); failure.zero(); maximum.zero();
        pf::linear(w, input.as<float>(), 1, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
        gpu::synchronize(); failure.download(&flag, sizeof(flag)); CHECK((flag & 4U) != 0);
        input.upload(&one, sizeof(one)); weights.upload(&bad, sizeof(bad)); failure.zero(); maximum.zero();
        refuses([&] { pf::PreparedLinear invalid(w); });
        pf::linear(w, input.as<float>(), 1, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
        gpu::synchronize(); failure.download(&flag, sizeof(flag)); CHECK((flag & 4U) != 0);
        weights.upload(&one, sizeof(one));
    }
    std::puts("prefill native FP16 midpoint/activation/weight conversion PASS");
}
void layout_tests() {
    for (const auto format : {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q4_k, WeightFormat::q6_k, WeightFormat::q8_0}) {
        constexpr unsigned width = 512, outputs = 67, rows = 65;
        const auto packed = fixture::packed(format, outputs, width);
        gpu::Buffer weights(packed.size()), input(rows * width * sizeof(float)), output(rows * outputs * sizeof(float));
        gpu::Buffer scratch(rows * width * sizeof(std::uint16_t)), failure(sizeof(unsigned)), maximum(sizeof(unsigned));
        weights.upload(packed.data(), packed.size());
        std::vector<float> x(rows * width);
        for (unsigned r = 0; r < rows; ++r) x[std::size_t(r) * width + (r * 23) % width] = 1.0F;
        input.upload(x.data(), input.size()); failure.zero(); maximum.zero();
        LinearView w{weights.as<std::uint8_t>(), outputs, width, packed_row_bytes(format, width), format};
        pf::linear(w, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
        gpu::synchronize();
        unsigned flag = 0; failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
        std::vector<float> actual(rows * outputs); output.download(actual.data(), output.size());
        w.data = packed.data();
        bytes(actual, pf::emulate_linear(w, x, rows));
    }
    std::puts("prefill packed/tile/layout PASS");
}
void tile_tests() {
    for (const auto format : {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q4_k, WeightFormat::q6_k, WeightFormat::q8_0}) {
        const bool quant = format == WeightFormat::q4_k || format == WeightFormat::q6_k || format == WeightFormat::q8_0;
        for (const unsigned width : quant ? std::vector<unsigned>{256, 512} : std::vector<unsigned>{1, 15, 16, 17, 257, 10240, 10752}) {
            for (const unsigned rows : width > 1024 ? std::vector<unsigned>{3} : std::vector<unsigned>{1, 8, 17}) {
                const unsigned outputs = 19;
                const auto packed = fixture::packed(format, outputs, width, width + 71);
                gpu::Buffer weights(packed.size()), input(std::size_t(rows) * width * sizeof(float));
                gpu::Buffer output(rows * outputs * sizeof(float)), reference(output.size());
                gpu::Buffer scratch(std::size_t(rows) * ((width + 15U) / 16U) * 16 * sizeof(std::uint16_t)), failure(sizeof(unsigned)), maximum(sizeof(unsigned));
                weights.upload(packed.data(), packed.size());
                std::vector<float> x(std::size_t(rows) * width);
                std::mt19937 rng(width + 91);
                for (auto& a : x) a = float(int(rng() % 65535) - 32767) * 0.0009765625F;
                input.upload(x.data(), input.size()); failure.zero(); maximum.zero();
                const LinearView w{weights.as<std::uint8_t>(), outputs, width, packed_row_bytes(format, width), format};
                pf::linear(w, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>());
                pf::linear(w, input.as<float>(), rows, reference.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(), nullptr, false);
                gpu::synchronize();
                unsigned flag = 0; failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
                std::vector<float> actual(rows * outputs), expected(rows * outputs);
                output.download(actual.data(), output.size()); reference.download(expected.data(), reference.size()); bytes(actual, expected);
                auto host_weights = w; host_weights.data = packed.data();
                bytes(actual, pf::emulate_linear(host_weights, x, rows));
                pf::PreparedLinear prepared(w);
                pf::linear(w, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(), nullptr, true, &prepared);
                gpu::synchronize(); failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
                output.download(actual.data(), output.size()); bytes(actual, expected);
            }
        }
    }
    std::puts("prefill shared/prepared/reference tile equality PASS");
}
ForwardResult decode(gpu::DeviceTarget& target, gpu::DeviceStream& stream, std::span<const unsigned> tokens) {
    auto tx = stream.begin();
    std::vector<RowPlan> rows;
    for (unsigned r = 0; r < tokens.size(); ++r) rows.push_back({1, 1, tx->frontier() + r, tokens[r], tx.get(), r, true});
    auto out = target.forward_rows(rows); tx->commit(tokens.size()); return out;
}
void split_tests() {
    const auto model = fixture::tiny(6);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    pf::DevicePrefill target(weights, tables.view());
    gpu::DeviceTarget decoder(weights, tables.view());
    std::vector<unsigned> tokens(35);
    for (unsigned r = 0; r < tokens.size(); ++r) tokens[r] = r % model.config.vocab;
    gpu::DeviceStream canonical(model.config, 1024);
    const auto expected = target.run(canonical, tokens);
    const auto digest = canonical.digest();
    { pf::DevicePrefill borrowed(target.borrow_device_weights(), tables.view(), pf::Residency::device);
      gpu::DeviceStream stream(model.config, 1024);
      const auto out = borrowed.run(stream, tokens);
      bytes(out.last_logits, expected.last_logits); bytes(out.tail_taps, expected.tail_taps); CHECK(stream.digest() == digest); }
    pf::Reference reference(weights, tables.view(), 1024);
    const auto host = reference.run(tokens);
    bytes(host.last_logits, expected.last_logits); bytes(host.tail_taps, expected.tail_taps);
    CHECK(reference.digest() == digest);
    { gpu::DeviceStream all(model.config, 1024); auto tx = all.begin();
      const auto out = target.forward(*tx, tokens, true);
      bytes(out.last_logits, expected.last_logits);
      bytes(std::vector<float>(out.all_logits.end() - model.config.vocab, out.all_logits.end()), expected.last_logits); }
    for (unsigned split = 0; split <= tokens.size(); ++split) {
        gpu::DeviceStream stream(model.config, 1024);
        pf::Prompt prompt(target, stream);
        prompt.append(std::span(tokens).first(split)); prompt.append(std::span(tokens).subspan(split));
        const auto out = prompt.finish();
        CHECK(out.frontier == tokens.size()); bytes(out.last_logits, expected.last_logits); bytes(out.tail_taps, expected.tail_taps);
        CHECK(stream.digest() == digest); refuses([&] { prompt.append({}); }); refuses([&] { (void)prompt.finish(); });
    }
    // Cross a fixed prompt chunk and ring edge; fragment every input separately.
    tokens.resize(521);
    for (unsigned r = 0; r < tokens.size(); ++r) tokens[r] = (r * 7) % model.config.vocab;
    gpu::DeviceStream a(model.config, 1024), b(model.config, 1024);
    const auto whole = target.run(a, tokens);
    pf::Prompt fragmented(target, b);
    for (const auto& token : tokens) fragmented.append({&token, 1});
    const auto one = fragmented.finish();
    bytes(whole.last_logits, one.last_logits); bytes(whole.tail_taps, one.tail_taps); CHECK(a.digest() == b.digest());
    // Decode rows from the identical WMMA state remain canonical, including
    // hypothetical verify inputs. The speculative verdict is coordinator-owned.
    const std::array<unsigned, 4> next = {3, 1, 5, 8};
    const auto batched = decode(decoder, a, next);
    for (unsigned r = 0; r < next.size(); ++r) {
        const auto serial = decode(decoder, b, {&next[r], 1});
        bytes(serial.logits, std::vector<float>(batched.logits.begin() + std::ptrdiff_t(r * model.config.vocab), batched.logits.begin() + std::ptrdiff_t((r + 1) * model.config.vocab)));
        bytes(serial.taps, std::vector<float>(batched.taps.begin() + std::ptrdiff_t(r * 5 * model.config.hidden), batched.taps.begin() + std::ptrdiff_t((r + 1) * 5 * model.config.hidden)));
    }
    CHECK(a.digest() == b.digest());
    // Rejected overlays do not replace committed prompt state.
    const auto before = a.digest();
    { auto tx = a.begin(); (void)target.forward(*tx, std::span(next).first(1)); }
    CHECK(a.digest() == before);
    std::puts("prefill split/commit/canonical decode PASS");
}
// PINNED_TAPS: pinned views (taps consumed, hidden rows) of a full and a
// partial chunk are byte-equal to the vectors of views=false; taps=false
// changes no logits, hidden row or committed KV.
// A model with fewer layers than a tap index leaves that tap's columns at
// zero on the pinned path too. The device tap buffer is poisoned first: freed
// blocks of its exact size filled with NaN bytes are what the allocator hands
// back, so a missing zero shows as NaN instead of passing on fresh memory.
void pinned_absent_taps_tests() {
    const auto model = fixture::tiny(6);
    const std::size_t tap_bytes = std::size_t(pf::chunk_rows) * model.config.hidden * sizeof(float) * 5;
    {
        const std::vector<std::uint8_t> poison(tap_bytes, 0xff);
        std::vector<gpu::Buffer> blocks;
        for (unsigned i = 0; i < 8; ++i) { blocks.emplace_back(tap_bytes); blocks.back().upload(poison.data(), tap_bytes); }
    }
    fixture::Tables tables;
    pf::DevicePrefill target(bind_host_weights(model), tables.view());
    std::vector<unsigned> tokens(37);
    for (unsigned r = 0; r < tokens.size(); ++r) tokens[r] = (r * 3 + 1) % model.config.vocab;
    gpu::DeviceStream stream(model.config, 1024);
    auto tx = stream.begin();
    const auto out = target.forward(*tx, tokens, false, true, false, true, true);
    const auto taps = out.tap_rows();
    CHECK(taps.size() == tokens.size() * std::size_t(out.tap_width));
    const auto hidden = model.config.hidden;
    for (unsigned t = 0; t < 5; ++t) {
        if (tap_layers[t] < model.config.layers) continue;
        for (unsigned r = 0; r < tokens.size(); ++r)
            for (unsigned i = 0; i < hidden; ++i) CHECK(taps[std::size_t(r) * out.tap_width + t * hidden + i] == 0.0F);
    }
    std::puts("prefill pinned taps zero absent tap layers PASS");
}
void pinned_views_tests() {
    const auto model = fixture::tiny(6);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    pf::DevicePrefill target(weights, tables.view());
    std::vector<unsigned> tokens(pf::chunk_rows + 9);
    for (unsigned r = 0; r < tokens.size(); ++r) tokens[r] = (r * 5 + 3) % model.config.vocab;
    const auto vec = [](std::span<const float> v) { return std::vector<float>(v.begin(), v.end()); };
    gpu::DeviceStream plain(model.config, 1024), views(model.config, 1024), untapped(model.config, 1024);
    for (unsigned start = 0; start < tokens.size(); start += pf::chunk_rows) {
        const auto chunk = std::span(tokens).subspan(start, std::min<std::size_t>(pf::chunk_rows, tokens.size() - start));
        auto ta = plain.begin();
        const auto a = target.forward(*ta, chunk, false, true, true, true, false);
        const auto a_taps = vec(a.tap_rows()), a_hidden = vec(a.hidden_rows());
        CHECK(a.pinned_taps.empty() && a_taps.size() == chunk.size() * a.tap_width && a_hidden.size() == chunk.size() * model.config.hidden);
        ta->commit(unsigned(chunk.size()));
        auto tb = views.begin();
        const auto b = target.forward(*tb, chunk, false, true, true, true, true);
        CHECK(!b.pinned_taps.empty() && !b.pinned_hidden.empty());
        bytes(b.last_logits, a.last_logits); bytes(vec(b.tap_rows()), a_taps); bytes(vec(b.hidden_rows()), a_hidden);
        tb->commit(unsigned(chunk.size()));
        auto tc = untapped.begin();
        const auto c = target.forward(*tc, chunk, false, true, true, false, true);
        CHECK(c.tap_rows().empty());
        bytes(c.last_logits, a.last_logits); bytes(vec(c.hidden_rows()), a_hidden);
        tc->commit(unsigned(chunk.size()));
    }
    CHECK(plain.digest() == views.digest() && plain.digest() == untapped.digest());
    std::puts("prefill pinned views PASS");
}
void alias_tests() {
    const auto model = fixture::tiny(42);
    fixture::Tables tables;
    pf::DevicePrefill target(bind_host_weights(model), tables.view());
    std::vector<unsigned> tokens(513);
    for (unsigned r = 0; r < tokens.size(); ++r) tokens[r] = (r * 7) % model.config.vocab;
    gpu::DeviceStream a(model.config, 1024), b(model.config, 1024);
    const auto expected = target.run(a, tokens);
    pf::Prompt prompt(target, b);
    for (const auto token : tokens) prompt.append({&token, 1});
    const auto out = prompt.finish();
    bytes(out.last_logits, expected.last_logits); bytes(out.tail_taps, expected.tail_taps); CHECK(a.digest() == b.digest());
    std::puts("prefill shared KV 22/23 and ring crossing PASS");
}
void c4_tests() {
    const auto model = fixture::tiny(6);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    pf::DevicePrefill prefill(weights, tables.view());
    gpu::DeviceTarget decoder(weights, tables.view());
    std::array<std::unique_ptr<gpu::DeviceStream>, 4> streams;
    std::array<std::unique_ptr<pf::Reference>, 4> reference;
    std::array<unsigned, 4> anchors{};
    for (unsigned i = 0; i < 4; ++i) {
        streams[i] = std::make_unique<gpu::DeviceStream>(model.config, 1024);
        reference[i] = std::make_unique<pf::Reference>(weights, tables.view(), 1024);
        std::vector<unsigned> tokens(1 + i * 5, i + 2);
        const auto out = prefill.run(*streams[i], tokens), expected = reference[i]->run(tokens);
        bytes(out.last_logits, expected.last_logits); CHECK(streams[i]->digest() == reference[i]->digest());
        anchors[i] = i + 2;
    }
    for (unsigned step = 0; step < 3; ++step) {
        std::array<std::unique_ptr<KvTxn>, 4> tx;
        std::vector<RowPlan> rows;
        for (unsigned i = 0; i < 4; ++i) {
            tx[i] = streams[i]->begin(); rows.push_back({i, 7, tx[i]->frontier(), anchors[i], tx[i].get(), 3 - i, true});
        }
        const auto out = decoder.forward_rows(rows);
        for (unsigned i = 0; i < 4; ++i) {
            tx[i]->commit(1); const auto expected = reference[i]->forward({&anchors[i], 1}, false);
            const auto first = std::size_t(3 - i) * out.vocab;
            bytes(expected.last_logits, std::vector<float>(out.logits.begin() + std::ptrdiff_t(first), out.logits.begin() + std::ptrdiff_t(first + out.vocab)));
            const auto tap = std::size_t(3 - i) * out.tap_width;
            bytes(expected.taps, std::vector<float>(out.taps.begin() + std::ptrdiff_t(tap), out.taps.begin() + std::ptrdiff_t(tap + out.tap_width)));
            CHECK(streams[i]->digest() == reference[i]->digest()); anchors[i] = (anchors[i] + 3) % model.config.vocab;
        }
    }
    std::puts("prefill unequal-context C4/permuted slots versus host C1 PASS");
}
void context_callback_tests() {
    auto model = fixture::tiny(1); model.config.max_positions = 2048;
    fixture::Tables tables(2048);
    pf::DevicePrefill prefill(bind_host_weights(model), tables.view());
    gpu::DeviceStream expected(model.config, 2048), stream(model.config, 2048);
    struct Context final : DraftContext {
        gpu::DeviceStream& stream;
        std::uint64_t frontier = 0;
        std::vector<float> taps;
        explicit Context(gpu::DeviceStream& state) : stream(state) {}
        void append(std::span<const float> values, std::span<const std::uint64_t> positions) override {
            CHECK(!positions.empty() && positions.size() <= 8 && positions.front() == frontier);
            CHECK(positions.back() < stream.frontier());
            taps.insert(taps.end(), values.begin(), values.end()); frontier += positions.size();
        }
    } context(stream);
    std::vector<unsigned> tokens(1049);
    for (unsigned i = 0; i < tokens.size(); ++i) tokens[i] = i % model.config.vocab;
    const auto whole = prefill.run(expected, tokens);
    (void)prefill.run(stream, std::span(tokens).first(16), &context);
    CHECK(context.frontier == 16);
    const auto extended = prefill.run(stream, std::span(tokens).subspan(16), &context);
    CHECK(context.frontier == tokens.size() && context.taps.size() == tokens.size() * 5 * model.config.hidden);
    bytes(whole.last_logits, extended.last_logits); bytes(whole.tail_taps, extended.tail_taps);
    CHECK(stream.digest() == expected.digest());
    bytes(extended.tail_taps, std::vector<float>(context.taps.end() - std::ptrdiff_t(extended.tail_taps.size()), context.taps.end()));
    std::puts("prefill committed draft callback and long stream extension PASS");
}
void overflow_test() {
    auto model = fixture::tiny(1);
    bool injected = false;
    for (auto& t : model.tensors)
        if (t.name == "model.language_model.embed_tokens.weight") {
            // BF16 embedding multiplication stays finite FP32 but exceeds FP16.
            auto data = std::make_shared<std::vector<std::uint8_t>>(std::size_t(t.view.outputs) * t.view.row_bytes);
            for (std::size_t i = 0; i < data->size(); i += 2) fixture::put16(data->data() + i, bf16_bits(2000.0F));
            t.view.data = data->data(); model.storage.push_back(data);
            injected = true;
        }
    CHECK(injected);
    fixture::Tables tables;
    pf::DevicePrefill target(bind_host_weights(model), tables.view());
    gpu::DeviceStream stream(model.config, 64);
    const auto before = stream.digest();
    { auto tx = stream.begin(); const unsigned token = 1;
      refuses([&] { (void)target.forward(*tx, {&token, 1}); }); refuses([&] { tx->commit(1); }); }
    CHECK(stream.frontier() == 0 && stream.digest() == before);
    std::puts("prefill FP16 overflow rollback PASS");
}
// The fused gate/up kernel with its v2 low panel feeds a corrected down with
// exactly the bytes of the separate path (FP32 gelu product, split, corrected).
void v2_pair_tests() {
    constexpr unsigned hidden = 512, inter = 1024;
    for (const unsigned rows : {3U, 70U, 256U, 300U}) for (const auto precision : {pf::LinearPrecision::hilo, pf::LinearPrecision::hilo_k128})
    for (const auto pair_mode : {pf::LinearPrecision::fp16, pf::LinearPrecision::hilo, pf::LinearPrecision::hilo_k128}) {
        const bool split = pair_mode != pf::LinearPrecision::fp16;
        const auto pg = fixture::packed(WeightFormat::bf16, inter, hidden, 101), pu = fixture::packed(WeightFormat::bf16, inter, hidden, 202);
        const auto pd = fixture::packed(WeightFormat::bf16, hidden, inter, 303);
        gpu::Buffer g(pg.size()), u(pu.size()), d(pd.size()), x(std::size_t(rows) * hidden * 4), gate(std::size_t(rows) * inter * 4), up(gate.size());
        gpu::Buffer fused(std::size_t(rows) * hidden * 4), separate(fused.size()), failure(4), maximum(4 * 4);
        gpu::Buffer pair_scratch(2 * std::size_t(rows) * hidden * 2), product(2 * std::size_t(rows) * inter * 2), scratch(product.size());
        g.upload(pg.data(), pg.size()); u.upload(pu.data(), pu.size()); d.upload(pd.data(), pd.size());
        std::vector<float> host(std::size_t(rows) * hidden);
        std::mt19937 rng(rows);
        for (auto& a : host) a = float(int(rng() % 20001) - 10000) / 2500.0F;
        x.upload(host.data(), x.size()); failure.zero(); maximum.zero();
        const LinearView wg{g.as<std::uint8_t>(), inter, hidden, packed_row_bytes(WeightFormat::bf16, hidden), WeightFormat::bf16};
        const LinearView wu{u.as<std::uint8_t>(), inter, hidden, packed_row_bytes(WeightFormat::bf16, hidden), WeightFormat::bf16};
        const LinearView wd{d.as<std::uint8_t>(), hidden, inter, packed_row_bytes(WeightFormat::bf16, inter), WeightFormat::bf16};
        pf::PreparedLinear prepared_g(wg), prepared_u(wu), prepared_d(wd);
        CHECK(pf::gelu_pair_supported(prepared_g, prepared_u));
        pf::gelu_pair(prepared_g, prepared_u, x.as<float>(), rows, pair_scratch.data(), product.data(), failure.as<unsigned>(), maximum.as<unsigned>(), nullptr, true, split,
                      pair_mode == pf::LinearPrecision::hilo_k128);
        pf::linear(wd, gate.as<float>(), rows, fused.as<float>(), product.data(), failure.as<unsigned>(), maximum.as<unsigned>() + 3,
                   nullptr, true, &prepared_d, nullptr, true, precision);
        pf::linear(wg, x.as<float>(), rows, gate.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>() + 3, nullptr, true, &prepared_g,
                   nullptr, false, pair_mode);
        pf::linear(wu, x.as<float>(), rows, up.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>() + 3, nullptr, true, &prepared_u,
                   nullptr, false, pair_mode);
        gpu::gelu_product(gate.as<float>(), up.as<float>(), rows * inter);
        pf::linear(wd, gate.as<float>(), rows, separate.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>() + 3,
                   nullptr, true, &prepared_d, nullptr, false, precision);
        gpu::synchronize();
        unsigned flag = 0; failure.download(&flag, 4); CHECK(flag == 0);
        std::vector<float> a(std::size_t(rows) * hidden), b(a.size());
        fused.download(a.data(), fused.size()); separate.download(b.data(), separate.size()); bytes(a, b);
    }
    std::puts("prefill fused gate/up (fp16, hilo and hilo_k128) low panel == separate v2 path PASS");
}
// g4-wmma-pf-v2 corrected linears against emulate_linear_v2, byte for byte:
// partial and multi-block K, one-row (head) and many-row tiles, BF16/F32/Q4_K,
// cancellation-heavy activations with -0 and wide magnitudes.
void v2_case(WeightFormat format, unsigned width, unsigned rows, unsigned outputs, pf::LinearPrecision precision);
void v2_tests() {
    for (const auto format : {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q4_k}) {
        const bool quant = format == WeightFormat::q4_k;
        for (const unsigned width : quant ? std::vector<unsigned>{256, 2560} : std::vector<unsigned>{17, 48, 128, 129, 2560, 10240})
            for (const unsigned rows : width > 4096 ? std::vector<unsigned>{1, 40} : std::vector<unsigned>{1, 7, 33, 130})
                for (const auto precision : {pf::LinearPrecision::hilo, pf::LinearPrecision::hilo_k128})
                    v2_case(format, width, rows, 83, precision);
    }
    // Production routes: outputs > 1024 select hilo_direct (unblocked) and,
    // with K >= 8192 blocked, staged_hilo; 256 x 2560 is all full tiles.
    struct Shape { unsigned width, rows, outputs; };
    for (const auto& shape : {Shape{2560, 256, 2560}, Shape{2560, 130, 1040}, Shape{10240, 130, 1040}})
        for (const auto precision : {pf::LinearPrecision::hilo, pf::LinearPrecision::hilo_k128})
            v2_case(WeightFormat::q4_k, shape.width, shape.rows, shape.outputs, precision);
    std::puts("prefill g4-wmma-pf-v2 hilo/hilo_k128 linears == emulator PASS");
}
void v2_case(WeightFormat format, unsigned width, unsigned rows, unsigned outputs, pf::LinearPrecision precision) {
    const auto stride = (width + 15U) / 16U * 16U;
    const auto packed = fixture::packed(format, outputs, width, width + 13);
    gpu::Buffer weights(packed.size()), input(std::size_t(rows) * width * sizeof(float)), output(std::size_t(rows) * outputs * sizeof(float));
    gpu::Buffer scratch(2 * std::size_t(rows) * stride * sizeof(std::uint16_t)), failure(sizeof(unsigned)), maximum(sizeof(unsigned));
    weights.upload(packed.data(), packed.size());
    std::vector<float> x(std::size_t(rows) * width);
    std::mt19937 rng(width * 7 + rows);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float u = float(int(rng() % 2000001) - 1000000) / 1000000.0F;
        const unsigned kind = rng() % 16;
        x[i] = kind == 0 ? -0.0F : kind == 1 ? u * 1700.0F : kind == 2 ? u * 1e-5F : kind == 3 ? 1.0F + u * 1e-3F : u * 8.0F;
    }
    input.upload(x.data(), input.size()); failure.zero(); maximum.zero();
    const LinearView w{weights.as<std::uint8_t>(), outputs, width, packed_row_bytes(format, width), format};
    pf::PreparedLinear prepared(w);
    pf::linear(w, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(),
               nullptr, true, &prepared, nullptr, false, precision);
    gpu::synchronize();
    unsigned flag = 0; failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
    std::vector<float> actual(std::size_t(rows) * outputs); output.download(actual.data(), output.size());
    auto host_weights = w; host_weights.data = packed.data();
    bytes(actual, pf::emulate_linear_v2(host_weights, x, rows, precision == pf::LinearPrecision::hilo_k128));
}
} // namespace
int main() try { conversion_tests(); layout_tests(); tile_tests(); v2_tests(); v2_pair_tests(); pinned_absent_taps_tests(); split_tests(); pinned_views_tests(); alias_tests(); c4_tests(); context_callback_tests(); overflow_test(); return 0; }
catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
