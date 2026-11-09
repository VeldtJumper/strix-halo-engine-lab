#include "strix/deepseek41_sampling.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

void greedy_and_filters() {
    strix::serve::SamplingParameters parameters;
    strix::deepseek41::TargetOnlySampler greedy(parameters);
    expect(!greedy.enabled(), "temperature zero stays greedy");
    expect(greedy.choose({}, 100, 9) == 9, "greedy needs no gathered scores");
    parameters.temperature = 1.0;
    parameters.top_k = 1;
    const std::array<float, 4> logits{0.0F, -1.0F, 3.0F, 1.0F};
    for (std::uint64_t seed = 0; seed < 64; ++seed) {
        parameters.seed = seed;
        strix::deepseek41::TargetOnlySampler sampler(parameters);
        expect(sampler.enabled(), "positive temperature enables the sampler");
        expect(sampler.choose(logits, 512, 0) == 2, "top-k one actually restricts the chosen token");
    }
    parameters.top_k = 0;
    parameters.top_p = 0.01;
    strix::deepseek41::TargetOnlySampler nucleus(parameters);
    expect(nucleus.choose(logits, 513, 0) == 2, "top-p is executed, not merely echoed");
    parameters.temperature = -1.0;
    bool rejected = false;
    try { strix::deepseek41::TargetOnlySampler invalid(parameters); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "invalid sampling parameters are rejected");
}

void seeded_requests() {
    strix::serve::SamplingParameters parameters;
    parameters.temperature = 1.0;
    parameters.seed = 17;
    const std::array<float, 4> logits{0.0F, 0.0F, 0.0F, 0.0F};
    strix::deepseek41::TargetOnlySampler first(parameters), replay(parameters), unrelated(parameters);
    parameters.seed = 18;
    strix::deepseek41::TargetOnlySampler other(parameters);
    std::vector<std::uint32_t> a, b;
    for (std::uint64_t position = 512; position < 576; ++position) {
        const auto token = first.choose(logits, position, 0);
        (void)unrelated.choose(logits, position + 91, 0);
        expect(token == replay.choose(logits, position, 0), "request-local seed replay survives interleaving");
        a.push_back(token);
        b.push_back(other.choose(logits, position, 0));
    }
    expect(a != b, "different seeds can choose different sequences");
}

void prefix_and_batch_contract() {
    std::vector<std::uint32_t> prompt(512, 1000);
    expect(strix::deepseek41::sampling_prefix_query(prompt, false).size() == 512, "greedy keeps exact whole-prefix hits");
    expect(strix::deepseek41::sampling_prefix_query(prompt, true).size() == 511, "sampling recomputes first-token scores");
    expect(strix::deepseek41::sampling_prefix_query(prompt, true, 32).size() == 480, "device prefill retains its minimum tail");
    prompt.push_back(1000);
    expect(strix::deepseek41::sampling_prefix_query(prompt, true, 32).size() < 512, "an appended token cannot select an undersized exact-hit tail");
    expect(strix::deepseek41::sampling_prefix_query({}, true).empty(), "empty prefix does not underflow");
    struct Sequence { bool sampled; int identity; } a{false, 1}, b{true, 2}, c{false, 3}, d{true, 4};
    std::vector<Sequence*> all{&a, &b, &c, &d};
    const auto groups = strix::deepseek41::split_sampling_batches(all);
    expect(groups[0] == std::vector<Sequence*>({&a, &c}), "greedy batch keeps its order and state owners");
    expect(groups[1] == std::vector<Sequence*>({&b, &d}), "sampled batch cannot enter greedy speculation");
}
}  // namespace

int main() {
    greedy_and_filters();
    seeded_requests();
    prefix_and_batch_contract();
    if (failures) return 1;
    std::cout << "deepseek41 target-only sampling contracts: PASS\n";
}
