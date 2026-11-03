// GLM-5 chat template: the C++ renderer equals HF transformers apply_chat_template on the model's chat_template.jinja
// (cases written by tools/glm/make_chat_fixtures.py), and the output parser splits reasoning, content and tool calls.
//   strix-glm-chat-tests tests/data/serve/glm_chat_cases.json
#include "strix/serve/glm_chat.hpp"
#include "strix/serve/json.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

using strix::serve::ChatTemplateInput;
using strix::serve::Json;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}
std::string escape(const std::string& s) {
    std::string out;
    strix::serve::append_json_string(out, s);
    return out;
}

ChatTemplateInput input_of(const Json& c) {
    ChatTemplateInput input;
    input.messages = *c.find("messages");
    if (const auto* t = c.find("tools"); t && t->is_array()) input.tools = *t;
    const auto* options = c.find("options");
    if (const auto* g = options->find("add_generation_prompt")) input.add_generation_prompt = g->as_bool();
    if (const auto* e = options->find("reasoning_effort")) input.reasoning_effort = e->as_string();
    if (const auto* k = options->find("clear_thinking")) input.preserve_thinking = !k->as_bool();
    return input;
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::serve;
    if (argc != 2) {
        std::cerr << "usage: strix-glm-chat-tests CASES.json\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto cases = parse_json(text);
    std::size_t index = 0, equal = 0;
    for (const auto& c : cases.find("render")->as_array()) {
        const auto& name = c.find("name")->as_string();
        const auto& want = c.find("expected")->as_string();
        std::string got;
        try { got = render_glm_chat(input_of(c)); }
        catch (const ChatTemplateError& e) { got = std::string("error: ") + e.what(); }
        if (got == want) ++equal;
        else {
            std::size_t k = 0;
            while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
            check(false, name + " differs at byte " + std::to_string(k) + ": got " + escape(got.substr(k, 80)) + " want " + escape(want.substr(k, 80)));
        }
        ++index;
    }
    std::cout << "glm chat: " << equal << " of " << index << " cases equal the HF template\n";
    // the stable prefix the API hands the engine (leading system messages and tools rendered alone)
    std::size_t prefixes = 0;
    for (const auto& c : cases.find("render")->as_array()) {
        const auto& messages = c.find("messages")->as_array();
        std::size_t lead = 0;
        while (lead + 1U < messages.size() && messages[lead].find("role")->as_string() == "system") ++lead;
        if (lead == 0U) continue;
        const auto input = input_of(c);
        ChatTemplateInput head = input;
        Json system = Json::array();
        for (std::size_t i = 0; i < lead; ++i) system.push(messages[i]);
        head.messages = system;
        head.add_generation_prompt = false;
        const auto full = render_glm_chat(input), part = render_glm_chat(head);
        check(full.compare(0, part.size(), part) == 0 && part.size() < full.size(), c.find("name")->as_string() + ": system + tools rendered alone begin the prompt");
        ++prefixes;
    }
    std::cout << "glm chat: " << prefixes << " stable prefixes checked\n";

    // rejected inputs
    {
        ChatTemplateInput input;
        input.messages = parse_json(R"([{"role":"user","content":{"a":1}}])");
        bool threw = false;
        try { (void)render_glm_chat(input); } catch (const ChatTemplateError&) { threw = true; }
        check(threw, "object content is rejected");
        input.messages = parse_json(R"([{"role":"tool","content":[{"type":"tool_reference","name":"x"}]}])");
        threw = false;
        try { (void)render_glm_chat(input); } catch (const ChatTemplateError&) { threw = true; }
        check(threw, "tool_reference without tools is rejected");
        input.messages = parse_json(R"([{"role":"user","content":"q"},{"role":"assistant","content":"",
            "tool_calls":[{"function":{"name":"f","arguments":"{\"a\": 1, \"b\": \"x\"}"}}]}])");
        input.add_generation_prompt = false;
        check(render_glm_chat(input).ends_with("<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>x</arg_value></tool_call>"),
              "JSON-string arguments are parsed");
    }

    // output parsing
    std::size_t parsed = 0;
    for (const auto& c : cases.find("parse")->as_array()) {
        const auto& name = c.find("name")->as_string();
        const auto* tools = c.find("tools");
        const auto out = parse_glm_output(c.find("text")->as_string(), true, tools ? *tools : Json());
        check(out.reasoning == c.find("reasoning")->as_string(), name + ": reasoning " + escape(out.reasoning));
        check(out.reasoning_complete == c.find("reasoning_complete")->as_bool(), name + ": reasoning_complete");
        check(out.content == c.find("content")->as_string(), name + ": content " + escape(out.content));
        check(out.tool_calls.dump() == c.find("tool_calls")->dump(), name + ": tool calls " + out.tool_calls.dump());
        const auto& complete = c.find("call_complete")->as_array();
        bool same = complete.size() == out.call_complete.size();
        for (std::size_t i = 0; same && i < complete.size(); ++i) same = complete[i].as_bool() == out.call_complete[i];
        check(same, name + ": call_complete");
        check(out.unfinished_call == c.find("unfinished_call")->as_bool(), name + ": unfinished_call");
        ++parsed;
    }
    std::cout << "glm chat: " << parsed << " parse cases checked\n";
    // a rendered assistant turn parses back (the generation prompt ends with <think>)
    {
        ChatTemplateInput input;
        input.messages = parse_json(R"([{"role":"user","content":"q"},{"role":"assistant","content":"Done.","reasoning_content":"why",
            "tool_calls":[{"function":{"name":"get","arguments":{"n":2,"s":"t"}}}]}])");
        input.tools = parse_json(R"([{"type":"function","function":{"name":"get","parameters":{"type":"object","properties":{"n":{"type":"integer"}}}}}])");
        input.add_generation_prompt = false;
        const auto prompt = render_glm_chat(input);
        const auto turn = prompt.substr(prompt.rfind("<|assistant|><think>") + 20U);
        const auto out = parse_glm_output(turn, false, input.tools);
        check(out.reasoning == "why" && out.content == "Done." && out.tool_calls.dump() == R"([{"name":"get","arguments":{"n":2,"s":"t"}}])",
              "round trip " + out.tool_calls.dump());
    }
    check(glm_stop_tokens.size() == 3U && glm_stop_tokens[0] == 154820U && glm_stop_tokens[1] == 154827U && glm_stop_tokens[2] == 154829U, "stop tokens");
    std::cout << "glm chat tests: failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
