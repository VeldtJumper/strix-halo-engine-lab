#include "strix/serve/glm_chat.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

// Port of the GLM-5 chat_template.jinja (macros visible_text, tool_to_json, render_tool_response,
// tool_references_to_response, id_of, is_list_of_outputs) as HF transformers renders it (trim_blocks, lstrip_blocks).
// tojson(ensure_ascii=False) is Json::dump_spaced; Jinja's {{ value }} of a scalar is Python str() ("None", "True").
namespace strix::serve {
namespace {

const std::string kThinkStart = "<think>", kThinkEnd = "</think>";
const std::string kCallStart = "<tool_call>", kCallEnd = "</tool_call>";
const std::string kKeyStart = "<arg_key>", kKeyEnd = "</arg_key>", kValueStart = "<arg_value>", kValueEnd = "</arg_value>";

[[noreturn]] void raise(const std::string& message) { throw ChatTemplateError(message); }

bool is_type(const Json* item, std::string_view type) {
    const auto* t = item ? item->find("type") : nullptr;
    return t && t->is_string() && t->as_string() == type;
}
// Python truthiness of a value (absent = Jinja undefined = false).
bool truthy(const Json* v) {
    if (!v || v->is_null()) return false;
    if (v->is_bool()) return v->as_bool();
    if (v->is_number()) return v->as_double() != 0.0;
    if (v->is_string()) return !v->as_string().empty();
    return v->size() > 0U;
}
// {{ value }}: Python str() of a scalar; undefined prints nothing.
std::string py_str(const Json* v, const char* what) {
    if (!v) return {};
    switch (v->kind()) {
    case Json::Kind::null: return "None";
    case Json::Kind::boolean: return v->as_bool() ? "True" : "False";
    case Json::Kind::number: return v->number_source();
    case Json::Kind::string: return v->as_string();
    default: raise(std::string("Unexpected ") + what + " type (an object or array printed as text).");
    }
}
// A non-empty array whose first element is an object (content.0 in the template).
const Json* first_item(const Json* v) {
    return v && v->is_array() && v->size() > 0U && v->as_array()[0].is_object() ? &v->as_array()[0] : nullptr;
}

// visible_text: a string, or the text parts of a list (media parts become a reminder), or str() of anything else.
std::string visible_text(const Json* content) {
    if (content && content->is_array()) {
        std::string out;
        for (const auto& item : content->as_array()) {
            if (item.is_string()) { out += item.as_string(); continue; }
            if (!item.is_object()) continue;
            if (is_type(&item, "text")) { out += py_str(item.find("text"), "text part"); continue; }
            const auto* type = item.find("type");
            if (!type || !type->is_string()) continue;
            const auto& t = type->as_string();
            if (t == "image" || t == "image_url" || t == "video" || t == "video_url" || t == "audio" || t == "audio_url" || t == "input_audio") {
                const std::string media = t == "input_audio" ? "audio" : t.substr(0, t.find("_url"));
                out += "<reminder>You are unable to process this " + media +
                       " because you don't have multi-modal input ability. Try different methods.</reminder>";
            }
        }
        return out;
    }
    return py_str(content, "content");
}

std::string role_of(const Json& message) {
    const auto* role = message.find("role");
    return role && role->is_string() ? role->as_string() : std::string{};
}

// A tool as the template sees it: its `function` member when it has one.
const Json& tool_spec(const Json& tool) {
    const auto* function = tool.find("function");
    if (!function) return tool;
    if (!function->is_object()) raise("A tool's function must be an object.");
    return *function;
}

// tool_to_json: the tool's members in order, without defer_loading and strict.
std::string tool_to_json(const Json& tool) {
    if (!tool.is_object()) raise("Every tool must be an object.");
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : tool.as_object()) {
        if (key == "defer_loading" || key == "strict") continue;
        if (!first) out += ", ";
        first = false;
        out += "\"" + key + "\": " + value.dump_spaced();
    }
    return out + "}";
}

std::string tool_response(const std::string& text) { return "<tool_response>" + text + "</tool_response>"; }

// tool_references_to_response: the request's tools named by the references (deferred ones too).
std::string tool_references(const Json& refs, const Json& tools) {
    if (!truthy(&tools)) raise("A tool_reference result needs the request's tools.");
    std::string out = "<tool_response><tools>\n";
    for (const auto& ref : refs.as_array()) {
        const auto* want = ref.find("name");
        for (const auto& entry : tools.as_array()) {
            const auto& tool = tool_spec(entry);
            const auto* name = tool.find("name");
            // Jinja: undefined == undefined
            const bool same = (!want && !name) || (want && name && want->dump() == name->dump());
            if (same) out += tool_to_json(tool) + "\n";
        }
    }
    return out + "</tools></tool_response>";
}

bool is_tool_reference_list(const Json* v) { return is_type(first_item(v), "tool_reference"); }
bool is_list_of_outputs(const Json& message) {
    const auto* first = first_item(message.find("content"));
    return first && first->find("output");
}

// One tool output (an entry of a list-of-outputs content).
std::string render_output(const Json& entry, const Json& tools) {
    const auto* output = entry.find("output");
    return is_tool_reference_list(output) ? tool_references(*output, tools) : tool_response(visible_text(output));
}

// render_tool_response
std::string render_tool_response(const Json& message, const Json& tools) {
    const auto* content = message.find("content");
    if (content && content->is_string()) return tool_response(content->as_string());
    if (is_tool_reference_list(content)) return tool_references(*content, tools);
    if (is_list_of_outputs(message)) {
        std::string out;
        for (const auto& entry : content->as_array()) out += render_output(entry, tools);
        return out;
    }
    return tool_response(visible_text(content));
}

// id_of: tool_call_id, else id, else empty.
std::string id_of(const Json& v) {
    if (const auto* id = v.find("tool_call_id"); truthy(id)) return py_str(id, "tool_call_id");
    if (const auto* id = v.find("id"); truthy(id)) return py_str(id, "id");
    return {};
}

// The <|observation|> block of the tool messages [begin, end]: in the order of the preceding assistant's tool calls
// when every call and every result has an id (results matching no call are dropped), else in message order.
std::string observation_block(const Json::Array& messages, std::size_t begin, std::size_t end, const Json& tools) {
    const Json* calls = nullptr;
    if (begin > 0U && role_of(messages[begin - 1U]) == "assistant" && truthy(messages[begin - 1U].find("tool_calls")))
        calls = messages[begin - 1U].find("tool_calls");
    bool can_sort = calls != nullptr;
    if (can_sort && !calls->is_array()) raise("tool_calls must be an array.");
    for (std::size_t k = begin; can_sort && k <= end; ++k) {
        if (is_list_of_outputs(messages[k])) {
            for (const auto& entry : messages[k].find("content")->as_array())
                if (id_of(entry).empty()) can_sort = false;
        } else if (id_of(messages[k]).empty()) can_sort = false;
    }
    if (can_sort)
        for (const auto& call : calls->as_array())
            if (id_of(call).empty()) can_sort = false;
    std::string out = "<|observation|>";
    if (!can_sort) {
        for (std::size_t k = begin; k <= end; ++k) out += render_tool_response(messages[k], tools);
        return out;
    }
    for (const auto& call : calls->as_array()) {
        const auto id = id_of(call);
        for (std::size_t k = begin; k <= end; ++k) {
            if (is_list_of_outputs(messages[k])) {
                for (const auto& entry : messages[k].find("content")->as_array())
                    if (id_of(entry) == id) out += render_output(entry, tools);
            } else if (id_of(messages[k]) == id) {
                out += render_tool_response(messages[k], tools);
            }
        }
    }
    return out;
}

// The arguments mapping of a tool call: an object as given, or a JSON string parsed into one.
Json call_arguments(const Json& call, const std::string& name) {
    const auto* arguments = call.find("arguments");
    if (arguments && arguments->is_object()) return *arguments;
    if (arguments && arguments->is_string()) {
        if (strip(arguments->as_string()).empty()) return Json::object();
        Json parsed;
        try { parsed = parse_request_json(arguments->as_string()); }
        catch (const JsonError&) { raise("Tool call arguments for function \"" + name + "\" are not valid JSON."); }
        if (!parsed.is_object()) raise("Tool call arguments for function \"" + name + "\" must be a JSON object.");
        return parsed;
    }
    raise("Tool call arguments for function \"" + name + "\" must be an object/mapping or a JSON string.");
}

std::string render_assistant(const Json& message, bool keep_reasoning) {
    std::string out = "<|assistant|>";
    auto content = visible_text(message.find("content"));
    std::optional<std::string> reasoning;
    if (const auto* r = message.find("reasoning_content"); r && r->is_string()) {
        reasoning = r->as_string();
    } else if (const auto end = content.find(kThinkEnd); end != std::string::npos) {
        const auto before = content.substr(0, end);
        const auto start = before.rfind(kThinkStart);
        reasoning = start == std::string::npos ? before : before.substr(start + kThinkStart.size());
        content = content.substr(content.rfind(kThinkEnd) + kThinkEnd.size());
    }
    out += keep_reasoning && reasoning ? kThinkStart + *reasoning + kThinkEnd : kThinkStart + kThinkEnd;
    out += strip(content);
    const auto* calls = message.find("tool_calls");
    if (!truthy(calls)) return out;
    if (!calls->is_array()) raise("tool_calls must be an array.");
    for (const auto& entry : calls->as_array()) {
        const auto* call = &entry;
        if (const auto* function = entry.find("function"); truthy(function)) call = function;
        const auto* name = call->find("name");
        if (!name || !name->is_string()) raise("Tool call is missing a function name.");
        out += kCallStart + name->as_string();
        const auto arguments = call_arguments(*call, name->as_string());
        for (const auto& [key, value] : arguments.as_object())
            out += kKeyStart + key + kKeyEnd + kValueStart + (value.is_string() ? value.as_string() : value.dump_spaced()) + kValueEnd;
        out += kCallEnd;
    }
    return out;
}

}  // namespace

std::string render_glm_chat(const ChatTemplateInput& input) {
    if (!input.messages.is_array()) raise("messages must be an array.");
    const auto& messages = input.messages.as_array();
    const auto& tools = input.tools;
    if (truthy(&tools) && !tools.is_array()) raise("tools must be an array.");
    const auto effort = input.reasoning_effort.value_or("max");
    std::string out = "[gMASK]<sop><|system|>Reasoning Effort: ";
    out += effort == "low" ? "Low" : effort == "high" ? "High" : "Max";
    if (truthy(&tools)) {
        out += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for (const auto& entry : tools.as_array()) {
            const auto& tool = tool_spec(entry);
            if (!truthy(tool.find("defer_loading"))) out += tool_to_json(tool) + "\n";
        }
        out += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
               "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}"
               "</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
    std::int64_t last_user = -1;
    for (std::size_t i = 0; i < messages.size(); ++i)
        if (role_of(messages[i]) == "user") last_user = static_cast<std::int64_t>(i);
    const bool keep_all = input.preserve_thinking && *input.preserve_thinking;  // clear_thinking=false
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (!message.is_object()) raise("Every message must be an object.");
        const auto role = role_of(message);
        if (role == "user") {
            out += "<|user|>" + visible_text(message.find("content"));
        } else if (role == "assistant") {
            out += render_assistant(message, keep_all || static_cast<std::int64_t>(i) > last_user);
        } else if (role == "tool") {
            if (i > 0U && role_of(messages[i - 1U]) == "tool") continue;  // rendered with the first of the run
            auto end = i;
            while (end + 1U < messages.size() && role_of(messages[end + 1U]) == "tool") ++end;
            out += observation_block(messages, i, end, tools);
        } else if (role == "system") {
            out += "<|system|>" + visible_text(message.find("content"));
        }  // other roles render nothing
    }
    if (input.add_generation_prompt) out += "<|assistant|>" + kThinkStart;
    return out;
}

namespace {
// As parse_assistant_output: the schema of one parameter of a request tool.
const Json* parameter_schema(const Json& tools, const std::string& function, const std::string& parameter) {
    if (!tools.is_array()) return nullptr;
    for (const auto& tool : tools.as_array()) {
        const auto* spec = tool.find("function");
        if (!spec) spec = &tool;
        const auto* name = spec->find("name");
        if (!name || !name->is_string() || name->as_string() != function) continue;
        const auto* parameters = spec->find("parameters");
        if (!parameters) parameters = spec->find("input_schema");
        const auto* properties = parameters ? parameters->find("properties") : nullptr;
        return properties ? properties->find(parameter) : nullptr;
    }
    return nullptr;
}
// Strings (or no schema) stay text; other types are parsed as JSON when they parse, else kept as text.
Json typed_value(const std::string& text, const Json* schema) {
    const auto* type = schema ? schema->find("type") : nullptr;
    const bool string_type = !type || (type->is_string() && type->as_string() == "string");
    if (string_type) return Json(text);
    try { return parse_json(text); } catch (const JsonError&) { return Json(text); }
}
}  // namespace

AssistantOutput parse_glm_output(const std::string& text, bool thinking, const Json& tools) {
    (void)thinking;  // the generation prompt always opens the reasoning block
    AssistantOutput out;
    out.tool_calls = Json::array();  // also when the reply ends inside the thinking block
    const auto end = text.find(kThinkEnd);
    if (end == std::string::npos) { out.reasoning = strip(text); return out; }
    out.reasoning = strip(text.substr(0, end));
    out.reasoning_complete = true;
    const auto rest = text.substr(end + kThinkEnd.size());
    const auto first_call = rest.find(kCallStart);
    out.content = strip(first_call == std::string::npos ? rest : rest.substr(0, first_call));
    for (auto at = first_call; at != std::string::npos; at = rest.find(kCallStart, at)) {
        const auto begin = at + kCallStart.size();
        const auto close = rest.find(kCallEnd, begin);
        const auto block = rest.substr(begin, close == std::string::npos ? std::string::npos : close - begin);
        at = close == std::string::npos ? rest.size() : close + kCallEnd.size();
        if (close == std::string::npos) out.unfinished_call = true;
        const auto name = strip(block.substr(0, block.find(kKeyStart)));
        if (name.empty()) continue;
        Json arguments = Json::object();
        for (auto p = block.find(kKeyStart); p != std::string::npos; p = block.find(kKeyStart, p)) {
            const auto key_end = block.find(kKeyEnd, p);
            if (key_end == std::string::npos) break;
            const auto key = strip(block.substr(p + kKeyStart.size(), key_end - p - kKeyStart.size()));
            const auto value_at = block.find(kValueStart, key_end);
            if (value_at == std::string::npos || !strip(block.substr(key_end + kKeyEnd.size(), value_at - key_end - kKeyEnd.size())).empty()) break;
            const auto value_begin = value_at + kValueStart.size();
            const auto value_end = block.find(kValueEnd, value_begin);
            const auto value = block.substr(value_begin, value_end == std::string::npos ? std::string::npos : value_end - value_begin);
            arguments.set(key, typed_value(value, parameter_schema(tools, name, key)));
            p = value_end == std::string::npos ? block.size() : value_end + kValueEnd.size();
        }
        Json call = Json::object();
        call.set("name", Json(name));
        call.set("arguments", std::move(arguments));
        out.tool_calls.push(std::move(call));
        out.call_complete.push_back(close != std::string::npos);
    }
    return out;
}

}  // namespace strix::serve
