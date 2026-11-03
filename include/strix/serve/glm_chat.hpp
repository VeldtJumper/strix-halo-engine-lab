#pragma once

#include "strix/serve/chat_template.hpp"

#include <array>
#include <cstdint>
#include <string>

// GLM-5 chat template (the model's chat_template.jinja, rendered natively): "[gMASK]<sop>", a reasoning-effort system
// block, <|system|>/<|user|>/<|assistant|> turns, reasoning as <think>...</think> (kept after the last user message or
// with clear_thinking=false), tool calls as <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
// and consecutive tool messages as one <|observation|> block of <tool_response> entries (in tool-call order when every
// id is known). The template has no non-thinking mode: the generation prompt always ends inside "<think>".
namespace strix::serve {

// enable_thinking is ignored (the template has none); reasoning_effort low | high, anything else renders as Max;
// preserve_thinking=true is the template's clear_thinking=false, false/unset clear earlier reasoning. A tool call's
// `arguments` may be an object or a JSON string (parsed here, as the template requires a mapping). Content the template
// would print as a Python repr (an object, or an array inside a text part) is rejected.
[[nodiscard]] std::string render_glm_chat(const ChatTemplateInput& input);

// The generated text of one assistant turn (without the stop token). Generation always starts inside the reasoning
// block, so `thinking` is not consulted: reasoning before </think>, then the content, then the tool calls (a value is
// JSON-parsed when its schema type in `tools` is not string and it parses, else kept as text).
[[nodiscard]] AssistantOutput parse_glm_output(const std::string& text, bool thinking, const Json& tools);

// The marker that starts a tool call in the content (streaming keeps everything from it for the final parse).
inline constexpr const char* glm_tool_marker = "<tool_call>";

// generation_config.json eos_token_id: <|endoftext|>, <|user|>, <|observation|>.
inline constexpr std::array<std::uint32_t, 3> glm_stop_tokens = {154820U, 154827U, 154829U};

}  // namespace strix::serve
