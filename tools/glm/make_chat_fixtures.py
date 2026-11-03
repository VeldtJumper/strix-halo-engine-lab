#!/usr/bin/env python3
"""Generate the GLM-5 chat-template cases checked by tests/glm_chat_tests.cpp.

Reference: HuggingFace transformers apply_chat_template(tokenize=False) with the model's chat_template.jinja.
usage: make_chat_fixtures.py --model-dir MODEL_DIR OUT.json
Output (JSON, UTF-8): {render: [{name, messages, tools, options, expected}], parse: [{name, text, tools, reasoning,
reasoning_complete, content, tool_calls, call_complete, unfinished_call}]}
options: add_generation_prompt, reasoning_effort, clear_thinking (the template kwargs; absent = not passed).
"""
import argparse
import json
import os

from transformers import PreTrainedTokenizerFast

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "units": {"type": "string", "enum": ["c", "f"]}},
    "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Búsqueda web — «unicode» 日本", "strict": True, "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}, "limit": {"type": "number"},
                                     "exact": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}}}}}
DEFERRED = {"type": "function", "function": {"name": "slow_tool", "defer_loading": True, "parameters": {"type": "object", "properties": {}}}}
DEFER_FALSE = {"type": "function", "function": {"name": "quick_tool", "defer_loading": False, "description": "x", "parameters": {}}}
BARE = {"name": "bare_tool", "description": "A tool given without the function wrapper.", "parameters": {"type": "object", "properties": {
    "n": {"type": "integer"}}}}

U = lambda c: {"role": "user", "content": c}
S = lambda c: {"role": "system", "content": c}
A = lambda c, **kw: dict({"role": "assistant", "content": c}, **kw)
T = lambda c, **kw: dict({"role": "tool", "content": c}, **kw)
CALL = lambda name, args, id_=None: dict({"type": "function", "function": {"name": name, "arguments": args}}, **({"id": id_} if id_ else {}))

RENDER_CASES = [
    ("basic", [U("Hello!")], None, {}),
    ("no_gen_prompt", [U("Hello!")], None, {"add_generation_prompt": False}),
    ("effort_low", [U("Hi")], None, {"reasoning_effort": "low"}),
    ("effort_high", [U("Hi")], None, {"reasoning_effort": "high"}),
    ("effort_medium_is_max", [U("Hi")], None, {"reasoning_effort": "medium"}),
    ("effort_max", [U("Hi")], None, {"reasoning_effort": "max"}),
    ("system_user", [S("You are a pirate."), U("¿Qué hora es?  \n")], None, {}),
    ("multi_system", [S("A"), U("q"), S("late system"), U("q2")], None, {}),
    ("multi_turn_reasoning_dropped", [U("2+2?"), A("4", reasoning_content="add them"), U("and 3+3?")], None, {}),
    ("multi_turn_clear_false", [U("2+2?"), A("4", reasoning_content="add them"), U("and 3+3?")], None, {"clear_thinking": False}),
    ("multi_turn_clear_true", [U("2+2?"), A("4", reasoning_content="add them"), U("and 3+3?")], None, {"clear_thinking": True}),
    ("assistant_last_kept", [U("2+2?"), A("  4  \n", reasoning_content="\nadd them\n")], None, {"add_generation_prompt": False}),
    ("think_in_content", [U("q"), A("<think>inline reasoning</think>\n\nThe answer.")], None, {"add_generation_prompt": False}),
    ("think_in_content_old", [U("q"), A("pre<think>r1</think>mid</think>tail"), U("next")], None, {}),
    ("no_reasoning_after_user", [U("q"), A("plain answer")], None, {"add_generation_prompt": False}),
    ("reasoning_then_no_reasoning", [U("q"), A("a1", reasoning_content="r1"), A("a2")], None, {"add_generation_prompt": False}),
    ("empty_reasoning", [U("q"), A("a", reasoning_content="")], None, {"add_generation_prompt": False}),
    ("null_content_user", [{"role": "user", "content": None}], None, {}),
    ("missing_content_assistant", [U("q"), {"role": "assistant", "tool_calls": [CALL("get_weather", {"city": "Paris"})]}], [WEATHER],
     {"add_generation_prompt": False}),
    ("list_content", [U([{"type": "text", "text": "Part one. "}, "raw string part ", {"type": "text", "text": "Part two."}]),
                      A([{"type": "text", "text": "Reply"}])], None, {"add_generation_prompt": False}),
    ("media_parts", [U([{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "what is this?"},
                        {"type": "input_audio", "input_audio": {}}, {"type": "video"}, {"type": "audio_url"}, {"type": "unknown", "x": 1}])],
     None, {}),
    ("tools_basic", [S("Be brief."), U("Weather in Paris?")], [WEATHER], {}),
    ("tools_strict_deferred_bare", [U("hi")], [SEARCH, DEFERRED, DEFER_FALSE, BARE], {}),
    ("tool_call_typed_args", [U("Weather?"), A("Let me check.", reasoning_content="need tool", tool_calls=[
        CALL("get_weather", {"city": "París", "days": 3, "units": "c"}),
        CALL("search", {"query": "x", "filters": {"lang": "es", "n": [1, 2.5, None, True]}, "limit": 1.5, "exact": False,
                        "tags": ["a", "ñ"]})])], [WEATHER, SEARCH], {"add_generation_prompt": False}),
    ("tool_call_no_content", [U("Weather?"), A("", tool_calls=[{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]),
                              T("sunny")], [WEATHER], {}),
    ("tool_call_bare_and_empty_args", [U("go"), A(None, tool_calls=[{"name": "bare_tool", "arguments": {}}]), T("ok")], [BARE], {}),
    ("tool_roundtrip_unsorted", [U("Weather in Paris and Rome?"), A("", reasoning_content="two calls", tool_calls=[
        CALL("get_weather", {"city": "Paris"}), CALL("get_weather", {"city": "Rome"})]), T("Paris: 20C"), T("Rome: 25C"),
        A("Paris 20C, Rome 25C.")], [WEATHER], {}),
    ("tool_sorted_by_id", [U("Weather?"), A("", tool_calls=[CALL("get_weather", {"city": "Paris"}, "call_1"), CALL("get_weather", {"city": "Rome"}, "call_2")]),
                           T("Rome: 25C", tool_call_id="call_2"), T("Paris: 20C", tool_call_id="call_1"), U("thanks")], [WEATHER], {}),
    ("tool_sorted_id_field", [U("w"), A("", tool_calls=[CALL("get_weather", {"city": "A"}, "b"), CALL("get_weather", {"city": "B"}, "a")]),
                              T("for a", id="a"), T("for b", id="b"), T("orphan", id="zzz")], [WEATHER], {}),
    ("tool_missing_id_not_sorted", [U("w"), A("", tool_calls=[CALL("get_weather", {"city": "A"}, "c1"), CALL("get_weather", {"city": "B"}, "c2")]),
                                    T("second", tool_call_id="c2"), T("first")], [WEATHER], {}),
    ("tool_list_of_outputs", [U("w"), A("", tool_calls=[CALL("get_weather", {"city": "A"}, "c1"), CALL("get_weather", {"city": "B"}, "c2")]),
                              T([{"tool_call_id": "c2", "output": "B out"}, {"tool_call_id": "c1", "output": [{"type": "text", "text": "A out"}]}])],
     [WEATHER], {}),
    ("tool_list_of_outputs_unsorted", [U("w"), A("x"), T([{"output": "o1"}, {"output": None}, {"output": [{"type": "image"}, "s"]}])], None, {}),
    ("tool_list_content", [U("w"), T([{"type": "text", "text": "part a"}, {"type": "text", "text": "part b"}]), T(None)], None, {}),
    ("tool_first_message", [T("orphan result"), U("q")], None, {}),
    ("tool_reference", [U("find tools"), A("", tool_calls=[CALL("search", {"query": "weather"}, "t1")]),
                        T([{"type": "tool_reference", "name": "get_weather"}, {"type": "tool_reference", "name": "bare_tool"}], tool_call_id="t1")],
     [SEARCH, WEATHER, BARE], {}),
    ("tool_reference_in_outputs", [U("find"), A("", tool_calls=[CALL("search", {"query": "w"}, "t1")]),
                                   T([{"tool_call_id": "t1", "output": [{"type": "tool_reference", "name": "slow_tool"}]}])],
     [SEARCH, DEFERRED], {}),
    ("unknown_role_ignored", [U("q"), {"role": "developer", "content": "dev"}, U("q2")], None, {}),
    ("special_chars", [U("Quotes \"x\" and \\ backslash, tabs\t, emoji 😀, <tags>")], None, {}),
    ("agentic_long", [S("sys"), U("plan"), A("step 1", reasoning_content="r1", tool_calls=[CALL("search", {"query": "a"}, "s1")]),
                      T("res a", tool_call_id="s1"), A("step 2", reasoning_content="r2", tool_calls=[CALL("search", {"query": "b"}, "s2")]),
                      T("res b", tool_call_id="s2")], [SEARCH], {"reasoning_effort": "high"}),
]

# Parser cases: the expected split is written here (no HF reference parser); tools type the values.
PARSE_CASES = [
    ("think_and_content", "Let me think.\n</think>\n\nThe answer is 4.", None,
     {"reasoning": "Let me think.", "reasoning_complete": True, "content": "The answer is 4.", "tool_calls": [], "call_complete": [], "unfinished_call": False}),
    ("unfinished_reasoning", "still thinking", None,
     {"reasoning": "still thinking", "reasoning_complete": False, "content": "", "tool_calls": [], "call_complete": [], "unfinished_call": False}),
    ("typed_tool_calls", "need weather</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
     "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call><tool_call>search<arg_key>query</arg_key><arg_value>42</arg_value>"
     "<arg_key>filters</arg_key><arg_value>{\"lang\": \"es\"}</arg_value><arg_key>exact</arg_key><arg_value>true</arg_value>"
     "<arg_key>limit</arg_key><arg_value>not json</arg_value><arg_key>other</arg_key><arg_value>7</arg_value></tool_call>", [WEATHER, SEARCH],
     {"reasoning": "need weather", "reasoning_complete": True, "content": "Checking.",
      "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}},
                     {"name": "search", "arguments": {"query": "42", "filters": {"lang": "es"}, "exact": True, "limit": "not json", "other": "7"}}],
      "call_complete": [True, True], "unfinished_call": False}),
    ("whitespace_between", "r</think><tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Rome</arg_value>\n</tool_call>", [WEATHER],
     {"reasoning": "r", "reasoning_complete": True, "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Rome"}}],
      "call_complete": [True], "unfinished_call": False}),
    ("unterminated_call", "r</think>ok<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par", [WEATHER],
     {"reasoning": "r", "reasoning_complete": True, "content": "ok", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Par"}}],
      "call_complete": [False], "unfinished_call": True}),
    ("no_args_call", "r</think><tool_call>bare_tool</tool_call>", [BARE],
     {"reasoning": "r", "reasoning_complete": True, "content": "", "tool_calls": [{"name": "bare_tool", "arguments": {}}],
      "call_complete": [True], "unfinished_call": False}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("out")
    a = ap.parse_args()
    with open(os.path.join(a.model_dir, "chat_template.jinja"), encoding="utf-8") as f:
        template = f.read()
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model_dir, "tokenizer.json"))
    render = []
    for name, messages, tools, options in RENDER_CASES:
        kwargs = dict(options)
        add_gen = kwargs.pop("add_generation_prompt", True)
        text = tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False, add_generation_prompt=add_gen, **kwargs)
        render.append({"name": name, "messages": messages, "tools": tools, "options": options, "expected": text})
    parse = [dict({"name": name, "text": text, "tools": tools}, **want) for name, text, tools, want in PARSE_CASES]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        json.dump({"render": render, "parse": parse}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(render)} render cases, {len(parse)} parse cases -> {a.out}")


if __name__ == "__main__":
    main()
