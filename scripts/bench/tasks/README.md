# Single-stream task suite

Eight chat tasks that separate copy-heavy outputs (where context drafting can confirm many tokens per target pass)
from new text (where it cannot). The text is project-owned: generated JSONL/CSV records, a file of this
repository (`src/gguf_reader.cpp`) and two original texts in `src/`. `build.py` writes `tasks.json`
deterministically; `build.py --check` fails when the file is stale.

| id | the model returns | max tokens |
|---|---|---:|
| jsonl-160 | 160 JSONL records with one field changed where a condition holds | 4,096 |
| jsonl-long | the same over 340 records | 8,192 |
| sql | CREATE TABLE + one INSERT per CSV row + index + aggregate query | 4,096 |
| code-edit | a ~5,000-token C++ file with one function renamed, in full | 8,192 |
| tr-zh-en | an English translation of a Chinese text | 4,096 |
| tr-es-en | an English translation of a Spanish text | 4,096 |
| prose-en | an original ~2,500-word story | 4,096 |
| code-new | a new Python module with tests | 4,096 |

## Protocol

One request at a time (C1) to the engine's `/v1/chat/completions`: stream, temperature 0, thinking off
(`chat_template_kwargs.enable_thinking=false`), `max_tokens` from the task, the request stops at the end of turn.
One warm-up request (the first task) is excluded.

- **Decode speed** = (N - 1) / (t_last - t_first), where N = completion tokens from `usage` and t_first / t_last
  are the client arrival times of the first and last content chunks. Load and prefill are excluded; TTFT is
  reported separately.
- **Identity**: `ids_sha256` of the generated token ids from the final chunk's `strix` receipt
  (`return_token_ids`); servers without it fall back to the text sha256 (marked `text:`).
- **Drafting**: `spec_stats` from the receipt when the engine reports it (passes, tokens per pass, passes per
  source, acceptance by depth, verify windows used).

```
python3 scripts/bench/tasks/run_tasks.py --url http://127.0.0.1:18880 --out res/off --label off
python3 scripts/bench/tasks/run_tasks.py --url http://127.0.0.1:18880 --out res/on --label on
python3 scripts/bench/tasks/run_tasks.py compare res/off/tasks-results.json res/on/tasks-results.json
```

`compare` prints the speed ratio per task and fails on any id mismatch: drafting must not change greedy output.
For an A/B, restart the server for every arm and alternate the arms (A B B A).
