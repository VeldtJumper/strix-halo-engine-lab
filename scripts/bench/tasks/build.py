#!/usr/bin/env python3
"""Builds scripts/bench/tasks/tasks.json: the single-stream task suite (project-owned text, deterministic).

Every task is one user message for the model's chat template with thinking off. The data tasks (JSONL, SQL) are
generated from a fixed linear congruential sequence, so the file is byte-identical on every machine; the code-edit
input is a file of this repository; the translation sources are original texts in tasks/src/.

usage: build.py [--check]   (--check: fail when tasks.json differs from what this script builds)
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "tasks.json"

NAMES = ["Ana", "Bruno", "Carla", "Diego", "Elena", "Felix", "Gloria", "Hugo", "Irene", "Jorge", "Karen", "Luis",
         "Marta", "Nora", "Oscar", "Paula", "Quim", "Rosa", "Sergio", "Tania", "Ulises", "Vera", "Walter", "Yago"]
CATEGORIES = ["garden", "kitchen", "office", "outdoor", "lighting", "storage"]
ITEMS = ["lamp", "shelf", "kettle", "planter", "bench", "basket", "clock", "mirror", "stool", "rack", "tray", "vase"]
ADJ = ["oak", "steel", "ceramic", "linen", "bamboo", "copper", "glass", "walnut", "wicker", "marble"]


class Lcg:
    """64-bit LCG (Knuth MMIX constants); the high bits are used."""

    def __init__(self, seed):
        self.state = seed & (2**64 - 1)

    def next(self, n):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        return (self.state >> 33) % n


def records(count, seed):
    rng = Lcg(seed)
    rows = []
    for i in range(1, count + 1):
        rows.append({"id": i, "name": NAMES[rng.next(len(NAMES))], "qty": 1 + rng.next(99), "status": "open"})
    return "\n".join(json.dumps(r, separators=(",", ":")) for r in rows)


def jsonl_task(count, seed):
    return ("Below are " + str(count) + " JSON records, one per line (JSONL). Return all " + str(count) +
            " records as JSONL in the same order and with the same keys, changing only one thing: set \"status\" "
            "to \"closed\" for every record whose \"qty\" is 50 or more. Leave every other value unchanged. Output "
            "only the JSONL lines, with no explanation and no code fence.\n\n" + records(count, seed))


def sql_task(count, seed):
    rng = Lcg(seed)
    lines = ["id,name,category,price,stock"]
    for i in range(1, count + 1):
        name = ADJ[rng.next(len(ADJ))] + " " + ITEMS[rng.next(len(ITEMS))]
        price = 5 + rng.next(29500) / 100
        lines.append(f"{i},{name},{CATEGORIES[rng.next(len(CATEGORIES))]},{price:.2f},{rng.next(400)}")
    return ("Write one PostgreSQL script for the product list below (CSV with a header row). The script must: "
            "(1) create a table products with suitable column types, a primary key and NOT NULL / CHECK "
            "constraints; (2) insert every row with one INSERT statement per row, in the given order; (3) create an "
            "index on category; (4) end with a query that returns, per category, the number of products and the "
            "total stock value (price * stock), ordered by total value descending. Output only the SQL.\n\n" +
            "\n".join(lines))


def code_edit_task():
    source = (ROOT / "src" / "gguf_reader.cpp").read_text(encoding="utf-8")
    return ("Here is a C++ source file. Rename the private member function read_bytes to read_exact everywhere in "
            "the file (its definition and every call). Change nothing else: keep every other line, comment and "
            "blank line exactly as it is. Return the complete updated file in a single ```cpp code block, with no "
            "explanation before or after it.\n\n```cpp\n" + source + "```")


def translation_task(language, path):
    text = (HERE / "src" / path).read_text(encoding="utf-8").strip()
    return ("Translate the following " + language + " text into natural, fluent English. Keep the paragraph "
            "structure. Output only the translation.\n\n" + text)


PROSE = ("Write an original mystery story of about 2,500 words set in a lighthouse on a remote island during a "
         "three-day storm. The keeper's logbook goes missing and one of four stranded visitors is lying about where "
         "they were on the first night. Use vivid but plain English, dialogue, and a fair solution that the reader "
         "could deduce from clues given earlier. Give the story a title.")

CODE_NEW = ("Write a complete, self-contained Python 3 module named ttl_cache.py that implements a thread-safe LRU "
            "cache with an optional per-entry time-to-live. Requirements: a TTLCache class with get, put, delete, "
            "clear, __len__ and __contains__; a maximum size with least-recently-used eviction; expired entries "
            "are never returned and are purged lazily; an injectable clock function for testing; hit/miss/eviction "
            "counters; a decorator cached(maxsize, ttl) for functions with hashable arguments; type hints and "
            "docstrings throughout. Then write a unittest.TestCase class with at least ten tests that use the "
            "injectable clock. Output only the code in one ```python block.")


def tasks():
    return [
        {"id": "jsonl-160", "category": "JSONL transform (160 lines)", "max_tokens": 4096, "text": jsonl_task(160, 1)},
        {"id": "jsonl-long", "category": "JSONL transform (long)", "max_tokens": 8192, "text": jsonl_task(340, 2)},
        {"id": "sql", "category": "SQL from CSV", "max_tokens": 4096, "text": sql_task(60, 3)},
        {"id": "code-edit", "category": "code edit, whole file", "max_tokens": 8192, "text": code_edit_task()},
        {"id": "tr-zh-en", "category": "translation zh->en", "max_tokens": 4096, "text": translation_task("Chinese", "zh.txt")},
        {"id": "tr-es-en", "category": "translation es->en", "max_tokens": 4096, "text": translation_task("Spanish", "es.txt")},
        {"id": "prose-en", "category": "English prose", "max_tokens": 4096, "text": PROSE},
        {"id": "code-new", "category": "code from scratch", "max_tokens": 4096, "text": CODE_NEW},
    ]


def render():
    doc = {"schema": "strix-bench-tasks/1", "mode": "chat", "thinking": False, "greedy": True, "concurrency": 1,
           "tasks": tasks()}
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main():
    text = render()
    if sys.argv[1:] == ["--check"]:
        if OUT.read_text(encoding="utf-8") != text:
            sys.exit("tasks.json is stale: run scripts/bench/tasks/build.py")
        return
    OUT.write_text(text, encoding="utf-8", newline="\n")
    for t in json.loads(text)["tasks"]:
        print(f"{t['id']:<11} {len(t['text']):>6} chars  max_tokens {t['max_tokens']}")


if __name__ == "__main__":
    main()
