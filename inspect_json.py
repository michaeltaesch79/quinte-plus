"""
Print the key structure of a downloaded JSON file (one example per list),
so the parser can be written against the real field names.

Usage:
    python inspect_json.py data/raw/programme/2026-10-26.json > structure.txt
"""
import json
import sys


def walk(node, path="", depth=0, max_depth=7):
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}{path} ...")
        return
    if isinstance(node, dict):
        for k, v in node.items():
            kind = type(v).__name__
            sample = "" if isinstance(v, (dict, list)) else f" = {str(v)[:60]!r}"
            print(f"{pad}{k} ({kind}){sample}")
            if isinstance(v, (dict, list)):
                walk(v, f"{path}.{k}", depth + 1, max_depth)
    elif isinstance(node, list):
        print(f"{pad}[{len(node)} items, first shown]")
        if node:
            walk(node[0], path + "[0]", depth + 1, max_depth)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        walk(json.load(f))
