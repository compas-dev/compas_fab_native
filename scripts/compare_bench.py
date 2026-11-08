"""Markdown comparison of two ci_bench JSON files (smaller is better).

Usage: python compare_bench.py head.json [base.json]
"""

import json
import sys


def load(path):
    with open(path) as f:
        return {row["name"]: row for row in json.load(f)}


def main():
    head = load(sys.argv[1])
    base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
    if base:
        print("| benchmark | base | this commit | change |\n|---|---:|---:|---:|")
    else:
        print("| benchmark | this commit |\n|---|---:|")
    for name, row in head.items():
        value = "{:.3f} {}".format(row["value"], row["unit"])
        if not base:
            print("| {} | {} |".format(name, value))
            continue
        if name not in base:
            print("| {} | – | {} | new |".format(name, value))
            continue
        b = base[name]["value"]
        change = (row["value"] - b) / b * 100.0
        flag = " :warning:" if change > 15.0 else (" :rocket:" if change < -15.0 else "")
        print("| {} | {:.3f} {} | {} | {:+.1f}%{} |".format(name, b, row["unit"], value, change, flag))
    if base:
        print("\nSingle-threaded, on a shared GitHub runner: changes within about ±10% are noise.")


if __name__ == "__main__":
    main()
