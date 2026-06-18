"""
TextEE -> master JSONL converter (the "first hop" for guideline generation).

Reads TextEE `processed_data/<dataset_name>/<split>/<file>.json` (one JSON object per line)
and emits the master JSONL consumed by
`synthesize_guidelines/synthesize_guidelines_w_neg_samples.py --master_file ...`.

Output: one line per (sentence, event_type) so each line carries a single event type, matching
the convention the downstream builder/scorers assume:

    {"text": "...", "event_mentions": {"Die(LifeEvent)": {"results": [
        {"mention": "killed", "victim": ["people"], "place": ["town"]}
    ]}}}

Event-type names are normalized with the repo's own `clean_event_name`, and argument roles are
lowercased with '-'/'.' replaced (matching `generate_schema.py`) so the names line up everywhere.
"""
import argparse
import json
import os
import sys

# Reuse the repo's canonical event-name normalization (and schema_splitters).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clean_event_name, schema_splitters  # noqa: E402


def _norm_role(role):
    return role.lower().replace("-", "_").replace(".", "_")


def convert_sentence(sent, dataset_name):
    """Return {event_type: {"results": [ {mention, role: [spans]}, ... ]}} for one TextEE sentence."""
    grouped = {}
    for ev in sent.get("event_mentions", []):
        event_type = clean_event_name(ev["event_type"], dataset_name)
        trigger = ev.get("trigger", {})
        mention = trigger.get("text", "") if isinstance(trigger, dict) else str(trigger)
        result = {"mention": mention}
        for arg in ev.get("arguments", []):
            role = _norm_role(arg["role"])
            span = arg.get("text", arg.get("mention", ""))
            result.setdefault(role, []).append(span)
        grouped.setdefault(event_type, {"results": []})["results"].append(result)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Convert TextEE processed data to the guideline master JSONL.")
    parser.add_argument("-i", "--dataset_directory", default="./../../TextEE/processed_data",
                        help="Root of TextEE processed_data.")
    parser.add_argument("-d", "--dataset_name", required=True,
                        help=f"Dataset name; must be a key in schema_splitters ({', '.join(schema_splitters)}).")
    parser.add_argument("--split", default="split1", help="Split subdir (default: split1).")
    parser.add_argument("--files", nargs="+", default=["train.json"],
                        help="Which split files to include (e.g. train.json dev.json).")
    parser.add_argument("-o", "--output", required=True, help="Output master JSONL path.")
    args = parser.parse_args()

    if args.dataset_name not in schema_splitters:
        raise SystemExit(f"Unknown dataset_name '{args.dataset_name}'. Known: {list(schema_splitters)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_sent, n_lines = 0, 0
    with open(args.output, "w") as out:
        for fname in args.files:
            path = os.path.join(args.dataset_directory, args.dataset_name, args.split, fname)
            if not os.path.exists(path):
                print(f"[warn] missing TextEE file: {path}")
                continue
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                sent = json.loads(line)
                n_sent += 1
                grouped = convert_sentence(sent, args.dataset_name)
                # Emit one line per event type so each carries a single event type.
                for event_type, payload in grouped.items():
                    out.write(json.dumps({"text": sent.get("text", ""),
                                          "event_mentions": {event_type: payload}}) + "\n")
                    n_lines += 1
    print(f"Read {n_sent} sentences -> wrote {n_lines} master lines to {args.output}")


if __name__ == "__main__":
    main()
