"""Reconstruct CSVs from a JATOS data.txt fallback log.

data.txt is NOT JSON-lines: it's a raw concatenation of JSON objects with no
separator at all, e.g. {"type":"trial","row":{...}}{"type":"trial","row":{...}}
so it must be parsed with json.JSONDecoder.raw_decode in a loop.
"""
import json
from pathlib import Path

import pandas as pd

TYPE_TO_CSV = {
    "trial": "trials.csv",
    "participant": "participants.csv",
    "digit_span": "digit_span.csv",
}

# digit_span entries are logged with "ts" instead of the CSV's "timestamp" column.
FIELD_RENAMES = {
    "digit_span.csv": {"ts": "timestamp"},
}

CANONICAL_COLUMNS = {
    "trials.csv": [
        "uid", "gt", "session", "attempt", "prompt", "gen", "subjective_score",
        "prompt_latency_secs", "generating_latency_secs", "rating_latency_secs", "ts",
    ],
    "participants.csv": [
        "uid", "age", "gender", "native_language",
        "feedback_difficulty", "feedback_clarity", "feedback_comment",
    ],
    "digit_span.csv": [
        "uid", "session", "trial_num", "sequence_length", "presented_sequence",
        "participant_response", "response_time_ms", "timestamp",
    ],
}


def parse_data_txt(path: Path) -> dict[str, list[dict]]:
    """Parse concatenated-JSON data.txt into {"trial": [...], "participant": [...], ...}.

    Entries come in two shapes: trial/participant rows nest their fields under
    a "row" key, but digit_span rows have their fields flat, as siblings of
    "type". Both are handled here.
    """
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows_by_type: dict[str, list[dict]] = {}
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        idx = end
        entry_type = obj.get("type")
        if entry_type is None:
            continue
        if "row" in obj:
            row = obj["row"]
        else:
            row = {k: v for k, v in obj.items() if k != "type"}
        rows_by_type.setdefault(entry_type, []).append(row)
    return rows_by_type


def write_reconstructed_csvs(files_dir: Path, rows_by_type: dict[str, list[dict]]) -> set[str]:
    """Write trials/participants/digit_span CSVs from parsed rows, skipping any
    CSV that already exists on disk. Returns the set of filenames written."""
    written = set()
    for entry_type, rows in rows_by_type.items():
        csv_name = TYPE_TO_CSV.get(entry_type)
        if csv_name is None or not rows:
            continue
        out_path = files_dir / csv_name
        if out_path.exists():
            continue
        files_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).rename(columns=FIELD_RENAMES.get(csv_name, {}))
        df = df.reindex(columns=CANONICAL_COLUMNS[csv_name])
        df.to_csv(out_path, index=False)
        written.add(csv_name)
    return written
