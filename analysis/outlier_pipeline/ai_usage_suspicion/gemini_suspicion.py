"""Gemini judge for AI-usage suspicion. Shares the rubric/scale/schema in common.py."""
import argparse
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

try:
    from . import common
except ImportError:  # running as a plain script from inside this folder
    import common

DEFAULT_MODEL = "gemini-3.5-flash" # instead of gemini-2.5-flash
SEED = 42


def score_prompt(
    client: genai.Client,
    prompt: str,
    model: str = DEFAULT_MODEL,
    usage: "common.UsageAccumulator | None" = None,
) -> common.PromptSuspicionResult:
    response = client.models.generate_content(
        model=model,
        contents=common.USER_TEMPLATE.format(prompt=prompt),
        config=types.GenerateContentConfig(
            system_instruction=common.SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=common.PromptSuspicionResult,
            temperature=0.1,   # low-variance baseline
            seed=SEED,         # reproducibility
        ),
    )
    if usage is not None and response.usage_metadata is not None:
        usage.add(
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
        )
    return response.parsed


def score_csv(input_csv: Path, output_csv: Path, model: str = DEFAULT_MODEL, limit: int | None = None) -> None:
    client = genai.Client()
    df = pd.read_csv(input_csv)
    scored = common.score_dataframe(df, lambda p: score_prompt(client, p, model), limit=limit)
    scored.to_csv(output_csv, index=False)
    print(f"Saved scored CSV to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prompts for AI-style suspicion with Gemini.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (cost guard).")
    args = parser.parse_args()
    score_csv(args.input_csv, args.output_csv, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
