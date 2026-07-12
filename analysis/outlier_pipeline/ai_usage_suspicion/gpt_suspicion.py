"""GPT judge for AI-usage suspicion. Shares the rubric/scale/schema in common.py."""
import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI

try:
    from . import common
except ImportError:  # running as a plain script from inside this folder
    import common

DEFAULT_MODEL = "gpt-5.5" # instead of gpt-5.4


def score_prompt(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    usage: "common.UsageAccumulator | None" = None,
) -> common.PromptSuspicionResult:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": common.SYSTEM_PROMPT},
            {"role": "user", "content": common.USER_TEMPLATE.format(prompt=prompt)},
        ],
        text_format=common.PromptSuspicionResult,
    )
    if usage is not None and response.usage is not None:
        usage.add(response.usage.input_tokens, response.usage.output_tokens)
    return response.output_parsed


def score_csv(input_csv: Path, output_csv: Path, model: str = DEFAULT_MODEL, limit: int | None = None) -> None:
    client = OpenAI()
    df = pd.read_csv(input_csv)
    scored = common.score_dataframe(df, lambda p: score_prompt(client, p, model), limit=limit)
    scored.to_csv(output_csv, index=False)
    print(f"Saved scored CSV to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prompts for AI-style suspicion with GPT.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (cost guard).")
    args = parser.parse_args()
    score_csv(args.input_csv, args.output_csv, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
