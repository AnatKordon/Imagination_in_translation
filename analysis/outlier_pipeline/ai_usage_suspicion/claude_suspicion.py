"""Claude judge for AI-usage suspicion. Shares the rubric/scale/schema in common.py."""
import argparse
from pathlib import Path

import anthropic
import pandas as pd

try:
    from . import common
except ImportError:  # running as a plain script from inside this folder
    import common

DEFAULT_MODEL = "claude-haiku-4-5" # claude-haiku-4-5, or better claude-sonnet-5


def score_prompt(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = DEFAULT_MODEL,
    usage: "common.UsageAccumulator | None" = None,
) -> common.PromptSuspicionResult:
    response = client.messages.parse(
        model=model,
        max_tokens=1024,  # 256 truncated sonnet-5's JSON mid-string -> parse failures
        system=common.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": common.USER_TEMPLATE.format(prompt=prompt)}],
        output_format=common.PromptSuspicionResult,
    )
    if usage is not None:
        usage.add(response.usage.input_tokens, response.usage.output_tokens)
    return response.parsed_output


def score_csv(input_csv: Path, output_csv: Path, model: str = DEFAULT_MODEL, limit: int | None = None) -> None:
    client = anthropic.Anthropic()
    df = pd.read_csv(input_csv)
    scored = common.score_dataframe(df, lambda p: score_prompt(client, p, model), limit=limit)
    scored.to_csv(output_csv, index=False)
    print(f"Saved scored CSV to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score prompts for AI-style suspicion with Claude.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (cost guard).")
    args = parser.parse_args()
    score_csv(args.input_csv, args.output_csv, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
