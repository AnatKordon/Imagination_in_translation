"""Shared rubric, output schema, and row-handling for the three AI-usage judges.

All three detectors (GPT, Gemini, Claude) import from here so an "80" means the
same thing to each model: identical criteria, identical 0-100 bands, identical
output contract. Only the per-SDK call mechanics differ in each detector file.
"""
import time

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env once for every detector (the original demos never did this themselves).
load_dotenv()

# Exclusion is decided by consensus.py, not here, but the constants live with the
# rubric so the rubric and the threshold it targets stay in sync.
THRESHOLD = 80          # a judge "flags" a prompt when its score is >= THRESHOLD
MIN_AGREEMENT = 2       # exclude a prompt when at least this many of 3 judges flag it

USER_TEMPLATE = "Image prompt:\n{prompt}"

# Shared rubric. Deliberately NOT "be conservative": the 2-of-3 >= 80 consensus
# already makes the overall filter strict, so each judge should be well-calibrated
# to the bands rather than reluctant to ever reach the top band.
#
# Key design point: in this experiment participants were INSTRUCTED to describe a
# target image in as much detail as possible so an AI image generator could recreate
# it. So length / detail / thoroughness are exactly what the task demands and are NOT
# evidence of AI. The discriminator is writing STYLE and REGISTER, not amount of detail.
SYSTEM_PROMPT = """
You are a linguistic-forensics assistant for a psychology experiment.

In this experiment, human participants were asked to describe a target image in as much detail as possible so that an AI image generator could recreate it. Long, highly detailed, exhaustive descriptions are exactly what the task demands, so ordinary thoroughness is expected and is not, by itself, evidence of AI. Do not raise the score merely because a description is long or elaborate.

Return a suspicion_score from 0 to 100 estimating how strongly the WRITING STYLE and REGISTER resemble AI-generated text, or text copied from an image-generation prompt, rather than a person describing what they see in their own words. Base your judgment primarily on writing STYLE and REGISTER (below), not on amount of detail. An unusually exhaustive, flat catalogue of many distinct objects — well beyond what a person typically bothers to mention — may count as a mild secondary cue, but only together with the style signals below; never treat detail or object count as decisive on its own. Do not penalize sophisticated vocabulary or strong writing by itself, and do not infer anything from perceived education level, fluency, or whether the writer is a native speaker.

Signals that RAISE suspicion (AI / image-generation-prompt style):
- Telegraphic noun-phrase fragments and stacked comma-separated modifiers instead of full sentences (caption / tag style), e.g. "cylindrical iron tower on a broad conical base, oxidized red and brown rust with peeling faded pink paint".
- No observer or narration voice anywhere ("we see", "there is", "on the left", "I notice"); the text just lists attributes.
- Flawless spelling, grammar, spacing, and capitalization throughout.
- Image-generation jargon or parameters: "photorealistic", "hyper-detailed", "cinematic lighting", "8k", "octane render", "bokeh", "35mm", "masterpiece", "trending on artstation", "--ar 16:9", "negative prompt".
- Confident, encyclopedic technical vocabulary delivered tersely; uniformly templated structure.

Signals that LOWER suspicion (human task-description style) — these hold EVEN when the text is very long and detailed:
- Full sentences and natural narration; a person walking through the scene spatially ("on the left", "in the center", "behind it", "we see", "there is").
- First-person or observer framing and hedging / uncertainty ("I think", "almost looks like", "maybe", "not sure", "it seems").
- Redundancy, restatement, or self-correction (saying the same thing twice, refining a guess).
- Typos, doubled spaces, inconsistent or mid-sentence capitalization, run-on sentences — mechanical imperfections are STRONG evidence of a real person typing freely, and should pull the score down.
- Conversational or informal asides.

Scoring bands:
0-20: clearly a person describing the image in their own words (narration, hedging, or typos present).
21-50: mostly human narration with a few polished or generic touches.
51-79: genuinely uncertain — some cues point to AI / generation-prompt style and some point to human writing, with no confident verdict either way.
80-100: strong evidence of AI-generated or copied image-generation-prompt text — fragment / tag style, jargon or parameters, flawless polish, and no observer voice.

Return only the requested structured fields. The explanation must be one concise sentence naming the specific textual cues behind the score.
""".strip()


class PromptSuspicionResult(BaseModel):
    suspicion_score: int = Field(
        ge=0, le=100, description="Suspicion score from 0 to 100."
    )
    explanation: str = Field(
        description="One concise sentence explaining the observable textual basis for the score."
    )


def _clean_prompt(prompt) -> str:
    return "" if pd.isna(prompt) else str(prompt)


def retry(score_one, prompt: str, max_retries: int = 3) -> PromptSuspicionResult:
    """Call a single-prompt scorer with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return score_one(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced after retries
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to score prompt after {max_retries} attempts: {last_error}")


def score_dataframe(
    df: pd.DataFrame,
    score_one,
    score_col: str = "suspicion_score",
    expl_col: str = "explanation",
    limit: int | None = None,
) -> pd.DataFrame:
    """Score the 'prompt' column of df, inserting score_col/expl_col right after it.

    - Empty/NaN prompts are skipped (score 0).
    - Identical prompt strings are scored once and reused (dedup cache) to save API calls.
    - Per-row exceptions are caught and recorded as pd.NA + an error message.
    - limit: only score the first `limit` rows (rest left as pd.NA) — a cost guard.
    """
    if "prompt" not in df.columns:
        raise ValueError('Input must contain a column named "prompt".')

    df = df.copy()
    scores: list = []
    explanations: list = []
    cache: dict[str, PromptSuspicionResult] = {}

    for idx, raw in enumerate(df["prompt"]):
        prompt = _clean_prompt(raw)

        if limit is not None and idx >= limit:
            scores.append(pd.NA)
            explanations.append(pd.NA)
            continue

        if not prompt.strip():
            scores.append(0)
            explanations.append("Empty prompt; no textual evidence of AI-style prompting.")
            continue

        try:
            if prompt in cache:
                result = cache[prompt]
            else:
                result = retry(score_one, prompt)
                cache[prompt] = result
            scores.append(result.suspicion_score)
            explanations.append(result.explanation)
        except Exception as exc:  # noqa: BLE001
            scores.append(pd.NA)
            explanations.append(f"Scoring failed: {type(exc).__name__}: {exc}")

        if (idx + 1) % 50 == 0:
            print(f"Scored {idx + 1} rows...")

    for col in (score_col, expl_col):
        if col in df.columns:
            df = df.drop(columns=[col])
    pos = df.columns.get_loc("prompt")
    df.insert(pos + 1, score_col, scores)
    df.insert(pos + 2, expl_col, explanations)
    return df
