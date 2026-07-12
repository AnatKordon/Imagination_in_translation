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

# (input, output) USD per 1,000,000 tokens, keyed by model id. The Claude rate is
# Anthropic's published pricing for claude-haiku-4-5. The GPT/Gemini rates are left
# as None on purpose: token counts below are always exact, but dollar figures only
# appear once you fill in a verified rate for that provider.
PRICES: dict[str, tuple[float | None, float | None]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),   # TODO: verify current Anthropic pricing
    "gpt-5.4-mini": (0.75, 4.50),       # current OpenAI pricing
    "gpt-5.5": (None, None),             # TODO: set from current OpenAI pricing
    "gpt-5.4-nano": (0.20, 1.25),       # TODO: set from current OpenAI pricing
    "gemini-2.5-flash": (0.30, 2.50),   # TODO: set from current Gemini pricing
    "gemini-3.1-pro": (None, None),     # TODO: set from current Gemini pricing
    "gemini-3.5-flash": (None, None),   # TODO: set from current Gemini pricing (current judge model)
}


class UsageAccumulator:
    """Running total of token usage for one judge, summed over real API calls.

    Only updated on a cache miss (score_dataframe scores each unique prompt once),
    so the totals reflect actual spend, not the number of trials.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def cost_usd(self, model: str) -> float | None:
        rate = PRICES.get(model)
        if not rate or rate[0] is None or rate[1] is None:
            return None
        return self.input_tokens / 1e6 * rate[0] + self.output_tokens / 1e6 * rate[1]


def print_usage_report(accumulators: dict[str, UsageAccumulator]) -> None:
    """Print per-model token usage and (where a rate is known) estimated cost."""
    print("\n=== AI-usage judge spend ===")
    total = 0.0
    any_unknown = False
    for model, acc in accumulators.items():
        cost = acc.cost_usd(model)
        if cost is None:
            any_unknown = True
            cost_str = "cost n/a (set PRICES)"
        else:
            total += cost
            cost_str = f"${cost:.4f}"
        print(f"  {model:<20} calls={acc.calls:<4} "
              f"in={acc.input_tokens:<8} out={acc.output_tokens:<7} {cost_str}")
    known = f"${total:.4f}" + (" (excludes models with no rate set)" if any_unknown else "")
    print(f"  {'TOTAL (priced)':<20} {known}")

USER_TEMPLATE = "Image prompt:\n{prompt}"

# Shared rubric. Judges score writing REGISTER, not amount of detail: long, detailed,
# object-by-object, spatially-organized description is EXPECTED from genuine participants in
# this task and is not, by itself, evidence of AI. The 80-100 band is a holistic "reads like
# text copied from a chatbot" verdict; the discriminator is HOW the detail is expressed.
# Human tells (generator-directed edit commands, first-person / meta references, lay
# estimates, mechanical slips) pull down; a flawless flowing catalogue with photographic /
# render parameters pulls up. The signal lists are weighted cues, not hard rules. This text
# was calibrated against a content-matched gold set of GPT-5 vs real-participant descriptions
# (see prompt_variants.py "v1" and eval_prompts.py).
SYSTEM_PROMPT = """
You are a linguistic-forensics assistant for a psychology experiment.

In this experiment, human participants were asked to describe a target image in as much detail as possible so that an AI image generator could recreate it. A typical genuine response is therefore LONG and DETAILED: it catalogues many objects with their colors, positions, and spatial layout (foreground/background, left/right, what is cut off by the frame), and it may open with phrasing like "photo realistic". None of that detail is, by itself, evidence of AI. What distinguishes a real participant is HOW the detail is expressed: in plain, everyday words, often while reasoning aloud, estimating, addressing the image generator, or referring to "the picture".

Your job is to estimate how strongly the text looks like it was COPIED from an AI chatbot (e.g. pasted from ChatGPT) rather than written by the participant. Return a suspicion_score from 0 to 100. Judge the writing STYLE and REGISTER, never the amount of detail, the number of objects, or the presence of spatial/framing description. Simple, unpolished, or non-native English is NOT evidence of AI; do not raise the score merely because the writing seems unsophisticated, terse, or non-native.

Signals that RAISE suspicion (toward copied AI text). Weigh them together; none is decisive alone, and any can occasionally appear in genuine writing:
- A polished, essayistic register sustained across the whole passage: flowing, well-formed sentences or clauses with curated, evocative vocabulary that goes beyond plainly naming what is present.
- Photographic or rendering parameters stated as if configuring an image rather than noticed by a viewer: e.g. "eye level", "wide angle", "low angle", "24mm"/"35mm lens", "deep depth of field", "deep focus", "centered composition", "photorealistic", "3D render", "CGI", "soft natural lighting", "sharp focus", "no people", and harder jargon like "8k", "octane render", "bokeh", "--ar 16:9", "negative prompt".
- Chatbot framing or boilerplate: an intro/sign-off, "Sure, here's...", "Certainly", headings, or markdown bullet/numbered lists.
- Uniformly flawless spelling, grammar, spacing, and capitalization sustained across a long passage, with no typos, hedging, or self-correction anywhere.

Signals that point to a GENUINE participant (these LOWER suspicion). Any one of these is strong evidence against copied AI text, even in a long, polished-looking description:
- Direct instructions or edits aimed at the image generator: imperatives like "Put...", "Add...", "Make it...", "Change... to...", "Mind the colors", "Tilt the angle", "the bed should be...". Pasted chatbot descriptions do not tell a generator what to do.
- First or second person, or casual meta references to the act of viewing/photographing, in plain words: "you can see", "I think", "the image shows", "the picture", "the pic", "the photographer is facing".
- Lay comparative estimates and rough reasoning: "about five stories tall", "one tenth the height", "75 percent of the size of the table", "barely in frame".
- Hedging, uncertainty, or self-correction: "almost looks like", "maybe", "not sure", "it seems".
- Mechanical imperfections: typos, doubled or missing words, doubled spaces, inconsistent capitalization, run-on sentences.
- Plain everyday vocabulary and casual counting ("3 chairs", "a tall plant").

Scoring bands (judge the OVERALL impression; the signal lists inform it but do not dictate a verdict):
0-30: reads clearly like a participant describing the image in their own words.
31-60: mostly reads human, with some clean or generic phrasing but nothing that clearly points to a chatbot.
61-79: genuinely mixed -- a real pull toward AI-style writing alongside human-looking cues, with no confident verdict.
80-100: taken as a whole, the text reads like writing generated by or copied from an AI chatbot rather than typed by the participant -- the AI-leaning cues dominate and there is little or no sign of a real person doing the task (no instruction aimed at the generator, no first-person or meta reference, no lay estimate, no hedging, no mechanical slip). Length, detail, spatial description, or a "photo realistic" opening are not, on their own, reasons to land here.

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
    label: str | None = None,
) -> pd.DataFrame:
    """Score the 'prompt' column of df, inserting score_col/expl_col right after it.

    - Empty/NaN prompts are skipped (score 0).
    - Identical prompt strings are scored once and reused (dedup cache) to save API calls.
    - Per-row exceptions are caught and recorded as pd.NA + an error message.
    - limit: only score the first `limit` rows (rest left as pd.NA) — a cost guard.
    - label: prefixed to progress lines so multi-judge runs are readable.
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
            tag = f"[{label}] " if label else ""
            print(f"{tag}scored {idx + 1}/{len(df)} rows...", flush=True)

    for col in (score_col, expl_col):
        if col in df.columns:
            df = df.drop(columns=[col])
    pos = df.columns.get_loc("prompt")
    df.insert(pos + 1, score_col, scores)
    df.insert(pos + 2, expl_col, explanations)
    return df
