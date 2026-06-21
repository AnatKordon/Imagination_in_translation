"""Tiny calibration probe: do the judges actually score AI-styled prompts >= THRESHOLD,
and human prompts low? Run once on a handful of hand-labeled prompts to sanity-check
the rubric and the THRESHOLD before spending on a full condition.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from . import claude_suspicion, common, gemini_suspicion, gpt_suspicion
except ImportError:  # running as a plain script from inside this folder
    import claude_suspicion
    import common
    import gemini_suspicion
    import gpt_suspicion

import anthropic  # noqa: E402
from google import genai  # noqa: E402
from openai import OpenAI  # noqa: E402

# (label, prompt). "ai" = should score high; "human" = should score low.
PROBE = [
    ("ai", "photorealistic lighthouse, hyper-detailed, intricate, cinematic lighting, 8k, masterpiece, dramatic volumetric fog, ultra sharp focus, octane render, trending on artstation"),
    ("human", "A blue four-seat velvet couch rests in front of a light gray wall with a wide, rectangular print of a parking garage in blue tint appears on the wall. In front of the couch is a grat coffee table with a glass of four orange tulips. The furniture rests on a plush rug comprised of magenta, royal blue, yellow, bright green and white circles, each about 75 percent of the size of the coffee table. There is also a modern white ceiling lamp above the table. Tilt the angle of the pic - the person who took the pic is standing a few feet away, facing directly toward the left corner of the couch that is touching the wall."),
    ("ai", "Cozy traditional bedroom interior with a dark wood four poster bed dressed in a cream duvet, ivory sheets, and layered yellow floral pillows; two framed botanical bird prints hang above the headboard; a dark wood nightstand holds a black shade table lamp, an old style clock, and a vase of orange flowers."),
    ("human", "This image shows a meeting or training room arranged for a conference. Light wood tables with silver or gray metal legs are set up in a large U-shaped, open toward the viewer. Vibrant orange-red chairs with black legs are tucked into the tables, facing inward. The room has light walls, a simple gray floor, and features a tall green potted plant in the background, adding a touch of color and nature. The overall atmosphere is functional and modern. the atmosphere is functional and modern. the is one piece of paper on the table."),
    ("human", "Warm room, mustard yellow walls. In the corner of the room, there is a tall green plant. A bed with a wooden bench at the foot. White sheets and mustard colored pillows. Rug on the floor. On the wall above the headboard, there are 2 white frames and a wooden nightstand with a base and orange flowers."),
    ("ai", "Old offshore caisson lighthouse leaning strongly to the right, cylindrical iron tower on a broad conical base, oxidized red and brown rust with peeling faded pink paint.")
]


def main() -> None:
    clients = {"gpt": OpenAI(), "gemini": genai.Client(), "claude": anthropic.Anthropic()}
    modules = {"gpt": gpt_suspicion, "gemini": gemini_suspicion, "claude": claude_suspicion}

    rows = []
    for label, prompt in PROBE:
        row = {"label": label, "prompt": prompt[:60] + "..."}
        for name, module in modules.items():
            result = common.retry(lambda p: module.score_prompt(clients[name], p), prompt)
            row[f"{name}_score"] = result.suspicion_score
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 65)
    print(df.to_string(index=False))
    print(f"\nTHRESHOLD={common.THRESHOLD}, MIN_AGREEMENT={common.MIN_AGREEMENT}. "
          "AI rows should clear the threshold on >=2 judges; human rows should not.")


if __name__ == "__main__":
    main()
