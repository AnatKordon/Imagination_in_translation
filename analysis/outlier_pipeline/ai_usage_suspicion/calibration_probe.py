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
    ("ai", "Old offshore caisson lighthouse leaning strongly to the right, cylindrical iron tower on a broad conical base, oxidized red and brown rust with peeling faded pink paint."),
    ("human", "photo realistic Small board room, white floor, foreground square table with mostly open center and a slight gap in the top center that takes up most of the frame, grey cylinder legs on the table, the left side of the table goes off screen but comes back onto screen, taken from the perspective of a corner of the table, left side of table 3 orange chairs with skinny black legs, 3rd chair mostly off screen, left top of square 3 chairs tucked under table facing the camera, right front of table same orange chairs, no chairs in top right of square, background left half behind table 3 full length posters that extend most of the picture, the left picture has a red heading with children on top with indiscernible text, the middle poster has blue heading with 1 child, the right poster has an orange heading with a baby and more text, right center of frame is a potted plant that is behind table with orange pot green leaves, to the right of the plant is a standing white board, above and to the right is another small white board affixed to the wall, the right is a larger whiteboard that extends off screen with a black eraser magnet on the top left, above the 2nd whiteboard is a metal grate speaker, behind the posters we can see a window with light streaming in but with white vertical shutters blocking a lot of the light"),
    ("human", "photo realistic bedroom, Main focus is the bed, The front of the bed ends in the middle of the frame and is the back goes in a diagonal left from the front and is flush with the wall. 3 long bedposts in frame extending out of the frame, the top left is not in the picture. White duvet folded like a hotel on top of bed, with Minimal black markings in the center spaced out, 7 pillows arranged neatly above the folded duvet like a hotel, The folded part of the duvet is a white and yellow checkered pattern that is only apparent on the folded over part of the duvet, 3 pillows have yellow and white designs, the back 2 pillows that are barely visible are beige red, and the rest have other indiscernible designs, in the foreground there is an open minimal wooden bench with a white cushion at the base of the bed. The bench has 2 open wooden handles that extend to the height of the bed, It has one sturdy square white with green checkered pillow leaning on the right handle, directly in the center there is a shallow basket with thin books forming 2 piles inside, the basket also has 2 handles on the right edge of frame we can see half of a dark finished wooden cabinet with the left door open On the bottom shelf we can see a stack of books that is cut off out of frame, on the 3rd shelf we can see half a bowl that is cut of because it is out of frame, in the back center on the right of the bed there is a night stand that is slightly taller than the bed with a small skinny light on top with a black shade, next to the lap is an artificial flower pot filled with 8 pinkish roses illuminated slightly by the lamp, to the right of the night stand and in the top corner of the room is a large skinny plant almost like a tree reaching the top of the frame with an empty trunk but at the top many full green leaves, the walls of the room are a light yellow, above the bed sits 2 portraits frame with a black border inside are pictures of what appears to be flower drawings, above the framed pictures is an arch window that reaches out of the top left of the frame of the picture, it has a spiny white frame with one spine going directly up in the middle and 2 spines from the lower point radiating diagonally to the left and right symmetrically. On the front left almost completely out of frame we can see a wicker chair with a cushion on top, we can barely see it though, the room is carpeted in possibly , in the front right of the picture on top of the carpet is a rug that is mostly out of frame it is mainly red with black patterns with a few visible tassels on the short edge that we see. there is an off screen light source coming from the front left corner that casts a shadow and illuminates the foreground.")
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
