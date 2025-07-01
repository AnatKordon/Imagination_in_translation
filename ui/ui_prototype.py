# It's a prototype of UI for the project. It doesn't actually send anything to the model or calculate image similarity yet.
# To call the app, write: cd UI streamlit run ui_prototype.py --server.port 8501 in the codespace terminal (in github or locally, then "cd UI" part is optional), but first, install the required libraries listed in the requirements.txt file with a single command: pip install -r requirements.txt.
# Note, that by default a user has to press ctrl+enter after filling in the text box to apply the text, count characters, send it to generation etc. 

from pathlib import Path
from uuid import uuid4 # used to create session / user IDs
from models import api_model # the model API wrapper
import random, csv, time 
import config
import streamlit as st # Streamlit UI framework
from PIL import Image, ImageOps # Pillow library to manipulate images


# ensuring all the required folders exist so .save() or logging never crash
for d in (config.GEN_DIR, config.LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Setting up the appearance
st.set_page_config(page_title="Image Description", layout="wide") # the page is full-width 

# Customising the buttons
st.markdown(
    """
    <style>
        button[kind="primary"]{background:#8B0000;color:white}
        button[kind="primary"]:hover{background:#A80000;color:white}
    </style>
    """,
    unsafe_allow_html=True,
)

# TO BE CHANGED: here goes actual IMAGE GENERATION. So far, it just pretends to run a diffusion model: it copies the fallback Mona Lisa picture to a new filename so we have something to display.
def generate_image(prompt: str, attempt: int, gt: Path,id: str) -> Path:
    params = config.params.copy() # copy the params dict so we don't change the original one
    params["prompt"] = prompt # set the prompt
    out = config.GEN_DIR / f"{gt.stem}_att{attempt}.png"
    api_model.send_generation_request(host="https://api.stability.ai/v2beta/stable-image/generate/sd3",params=params,
                                      iteration=attempt,user_id=id)
    ImageOps.contain(Image.open(config.FALLBACK), (512, 512)).save(out)
    return out

# TO BE CHANGED: here goes actual SIMILARITY SCORE. So far, it's just random numbers.
def similarity(_: Path, __: Path) -> float:
    return random.uniform(0.25, 0.9)

# TO BE CHANGED: here goes LOG saving. Here is a simple function that appends one row per generation to a specific CSV log for each session. Creates the file and header on first write.
def log_row(**kw):
    f = config.LOG_DIR / f"{kw['uid']}.csv"
    first = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=kw.keys())
        if first:
            w.writeheader()
        w.writerow(kw)


# A helper for st.rerun() function in Steamlit. It is named differently in different versions of Steamlit, so we just make sure that we have something of this kind.
rerun = st.rerun if hasattr(st, "rerun") else st.experimental_rerun

# Function returning a new widget key for the textbox every time we load a new target (e.g. a new ground truth image is presented)
def fresh_key() -> str:
    return f"prompt_{uuid4().hex}"

# Function that loads a new ground-truth image (or finish the whole thing if none left in the folder) and reset all per-target variables. Then force an immediate rerun.
def next_gt():
    remaining = [p for p in config.GT_DIR.glob("*.[pj][pn]g") if p.name not in S.used]
    if not remaining:
        S.finished = True
        rerun()

    S.gt_path = random.choice(remaining)
    S.used.add(S.gt_path.name)

    S.attempt = 1
    S.generated = False
    S.gen_path = None
    S.last_score = 0.0

    S.text_key = fresh_key() # new widget key so the existing widget value is not overwritten
    rerun()


# Defining a st.session_state, which is Streamlit’s dictionary like place for keeping data between reruns during a single session (for a given user). 
for k, v in {
    "uid": uuid4().hex, # user/session ID
    "used": set(), # set of ground-truth images already shown
    "gt_path": None, # TO BE CHANGED: path to the current ground-truth
    "attempt": 1, # current attempt counter (from 1 to 5)
    "generated": False, # TO BE CHANGED: telling whether we have a generated image to show or not
    "gen_path": None, # TO BE CHANGED: path to generated image
    "finished": False, # True when pool exhausted
    "last_score": 0.0, # similarity of last generation
    "text_key": fresh_key(), # widget key for the textbox
}.items():
    st.session_state.setdefault(k, v)
S = st.session_state

if S.gt_path is None:      # loading the next ground truth picture
    next_gt()

# The finish screen (appears when the user presses "exit" button or there are no more ground truth pictures).
if S.finished:
    st.markdown(
        "<h2 style='text-align:center'>The session is finished.<br>Thank you for participating!</h2>",
        unsafe_allow_html=True,
    )
    st.stop()

# Layout of the textbox and pictures (next to each other)
left, right = st.columns([1, 2])

# left column: textbox for descriptive prompt and "generate" and "exit" buttons.
with left:
    st.write(f"**Your ID:** `{S.uid}`")
    st.markdown("**Please, describe the picture as precisely as possible. You have up to 5 attempts to improve your description.**")

# Textbox (unique key per target)
    S.prompt = st.text_area(
        "The picture shows...",
        key=S.text_key,
        height=140,
        placeholder="Type an accurate description of the target image",
    )

# Character counters (below the box)
    c1, c2 = st.columns(2)
    c1.caption(f"{len(S.prompt)} characters")
    c2.caption(f"{S.attempt} / {config.MAX_ATTEMPTS}")

# the generation is disabled if the image is already generated, we have empty prompt, or there were more than five attempts
    gen_disabled = (
        S.generated
        or not str(S.prompt).strip()
        or S.attempt > config.MAX_ATTEMPTS
    )
# TO BE CHANGED: "Generate" button should send the prompt for a real IMAGE GENERATION, but now it just runs a dummy generator, saves some logs, then reruns
    if st.button("Generate", type="primary", disabled=gen_disabled):
        S.gen_path = generate_image(S.prompt, S.attempt, S.gt_path,S.uid) # generate the image
        S.last_score = similarity(S.gt_path, S.gen_path)
        log_row(
            uid=S.uid,
            gt=S.gt_path.name,
            attempt=S.attempt,
            prompt=S.prompt,
            gen=S.gen_path.name,
            similarity=round(S.last_score, 4),
            ts=int(time.time()),
        )
        S.generated = True
        rerun()

# flags session as finished and rerun
    if st.button("Exit"):
        S.finished = True
        rerun()

# Right column displays the ground truth (target) image and the generated one (next to each other) together with similarity scores, "accept" and "try again" buttons.
with right:
    gt_c, gen_c = st.columns(2, gap="medium")
  # always shows ground truth picture
    with gt_c:
        st.image(
            ImageOps.contain(Image.open(S.gt_path), (int(config.IMG_H * 1.8), config.IMG_H)),
            caption="Target image",
            clamp=True,
        )
# the generated picture is shown together with the similarity score and "accept" and "try again" buttons
    if S.generated:
        with gen_c:
            st.image(
                ImageOps.contain(Image.open(S.gen_path), (int(config.IMG_H * 1.8), config.IMG_H)),
                caption="Generated image",
                clamp=True,
            )

        st.caption("Similarity:")
        st.progress(int(S.last_score * 100))
        st.write(f"**{S.last_score * 100:.1f}%**")

        a_col, t_col = st.columns(2)

        if a_col.button("Accept"):
            next_gt()
# "Try again" button is disabled on 5th attempt
        if t_col.button("Try again", disabled=S.attempt >= config.MAX_ATTEMPTS):
            S.generated = False
            S.attempt += 1
            rerun()
    else:
        gen_c.empty() # keeps column widths
