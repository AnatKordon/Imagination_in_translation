# To call the app, write:  run ui/ui_prototype.py --server.port 8501 in the codespace terminal  but first, install the required libraries listed in the requirements.txt file with a single command: pip install -r requirements.txt.
# Note, that by default a user has to press ctrl+enter after filling in the text box to apply the text, count characters, send it to generation etc. 
from pathlib import Path
import config       
import streamlit.components.v1 as components
from models import api_model # the model API wrapper for Stability AI
# removed it for now as we are not using it and i don't wantit to be imported
#from models import gpt_model # the model API wrapper for open ai
from drive_utils import build_drive_from_token_dict, ensure_folder, ensure_path, update_or_insert_file, extract_folder_id, get_token_dict_from_secrets_or_env
# from similarity import vgg_similarity # the similarity function 
from uuid import uuid4 # used to create session / user IDs
# from drive_utils import get_drive_service, create_folder, upload_file, extract_folder_id_from_url
import random, csv, time 
import time
import os
import numpy as np 
import re
import streamlit as st # Streamlit UI framework
from PIL import Image, ImageOps # Pillow library to manipulate images
import hashlib
import base64
from io import BytesIO
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# Google Drive setup
token_dict = get_token_dict_from_secrets_or_env(st)
service = build_drive_from_token_dict(token_dict)
FOLDER_ID = extract_folder_id(config.DRIVE_FOLDER) # main drive folder

# creates a folder and image root for every ppt in google drive
def init_drive_tree_for_participant():
    """
    Ensure Drive folders:
    <DRIVE_FOLDER>/participants_data/<UID>/gen_images/
    Save their ids in session so we don't repeat list/create calls.
    """
    root = ensure_folder(service, "participants_data", FOLDER_ID)
    participant = ensure_folder(service, S.uid, root)
    images_root = ensure_folder(service, "gen_images", participant)

    S.drive_root_id = root
    S.participant_folder_id = participant
    S.images_root_id = images_root
    # session folder is created per target in update_session_folder()

    # Write created-at timestamp once (UTC)
    if not getattr(S, "participant_created_at_uploaded", False):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = config.LOG_DIR / f"{S.uid}_created_at_utc.txt"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(ts, encoding="utf-8")
        update_or_insert_file(
            service,
            meta,
            S.participant_folder_id,
            dest_name="created_at_utc.txt",
            mime_type="text/plain",
        )
        S.participant_created_at_uploaded = True

def update_session_folder():
    """Ensure session_<nn> folder exists under .../gen_images/ and store id."""
    if "images_root_id" not in S:
        init_drive_tree_for_participant()
    session_name = f"session_{S.session:02d}"
    S.session_drive_folder_id = ensure_folder(service, session_name, S.images_root_id)

def image_dest_name(uid: str, session: int, attempt: int, idx: int, suffix: str) -> str:
    # idx is 1-based for readability
    return f"{uid}_session{session:02d}_attempt{attempt:02d}_img{idx:02d}{suffix}"

def info_csv_name(uid: str) -> str:
    return f"participant_{uid}_info.csv"

def log_csv_name(uid: str) -> str:
    return f"participant_{uid}_log.csv"

def upload_participant_info(uid: str, info_csv_path: Path):
    init_drive_tree_for_participant()
    update_or_insert_file(
        service, info_csv_path, S.participant_folder_id,
        dest_name=info_csv_name(uid), mime_type="text/csv"
    )

def upload_participant_log(uid: str):
    log_csv = config.LOG_DIR / f"{uid}.csv"
    update_or_insert_file(
        service, log_csv, S.participant_folder_id,
        dest_name=log_csv_name(uid), mime_type="text/csv"
    )

def on_saved(path: Path, idx: int, seed: str):
    dest = image_dest_name(S.uid, S.session, S.attempt, idx=idx, suffix=path.suffix)  # set idx appropriately
    update_or_insert_file(service, path, S.session_drive_folder_id, dest_name=dest)

# Image size setup - Fixed bounding boxes (size should change if a single image or 2)
GT_BOX  = (300, 300)   # target image size
GEN_BOX = (300, 300)   # each generated image

# def show_img_fixed(path, box, caption=None):
#     """Open, bound to box while preserving aspect, and render at a fixed width."""
#     img = ImageOps.contain(Image.open(path), box)
#     st.image(img, width=box[0], clamp=True, caption=caption)

@st.cache_data
def load_gt_images_base64():
    """Load all GT images as base64 once and cache them"""
    gt_data = {}
    for gt_path in config.GT_DIR.glob("*.[pj][pn]g"):
        img = ImageOps.contain(Image.open(gt_path), GT_BOX)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        gt_data[gt_path.name] = base64.b64encode(buffered.getvalue()).decode()
    return gt_data

def show_img_fixed(image_source, box, caption=None):
    """
    Display image from either file path, bytes, or PIL Image object.
    """
    try:
        # Handle different input types
        if isinstance(image_source, (str, Path)):
            if not image_source or not os.path.exists(image_source):
                st.error("Image file not found")
                return
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(BytesIO(image_source))
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            st.error("Invalid image source")
            return
        
        # Resize to fit box
        img = ImageOps.contain(img, box)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        html = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" 
                 style="width: {box[0]}px; height: auto; max-width: 100%;">
        </div>
        """
        if caption:
            html += f'<p style="text-align: center; font-size: 14px; color: gray;">{caption}</p>'
        
        st.markdown(html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def show_img_base64(img_base64, box, caption=None):
    """Display pre-converted base64 image (fastest)"""
    html = f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" 
             style="width: {box[0]}px; height: auto; max-width: 100%;">
    </div>
    """
    if caption:
        html += f'<p style="text-align: center; font-size: 14px; color: gray;">{caption}</p>'
    
    st.markdown(html, unsafe_allow_html=True)

#  handling resizing as a function
def resize_inplace(p: Path, size=(512, 512)) -> None:
    img = Image.open(p)
    img = ImageOps.contain(img, size)  # preserves aspect ratio inside 512x512
    img.save(p)

#generate a unique seed per image
def seed_from(gt_filename: str) -> int:
    # Deterministic 32-bit seed from any string
    return int(hashlib.sha256(gt_filename.encode("utf-8")).hexdigest(), 16) % (2**32)  # 32-bit

#new generation of 4 images
def generate_images(prompt: str, negative_prompt: str, seed: int, session: int, attempt: int, gt: Path, uid: str) -> list[Path]:
    params = config.params.copy()
    params["prompt"] = prompt
    params["gt"] = str(gt)
    if negative_prompt:
        params["negative_prompt"] = negative_prompt  # Stability API accepts this
    # initialize collecting paths
    local_paths = []
    returned_seeds = []
    images_bytes = []
    N_OUT = config.N_OUT  # wither single or multiple images generation

    if config.API_CALL == "stability_ai":
        for i in range(N_OUT):  # generate 4 images
            params["seed"] = seed  # keep seed same, not changing it
            local_path, returned_seed, image_bytes = api_model.send_generation_request(
                host="https://api.stability.ai/v2beta/stable-image/control/structure", # chagne from sd3 to structure
                params=params,
                iteration=attempt,
                session_num=session,
                user_id=uid,
                img_index=i+1,
                on_image_saved=on_saved
                # on_image_saved=on_saved(local_path, i+1, params["seed"])
            )
            try:
                resize_inplace(local_path, (512, 512))
            except Exception as e:
                print(f"❌ Error resizing image {local_path}: {e}")
            local_paths.append(local_path)
            returned_seeds.append(returned_seed)
            images_bytes.append(image_bytes)

    # elif config.API_CALL == "open_ai":  # it inherently generates 4 images
    #     #currently unavailable - i commented out the import and the installation in requirements.txt
    #     paths = gpt_model.send_gpt_request(
    #         #I should add loging of the revised prompt of selected image...
    #         prompt=prompt,
    #         iteration=attempt,
    #         session_num=session,
    #         user_id=uid,
    #     )
    #     for p in paths:
    #         try:
    #             resize_inplace(p, (512, 512))
    #         except Exception as e:
    #             print(f"❌ Error resizing image {p}: {e}")
    #     local_paths.extend(paths)
    else:
        st.error(f"❌ Unknown API_CALL value: {config.API_CALL}, please contact experiment owner")
    
    return local_paths, returned_seeds, images_bytes

def log_row(**kw):
    f = config.LOG_DIR / f"{kw['uid']}.csv"
    
    first = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=kw.keys())
        if first:
            w.writeheader()
        w.writerow(kw)
    # f = str(f)   # remove unused line

def log_participant_info(uid: str, age: int, gender: str, native: str) -> Path:
    """
    Saves a participant info CSV file with their demographic details.
    Returns the Path to the saved CSV.
    """
    info_data = {
        "uid": uid,
        "age": age,
        "gender": gender,
        "native_language": native,
    }
    
    filename = f"participant_{uid}_info.csv"
    path = config.LOG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=info_data.keys())
        writer.writeheader()
        writer.writerow(info_data)

    return path


# Function returning a new widget key for the textbox every time we load a new target (e.g. a new ground truth image is presented)
def fresh_key() -> str:
    return f"prompt_{uuid4().hex}"

def fresh_neg_key() -> str:
    return f"neg_{uuid4().hex}"

#marking that image was rated
def mark_rated():
    # Use a per-session/attempt key so it resets automatically each attempt
    rated_key = f"rated_{S.session}_{S.attempt}"
    st.session_state[rated_key] = True
    # Log the time when the rating was made
    st.session_state[f"rated_time_{S.session}_{S.attempt}"] = time.time()

def is_rated() -> bool:
    return st.session_state.get(f"rated_{S.session}_{S.attempt}", False)

def commit_attempt_log():
    # Read the slider’s value from session (see the slider key below)
    rating_key = f"subjective_score_{S.session}_{S.attempt}"
    rating = st.session_state.get(rating_key, None)
    if rating is None:
        st.warning("No similarity rating found for this attempt. Please move the slider before continuing.")
        return
    
    #latency measures:
    metrics = getattr(S, "attempt_metrics", {}) or {}
    # compute rating latency if we have both times
    rated_epoch = st.session_state.get(f"rated_time_{S.session}_{S.attempt}")
    rating_latency = None
    if rated_epoch and metrics.get("generated_epoch"):
        rating_latency = round(rated_epoch - metrics["generated_epoch"], 3)

    for meta in S.last_gen_meta:
        p = Path(meta["path"])
        log_row(
            uid=S.uid,
            gt=S.gt_path.name,
            session=S.session,
            attempt=S.attempt,
            img_index=meta["img_index"],
            request_seed=meta["request_seed"] if config.API_CALL == "stability_ai" else "",
            returned_seed=meta["returned_seed"],
            prompt=S.prompt,
            negative_prompt=S.neg_prompt, 
            gen=p.name,
            subjective_score=rating,
            prompt_latency_secs=metrics.get("prompt_latency_secs"),
            model_latency_secs=metrics.get("model_latency_secs"),
            rating_latency_secs=rating_latency,
            generated_at_utc=metrics.get("generated_at_utc"),
            ts=int(time.time()),
        )
    upload_participant_log(S.uid)


# --- logging attempt timing helpers ---
def start_attempt():
    """Call whenever a new attempt becomes active (first attempt of a GT or after 'Another try')."""
    S.attempt_started_at = time.time()
    S.attempt_metrics = {}         # clear per-attempt metrics
    # rotate textbox key only when you want the box cleared
    # S.text_key = fresh_key()

# Function that loads a new ground-truth image (or finish the whole thing if none left in the folder) and reset all per-target variables. Then force an immediate rerun.
def next_gt():
    # Stop after MAX_SESSIONS
    if S.finished and not S.feedback_submitted:
        # Don't redirect yet, let the feedback screen show
        return
        
    if S.finished and S.feedback_submitted:
        # Now redirect after feedback is submitted
        st.success("Thank you for your feedback!")
        # Try JavaScript redirect
        st.markdown(f"""
        [**Click here** to get to back to Prolific and receive your credit]({config.PROLIFIC_URL}).
        """)
        st.caption("If the link doesn't work, please copy-paste the URL into your browser:\n https://app.prolific.com/submissions/complete?cc=C1OJX362 \n Or paste the following code directly inside prolific: C1OJX362")
        st.stop()
        

    remaining = [p for p in config.GT_DIR.glob("*.[pj][pn]g") if p.name not in S.used]
    if not remaining:
        S.finished = True
        rerun()

    # if a new session starts pick a new gt image: 
    S.gt_path = random.choice(remaining)
    S.used.add(S.gt_path.name) # keep same gt

    # New session setup:
    S.session += 1 # increase session counter
    start_attempt()
    S.seed = seed_from(S.gt_path.name)# instead of random - fixed per gt image: np.random.randint(1, 0, 2**32 - 1) # randomise the seed for the next generation
    S.attempt = 1
    S.generated = False
    S.gen_path = None
    S.gen_paths = [] # clear list on a new target
    S.last_prompt = "" # reset for new session - somehow doesn't work 
    S.neg_prompt = ""

    # S.last_score = 0.0

    # creating a new sussion folder for next session:
    update_session_folder()

    # st.session_state["prompt_text"] = ""  # because your text_area uses key="prompt_text"

    S.text_key = fresh_key() # new widget key so the existing widget value is not overwritten
    S.neg_text_key = fresh_neg_key()
    rerun()


# A helper for st.rerun() function in Steamlit. It is named differently in different versions of Steamlit, so we just make sure that we have something of this kind.
# It restarts your script from the top, keeping st.session_state intact. Use it after state changes that should immediately update the UI (e.g., after generating images, after moving to the next GT).
rerun = st.rerun if hasattr(st, "rerun") else st.experimental_rerun

# Defining a **st.session_state** - which is Streamlit’s dictionary like place for keeping data between reruns during a single session (for a given user). 
#this is a Session state init - sets values the first time the app runs
for k, v in {
    "uid": uuid4().hex, # user/session ID
    "used": set(), # set of ground-truth images already shown
    "gt_path": None, # TO BE CHANGED: path to the current ground-truth,
    "session":0, # The number of seesion per user (for the same user, we can have multiple sessions, e.g. if the user closes the browser and comes back later)
    "attempt": 1, # current attempt counter (from 1 to 5),
    "seed": np.random.randint(1,4000000), # seed for the image generation - randomized
    "generated": False, # TO BE CHANGED: telling whether we have a generated image to show or not
    "gen_path": None, #for only a single image
    "gen_paths": [], # used to be initialized as None, but now therre are multiple images
    "finished": False, # True when pool exhausted
    # "last_score": 0.0, # similarity of last generation
    # "cosine_distance": 0.0, # cosine distance of last generation
    "text_key": fresh_key(), # widget key for the textbox
    "last_prompt": "", # stores the last image description
    "neg_text_key": fresh_neg_key(),
    "neg_prompt": "",
    "subjective_score": None, # stores the last subjective score
    "is_generating": False, # Adds a deafult state - True when waiting for the model to generate
    "feedback_submitted": False,  # Add this new variable
    "attempt_started_at": None, #when prompt writing started
    "attempt_metrics": {},   # holds timings for the current attempt
}.items():
    st.session_state.setdefault(k, v)
S = st.session_state



st.set_page_config(page_title="Imagination in Translation", layout="wide")
S.setdefault("last_gen_meta", []) # list of dicts with metadata of last generation (one dict per image)
st.markdown("""
<style>
.small-ital-grey { font-size:0.95rem; color:#666; font-style: italic; margin:.25rem 0 .4rem 0; }
</style>
""", unsafe_allow_html=True)


# Hide the built-in fullscreen control on media (images, charts, etc.)
st.markdown("""
    <style>
     button[kind="primary"]{background:#8B0000;color:white}
    button[kind="primary"]:hover{background:#A80000;color:white}
    /* Streamlit uses a dedicated button for fullscreen */
    [data-testid="StyledFullScreenButton"] { display: none !important; }
    /* Older builds expose a title attr */
    button[title="View fullscreen"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# Load at startup
if 'gt_images_cache' not in st.session_state:
    st.session_state.gt_images_cache = load_gt_images_base64()

# --- Consent gate (shown once) ---
if "consent_given" not in S:
    S.consent_given = False

if not S.consent_given:
    with st.form("consent_form", clear_on_submit=False):
        st.subheader("Participant Information & Consent")
        st.markdown("""
                    In this study you will be asked to describe images in the most accurate way possible. 
                    - Once you finish and click **generate**, an image will appear based on your description.
                    - It may take up to 30 seconds to generate an image. Please wait patiently.
                    - Once an image is generated you will have to **rate** it's similarity to the original image you were shown on a scale of 1 (least similar) to 100 (most similar).
                    - If the image is not similar to the original image, update your description to make them as similar as possible.
                    - After you finish the three attempts, a new image will be presented and you will repeat the process.
                    - You will get ***3 attempts** per image, in each you may modify your previous description, or expand it to include more aspects of the original image that are missing in the generated image (you can't use an identical description).
                    - ***Duration**: The experiment should take about 40 minutes to complete.
                    """)
        # Informed consent – translation & adaptation of your bullets
        st.markdown("""
                    **By clicking *I agree and continue*, I confirm that:**
                    - I **voluntarily** agree to participate in this study.
                    - No graphic or unpleasent content is expected to be shown during the study.
                    - No risks are expected from participation in this study.
                    - I may **stop my participation at any time** by closing the window, **without penalty** or loss of benefits.
                    - The research team will **protect my privacy and confidentiality**. 
                    - I can **ask questions at any time** and receive answers from the research team at **anat.korol@mail.tau.ac.il**.
                        """)
        
        agree = st.checkbox("I have read the study information and **I consent** to participate.")
        submit = st.form_submit_button("I agree and continue")
        
        if submit:
            if not agree:
                st.warning("Please, consent to participate in the study in order to proceed, or exit.")

            else:
                S.consent_given = True

                #(Optional) Log consent
                # log_row(uid=S.uid, event="consent")  # add 'event' to your CSV writer if you like

                st.rerun()
    st.stop()

# Participant info form (shown only once at the start of the session) - makes sure there's a button
if "participant_info_submitted" not in S:
    S.participant_info_submitted = False
# if form didn't appear yet, show it 
if not S.participant_info_submitted:
    with st.form("participant_info_form"):
        st.header("Participant Information")
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        native = st.radio("Are you a native English speaker?", ["No", "Yes"])
        submit = st.form_submit_button("Submit")
        st.markdown("##### _Note: Loading the experiment may take 10-20 seconds, please be patient..._")

        if submit:
            # S.transition_loading = True  
            S.participant_age = age
            S.participant_gender = gender
            S.participant_native = native
            S.participant_info_submitted = True
            
            # Save locally first
            info_path = log_participant_info(S.uid, age, gender, native)
            print(f"Participant info saved to {info_path}")
            # upload to drive pipeline
            # make sure participant's tree exists
            init_drive_tree_for_participant() 
            # upload the participant info (one-time, overwrite-safe by name)
            upload_participant_info(S.uid, info_path)
        # <-- add this
            st.rerun()
    st.stop()

# --- Feedback screen (shown when experiment is complete) ---
if S.finished and not S.feedback_submitted:
    st.markdown("""
    <div style="text-align: center; margin: 50px 0;">
        <h2>Experiment Complete!</h2>
        <p>Thank you for participating in our study.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form"):
        st.subheader("Optional Feedback")
        st.markdown("""
        We would appreciate your thoughts about the experiment. 
        Your feedback helps us improve future studies.
        """)
        
        feedback = st.text_area(
            "Please share any thoughts, observations, or suggestions about the experiment:",
            height=150,
            placeholder="How was your experience? Any technical issues? Suggestions for improvement?"
        )
        

        difficulty = st.select_slider(
            "How would you rate the overall difficulty of the task?",
            options=["Very Easy", "Easy", "Moderate", "Difficult", "Very Difficult"],
            value="Moderate"
        )
        
        clarity = st.select_slider(
            "How clear were the instructions?",
            options=["Very Unclear", "Unclear", "Somewhat Clear", "Clear", "Very Clear"],
            value="Somewhat Clear"
        )
    
        # Buttons in columns like your other forms
        col1, col2 = st.columns([1, 1])
        with col1:
            skip = st.form_submit_button("Skip feedback", use_container_width=True)
        with col2:
            submit_feedback = st.form_submit_button("Submit feedback", type="primary", use_container_width=True)
        
        if submit_feedback or skip:
            # Log the feedback if provided
            if submit_feedback and feedback.strip():
                feedback_data = {
                    "uid": S.uid,
                    "feedback": feedback.strip(),
                    "difficulty": difficulty,
                    "clarity": clarity,
                    "ts": int(time.time())
                }
                
                # Save feedback to CSV
                feedback_file = config.LOG_DIR / f"{S.uid}_feedback.csv"
                with feedback_file.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
                    writer.writeheader()
                    writer.writerow(feedback_data)
                
                # Upload to Drive
                try:
                    update_or_insert_file(
                        service, feedback_file, S.participant_folder_id,
                        dest_name=f"participant_{S.uid}_feedback.csv",
                        mime_type="text/csv"
                    )
                except Exception as e:
                    print(f"Failed to upload feedback: {e}")
            
            S.feedback_submitted = True
            rerun()
    
    st.stop()

if S.gt_path is None:      
    next_gt()

# The finish screen (appears when the user presses "exit" button or there are no more ground truth pictures).
# if S.finished:
#     st.markdown(
#         "<h2 style='text-align:center'>The session is finished.<br>Thank you for participating!</h2>",
#         unsafe_allow_html=True,
#     )
#     st.stop()

# Layout of the textbox and pictures (next to each other)
left, right = st.columns([1, 2], gap="large")

# st.markdown(
#     "**Please, describe the picture as precisely as possible in English only. You have 4 attempts to improve your description. \n'Press ctrl + enter buttons after you are done typing to apply the text. Note that you cannot use the same description twice.**"
# )
# left column: textbox for descriptive prompt and "generate" and "exit" buttons.
with left:
    st.markdown(
        "**Please, describe the picture as precisely as possible in English only. "
        "You have 3 attempts to describe every image, feel free to edit/change your description as you like.**"
    )

    # --- FORM: text + submit button live inside the same form ---
    with st.form("gen_form", clear_on_submit=False):
        text_disabled = S.generated
        prompt_val = st.text_area(
            "The picture shows...",
            key=S.text_key, # I think this shoulld be S.text_key
            height=220,
            placeholder="Type an accurate description of the target image.",
            disabled=text_disabled
        )
        c1, c2 = st.columns(2) # add count for characters and attempt number
        c1.caption(f"{len(prompt_val)} characters")
        c2.caption(f"{S.attempt} / {config.REQUIRED_ATTEMPTS}")

        #negative prompt area
        neg_text_disabled = S.generated  or (S.attempt == 1)# same behavior as main textbox
        neg_val = st.text_area(
            "Negative prompt: if there is an item/object in the generated image that does not appear in the original image and you want to remove it, you can write it down here: (optional from second attempt onwards)",
            key=S.neg_text_key,
            height=100,
            placeholder="e.g., objects, people, text.",
            disabled=neg_text_disabled
        )


        # ---- two side-by-side form submit buttons - generate or another try/done ----
        gen_button, next_button = st.columns([1, 1])
        
        #because it's form, gen is disabled when image is generated (not for wrong text inputs because character counter doesn't update so it locks it in generation
        gen_disabled = S.generated or S.is_generating

        #generate button:
        gen_clicked = gen_button.form_submit_button("Generate", type="primary", disabled=gen_disabled, use_container_width=True)

        # label toggles when last attempt reached
        is_final   = (S.attempt >= config.REQUIRED_ATTEMPTS)
        next_label = "Done: Next image" if is_final else "Another try"

        rated = is_rated()  # my helper
        next_disabled = (not S.generated) or (not rated)
        #Anoter try or done button:
        next_clicked = next_button.form_submit_button(next_label, disabled=next_disabled, use_container_width=True)

    if gen_clicked:
        if S.attempt_started_at:
            S.attempt_metrics = S.attempt_metrics or {}
            S.attempt_metrics["prompt_latency_secs"] = round(time.time() - S.attempt_started_at, 3)
        else:
            # fallback if something started without calling start_attempt()
            S.attempt_metrics = {"prompt_latency_secs": None}
        
        #regular text:
        raw_text = st.session_state.get(S.text_key, "")
        #negative prompt text
        raw_neg = st.session_state.get(S.neg_text_key, "")
        # allow only these chars
        allowed_full_line = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\:\;\'\"\-\(\)]*$')

        #start collecting error
        errors = []

        
        # --- MAIN PROMPT ---
        if not raw_text.strip():
            errors.append("Please enter a description.")
        if not allowed_full_line.match(raw_text):
            errors.append("Please use only letters, numbers, spaces, and . , ! ? : ; ' \" - ( ).")
        if any(i in raw_text for i in config.websites):
            errors.append("Please, do not use links in your description.")
        if S.attempt > 1 and raw_text.strip() == S.last_prompt.strip():
            errors.append("Please modify your description before generating again.")

        # --- NEGATIVE PROMPT (optional) ---
        neg_used = ""
        if not errors:
            if S.attempt == 1:
                # Ignore anything they somehow typed (defense in depth)
                neg_used = ""
            else:
                if S.attempt > 1 and raw_neg.strip():
                    if not allowed_full_line.match(raw_neg):
                        errors.append("Negative prompt: Please use only letters, numbers, spaces, and . , ! ? : ; ' \" - ( ).")
                    if any(i in raw_neg for i in config.websites):
                        errors.append("Negative prompt: links are not allowed.")
                    # only truncate after it passes symbol/link validation
                    neg_used = raw_neg[: config.MAX_LENGTH - 1].strip()

        #if only neg prompt is modified - don't allow it:
        if not errors and S.attempt > 1 and raw_text.strip() == S.last_prompt.strip() and neg_used and neg_used != S.neg_prompt:
            errors.append("Please modify your main description, not only the negative prompt.")
        
        # Validate main prompt
        # --- collect validation errors without st.stop() ---
        
        
        if errors:
            for e in errors:
                st.error(e)            # show messages
             # also make sure we are not stuck in a generating state
            S.is_generating = False
            S.generated = False

            # IMPORTANT: do not st.stop(), do not generate, simply fall through
        else:

            # --- proceed to generation ---
            S.neg_prompt = neg_used
            prompt_used  = raw_text[: config.MAX_LENGTH - 1]
            S.prompt     = prompt_used.strip()

            S.gen_paths = []
            S.generated = False # because it's before generation
            try:
                #  ensure session folder exists BEFORE calling api (in case a callback uploads)
                update_session_folder() 
                # API call
                S.is_generating = True
                with st.spinner("Generating the image may take 20-30 seconds…"):
                    t0 = time.time() # starting clock for model latency measurement
                    S.gen_paths, returned_seeds, images_bytes = generate_images(S.prompt, S.neg_prompt, S.seed, S.session, S.attempt, S.gt_path, S.uid) # generate the image ## Note: i only return one image bytes
                    model_latency = time.time() - t0
                S.images_bytes = images_bytes 
                S.is_generating = False
                
                # store model latency + timestamps
                S.attempt_metrics["model_latency_secs"] = round(model_latency, 3)
                S.attempt_metrics["generated_at_utc"]   = datetime.now(timezone.utc).isoformat()
                S.attempt_metrics["generated_epoch"]    = time.time()

                print(f"Generated image/images saved: {[Path(p).name for p in S.gen_paths]}")
                

                # Upload each generated image to Drive (names already include attempt/img index)
                for i, gen_path in enumerate(S.gen_paths, start=1):
                    gen_path = Path(gen_path)
                    # uploading to google drive:
                    dest = image_dest_name(S.uid, S.session, S.attempt, i, gen_path.suffix) # including index
                    update_or_insert_file(service, gen_path, S.session_drive_folder_id, dest_name=dest)
                S.last_gen_meta = []
                # log only relevant parts locally (wait for full ogging after similarity ratins)
                for i, gen_path in enumerate(S.gen_paths, start=1):
                    gen_path = Path(gen_path)
                    S.last_gen_meta.append({
                        "path": str(gen_path),   # safe to stringify
                        "img_index": i,
                        "request_seed": S.seed if config.API_CALL == "stability_ai" else "",
                        "returned_seed": str(returned_seeds[i - 1]) if returned_seeds else "",
                        "image_bytes": images_bytes[i - 1] if images_bytes else None
                        })
                #     log_row(
                #         uid=S.uid,
                #         participant_age=S.participant_age,
                #         participant_gender=S.participant_gender,
                #         participant_native=S.participant_native,
                #         gt=S.gt_path.name,
                #         session=S.session,
                #         attempt=S.attempt,
                #         img_index=i,  # for multiple image generation
                #         request_seed=S.seed if config.API_CALL == "stability_ai" else "",
                #         returned_seed=str(returned_seeds[i - 1]) if returned_seeds else "",  # because openai doesn't return a seed
                #         prompt=S.prompt,
                #         gen=gen_path.name,
                #         # similarity=score,
                #         # subjective_score=S.subjective_score if "subjective_score" in S else None,
                #         ts=int(time.time())
                #     )
                
                # upload_participant_log(S.uid)  # outside the image loop, uploading the participant log
                    
                S.generated = True
                S.last_prompt = S.prompt.strip()  # save the last prompt to check if it is the same as the current one
                rerun()
                # except errors regarding generation!
            except Exception as e:
                msg = str(e)
                print(e)
                KNOWN_ERRORS = config.KNOWN_ERRORS
                if KNOWN_ERRORS["required_field"] in msg:
                    st.error("Some required field is missing. Please check the inputs.")
                elif KNOWN_ERRORS["content_moderation"] in msg:
                    st.error("Your request was flagged for unsafe content. Please rephrase.")
                elif KNOWN_ERRORS["payload_too_large"] in msg:
                    st.error("Your prompt is too large. Try shortening it.")
                elif KNOWN_ERRORS["language_not_supported"] in msg:
                    st.error("Only English is supported. Please write in English.")
                elif KNOWN_ERRORS["rate_limit"] in msg:
                    st.error("Too many requests. Please wait a moment and try again.")
                elif KNOWN_ERRORS["server_error"] in msg:
                    st.error("Server error. Please try again shortly.")
                elif KNOWN_ERRORS["Invalid_Api"] in msg:
                    st.error("Invalid API key. Please check readme for more details.")
                else:
                    st.error(f"Unexpected error: {msg}")
                S.is_generating = False
                S.generated = False
                S.gen_paths = []

    elif next_clicked:
        # log with rating then either another attempt or next GT
        commit_attempt_log()  # uses the slider key; we fixed that earlier
        if is_final:
            next_gt()
            S.text_key = fresh_key()
            # S.last_prompt = ""     # allow changed-text validation to work correctly
        else:
            # prepare next attempt and clear per-attempt artifacts
            old_attempt = S.attempt
            S.attempt += 1
            S.generated = False
            S.gen_paths = []
            S.last_gen_meta = []
            # clear rating state for the old attempt
            st.session_state.pop(f"subjective_score_{S.session}_{old_attempt}", None)
            st.session_state.pop(f"rated_{S.session}_{old_attempt}", None)
            # blank the textbox by rotating the key
            # S.text_key = fresh_key()
            # S.last_prompt = ""     # allow changed-text validation to work correctly
            
            start_attempt()   # <-- start timing for the new attempt

            rerun()

#make stramlit not show the fullscreen option button!
#make streamlit not show the fullscreen option button!
st.markdown("""
<style>
/* Hide fullscreen button - multiple selectors for different Streamlit versions */
[data-testid="StyledFullScreenButton"] {
    display: none !important;
}
[data-testid="baseButton-header"] {
    display: none !important;
}
[data-testid="fullScreenButton"] {
    display: none !important;
}
button[title="View fullscreen"] {
    display: none !important;
}
.element-container button[kind="secondary"] {
    display: none !important;
}
/* Hide any button in the image container that has fullscreen-like properties */
.stImage button {
    display: none !important;
}
[data-testid="stImage"] button {
    display: none !important;
}
/* Additional fallback */
div[data-testid="stImage"] > button {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Right column displays the ground truth (target) image and the generated one (next to each other) together with similarity scores, "accept" and "try again" buttons.
with right:
    gt_display, gen_display = st.columns([1, 1], gap="medium")
    with gt_display:
        st.markdown('<div class="small-ital-grey">Target image</div>', unsafe_allow_html=True)
        show_img_fixed(S.gt_path, GT_BOX)  # presenting as html to avoid fullscreen
    with gen_display:
        if not S.generated:
            st.caption("_Generated image_")
            st.markdown('<div class="small-ital-grey">Click **Generate** to view the image.</div>', unsafe_allow_html=True)
        elif not S.gen_paths:
            st.markdown('<div class="small-ital-grey">Generated image</div>', unsafe_allow_html=True)
            st.warning("No image was produced. Please try again.")
        else:
            if len(S.gen_paths) == 1:
                st.markdown('<div class="small-ital-grey">Generated image</div>', unsafe_allow_html=True)
                show_img_fixed(S.images_bytes[0], GEN_BOX)
            else:  # 2 images
                st.markdown("### Generated images", unsafe_allow_html=True)
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    show_img_fixed(S.images_bytes[0], GEN_BOX)
                with c2:
                    show_img_fixed(S.images_bytes[1], GEN_BOX)


    # st.markdown("###### Target image:")

    # # Make GT roughly half-width responsively:
    # # - Only use the first column; leave the second empty.
    # # - Tweak the ratio to change the GT size:
    # #   [1,1]  -> ~50% of the container
    # #   [2,1]  -> ~66%
    # #   [1,2]  -> ~33%
    # col_gt, _ = st.columns([1, 1], gap="medium")  
    # with col_gt:
    #     st.image(Image.open(S.gt_path), use_container_width=True, clamp=True, caption="")

    # st.markdown("---")
    # # st.subheader("Generated images")

    # if S.generated and len(S.gen_paths) == 2:
    #     c1, c2 = st.columns(2, gap="large")
    #     with c1:
    #         st.image(Image.open(S.gen_paths[0]), use_container_width=True, clamp=True)
    #     with c2:
    #         st.image(Image.open(S.gen_paths[1]), use_container_width=True, clamp=True)


    st.markdown(" ")  # spacer
    if S.generated:
        st.caption("_Rate similarity_")
        # S.subjective_score = 
        st.slider(
            "Please, rate the Similarity between the **Generated Image** and **Target Image** (0 = not similar, 100 = very similar)",
            min_value=0,
            max_value=100,
            value=50,  # default position
            step=1,
            key=f"subjective_score_{S.session}_{S.attempt}", 
            disabled=not S.generated,
            on_change=mark_rated
        )
    
    # After displaying images / slider...
    # rated = is_rated()

    # done_disabled = (not S.generated) or (not rated) or (S.attempt < config.REQUIRED_ATTEMPTS)
    # try_disabled  = (not S.generated) or (not rated) or (S.attempt >= config.REQUIRED_ATTEMPTS)

    # # done_disabled = (S.attempt < config.REQUIRED_ATTEMPTS) or not S.generated
    # # try_disabled = S.attempt >= config.REQUIRED_ATTEMPTS or not S.subjective_score
    # try_col, done_col = st.columns(2)
    # # "Try again" button is disabled on 5th attempt
    # if try_col.button("Another try", disabled=try_disabled):
    #     # if not S.generated or not rated:
    #     #     st.warning("Rate similarity of the generated image to the original image first.")
    #         # st.stop()
    #     commit_attempt_log()  # log the attempt before changing state
    #     # S.seed = np.random.randint(1, 4000000) - I don't want randomization per attempts
    #     S.generated = False
    #     st.session_state.pop(f"rated_{S.session}_{S.attempt}", None)
    #     S.gen_paths = []
    #     S.last_gen_meta = []
    #     S.attempt += 1

    #     rerun()
    # else:
    #     st.caption("Click **Generate** to view images.")

    # if done_col.button("DONE : Next image", disabled=done_disabled):
    #     commit_attempt_log()  # <-- log with the slider’s value
    #     # if not S.generated or not rated or (S.attempt < config.REQUIRED_ATTEMPTS):
    #     #     # st.warning("Rate similarity of the generated image to the original image first.")
    #     #     # st.stop()
        
    #     next_gt()