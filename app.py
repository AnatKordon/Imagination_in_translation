# To call the app, write:  run ui/ui_prototype.py --server.port 8501 in the codespace terminal  but first, install the required libraries listed in the requirements.txt file with a single command: pip install -r requirements.txt.
# Note, that by default a user has to press ctrl+enter after filling in the text box to apply the text, count characters, send it to generation etc. 
from pathlib import Path
import config       
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

# from google.oauth2 import service_account
# from mimetypes import guess_type # for uploading to google drive - png or csv

from dotenv import load_dotenv
load_dotenv()


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

def on_saved(path: Path, seed: str):
    dest = image_dest_name(S.uid, S.session, S.attempt, idx=1, suffix=path.suffix)  # set idx appropriately
    update_or_insert_file(service, path, S.session_drive_folder_id, dest_name=dest)

st.set_page_config(page_title="Imagination in Translation", layout="wide")


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

#  handling resizing as a function
def resize_inplace(p: Path, size=(512, 512)) -> None:
    img = Image.open(p)
    img = ImageOps.contain(img, size)  # preserves aspect ratio inside 512x512
    img.save(p)


#new generation of 4 images
def generate_images(prompt: str, seed: int, session: int, attempt: int, gt: Path, uid: str) -> list[Path]:
    params = config.params.copy()
    params["prompt"] = prompt
    local_paths = []
    returned_seeds = []
    N_OUT = 2 #multiple images generation
    
    if config.API_CALL == "stability_ai":
        for i in range(N_OUT):  # generate 4 images
            params["seed"] = seed + i  # vary seed to get diversity - this is the requested seed
            local_path, returned_seed = api_model.send_generation_request(
                host="https://api.stability.ai/v2beta/stable-image/generate/sd3",
                params=params,
                iteration=attempt,
                session_num=session,
                user_id=uid,
                img_index=i+1,
                on_image_saved=on_saved
            )
            try:
                resize_inplace(local_path, (512, 512))
            except Exception as e:
                print(f"❌ Error resizing image {local_path}: {e}")
            local_paths.append(local_path)
            returned_seeds.append(returned_seed)

    elif config.API_CALL == "open_ai":  # it inherently generates 4 images
        #currently unavailable - i commented out the import and the installation in requirements.txt
        paths = gpt_model.send_gpt_request(
            #I should add loging of the revised prompt of selected image...
            prompt=prompt,
            iteration=attempt,
            session_num=session,
            user_id=uid,
        )
        for p in paths:
            try:
                resize_inplace(p, (512, 512))
            except Exception as e:
                print(f"❌ Error resizing image {p}: {e}")
        local_paths.extend(paths)
    else:
        st.error(f"❌ Unknown API_CALL value: {config.API_CALL}, please contact experiment owner")
    
    return local_paths, returned_seeds

#similarity scores for all gen images
# def similarities(GT_path: Path, GEN_paths: list[Path]) -> list[float]:
#     embedder = st.session_state.vgg_embedder  # use cached model
#     GT_embedding = embedder.get_embedding(img_path=str(GT_path))
#     scores = []
#     for gen_path in GEN_paths:
#         GEN_embedding = embedder.get_embedding(img_path=str(gen_path))
#         score, _ = vgg_similarity.compute_similarity_score(GT_embedding, GEN_embedding)
#         scores.append(score)
#     return scores

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

    # if a new session starts (still same gt_image): 
    S.gt_path = random.choice(remaining)
    S.used.add(S.gt_path.name) # keep same gt

    S.session += 1 # increase session counter
    # update_gen_folder()
    S.seed = np.random.randint(1, 4000000) # randomise the seed for the next generation
    S.attempt = 1
    S.generated = False
    S.gen_path = None
    S.gen_paths = [] # clear list on a new target
    # S.last_score = 0.0

    # creating a new sussion folder for next session:
    update_session_folder()

    S.text_key = fresh_key() # new widget key so the existing widget value is not overwritten
    rerun()


# Defining a st.session_state, which is Streamlit’s dictionary like place for keeping data between reruns during a single session (for a given user). 
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
}.items():
    st.session_state.setdefault(k, v)
S = st.session_state

if "participant_info_submitted" not in S:
    S.participant_info_submitted = False
if not S.participant_info_submitted:
    with st.form("participant_info_form"):
        st.header("Participant Information")
        age = st.number_input("Age", min_value=10, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        native = st.radio("Are you a native English speaker?", ["Yes", "No"])

        submit = st.form_submit_button("Submit")
        if submit:
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
            st.rerun()
    st.stop()
if S.gt_path is None:      
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
    # st.write(f"**Your ID:** `{S.uid}`")
    st.markdown("**Please, describe the picture as precisely as possible. You have up to 5 attempts to improve your description. Press ctrl + enter buttons after you are done typing to apply the text. Note that you cannot use the same description twice.**")

# Textbox (unique key per target)
    S.prompt = st.text_area(
        "The picture shows...",
        key=S.text_key,
        height=140,
        placeholder="Type an accurate description of the target image. After finished press ctrl+enter to apply the text.",
    )

    ## ERROR HANDLING for the prompt text
    
    ## checks whether the user has changed the description since the last attempt
    same_prompt = S.prompt.strip() == S.last_prompt.strip()

    ##check for symbols##
    symbols = bool(re.search(r"[^a-zA-Z0-9\s.,!?'\"\()-]", S.prompt))
    if symbols:
        st.error("Please, use only letters, numbers, spaces and punctuation marks (.,!?') in your description. Other symbols are not allowed.")

    ## Check for https links ##
    http = any(i in S.prompt for i in config.websites)
    if http:
        st.error("Please, do not use links in your description. Only text is allowed.")

    ##Check for legth##
    if len(S.prompt) > config.MAX_LENGTH-1:
        st.error(f"Your description is too long. Only the first {config.MAX_LENGTH} characters will be used.")
        S.prompt = S.prompt[:config.MAX_LENGTH-1]



# Character counters (below the box)
    c1, c2 = st.columns(2)
    c1.caption(f"{len(S.prompt)} characters")
    c2.caption(f"{S.attempt} / {config.MAX_ATTEMPTS}")
   
    if same_prompt and not S.generated and S.attempt > 1:
        st.info("Please modify your description before generating again.")

    gen_disabled = (
        S.generated
        or not str(S.prompt).strip()
        or S.attempt > config.MAX_ATTEMPTS
        or symbols
        or http
        or same_prompt
    )

    if st.button("Generate", type="primary", disabled=gen_disabled):
        try:
            S.gen_paths, returned_seeds = generate_images(S.prompt, S.seed, S.session, S.attempt, S.gt_path, S.uid) # generate the image
            print(f"Generated images saved: {[Path(p).name for p in S.gen_paths]}")
            
            # trying to upload
            update_session_folder() # ensure session folder id
            
        #  I removed calculating vgg similarity score
        # try:
        #     S.last_scores = similarities(S.gt_path, S.gen_paths)
        # except Exception as e:
        #     print(e)
        #     st.error(f"Error calculating similarity: {e}")
        #     S.last_scores = [0.0] * len(S.gen_paths)

            # Upload each generated image to Drive (names already include attempt/img index)
            for i, gen_path in enumerate(S.gen_paths, start=1):
                gen_path = Path(gen_path)
                # uploading to google drive:
                dest = image_dest_name(S.uid, S.session, S.attempt, i, gen_path.suffix) # including index
                update_or_insert_file(service, gen_path, S.session_drive_folder_id, dest_name=dest)

            # Log locally (now with index + true/returned seed)
            for i, gen_path in enumerate(S.gen_paths, start=1):
                gen_path = Path(gen_path)
                log_row(
                    uid=S.uid,
                    participant_age=S.participant_age,
                    participant_gender=S.participant_gender,
                    participant_native=S.participant_native,
                    gt=S.gt_path.name,
                    session=S.session,
                    attempt=S.attempt,
                    img_index=i,  # for multiple image generation
                    request_seed=S.seed + i if config.API_CALL == "stability_ai" else "",
                    returned_seed=str(returned_seeds[i - 1]) if returned_seeds else "",  # because openai doesn't return a seed
                    prompt=S.prompt,
                    gen=gen_path.name,
                    # similarity=score,
                    subjective_score=S.subjective_score if "subjective_score" in S else None,
                    ts=int(time.time())
                )
           
            upload_participant_log(S.uid)  # outside the image loop, uploading the participant log
                
            S.generated = True
            S.last_prompt = S.prompt.strip()  # save the last prompt to check if it is the same as the current one
            rerun()
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
                S.generated = False
                S.gen_paths = []

        # flags session as finished and rerun
    if st.button("Exit"):
        S.finished = True
        rerun()

# Right column displays the ground truth (target) image and the generated one (next to each other) together with similarity scores, "accept" and "try again" buttons.
with right:
    st.markdown("##### Target image:")

    # Make GT roughly half-width responsively:
    # - Only use the first column; leave the second empty.
    # - Tweak the ratio to change the GT size:
    #   [1,1]  -> ~50% of the container
    #   [2,1]  -> ~66%
    #   [1,2]  -> ~33%
    col_gt, _ = st.columns([1, 1], gap="medium")  
    with col_gt:
        st.image(Image.open(S.gt_path), use_container_width=True, clamp=True, caption="")

    st.markdown("---")
    # st.subheader("Generated images")

    if S.generated and len(S.gen_paths) == 2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(Image.open(S.gen_paths[0]), use_container_width=True, clamp=True)
        with c2:
            st.image(Image.open(S.gen_paths[1]), use_container_width=True, clamp=True)

            #removing the single similarity score
            # st.caption("Similarity:")
            # st.progress(int(S.last_score))
            # st.write(f"**{S.last_score:.1f}%**")
            
        st.markdown(" ")  # spacer
        S.subjective_score = st.slider(
            "Subjective Similarity (0 = not similar, 100 = very similar)",
            min_value=0,
            max_value=100,
            value=50,  # default position
            step=1,
            key=f"subjective_score_{S.session}_{S.attempt}",
        )
        
        a_col, t_col = st.columns(2)
        if a_col.button("DONE : Next image"):
            next_gt()

        # "Try again" button is disabled on 5th attempt
        if t_col.button("Another try", disabled=S.attempt >= config.MAX_ATTEMPTS):
            S.seed = np.random.randint(1, 4000000) 
            S.generated = False
            S.gen_paths = []
            S.attempt += 1
            rerun()
    else:
        st.caption("Click **Generate** to view images.")
