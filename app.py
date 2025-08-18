# To call the app, write:  run ui/ui_prototype.py --server.port 8501 in the codespace terminal  but first, install the required libraries listed in the requirements.txt file with a single command: pip install -r requirements.txt.
# Note, that by default a user has to press ctrl+enter after filling in the text box to apply the text, count characters, send it to generation etc. 
from pathlib import Path
import config       
from uuid import uuid4 # used to create session / user IDs
from models import api_model # the model API wrapper
from similarity import vgg_similarity # the similarity function 
from drive_utils import get_drive_service, create_folder, upload_file
import random, csv, time 
import time
import os
import numpy as np 
import re
import streamlit as st # Streamlit UI framework
from PIL import Image, ImageOps # Pillow library to manipulate images

from google.oauth2 import service_account
from mimetypes import guess_type # for uploading to google drive - png or csv

from dotenv import load_dotenv

# ensuring all the required folders exist so .save() or logging never crash
for d in (config.GEN_DIR, config.LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Setting up the appearance
st.set_page_config(page_title="Imagination in Translation", layout="wide") # the page is full-width 

load_dotenv()
#load google drive api db:
if "google" in st.secrets:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["google"],
        scopes=[
            "https://www.googleapis.com/auth/drive.file",  # Only files created by the app
            "https://www.googleapis.com/auth/drive.folder"  # Folder operations
        ]
    )
    #for local host
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    creds = service_account.Credentials.from_service_account_file(
        Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        scopes=[
            "https://www.googleapis.com/auth/drive.file",  # Only files created by the app
            "https://www.googleapis.com/auth/drive.folder"  # Folder operations
        ]
    )
else:
    creds = None
    print("❌ Error, No Google Drive credentials found. contact experiment host")

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



def generate_image(prompt: str,seed:int,session:int ,attempt: int, gt: Path,id: str) -> Path:
    params = config.params.copy() # copy the params dict so we don't change the original one
    #adding changing params
    params["prompt"] = prompt # set the prompt
    params["seed"] = seed # set the seed
    path = api_model.send_generation_request(host="https://api.stability.ai/v2beta/stable-image/generate/sd3",params=params,
                                      iteration=attempt,session_num=session,user_id=id)

    ImageOps.contain(Image.open(path), (512, 512))
    return path


def similarity(GT_path: Path, GEN_path: Path) -> float:
    ## use vgg to create embeddings and calculate similarity
    embedder = vgg_similarity.get_vgg_embedder()  # get the VGG embedder
    GT_path = str(GT_path)  # convert Path to string for the similarity function
    GEN_path = str(GEN_path)  # convert Path to string for the similarity function
    GT_embedding = embedder.get_embedding(img_path=GT_path)  # get embedding for the ground truth image
    GEN_embedding = embedder.get_embedding(img_path=GEN_path)  # get embedding for the generated image

    objective_similarity_score, cosine_distance = vgg_similarity.compute_similarity_score(GT_embedding, GEN_embedding)

    return objective_similarity_score , cosine_distance


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

def update_gen_folder():
    if "drive_service" in S and "participant_drive_folder_id" in S:
        gen_images_root = create_folder(S.drive_service, "gen_images", S.participant_drive_folder_id)
        S.gen_drive_folder_id = create_folder(S.drive_service, f"session_{S.session:02d}", gen_images_root)

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
    S.session += 1 # increase session counter
    update_gen_folder()
    S.seed = np.random.randint(1, 4000000) # randomise the seed for the next generation
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
    "gt_path": None, # TO BE CHANGED: path to the current ground-truth,
    "session":0, # The number of seesion per user (for the same user, we can have multiple sessions, e.g. if the user closes the browser and comes back later)
    "attempt": 1, # current attempt counter (from 1 to 5),
    "seed": np.random.randint(1,4000000), # seed for the image generation - randomized
    "generated": False, # TO BE CHANGED: telling whether we have a generated image to show or not
    "gen_path": None, # TO BE CHANGED: path to generated image
    "finished": False, # True when pool exhausted
    "last_score": 0.0, # similarity of last generation
    "cosine_distance": 0.0, # cosine distance of last generation
    "text_key": fresh_key(), # widget key for the textbox
    "last_prompt": "", # stores the last image description
}.items():
    st.session_state.setdefault(k, v)
S = st.session_state

#store the google drive service in the session state
if creds and "drive_service" not in S:
    service = get_drive_service(creds)
    S.drive_service = service
    
    # Create root folder once
    root_folder_id = create_folder(service, "participants_data")
    print(f"Root folder created with ID: {root_folder_id}")
    # Create participant folder once and store its ID
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # e.g., 20250817-134512
    folder_name = f"{timestamp}_{S.uid}"
    participant_folder_id = create_folder(service, folder_name, root_folder_id)
    print(f"Participant folder created with ID: {participant_folder_id}")
    S.participant_drive_folder_id = participant_folder_id

    # Create subfolders for generated images
    gen_images_root = create_folder(service, "gen_images", participant_folder_id)
    S.gen_drive_folder_id = create_folder(service, f"session_{S.session:02d}", gen_images_root)

if "participant_info_submitted" not in S:
    S.participant_info_submitted = False
# If the participant info is not submitted, show the form to collect it.
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
            
            info_path = log_participant_info(S.uid, age, gender, native)

            # Upload to Google Drive
            if "drive_service" in S and "participant_drive_folder_id" in S:
                upload_file(S.drive_service, info_path, "text/csv", S.participant_drive_folder_id)

            st.success("Thank you!")
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
            S.gen_path = generate_image(S.prompt,S.seed,S.session ,S.attempt, S.gt_path,S.uid) # generate the image
            print(f"Generated image saved to {S.gen_path}")
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
                
        try:
            S.last_score,S.cosine_distance = similarity(S.gt_path, S.gen_path)
        except Exception as e:
            print(e)
            st.error(f"Error calculating similarity: {e}")
            S.last_score = 0.0


        log_row(
            uid=S.uid,
            # participant_age=S.participant_age,
            # participant_gender= S.participant_gender,
            # participant_native=S.participant_native,
            gt=S.gt_path.name,
            session = S.session,
            attempt=S.attempt,
            seed=S.seed,
            prompt=S.prompt,
            gen=S.gen_path.name,
            similarity=round(S.last_score, 4),
            cosine_distance=round(S.cosine_distance, 4),
            subjective_score=S.subjective_score if "subjective_score" in S else None,
            ts=int(time.time()),
        )
        S.generated = True
        S.last_prompt = S.prompt.strip()  # save the last prompt to check if it is the same as the current one
        rerun()

        if creds and "drive_service" in S:
            # Upload image
            img_mime = guess_type(S.gen_path)[0] or "image/png"
            upload_file(S.drive_service, Path(S.gen_path), img_mime, S.gen_drive_folder_id)

            # Upload CSV log file with official MIME type for CSV files.
            csv_path = config.LOG_DIR / f"{S.uid}.csv"

            upload_file(S.drive_service, csv_path, "text/csv", S.participant_drive_folder_id)

        # Upload info if not uploaded yet
        if not getattr(S, "info_uploaded", False):
            info_path = config.LOG_DIR / f"participant_{S.uid}_info.csv"
            assert info_path.exists(), f"Participant info file does not exist: {info_path}"
            if info_path.exists():
                upload_file(S.drive_service, info_path, "text/csv", S.participant_drive_folder_id)
            S.info_uploaded = True
        
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
            try:
                st.image(
                    ImageOps.contain(Image.open(S.gen_path), (int(config.IMG_H * 1.8), config.IMG_H)),
                    caption="Generated image",
                    clamp=True,
                )
            except (FileNotFoundError, AttributeError, ValueError) as e:
                st.error(f"Generated image cannot be displayed: Try again later")
                S.generated = False
                S.gen_path = None
                st.stop()


        st.caption("Similarity:")
        st.progress(int(S.last_score))
        st.write(f"**{S.last_score:.1f}%**")

        S.subjective_score = st.radio(
            "Subjective Similarity (1 = not similar, 6 = very similar)",
            [1, 2, 3, 4, 5, 6 ],
            horizontal=True,
            key=f"subjective_score_{S.session}_{S.attempt}",
        )

        a_col, t_col = st.columns(2)

        if a_col.button("DONE : Next image"):
            next_gt()
# "Try again" button is disabled on 5th attempt
        if t_col.button("Another try", disabled=S.attempt >= config.MAX_ATTEMPTS):
            S.seed = np.random.randint(1, 4000000) 
            S.generated = False
            S.attempt += 1
            rerun()
    else:
        gen_c.empty() # keeps column widths
