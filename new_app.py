import streamlit as st
import numpy as np
from pathlib import Path
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

from uuid import uuid4
import config 
st.set_page_config(page_title="Imagination in Translation", layout="wide")

# Function returning a new widget key for the textbox every time we load a new target (e.g. a new ground truth image is presented)
def fresh_key() -> str:
    return f"prompt_{uuid4().hex}"
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

    S.text_key = fresh_key() # new widget key so the existing widget value is not overwritten
    rerun()

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


# A helper for st.rerun() function in Steamlit. It is named differently in different versions of Steamlit, so we just make sure that we have something of this kind.
rerun = st.rerun if hasattr(st, "rerun") else st.experimental_rerun

# Defining a **st.session_state** - which is Streamlit’s dictionary like place for keeping data between reruns during a single session (for a given user). 
for k, v in {
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
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        native = st.radio("Are you a native English speaker?", ["Yes", "No"])

        submit = st.form_submit_button("Submit")
        if submit:
            S.participant_age = age
            S.participant_gender = gender
            S.participant_native = native
            S.participant_info_submitted = True
            
            
            st.rerun()
    st.stop()
if S.gt_path is None:      
    next_gt()