# This is a file where we run the prompt column through an llm that corrects the spelling errors, to be used for the nlp analyses

# extracting semantic tags based on the DSG framework using open ai api
""" 
This will include 4 calls:
1. extract tuples from user description 
2. extract questions from tupples 
3. extract dependencies from tupples 
4. only in the final prompt provide the image and not the description, and answer the questions in 2 based on it
"""

from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths
import json
import pandas as pd

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# from config import PROCESSED_DIR

df = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/ppt_w_gpt_trials.csv").copy()
df = df.head(2) # for testing

def correct_text(prompt: str) -> str:
    if pd.isna(prompt):
        return prompt
    
    SYSTEM = """
    You are a helpful assistant that corrects spelling and grammar errors in text descriptions of images.
    Your task is to take the input PROMPT and return a corrected version with proper spelling and grammar.
    Do NOT change the meaning or content of the prompt, only fix errors.
    If uncertain, keep the original text as much as possible.
    Return ONLY the corrected text, no explanations or additional content.
    """
    USER_PROMPT = f"Correct the following prompt:\n{prompt}"

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role":"system","content": SYSTEM},
            {"role": "user", "content": USER_PROMPT},
            ],
        temperature=0.0,  
        top_p=1.0,
        max_output_tokens=5000,
    )
    return resp.output_text.strip()

df["prompt_corrected"] = df["prompt"].apply(correct_text)
df.to_csv("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/ppt_w_gpt_prompt_corrected.csv", index=False)
