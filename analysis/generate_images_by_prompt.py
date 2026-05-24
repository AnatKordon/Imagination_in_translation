import os
import base64
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your new condition directory
condition = "perception no feedback"
BASE_DIR = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/participants_data/pilot-2_29032026_perception_no-feedback")
DATA_PATH = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/29032026_pilot_2_perception_no_feedback")
DATA = DATA_PATH / "ppt_trials_w_similarity_trials.csv"
OUT_PATH = DATA_PATH / "ppt_trials_w_similarity_trials_with_gen.csv"

def build_gpt_image_filename(uid: str, session: int, attempt: int) -> str:
    """
    Constructs filename: uv0vccs51773064211294_session01_attempt01.png
    """
    return f"{uid}_session{session:02d}_attempt{attempt:02d}.png"

def build_jatos_path(row, base_path: Path) -> Path:
    """
    Matches hierarchy in image_3b2737.png:
    <BASE>/jatos_results_files_.../study_result_<ID>/comp-result_<ID>/files/<filename>
    """
    study_id = str(row["study_result"]) 
    comp_id  = str(row["comp_result"])
    filename = row["gen"]
    print(f'filename: {filename}, study_id: {study_id}, comp_id: {comp_id}')
    # Dynamically find the JATOS folder (since the timestamp varies)
    try:
        jatos_dir = next(base_path.glob("jatos_results_files_*"))
    except StopIteration:
        raise FileNotFoundError(f"Could not find jatos_results_files folder in {base_path}")

    return (
        jatos_dir / 
        study_id / 
        comp_id / 
        "files" / 
        filename
    )

def generate_gpt_image_from_prompt(prompt: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    with open(out_path, "wb") as f:
        f.write(img_bytes)
    
    return getattr(resp.data[0], "revised_prompt", None)


def main():
    if not DATA.exists():
        print(f"❌ Original input CSV not found: {DATA}")
        return

    print(f"📂 Reading complete fresh data from: {DATA.name}")
    df = pd.read_csv(DATA).copy()
    #  df = df.head(3)

    print(f"🚀 Processing images for entire dataset ({len(df)} total rows)...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Dynamically build and assign the explicit filename for this row
        filename = build_gpt_image_filename(row['uid'], int(row['session']), int(row['attempt']))
        df.at[idx, 'gen'] = filename
        
        try:
            # 2. Reconstruct the target path on your HDD
            out_path = build_jatos_path(df.loc[idx], BASE_DIR)
            
            # 3. PURE DISK SKIP LOGIC: If the image is already there, don't pay the API to make it again
            if out_path.exists():
                continue

            # 4. Generate and Save (Only runs if file doesn't exist)
            prompt = str(row["prompt"])
            revised = generate_gpt_image_from_prompt(prompt, out_path)
            df.at[idx, "revised_prompt"] = revised
            
            # Periodic save to disk every 5 entries to preserve text records/filenames
            if idx % 5 == 0:
                df.to_csv(OUT_PATH, index=False)

        except Exception as e:
            print(f"❌ Error at index {idx}: {e}")
            continue

    # Final save to write the full table out to OUT_PATH
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Finished! Full CSV updated at {OUT_PATH.name} and missing images generated.")

if __name__ == "__main__":
    main()