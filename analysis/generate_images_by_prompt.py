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
condition = "delay no feedback"
BASE_DIR = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/participants_data/pilot-2_07052026_delayed-memory_digit-span_no-feedback")
DATA_PATH = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/07052026_pilot_2_delayed_memory_digit_span_no_feedback")
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
    # 1. Choose the smart source path: pick OUT_PATH if recovery is needed, otherwise original DATA
    src_csv = OUT_PATH if OUT_PATH.exists() else DATA
    
    if not src_csv.exists():
        print(f"❌ Input CSV not found: {src_csv}")
        return

    print(f"📂 Reading data from: {src_csv.name}")
    df = pd.read_csv(src_csv).copy()
    
    # Optional: Remove or adjust this line if you want to process the whole dataset!
    df = df.head(5) 

    # Initialize column tracking structure if missing
    if "gen" not in df.columns:
        df["gen"] = pd.NA
    if "revised_prompt" not in df.columns:
        df["revised_prompt"] = pd.NA


    print(f"🚀 Processing images for: {BASE_DIR.name}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Build the explicit filename using your function
        filename = build_gpt_image_filename(row['uid'], int(row['session']), int(row['attempt']))
        # 2. Assign it to the dataframe copy and local dictionary
        df.at[idx, 'gen'] = filename
        row['gen'] = filename
      
        try:
            # 3. Reconstruct the full path
            out_path = build_jatos_path(row, BASE_DIR)
            
            # CRASH RECOVERY: If the image is on disk AND we already saved a prompt metadata, skip safely
            if out_path.exists():
                print(f"⏭️ Skipping entry {idx} (Image and metadata already exist on disk)")
                continue

            # 4. Generate and Save (only runs if the check above falls through)
            prompt = str(row["prompt"])
            revised = generate_gpt_image_from_prompt(prompt, out_path)
            df.at[idx, "revised_prompt"] = revised
            
            # Periodic file save to trace execution progress accurately
            if idx % 5 == 0:
                df.to_csv(OUT_PATH, index=False)

            # Periodic save to backup text data after every successful image generation
            df.to_csv(OUT_PATH, index=False)

        except Exception as e:
            print(f"❌ Error at index {idx}: {e}")
            continue

    # Final save for all 'gen' filenames and 'revised_prompts'
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Finished! CSV updated and images saved to the correct 'files' subfolders.")

if __name__ == "__main__":
    main()