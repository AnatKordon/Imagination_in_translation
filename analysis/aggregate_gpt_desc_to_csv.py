# turns the 3 csvs with gpt descriptions into one combined csv 
# that follows the same structure as the participants columns and format, in order to later call the generation of the images based on that
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
gpt_dir = config.EXPERIMENT_DIR / "gpt-5_descriptions_as_ppt"  # These are generations in the same imagination in translation app as participants, only with gpt descriptions differing in verbosity

def load_and_transform_gpt_descriptions(gpt_dir: Path):
    """
    Load the 3 GPT description CSVs (low, medium, high verbosity) and transform them
    to match the all_trials.csv format.
    """

    if not gpt_dir.exists():
        raise FileNotFoundError(f"GPT descriptions directory not found: {gpt_dir}")
    
    # Load the three verbosity CSVs
    low_df = pd.read_csv(gpt_dir / "gpt-5_image_descriptions_verbosity-low.csv")
    medium_df = pd.read_csv(gpt_dir / "gpt-5_image_descriptions_verbosity-medium.csv")
    high_df = pd.read_csv(gpt_dir / "gpt-5_image_descriptions_verbosity-high.csv")
    
    # Add verbosity labels
    low_df['verbosity'] = 'low'
    medium_df['verbosity'] = 'medium'
    high_df['verbosity'] = 'high'
    
    # Combine all verbosity levels
    all_desc = pd.concat([low_df, medium_df, high_df], ignore_index=True)
    
    # Create session mapping (1-5) based on unique gt images
    unique_gts = all_desc['image'].unique()
    gt_to_session = {gt: idx + 1 for idx, gt in enumerate(sorted(unique_gts))}
    print(gt_to_session)

    # Map verbosity to attempt number
    verbosity_to_attempt = {'low': 1, 'medium': 2, 'high': 3}
    
    # Transform to match all_trials.csv format
    transformed = pd.DataFrame({
        'uid': 'gpt-5',
        'gt': all_desc['image'],
        'session': all_desc['image'].map(gt_to_session),
        'attempt': all_desc['verbosity'].map(verbosity_to_attempt),
        'prompt': all_desc['description'],
        'gen': all_desc.apply(
            lambda row: f"gpt-5_session{gt_to_session[row['image']]:02d}_attempt{verbosity_to_attempt[row['verbosity']]:02d}.png",
            axis=1
        ),
        'subjective_score': pd.NA,  # No scores for GPT generations
        'prompt_latency_secs': pd.NA,
        'generating_latency_secs': pd.NA,
        'rating_latency_secs': pd.NA,
        'ts': pd.NA,
        'study_result': 'study_result_gpt-5',
        'comp_result': 'comp_result_gpt-5',
        'verbosity': all_desc['verbosity']
    })
    
    # Sort by session and attempt for consistency
    transformed = transformed.sort_values(['session', 'attempt']).reset_index(drop=True)
    
    return transformed

def main():
    try:
        gpt_trials = load_and_transform_gpt_descriptions()
        
        print("GPT trials shape:", gpt_trials.shape)
        print("\nFirst few rows:")
        print(gpt_trials.head(9))
        
        # Save to processed data directory
        out_file = config.PROCESSED_DIR / "gpt_trials.csv"
        gpt_trials.to_csv(out_file, index=False)
        print(f"\nWrote: {out_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total rows: {len(gpt_trials)}")
        print(f"Sessions: {gpt_trials['session'].nunique()}")
        print(f"Attempts per session: {gpt_trials['attempt'].nunique()}")
        print(f"Unique GT images: {gpt_trials['gt'].nunique()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()