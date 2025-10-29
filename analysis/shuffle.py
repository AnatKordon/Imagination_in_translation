import sys
import pandas as pd
from pathlib import Path

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import config as config

# preparing shuffled data - mixing "gen" and "prompt" columns to check what happens to similarity scores
# I shuffled prompt and gen columns - but i didn't shuffle within participants (on the whole dataframe)

# taking original participant data:
df = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/08092025_pilot/participants_log_cleaned_pilot_08092025.csv").copy()

#shuffle (a single time so may be some that stay in same place but low chances):
rng = np.random.default_rng(42)   # set/remove seed as you like
n = len(df)

# --- Simple, independent shuffles (may leave a few self-matches) ---
perm_gen    = rng.permutation(n)
perm_prompt = rng.permutation(n)
shuffle_df = df.copy()
shuffle_df['gen']    = shuffle_df['gen'].to_numpy()[perm_gen]
shuffle_df['prompt'] = shuffle_df['prompt'].to_numpy()[perm_prompt]

# save
shuffle_df.to_csv(config.PROCESSED_DIR / "participants_log_cleaned_pilot_08092025_shuffled.csv")