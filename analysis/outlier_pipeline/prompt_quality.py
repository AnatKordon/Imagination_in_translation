"""Flag image-description prompts that are too short to be a real description."""
import pandas as pd

MIN_WORDS = 8


def word_count(prompt) -> int:
    return len(str(prompt).split())


def flag_short_attempts(trials_df: pd.DataFrame) -> pd.Series:
    """Bool per row: True if that attempt's prompt has fewer than MIN_WORDS words."""
    return trials_df["prompt"].apply(word_count) < MIN_WORDS
