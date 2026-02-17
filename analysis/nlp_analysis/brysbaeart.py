# reproducing Ori's code from R to estimate concreteness ratings of each text.
""" we will have multiple scores: 
1. regular - without words that don't appear
2. with all words - assigning averages to missing words (these could also be words with typos)
3. average per sentence - we will have an average score per sentence (marked by dots, exclamation marks, question marks - after we count how many sentences are there, or maybe it better per x amount of words?)
4. average per x words - e.g., every 10 words we will have an average
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


import re
import pandas as pd
import spacy
import contractions
from num2words import num2words

# ---------- Load NLP (lemmatization + sentence/tokenization) ----------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # fast; we only need tokenizer+lemmatizer


# if you already have nlp, just ensure sentencizer exists
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# ---------- 1) Load Brysbaert dictionary ----------
# Expect columns like: Word, Conc.M (adjust if your file differs)
brys_dic = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/analysis/nlp_analysis/brysbaert/brysbaert_dic.csv")
brys_bigram = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/analysis/nlp_analysis/brysbaert/bigrams_brysbaert.csv")
brys = pd.concat([brys_dic, brys_bigram], ignore_index=True)

# Lowercase keys for matching; keep multiword entries too (e.g., "turning point") if present
brys_dict = dict(zip(brys["Word"].str.lower(), brys["Conc.M"]))

# ---------- 2) Text cleaning per your description ----------
_punct_re = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep apostrophes during expansion phase
_ws_re = re.compile(r"\s+")

def digits_to_words(text: str) -> str:
    # Replace standalone integers with words (e.g., "12" -> "twelve")
    def repl(m):
        try:
            return num2words(int(m.group(0)))
        except Exception:
            return m.group(0)
    return re.sub(r"\b\d+\b", repl, text)

def cleantext(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    # 1) normalize whitespace early
    text = _ws_re.sub(" ", str(text)).strip()

    # 2) expand contractions: "I've" -> "I have"
    text = contractions.fix(text)

    # 3) convert digits to words
    text = digits_to_words(text)

    # 4) remove punctuation (after contraction expansion)
    text = _punct_re.sub(" ", text)

    # 5) collapse whitespace again + lowercase
    text = _ws_re.sub(" ", text).strip().lower()
    return text

def wordcount(text: str) -> int:
    # Count tokens after cleaning (simple split)
    ct = cleantext(text)
    return 0 if not ct else len(ct.split())

# ---------- 3) Concreteness scoring with bigrams + lemmatization fallback ----------
def conc_score(text: str, brys_dict=None, samp_ave=None):
    """
    Returns a dict with:
      - mean_score: mean of scores used (None if no tokens contributed)
      - sum_score: sum of scores used
      - n_tokens: number of tokens in cleaned text
      - n_scored: number of tokens/bigrams that contributed a score
      - n_missing: number of tokens that were missing after lemma lookup
      - coverage: n_scored / n_tokens (rough; bigrams count as 1 unit)
    Missing handling:
      - If samp_ave is None: missing tokens are skipped (do not contribute)
      - If samp_ave is given: missing tokens contribute samp_ave
    """
    if brys_dict is None:
        brys_dict = globals()["brys_dict"]

    ct = cleantext(text)
    if not ct:
        return {
            "mean_score": None,
            "sum_score": 0.0,
            "n_tokens": 0,
            "n_scored": 0,
            "n_missing": 0,
            "coverage": 0.0,
        }

    tokens = ct.split()
    n_tokens = len(tokens)

    scores = []
    n_scored = 0
    n_missing = 0
    i = 0

    while i < len(tokens):
        # Try bigram first
        if i + 1 < len(tokens):
            bg = f"{tokens[i]} {tokens[i+1]}"
            if bg in brys_dict:
                scores.append(float(brys_dict[bg]))
                n_scored += 1
                i += 2
                continue

        # Unigram
        w = tokens[i]
        if w in brys_dict:
            scores.append(float(brys_dict[w]))
            n_scored += 1
            i += 1
            continue

        # Lemma fallback
        doc = nlp(w)
        lemma = doc[0].lemma_.lower() if len(doc) else w
        if lemma in brys_dict:
            scores.append(float(brys_dict[lemma]))
            n_scored += 1
        else:
            n_missing += 1
            if samp_ave is not None:
                scores.append(float(samp_ave))
                n_scored += 1
            # else: skip token entirely

        i += 1

    if n_scored == 0:
        return {
            "mean_score": None,
            "sum_score": 0.0,
            "n_tokens": n_tokens,
            "n_scored": 0,
            "n_missing": n_missing,
            "coverage": 0.0,
        }

    sum_score = float(sum(scores))
    mean_score = sum_score / len(scores)
    coverage = n_scored / max(1, n_tokens)

    return {
        "mean_score": mean_score,
        "sum_score": sum_score,
        "n_tokens": n_tokens,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "coverage": coverage,
    }

def sentence_macro_mean(text: str, samp_ave=None):
    doc = nlp(text if isinstance(text, str) else "")
    sent_means = []
    for sent in doc.sents:
        s = conc_score(sent.text, samp_ave=samp_ave)
        if s["mean_score"] is not None:
            sent_means.append(s["mean_score"])
    return (sum(sent_means) / len(sent_means)) if sent_means else None


def window_k_macro_mean(text: str, k=10, samp_ave=None):
    ct = cleantext(text)
    if not ct:
        return None
    toks = ct.split()
    if len(toks) == 0:
        return None

    # non-overlapping windows of size k (last window may be shorter)
    means = []
    for start in range(0, len(toks), k):
        chunk = " ".join(toks[start:start+k])
        s = conc_score(chunk, samp_ave=samp_ave)
        if s["mean_score"] is not None:
            means.append(s["mean_score"])

    return (sum(means) / len(means)) if means else None


# ---------- 4) R-pipeline equivalent in pandas ----------
def compute_concreteness(df, text_col="prompt"):
    out = df.copy()

    # stripWhitespace
    out[text_col] = out[text_col].astype(str).str.replace(_ws_re, " ", regex=True).str.strip()

    # clean + count
    out["cleantext"] = out[text_col].apply(cleantext)
    out["wordcount"] = out[text_col].apply(wordcount)

    # ---- First run (ignore missing) ----
    first_stats = out[text_col].apply(lambda t: conc_score(t, samp_ave=None))
    out["brys_score_first"] = first_stats.apply(lambda d: d["mean_score"])
    out["brys_sum_first"]   = first_stats.apply(lambda d: d["sum_score"])
    out["n_missing"]        = first_stats.apply(lambda d: d["n_missing"])
    out["n_tokens"]         = first_stats.apply(lambda d: d["n_tokens"])
    out["n_scored"]         = first_stats.apply(lambda d: d["n_scored"])
    out["coverage"]         = first_stats.apply(lambda d: d["coverage"])

    # sample average across prompts (mean, na.rm=TRUE)
    brys_ave = pd.to_numeric(out["brys_score_first"], errors="coerce").mean(skipna=True)
    out["brys_sample_average"] = brys_ave

    # ---- Second run (fill missing tokens with sample average) ----
    filled_stats = out[text_col].apply(lambda t: conc_score(t, samp_ave=brys_ave))
    out["brys_score"]     = filled_stats.apply(lambda d: d["mean_score"])
    out["brys_sum_score"] = filled_stats.apply(lambda d: d["sum_score"])

    # ---- Length-robust measures (use filled or first; I suggest filled) ----
    out["sent_macro_mean"] = out[text_col].apply(lambda t: sentence_macro_mean(t, samp_ave=brys_ave))
    out["window10_macro_mean"] = out[text_col].apply(lambda t: window_k_macro_mean(t, k=10, samp_ave=brys_ave))

    return out



# ---------------- Example usage ----------------
# df = pd.read_csv(config.PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv").copy()
df = pd.read_csv("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/ppt_w_gpt_trials.csv").copy()
out = compute_concreteness(df, text_col="prompt")
#OUT_PATH = config.PROCESSED_DIR / "nlp_analysis" / "ppt_w_gpt_brysbaert_concreteness.csv"
OUT_PATH = "/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/10122025_pilot_2/ppt_w_gpt_brysbaert_concreteness.csv"
out.to_csv(OUT_PATH, index=False)
