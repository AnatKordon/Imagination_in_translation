# python -m spacy download en_core_web_sm
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # now we can reuse your paths

import spacy
#load spacy pipeline
nlp = spacy.load("en_core_web_sm")

# load participant data
from config import PROCESSED_DIR
df = pd.read_csv(PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv").copy()
OUT_PATH = PROCESSED_DIR / "nlp_analysis"

# Create a stable document id (I'm not sure it's nessecary)
# This keeps merges easy later.
df = df.reset_index(drop=True)
df["doc_id"] = df.index

import pandas as pd
import spacy





# ---------- spaCy pipeline ----------
nlp = spacy.load("en_core_web_sm")

# Optional speed: you do NOT need NER for POS/dep parsing.
# (POS + dependency parsing requires tagger+parser; keep those.)
if "ner" in nlp.pipe_names:
    nlp.disable_pipes("ner")

# ---------- Process with nlp.pipe ----------
texts = df["prompt"].fillna("").tolist()

token_rows = []
np_rows = [] # counting noun chukn heads
# nlp.pipe is much faster than looping nlp(text) one by one
for doc_id, doc in zip(df["doc_id"], nlp.pipe(texts, batch_size=16)):
    # metadata to carry into token table
    meta = df.loc[doc_id, ["uid", "gt", "session", "attempt"]].to_dict()

    noun_chunks = list(doc.noun_chunks)
    np_heads = [nc.root.lemma_.lower() for nc in noun_chunks]
    unique_np_heads = sorted(set(np_heads))

    np_rows.append({
        "doc_id": doc_id,
        "n_noun_chunks": len(noun_chunks),
        "n_unique_np_heads": len(unique_np_heads),
        "np_heads": ", ".join(unique_np_heads),  # optional
    })

    for tok in doc:
        token_rows.append({
            "doc_id": doc_id,
            **meta,

            "token_i": tok.i,

            "text": tok.text,
            "lower": tok.text.lower(),
            "lemma": tok.lemma_,

            "pos": tok.pos_,      # coarse: NOUN/VERB/ADJ...
            "tag": tok.tag_,      # fine: NN/NNS/VBD/JJ...

            "dep": tok.dep_,      # role label: nsubj/obj/amod...
            "head_i": tok.head.i,
            "head_text": tok.head.text,

            "start_char": tok.idx,
            "end_char": tok.idx + len(tok),

            "is_alpha": tok.is_alpha,
            "is_stop": tok.is_stop,
            "is_punct": tok.is_punct,
            "like_num": tok.like_num,
        })

tokens_df = pd.DataFrame(token_rows)

# ---------- Doc-level POS counts ----------
# Most analyses exclude punctuation; you can decide.
content_tokens = tokens_df[~tokens_df["is_punct"]].copy()

pos_counts = (
    content_tokens
    .groupby(["doc_id", "pos"])
    .size()
    .unstack(fill_value=0)
)

# Make columns explicit: pos_NOUN, pos_VERB...
pos_counts.columns = [f"pos_{c}" for c in pos_counts.columns]
print(f"POS counts columns: {pos_counts.columns}")

docs_df = df.merge(pos_counts, left_on="doc_id", right_index=True, how="left")

# Add total token count (excluding punctuation)
docs_df["n_tokens"] = content_tokens.groupby("doc_id").size().reindex(docs_df["doc_id"]).values

# Fill missing counts with 0
count_cols = [c for c in docs_df.columns if c.startswith("pos_")]
docs_df[count_cols] = docs_df[count_cols].fillna(0).astype(int)

# Optional: proportions (nice for comparisons)
for c in count_cols:
    docs_df[c + "_prop"] = docs_df[c] / docs_df["n_tokens"].replace(0, pd.NA)

np_df = pd.DataFrame(np_rows)
docs_df = docs_df.merge(np_df, on="doc_id", how="left")

#fill na if missing:
docs_df["n_noun_chunks"] = docs_df["n_noun_chunks"].fillna(0).astype(int)
docs_df["n_unique_np_heads"] = docs_df["n_unique_np_heads"].fillna(0).astype(int)
docs_df["np_heads"] = docs_df["np_heads"].fillna("")
# ---------- Save ----------
# tokens_df can be big-ish; parquet is ideal (fast, preserves types).
# tokens_df.to_parquet("tokens_df.parquet", index=False)
# docs_df.to_parquet("docs_df.parquet", index=False)

# If you prefer CSV:
tokens_df.to_csv(OUT_PATH / "tokens_df.csv", index=False)
docs_df.to_csv(OUT_PATH / "docs_df.csv", index=False)