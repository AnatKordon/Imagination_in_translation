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
# df = pd.read_csv(PROCESSED_DIR / "ppt_w_gpt_w_similarity_trials.csv").copy()
OUT_PATH = PROCESSED_DIR / "nlp_analysis"

# # Create a stable document id (I'm not sure it's nessecary)
# # This keeps merges easy later.
# df = df.reset_index(drop=True)
# df["doc_id"] = df.index

import pandas as pd
import spacy
from spacy import displacy
# df = df.head(2) # For testing; remove in real runs



# ---------- spaCy pipeline ----------
nlp = spacy.load("en_core_web_sm")

import spacy
from spacy import displacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
if "ner" in nlp.pipe_names:
    nlp.disable_pipes("ner")

# Colors for POS (coarse tags)
POS_COLORS = {
    "NOUN": "#a7c7ff",
    "PROPN": "#a7c7ff",
    "ADJ":  "#b8f2c5",
    "ADV":  "#ffd7a7",
    "VERB": "#f7b2d9",
    "AUX":  "#f0d0b0",
    "ADP":  "#e3c9ff",
    "DET":  "#dddddd",
    "PRON": "#fff1a8",
    "NUM":  "#c7f0ff",
    "CCONJ":"#e6e6e6",
    "SCONJ":"#e6e6e6",
    "PART": "#e6e6e6",
    "PUNCT":"#ffffff",
    "SPACE":"#ffffff",
}

NP_COLORS = {"NP": "#c7f0ff"}

CSS_TWEAKS = """
<style>
  body { font-family: Arial, sans-serif; margin: 24px; }
  .block-title { margin: 14px 0 6px; font-weight: 700; font-size: 16px; }
  .spans { line-height: 2.6 !important; }
  .spans span { margin-right: 0.25rem; }
  .spans .label { font-size: 10px !important; padding: 2px 6px !important; border-radius: 6px !important; }
  hr { margin: 18px 0; }
</style>
"""

def _render_span(doc, spans, colors, title):
    """Helper: render one displaCy span block."""
    data = {
        "text": doc.text,
        "tokens": [t.text for t in doc],
        "spans": spans,
        "title": title,
    }
    # displacy.render returns an HTML fragment; we wrap it in a div for clarity
    html = displacy.render(data, style="span", manual=True, options={"colors": colors})
    return f'<div class="block-title">{title}</div>{html}'

def render_pos_and_np_report(prompt: str, title: str = "POS + Noun chunks"):
    doc = nlp(prompt)

    # --- POS spans: one span per token (excluding SPACE)
    pos_spans = [
        {"start_token": i, "end_token": i + 1, "label": t.pos_}
        for i, t in enumerate(doc)
        if not t.is_space
    ]

    # --- NP spans: one span per noun chunk
    noun_chunks = list(doc.noun_chunks)
    np_spans = [
        {"start_token": nc.start, "end_token": nc.end, "label": "NP"}
        for nc in noun_chunks
    ]

    # --- Counts
    tok_counts = Counter(t.pos_ for t in doc if not t.is_punct and not t.is_space)
    noun_tokens = tok_counts["NOUN"] + tok_counts["PROPN"]
    adj_tokens  = tok_counts["ADJ"]
    adv_tokens  = tok_counts["ADV"]

    np_count = len(noun_chunks)
    unique_np_heads = sorted(set(nc.root.lemma_ for nc in noun_chunks))

    # Optional extra counts you liked
    adp = tok_counts["ADP"]
    verbs = tok_counts["VERB"] + tok_counts["AUX"]

    counts_html = f"""
    <hr/>
    <div style="font-size:14px;">
      <div><b>Token POS counts</b> (excl. punctuation): NOUN(+PROPN)={noun_tokens} &nbsp; ADJ={adj_tokens} &nbsp; ADV={adv_tokens}</div>
      <div><b>Noun chunks</b>: {np_count} &nbsp; | &nbsp; <b>Unique noun chunk heads</b>: {len(unique_np_heads)}</div>
      <div style="margin-top:6px; color:#444;"><b>Chunk heads:</b> {", ".join(unique_np_heads[:40])}{(" ..." if len(unique_np_heads) > 40 else "")}</div>
      <div style="margin-top:6px; color:#444;"><b>Other POS counts:</b> ADP (spatial relations)={adp} &nbsp; VERB(+AUX)={verbs}</div>
    </div>
    """

    # --- Render blocks (stacked)
    html_pos = _render_span(doc, pos_spans, POS_COLORS, "Layer 1: POS for every token")
    html_np  = _render_span(doc, np_spans,  NP_COLORS,  "Layer 2: Noun chunks (NP)")

    full = f"""
    <h2>{title}</h2>
    {CSS_TWEAKS}
    {html_pos}
    {html_np}
    {counts_html}
    """
    return full


from pathlib import Path

prompt = "A living room with white walls. There are circular floor mats. The floor mats are blue and purple."
html = render_pos_and_np_report(prompt, title="Demo: POS + noun chunks")

Path("pos_np_report.html").write_text(html, encoding="utf-8")
print("Saved: pos_np_report.html")