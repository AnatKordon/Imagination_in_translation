from pathlib import Path
import sys
from annotated_types import doc
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
from spacy import displacy
df = df.head(3) # For testing; remove in real runs



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
    "PRON": "#fff1a8",
    "NUM":  "#c7f0ff",
    "PART": "#e6e6e6",
}
    #unused tags:
    # "PUNCT":"#ffffff",
    # "SPACE":"#ffffff",
    # "DET":  "#dddddd",
    # "CCONJ":"#e6e6e6",
    # "SCONJ":"#e6e6e6",


NP_COLORS = {"NP": "#c7f0ff"}

POS_KEEP = set(POS_COLORS.keys())  # only POS you defined colors for
# If you also want DET, CCONJ etc later, add them to POS_COLORS (or POS_KEEP)
 
CSS_TWEAKS = """
<style>
  body { font-family: Arial, sans-serif; margin: 24px; }
  .spans { line-height: 2.6 !important; }
  .spans span { margin-right: 0.25rem; }
  .spans .label { font-size: 10px !important; padding: 2px 6px !important; border-radius: 6px !important; }
  hr { margin: 18px 0; }

  /* --- Hide the sentence text in the second (NP) layer, but keep spacing --- */
  .np-layer .token-text { visibility: hidden !important; }
</style>
"""

def _mark_token_text(html: str) -> str:
    # displaCy span markup contains the token text inside a <span class="token-text"> in newer versions,
    # but not always. If not present, we add a class to the first inner span that contains the token.
    if 'class="token-text"' in html:
        return html
    # heuristic: wrap token text spans
    return html.replace("<span>", '<span class="token-text">', 1)


def hide_token_text_in_html(html: str) -> str:
    """
    Remove visible token text from a displaCy span render,
    but keep the underline/labels. Works across spaCy versions.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: no-op if bs4 isn't installed
        return html

    soup = BeautifulSoup(html, "html.parser")

    # In displaCy span markup, each token is typically inside a <span> element
    # and the token text is a text node inside it. We replace text nodes with a non-breaking space.
    for span in soup.find_all("span"):
        # If the span contains direct text (not just child tags), blank it out
        if span.string and span.string.strip():
            span.string.replace_with("\u00A0")  # &nbsp;

    return str(soup)


def _render_span(doc, spans, colors, wrapper_class="layer", hide_text=False):
    data = {"text": doc.text, "tokens": [t.text for t in doc], "spans": spans}
    html = displacy.render(data, style="span", manual=True, options={"colors": colors})
    if hide_text:
        html = hide_token_text_in_html(html)
    return f'<div class="{wrapper_class}">{html}</div>'



def render_pos_and_np_report(prompt: str, title: str = "POS + Noun chunks"):
    doc = nlp(prompt)

    # --- POS spans: one span per token (excluding SPACE)
    pos_spans = [
    {"start_token": i, "end_token": i + 1, "label": t.pos_}
    for i, t in enumerate(doc)
    if (not t.is_space) and (t.pos_ in POS_KEEP) and (not t.is_punct)
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
    html_pos = _render_span(doc, pos_spans, POS_COLORS, wrapper_class="pos-layer", hide_text=False)
    html_np = _render_span(doc, np_spans,  NP_COLORS,  wrapper_class="np-layer",  hide_text=True)
   
    full = f"""
    <h2>{title}</h2>
    {CSS_TWEAKS}
    {html_pos}
    {html_np}
    {counts_html}
    """
    return full


from pathlib import Path

html = render_pos_and_np_report(df.loc[0, "prompt"], title="Demo: POS + noun chunks")

Path("pos_np_report.html").write_text(html, encoding="utf-8")
print("Saved: pos_np_report.html")