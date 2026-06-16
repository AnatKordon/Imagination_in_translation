"""
Prompt Evolution Analysis & Visualization
===========================================
Analyze how participants modify descriptions across attempts and visualize changes.

Hardcoded — reads condition from condition_maps.yaml, no CLI args needed.

Output structure:
    ANALYSIS_DIR/
        per_participant/
            {uid}/
                {uid}_{gt_stem}_session{session}.png
        per_gt_image/
            {gt_stem}/
                {gt_stem}_session{session}.png
        summary_metrics.csv   (appended columns if original CSV provided)
"""

import re
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
from PIL import Image, ImageOps, ImageDraw, ImageFont

# ============================================================================
# PROJECT ROOT & CONFIG — hardcoded relative to this file's location
# ============================================================================
# One folder up from current notebook location
# project_root = Path.cwd().parent.parent.resolve()

# # Add subdirectories to path
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(project_root / 'config'))
# print(f"Project root: {project_root}")
# One folder up from current notebook location
project_root = Path.cwd().resolve()

# Add subdirectories to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'config'))
print(f"Project root: {project_root}")

# Reuse the project config instead of re-parsing condition_maps.yaml.
# Only the names this script actually uses are bound here.
import config
CONDITION        = config.CONDITION
CSV_PATH         = config.CSV_PATH
ANALYSIS_DIR     = config.ANALYSIS_DIR
PARTICIPANTS_DIR = config.PARTICIPANTS_DIR
# task/feedback metadata for the active condition (current grid or legacy pilot)
_cm = config.mapping_data["CONDITIONS"].get(CONDITION) or config.LEGACY.get(CONDITION, {})
TASK, FEEDBACK = _cm.get("task"), _cm.get("feedback")

print(f"Condition     : {CONDITION}")
print(f"CSV           : {CSV_PATH}")
print(f"Analysis out  : {ANALYSIS_DIR}")
print(f"Participants  : {PARTICIPANTS_DIR}")

# ============================================================================
# OPTIONAL: spaCy
# ============================================================================
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
except Exception:
    print("Warning: spaCy not found — falling back to simple tokenizer.")
    nlp_spacy = None

# ============================================================================
# CONSTANTS
# ============================================================================
IMG_EXTS       = {".png", ".jpg", ".jpeg"}
SEED_TAG       = re.compile(r"_seed\d+(?=\.\w+$)", re.IGNORECASE)
TEXT_MAX_WIDTH = 55   # chars per line in text panels
COLOR_ADD      = (76,  175,  80)   # green
COLOR_REMOVE   = (244,  67,  54)   # red
COLOR_NEUTRAL  = (50,   50,  50)   # dark grey

try:
    FONT      = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      10)
    FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
except Exception:
    FONT = FONT_BOLD = ImageFont.load_default()

# ============================================================================
# PATH HELPERS
# ============================================================================

def normalize_name(name: str) -> str:
    name = name.strip()
    return SEED_TAG.sub("", name)

def path_from_row(row) -> Path:
    """Reconstruct full path using JATOS folder structure."""
    filename     = normalize_name(str(row["gen"]))
    study_result = str(row["study_result"]).strip()
    comp_result  = str(row["comp_result"]).strip()
    return PARTICIPANTS_DIR / study_result / comp_result / "files" / filename

def read_image(path: Optional[Path], box=(300, 300)) -> Image.Image:
    """Load image or return a grey placeholder."""
    if path is None or not path.exists():
        im = Image.new("RGB", box, (220, 220, 220))
        d  = ImageDraw.Draw(im)
        msg = "missing" if path is None else str(path.name)
        d.text((8, box[1] // 2 - 7), msg[:30], fill=(100, 100, 100), font=FONT)
        return im
    try:
        im = Image.open(path).convert("RGB")
        return ImageOps.contain(im, box)
    except Exception as e:
        im = Image.new("RGB", box, (220, 220, 220))
        ImageDraw.Draw(im).text((8, box[1] // 2 - 7), f"err: {e}"[:30],
                                fill=(180, 60, 60), font=FONT)
        return im

# ============================================================================
# TOKENISATION & DIFF
# ============================================================================

def count_tokens(text: str) -> int:
    if nlp_spacy:
        return len(nlp_spacy(text))
    return len(re.findall(r'\b\w+\b', text.lower()))

def compare_prompts(p1: str, p2: str) -> Dict[str, Any]:
    """Word-level diff between two prompts. Returns changes list + metrics."""
    w1, w2 = p1.split(), p2.split()
    matcher = SequenceMatcher(None, w1, w2)

    changes: List[Tuple[str, str]] = []
    added = removed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for w in w1[i1:i2]:
                changes.append((w, 'equal'))
        elif tag == 'replace':
            for w in w1[i1:i2]: changes.append((w, 'remove')); removed += 1
            for w in w2[j1:j2]: changes.append((w, 'add'));    added   += 1
        elif tag == 'delete':
            for w in w1[i1:i2]: changes.append((w, 'remove')); removed += 1
        elif tag == 'insert':
            for w in w2[j1:j2]: changes.append((w, 'add'));    added   += 1

    n1, n2 = max(len(w1), 1), max(len(w2), 1)
    metrics = {
        'words_1':     len(w1),
        'words_2':     len(w2),
        'tokens_1':    count_tokens(p1),
        'tokens_2':    count_tokens(p2),
        'added':       added,
        'removed':     removed,
        'pct_added':   round(added   / n2 * 100, 1),
        'pct_removed': round(removed / n1 * 100, 1),
        'pct_changed': round((added + removed) / n2 * 100, 1),
        'tokens_changed': added + removed,
    }
    metrics.update(_change_positions(changes))
    return {'changes': changes, 'metrics': metrics}

def _change_positions(changes: List[Tuple[str, str]]) -> Dict:
    total = len(changes)
    if total == 0:
        return {'pos_beginning': 0, 'pos_early_mid': 0, 'pos_late_mid': 0, 'pos_end': 0}
    counts = {'pos_beginning': 0, 'pos_early_mid': 0, 'pos_late_mid': 0, 'pos_end': 0}
    for i, (_, ct) in enumerate(changes):
        if ct == 'equal':
            continue
        pct = i / total * 100
        if pct < 25:   counts['pos_beginning']  += 1
        elif pct < 50: counts['pos_early_mid']  += 1
        elif pct < 75: counts['pos_late_mid']   += 1
        else:          counts['pos_end']         += 1
    return counts

# ============================================================================
# PIL TEXT RENDERING (coloured diff)
# ============================================================================

def _wrap_changes(changes, width=TEXT_MAX_WIDTH):
    """Return list-of-lines, each line = list of (word_with_space, color, bold)."""
    lines, line, length = [], [], 0
    for word, ct in changes:
        color  = COLOR_ADD if ct == 'add' else (COLOR_REMOVE if ct == 'remove' else COLOR_NEUTRAL)
        bold   = ct != 'equal'
        chunk  = word + ' '
        if length + len(chunk) > width and line:
            lines.append(line); line = []; length = 0
        line.append((chunk, color, bold))
        length += len(chunk)
    if line:
        lines.append(line)
    return lines

def render_text_image(changes, width=TEXT_MAX_WIDTH, img_width=420) -> Image.Image:
    """Render colour-coded diff text to a PIL Image."""
    lines      = _wrap_changes(changes, width)
    lh         = 16
    img_height = max(len(lines) * lh + 20, 60)
    img        = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw       = ImageDraw.Draw(img)
    y = 6
    for line in lines:
        x = 6
        for text, color, bold in line:
            font = FONT_BOLD if bold else FONT
            draw.text((x, y), text, fill=color, font=font)
            # estimate char width via getbbox if available, else fallback
            try:
                bb = font.getbbox(text)
                x += bb[2] - bb[0]
            except AttributeError:
                x += len(text) * 6
        y += lh
    return img

# ============================================================================
# METRICS TABLE (matplotlib → PIL)
# ============================================================================

def render_metrics_table(m12: Dict, m23: Dict) -> Image.Image:
    headers = ['Transition', 'Words', 'Tokens', '% Added', '% Removed', '% Changed',
               'Words Δ', 'Pos: B/EM/LM/E']

    def _pos(m):
        return (f"{m['pos_beginning']}/{m['pos_early_mid']}/"
                f"{m['pos_late_mid']}/{m['pos_end']}")

    rows = []
    for label, m in [('1→2', m12['metrics']), ('2→3', m23['metrics'])]:
        rows.append([
            f'Attempt {label}',
            f"{m['words_1']}→{m['words_2']}",
            f"{m['tokens_1']}→{m['tokens_2']}",
            f"{m['pct_added']}%",
            f"{m['pct_removed']}%",
            f"{m['pct_changed']}%",
            str(m['tokens_changed']),
            _pos(m),
        ])

    fig, ax = plt.subplots(figsize=(14, 1.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                   colWidths=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for i in range(len(headers)):
        tbl[(0, i)].set_facecolor('#DDDDDD')
        tbl[(0, i)].set_text_props(weight='bold')
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(arr)

# ============================================================================
# LEGEND BAR
# ============================================================================

def render_legend(width=900) -> Image.Image:
    h   = 22
    img = Image.new('RGB', (width, h), (250, 250, 250))
    d   = ImageDraw.Draw(img)
    items = [("■ Added",   COLOR_ADD), ("■ Removed", COLOR_REMOVE), ("■ Unchanged", COLOR_NEUTRAL)]
    x = 10
    for label, color in items:
        d.text((x, 4), label, fill=color, font=FONT_BOLD)
        x += len(label) * 7 + 20
    return img

# ============================================================================
# SINGLE PANEL: one (uid, gt, session)
# ============================================================================

def build_panel(uid: str, gt: str, session: int,
                df_sub: pd.DataFrame) -> Optional[Image.Image]:
    """Build a composite panel for one (uid, gt, session) triplet."""

    # ---- collect the 3 attempts ----
    attempts = {}
    for a in [1, 2, 3]:
        rows = df_sub[df_sub['attempt'] == a]
        if rows.empty:
            return None
        attempts[a] = rows.iloc[0]

    p1, p2, p3 = [str(attempts[a]['prompt']) for a in [1, 2, 3]]
    c12 = compare_prompts(p1, p2)
    c23 = compare_prompts(p2, p3)

    # ---- text images ----
    # attempt 1: no diff (all neutral)
    ti1 = render_text_image([(w, 'equal') for w in p1.split()])
    ti2 = render_text_image(c12['changes'])
    ti3 = render_text_image(c23['changes'])

    # ---- generated images ----
    gi = []
    for a in [1, 2, 3]:
        p = path_from_row(attempts[a])
        gi.append(read_image(p, box=(300, 300)))

    # ---- metrics table ----
    metrics_img = render_metrics_table(c12, c23)
    legend_img  = render_legend()

    # ---- layout dimensions ----
    col_w    = 420
    img_h    = 300
    pad      = 8
    title_h  = 28
    legend_h = 22
    met_h    = metrics_img.height
    # max text panel height across all 3
    max_text_h = max(ti1.height, ti2.height, ti3.height)

    total_w = col_w * 3 + pad * 4
    total_h = title_h + legend_h + met_h + max_text_h + img_h + pad * 6

    canvas = Image.new('RGB', (total_w, total_h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)

    # title
    draw.rectangle([(0, 0), (total_w, title_h)], fill=(60, 60, 80))
    draw.text((10, 6), f"{uid}  |  GT: {Path(gt).stem}  |  Session {session}",
              fill=(255, 255, 255), font=FONT_BOLD)

    # legend
    canvas.paste(legend_img, (pad, title_h + pad))
    y = title_h + legend_h + pad * 2

    # metrics
    canvas.paste(metrics_img, (pad, y))
    y += met_h + pad

    # text panels + gen images
    col_labels = ['Attempt 1 (baseline)', 'Attempt 2 (changes vs 1)', 'Attempt 3 (changes vs 2)']
    for col, (ti, gi_img, label) in enumerate(zip([ti1, ti2, ti3], gi, col_labels)):
        x = pad + col * (col_w + pad)

        # column header
        draw.rectangle([(x, y), (x + col_w, y + 18)], fill=(200, 200, 220))
        draw.text((x + 4, y + 2), label, fill=(30, 30, 60), font=FONT_BOLD)

        # text panel (fit to col_w)
        ti_resized = ti.crop((0, 0, min(ti.width, col_w), ti.height))
        canvas.paste(ti_resized, (x, y + 20))

        # generated image (below text, anchored to bottom of text zone)
        gi_y = y + 20 + max_text_h + pad
        canvas.paste(gi_img, (x + (col_w - gi_img.width) // 2, gi_y))

    return canvas

# ============================================================================
# ORCHESTRATION
# ============================================================================

def generate_per_participant(df: pd.DataFrame, out_dir: Path):
    ppt_dir = out_dir / 'per_participant'
    ppt_dir.mkdir(parents=True, exist_ok=True)

    for uid, df_uid in df.groupby('uid'):
        if pd.isna(uid):
            continue
        uid_dir = ppt_dir / str(uid)
        uid_dir.mkdir(exist_ok=True)

        for (gt, session), df_sub in df_uid.groupby(['gt', 'session']):
            gt_stem = Path(str(gt)).stem
            out_path = uid_dir / f'{uid}_{gt_stem}_session{int(session)}.png'
            if out_path.exists():
                print(f"  skip (exists) {out_path.name}")
                continue
            panel = build_panel(str(uid), str(gt), int(session), df_sub)
            if panel:
                panel.save(out_path, dpi=(150, 150))
                print(f"  ✓ {out_path.relative_to(out_dir)}")
            else:
                print(f"  ⊘ skipped {uid} | {gt_stem} | s{session} (missing attempt)")

def generate_per_gt(df: pd.DataFrame, out_dir: Path):
    gt_dir = out_dir / 'per_gt_image'
    gt_dir.mkdir(parents=True, exist_ok=True)

    for (gt, session), df_sub in df.groupby(['gt', 'session']):
        gt_stem  = Path(str(gt)).stem
        this_dir = gt_dir / gt_stem
        this_dir.mkdir(exist_ok=True)
        out_path = this_dir / f'{gt_stem}_session{int(session)}.png'

        participants = df_sub['uid'].dropna().unique()
        panels = []
        for uid in participants:
            df_uid = df_sub[df_sub['uid'] == uid]
            panel = build_panel(str(uid), str(gt), int(session), df_uid)
            if panel:
                panels.append(panel)

        if not panels:
            print(f"  ⊘ no panels for {gt_stem} s{session}")
            continue

        # Stack vertically
        total_w = max(p.width  for p in panels)
        total_h = sum(p.height for p in panels) + 4 * len(panels)
        canvas  = Image.new('RGB', (total_w, total_h), (200, 200, 200))
        y = 0
        for p in panels:
            canvas.paste(p, (0, y))
            y += p.height + 4

        canvas.save(out_path, dpi=(150, 150))
        print(f"  ✓ {out_path.relative_to(out_dir)}")

# ============================================================================
# SUMMARY CSV
# ============================================================================

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (uid, gt, session) compute diff metrics for 1→2 and 2→3.
    Returns a long-form DataFrame with one row per transition.
    """
    records = []
    for (uid, gt, session), df_sub in df.groupby(['uid', 'gt', 'session']):
        attempts = {}
        for a in [1, 2, 3]:
            rows = df_sub[df_sub['attempt'] == a]
            if not rows.empty:
                attempts[a] = rows.iloc[0]

        if not all(a in attempts for a in [1, 2, 3]):
            continue

        p1, p2, p3 = [str(attempts[a]['prompt']) for a in [1, 2, 3]]

        for label, c in [('1to2', compare_prompts(p1, p2)),
                         ('2to3', compare_prompts(p2, p3))]:
            m = c['metrics']
            rec = {
                'uid':        uid,
                'gt':         gt,
                'session':    session,
                'transition': label,
                'condition':  CONDITION,
                'task':       TASK,
                'feedback':   FEEDBACK,
            }
            rec.update(m)
            records.append(rec)

    return pd.DataFrame(records)

# ============================================================================
# MAIN
# ============================================================================

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print(f"\nLoading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).copy()

    required = ['uid', 'gt', 'session', 'attempt', 'prompt', 'gen',
                'study_result', 'comp_result']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df['attempt'] = df['attempt'].astype(int)
    df['session'] = df['session'].astype(int)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # --- per-participant panels ---
    print("\n=== Per-participant panels ===")
    generate_per_participant(df, ANALYSIS_DIR)

    # --- per-gt panels ---
    print("\n=== Per-GT image panels ===")
    generate_per_gt(df, ANALYSIS_DIR)

    # --- summary CSV ---
    print("\n=== Summary metrics CSV ===")
    summary = compute_summary_metrics(df)
    summary_path = ANALYSIS_DIR / 'summary_metrics.csv'
    summary.to_csv(summary_path, index=False)
    print(f"  ✓ {summary_path}")
    print(f"  {len(summary)} rows, columns: {list(summary.columns)}")

    print(f"\n✓ Done. All outputs under: {ANALYSIS_DIR}")

if __name__ == '__main__':
    main()


# """
# Prompt Evolution Analysis & Visualization
# ===========================================
# Analyze how participants modify descriptions across attempts and visualize changes.

# Usage:
#     python analysis/nlp_analysis/qa_additions_in_attempts.py <csv_path> <output_folder>
    
# Example:
#         python analysis/nlp_analysis/qa_additions_in_attempts.py \
#             Data/processed_data/comparing_conditions/3_conditions_with_digit_span.csv \
#             analysis/comparing_conditions
# Output:
#     - output_folder/per_participant/: One PNG per participant
#     - output_folder/per_gt_image/: One PNG per ground truth image
# """

# import argparse
# import textwrap
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Any
# import sys
# import re
# from difflib import SequenceMatcher
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageOps, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import yaml
# import pandas as pd
# from pathlib import Path
# import sys

# # One folder up from current notebook location
# project_root = Path.cwd().parent.parent.resolve()

# # Add subdirectories to path
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(project_root / 'config'))
# print(f"Project root: {project_root}")
# import config
# project_folder = Path.cwd().resolve().parent 

# YAML_PATH = project_folder / "condition_maps.yaml"
# with open(YAML_PATH, "r") as f:
#     mapping_data = yaml.safe_load(f)

# CONDITION = mapping_data["CURRENT_CONDITION"]
# info = mapping_data["CONDITIONS"][CONDITION]
# csv_path = info["df"]
# output_dir =  info["analysis_sub"] # is this accurate?
                  
# # Import spaCy for tokenization
# try:
#     import spacy
#     nlp = spacy.load("en_core_web_sm")
# except:
#     print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
#     nlp = None


# # ============================================================================
# # CONSTANTS & CONFIGURATION
# # ============================================================================

# IMG_EXTS = {".png", ".jpg", ".jpeg"}
# SEED_TAG = re.compile(r"_seed\d+(?=\.\w+$)", re.IGNORECASE)

# # Text rendering constants
# TEXT_FONTSIZE = 9
# TEXT_LINE_HEIGHT = 1.4
# TEXT_MAX_WIDTH = 60  # characters per line
# TEXT_MAX_LINES = 40  # max lines to display

# # Colors for changes (RGB tuples 0-1 scale for matplotlib, 0-255 for PIL)
# COLOR_ADD = (76, 175, 80)      # Green
# COLOR_REMOVE = (244, 67, 54)   # Red
# COLOR_MODIFY = (255, 193, 7)   # Yellow
# COLOR_NEUTRAL = (50, 50, 50)   # Dark grey for unchanged text

# # PIL font (will load if available)
# try:
#     TEXT_FONT_PIL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
#     TEXT_FONT_PIL_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
# except:
#     TEXT_FONT_PIL = ImageFont.load_default()
#     TEXT_FONT_PIL_BOLD = ImageFont.load_default()

# # ============================================================================
# # IMAGE UTILITIES (reused from visualize_per_ppt.py)
# # ============================================================================

# def normalize_name(name: str) -> str:
#     """Remove seed tag from filename."""
#     name = name.strip()
#     name = SEED_TAG.sub("", name)
#     return name

# def read_image(path: Optional[Path], box=(320, 320)) -> Image.Image:
#     """Read image or create placeholder, fit to box."""
#     if path is None or not path.exists():
#         w, h = box
#         im = Image.new("RGB", box, (240, 240, 240))
#         d = ImageDraw.Draw(im)
#         d.text((10, h // 2 - 7), "missing", fill=(80, 80, 80))
#         return im
#     im = Image.open(path).convert("RGB")
#     return ImageOps.contain(im, box)

# def path_from_row(row) -> Path:
#     """Reconstruct full path from CSV row."""
#     filename = normalize_name(str(row["gen"]))
#     uid = str(row["uid"]).strip()
#     session = int(row["session"])
#     return (
#         Path(config.PARTICIPANTS_DIR)
#         / uid
#         / "gen_images"
#         / f"session_{session:02d}"
#         / filename
#     )

# def hide_axes(ax):
#     """Remove ticks and spines from axes."""
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for s in ax.spines.values():
#         s.set_visible(False)

# # ============================================================================
# # TEXT DIFF & TOKENIZATION
# # ============================================================================

# def tokenize_text(text: str) -> List[str]:
#     """Tokenize text into words."""
#     if nlp:
#         doc = nlp(text.lower())
#         return [token.text for token in doc]
#     else:
#         # Fallback: simple split on whitespace and punctuation
#         text = text.lower()
#         tokens = re.findall(r'\b\w+\b', text)
#         return tokens

# def count_tokens(text: str) -> int:
#     """Count tokens in text using spaCy or fallback."""
#     if nlp:
#         return len(nlp(text))
#     else:
#         return len(tokenize_text(text))

# def compare_prompts(prompt_1: str, prompt_2: str) -> Dict[str, Any]:
#     """
#     Compare two prompts and identify additions, removals, modifications.
    
#     Returns:
#         Dict with:
#         - 'tokens_1', 'tokens_2': token counts
#         - 'words_1', 'words_2': word counts
#         - 'changes': list of (word, change_type) tuples
#         - 'metrics': dict with counts of additions, removals, modifications
#     """
#     words_1 = prompt_1.split()
#     words_2 = prompt_2.split()
    
#     # Use SequenceMatcher for word-level diff
#     matcher = SequenceMatcher(None, words_1, words_2)
#     opcodes = matcher.get_opcodes()
    
#     changes = []  # list of (word, change_type)
#     added_count = 0
#     removed_count = 0
#     modified_count = 0
    
#     for tag, i1, i2, j1, j2 in opcodes:
#         if tag == 'equal':
#             for word in words_1[i1:i2]:
#                 changes.append((word, 'equal'))
#         elif tag == 'replace':
#             # Words that are replaced
#             for word in words_1[i1:i2]:
#                 changes.append((word, 'remove'))
#                 removed_count += 1
#             for word in words_2[j1:j2]:
#                 changes.append((word, 'add'))
#                 added_count += 1
#         elif tag == 'delete':
#             for word in words_1[i1:i2]:
#                 changes.append((word, 'remove'))
#                 removed_count += 1
#         elif tag == 'insert':
#             for word in words_2[j1:j2]:
#                 changes.append((word, 'add'))
#                 added_count += 1
    
#     # Calculate metrics
#     total_words_2 = len(words_2) if words_2 else 1
#     total_words_1 = len(words_1) if words_1 else 1
    
#     metrics = {
#         'words_1': len(words_1),
#         'words_2': len(words_2),
#         'tokens_1': count_tokens(prompt_1),
#         'tokens_2': count_tokens(prompt_2),
#         'added': added_count,
#         'removed': removed_count,
#         'pct_added': (added_count / total_words_2) * 100 if total_words_2 > 0 else 0,
#         'pct_removed': (removed_count / total_words_1) * 100 if total_words_1 > 0 else 0,
#         'pct_changed': ((added_count + removed_count) / total_words_2) * 100 if total_words_2 > 0 else 0,
#         'tokens_changed': added_count + removed_count,
#     }
    
#     # Position analysis: where do changes occur?
#     position_data = analyze_change_positions(changes)
#     metrics.update(position_data)
    
#     return {
#         'changes': changes,
#         'metrics': metrics,
#     }

# def analyze_change_positions(changes: List[Tuple[str, str]]) -> Dict[str, Any]:
#     """Analyze where in the text changes occur."""
#     total = len(changes)
#     if total == 0:
#         return {'change_positions': 'N/A'}
    
#     q_size = total // 4
#     positions = {
#         'beginning': 0,      # 0-25%
#         'early_mid': 0,      # 25-50%
#         'late_mid': 0,       # 50-75%
#         'end': 0,            # 75-100%
#     }
    
#     for i, (word, change_type) in enumerate(changes):
#         if change_type != 'equal':
#             pct = (i / total) * 100
#             if pct < 25:
#                 positions['beginning'] += 1
#             elif pct < 50:
#                 positions['early_mid'] += 1
#             elif pct < 75:
#                 positions['late_mid'] += 1
#             else:
#                 positions['end'] += 1
    
#     return {'change_positions': positions}

# # ============================================================================
# # VISUALIZATION: METRICS TABLE
# # ============================================================================

# def render_metrics_table(metrics_1_to_2: Dict, metrics_2_to_3: Dict) -> Image.Image:
#     """
#     Create a PIL Image with metrics table for transitions.
    
#     Shows:
#         - Word count, token count
#         - % added, % removed, % changed
#         - Tokens changed
#         - Change positions
#     """
#     # Create figure
#     fig, ax = plt.subplots(figsize=(14, 3.5))
#     ax.axis('tight')
#     ax.axis('off')
    
#     # Build table data
#     headers = ['Transition', 'Words', 'Tokens', '% Added', '% Removed', '% Changed', 'Tokens Δ', 'Position']
    
#     row_1_2 = [
#         'Attempt 1→2',
#         f"{metrics_1_to_2['metrics']['words_1']}→{metrics_1_to_2['metrics']['words_2']}",
#         f"{metrics_1_to_2['metrics']['tokens_1']}→{metrics_1_to_2['metrics']['tokens_2']}",
#         f"{metrics_1_to_2['metrics']['pct_added']:.1f}%",
#         f"{metrics_1_to_2['metrics']['pct_removed']:.1f}%",
#         f"{metrics_1_to_2['metrics']['pct_changed']:.1f}%",
#         f"{metrics_1_to_2['metrics']['tokens_changed']}",
#         _format_position(metrics_1_to_2['metrics']['change_positions']),
#     ]
    
#     row_2_3 = [
#         'Attempt 2→3',
#         f"{metrics_2_to_3['metrics']['words_1']}→{metrics_2_to_3['metrics']['words_2']}",
#         f"{metrics_2_to_3['metrics']['tokens_1']}→{metrics_2_to_3['metrics']['tokens_2']}",
#         f"{metrics_2_to_3['metrics']['pct_added']:.1f}%",
#         f"{metrics_2_to_3['metrics']['pct_removed']:.1f}%",
#         f"{metrics_2_to_3['metrics']['pct_changed']:.1f}%",
#         f"{metrics_2_to_3['metrics']['tokens_changed']}",
#         _format_position(metrics_2_to_3['metrics']['change_positions']),
#     ]
    
#     table = ax.table(
#         cellText=[row_1_2, row_2_3],
#         colLabels=headers,
#         cellLoc='center',
#         loc='center',
#         colWidths=[0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.2],
#     )
    
#     table.auto_set_font_size(False)
#     table.set_fontsize(8)
#     table.scale(1, 2)
    
#     # Style header
#     for i in range(len(headers)):
#         table[(0, i)].set_facecolor('#E8E8E8')
#         table[(0, i)].set_text_props(weight='bold')
    
#     # Convert to PIL image
#     fig.canvas.draw()
#     image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     img = Image.fromarray(image_from_plot)
#     plt.close(fig)
    
#     return img

# def _format_position(positions_dict: Dict) -> str:
#     """Format position dict to readable string."""
#     if isinstance(positions_dict, str):
#         return positions_dict
#     return (f"B:{positions_dict['beginning']} EM:{positions_dict['early_mid']} "
#             f"LM:{positions_dict['late_mid']} E:{positions_dict['end']}")

# # ============================================================================
# # VISUALIZATION: COLORED PROMPT TEXT
# # ============================================================================

# def render_prompt_with_changes(changes: List[Tuple[str, str]], width: int = TEXT_MAX_WIDTH) -> Image.Image:
#     """
#     Render prompt text with color-coded changes.
#     - Green: additions (bold)
#     - Red: removals (bold)
#     - Yellow: modifications (bold)
#     - Black: unchanged
    
#     Returns PIL Image.
#     """
#     # Wrap text into lines
#     lines = _wrap_with_changes(changes, width)
    
#     # Calculate image size
#     line_height = int(TEXT_FONTSIZE * TEXT_LINE_HEIGHT)
#     img_height = len(lines) * line_height + 20
#     img_width = width * 7  # rough estimate (char width ~7px)
    
#     img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
#     draw = ImageDraw.Draw(img)
    
#     y_pos = 10
#     for line_content in lines:  # list of (text, color, bold) tuples
#         x_pos = 10
#         for text, color, is_bold in line_content:
#             font = TEXT_FONT_PIL_BOLD if is_bold else TEXT_FONT_PIL
#             draw.text((x_pos, y_pos), text, fill=color, font=font)
#             x_pos += len(text) * 6  # approximate char width
#         y_pos += line_height
    
#     return img

# def _wrap_with_changes(changes: List[Tuple[str, str]], width: int) -> List[List[Tuple[str, Tuple, bool]]]:
#     """
#     Wrap text preserving change information.
#     Returns list of lines, each line is list of (text, color, is_bold).
#     """
#     lines = []
#     current_line = []
#     current_length = 0
    
#     for word, change_type in changes:
#         # Determine color and bold
#         if change_type == 'equal':
#             color = COLOR_NEUTRAL
#             is_bold = False
#         elif change_type == 'add':
#             color = COLOR_ADD
#             is_bold = True
#         elif change_type == 'remove':
#             color = COLOR_REMOVE
#             is_bold = True
#         else:  # modify
#             color = COLOR_MODIFY
#             is_bold = True
        
#         word_with_space = word + ' '
        
#         # Check if word fits in current line
#         if current_length + len(word_with_space) > width:
#             if current_line:
#                 lines.append(current_line)
#                 current_line = []
#                 current_length = 0
        
#         current_line.append((word_with_space, color, is_bold))
#         current_length += len(word_with_space)
    
#     if current_line:
#         lines.append(current_line)
    
#     return lines

# # ============================================================================
# # PANEL GENERATION: ATTEMPT COMPARISON
# # ============================================================================

# def create_attempt_comparison_panel(
#     uid: str, gt: str, session: int, df_subset: pd.DataFrame,
# ) -> Optional[Image.Image]:
#     """
#     Create a single panel showing 3 attempts with metrics.
#     Layout: [Metrics Table] / [Attempt1 text + img | Attempt2 text + img | Attempt3 text + img]
#     """
#     # Get attempts 1, 2, 3
#     attempts = {}
#     for attempt in [1, 2, 3]:
#         row = df_subset[df_subset['attempt'] == attempt]
#         if not row.empty:
#             attempts[attempt] = row.iloc[0]
#         else:
#             attempts[attempt] = None
    
#     # Skip if any attempt missing
#     if any(attempts[i] is None for i in [1, 2, 3]):
#         return None
    
#     # Extract prompts
#     prompt_1 = str(attempts[1]['prompt'])
#     prompt_2 = str(attempts[2]['prompt'])
#     prompt_3 = str(attempts[3]['prompt'])
    
#     # Compare 1->2 and 2->3
#     comp_1_2 = compare_prompts(prompt_1, prompt_2)
#     comp_2_3 = compare_prompts(prompt_2, prompt_3)
    
#     # Render components
#     metrics_img = render_metrics_table(comp_1_2, comp_2_3)
    
#     # For each attempt, render text with changes
#     text_img_1 = render_prompt_with_changes(
#         [(w, 'equal') for w in prompt_1.split()], TEXT_MAX_WIDTH
#     )
#     text_img_2 = render_prompt_with_changes(comp_1_2['changes'], TEXT_MAX_WIDTH)
#     text_img_3 = render_prompt_with_changes(comp_2_3['changes'], TEXT_MAX_WIDTH)
    
#     # Load images
#     img_1 = read_image(path_from_row(attempts[1]), box=(280, 280))
#     img_2 = read_image(path_from_row(attempts[2]), box=(280, 280))
#     img_3 = read_image(path_from_row(attempts[3]), box=(280, 280))
    
#     # Build composite layout using matplotlib
#     fig = plt.figure(figsize=(15, 12))
    
#     # Top: metrics
#     ax_metrics = plt.subplot(2, 1, 1)
#     ax_metrics.imshow(metrics_img)
#     ax_metrics.axis('off')
    
#     # Bottom: 3 columns for attempts
#     for col, (text_img, gen_img, attempt_num) in enumerate([
#         (text_img_1, img_1, 1),
#         (text_img_2, img_2, 2),
#         (text_img_3, img_3, 3),
#     ]):
#         ax_text = plt.subplot(4, 3, 4 + col)
#         ax_text.imshow(text_img)
#         ax_text.axis('off')
#         ax_text.set_title(f'Attempt {attempt_num} Text', fontsize=10, weight='bold')
        
#         ax_img = plt.subplot(4, 3, 7 + col)
#         ax_img.imshow(np.asarray(gen_img))
#         ax_img.axis('off')
#         ax_img.set_title(f'Attempt {attempt_num} Image', fontsize=10)
    
#     fig.suptitle(f'{uid} | {gt} | Session {session}', fontsize=12, weight='bold')
#     plt.tight_layout()
    
#     # Convert to PIL image
#     fig.canvas.draw()
#     image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     result = Image.fromarray(image_from_plot)
#     plt.close(fig)
    
#     return result

# # ============================================================================
# # MAIN ORCHESTRATION
# # ============================================================================

# def generate_per_participant_panels(csv_path: Path, out_dir: Path):
#     """
#     For each participant: generate one PNG per (gt, session).
#     Output: out_dir/per_participant/{uid}/{uid}_{gt_stem}_{session}.png
#     """
#     df = pd.read_csv(csv_path)
    
#     # Validate columns
#     required_cols = ['uid', 'gt', 'session', 'attempt', 'prompt', 'gen']
#     if not all(col in df.columns for col in required_cols):
#         raise ValueError(f"CSV missing required columns. Needs: {required_cols}")
    
#     df['attempt'] = df['attempt'].astype(int)
    
#     per_ppt_dir = out_dir / 'per_participant'
#     per_ppt_dir.mkdir(parents=True, exist_ok=True)
    
#     # Group by uid
#     for uid in df['uid'].unique():
#         if pd.isna(uid):
#             continue
        
#         df_uid = df[df['uid'] == uid]
#         uid_dir = per_ppt_dir / str(uid)
#         uid_dir.mkdir(exist_ok=True)
        
#         # For each (gt, session) combination
#         for (gt, session), df_combo in df_uid.groupby(['gt', 'session']):
#             gt_stem = Path(gt).stem
            
#             panel_img = create_attempt_comparison_panel(uid, gt, session, df_combo)
#             if panel_img:
#                 out_path = uid_dir / f'{uid}_{gt_stem}_{session}.png'
#                 panel_img.save(out_path, dpi=(150, 150))
#                 print(f"✓ Saved {out_path}")
#             else:
#                 print(f"⊘ Skipped {uid} | {gt} | Session {session} (missing attempts)")

# def generate_per_gt_panels(csv_path: Path, out_dir: Path):
#     """
#     For each GT image: generate one PNG per session showing all participants.
#     Output: out_dir/per_gt_image/{gt_stem}/{gt_stem}_{session}.png
#     """
#     df = pd.read_csv(csv_path)
    
#     # Validate columns
#     required_cols = ['uid', 'gt', 'session', 'attempt', 'prompt', 'gen']
#     if not all(col in df.columns for col in required_cols):
#         raise ValueError(f"CSV missing required columns. Needs: {required_cols}")
    
#     df['attempt'] = df['attempt'].astype(int)
    
#     per_gt_dir = out_dir / 'per_gt_image'
#     per_gt_dir.mkdir(parents=True, exist_ok=True)
    
#     # Group by (gt, session)
#     for (gt, session), df_combo in df.groupby(['gt', 'session']):
#         gt_stem = Path(gt).stem
#         gt_dir = per_gt_dir / gt_stem
#         gt_dir.mkdir(exist_ok=True)
        
#         # Create grid showing all participants for this GT & session
#         participants = df_combo['uid'].unique()
#         n_rows = len(participants)
        
#         fig = plt.figure(figsize=(16, n_rows * 5))
        
#         for row_idx, uid in enumerate(participants):
#             df_uid_gt = df_combo[df_combo['uid'] == uid]
            
#             panel_img = create_attempt_comparison_panel(uid, gt, session, df_uid_gt)
#             if panel_img:
#                 ax = plt.subplot(n_rows, 1, row_idx + 1)
#                 ax.imshow(panel_img)
#                 ax.axis('off')
        
#         fig.suptitle(f'GT: {gt} | Session {session}', fontsize=14, weight='bold', y=0.995)
#         plt.tight_layout()
        
#         out_path = gt_dir / f'{gt_stem}_session{session}.png'
#         fig.savefig(out_path, dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         print(f"✓ Saved {out_path}")

# def main(csv_path: str, output_dir: str):
#     """Main entry point."""
#     csv_path = Path(csv_path)
#     output_dir = Path(output_dir)
    
#     if not csv_path.exists():
#         raise FileNotFoundError(f"CSV not found: {csv_path}")
    
#     print(f"Loading CSV: {csv_path}")
#     print(f"Output directory: {output_dir}")
#     print()
    
#     print("Generating per-participant panels...")
#     generate_per_participant_panels(csv_path, output_dir)
    
#     print("\nGenerating per-GT panels...")
#     generate_per_gt_panels(csv_path, output_dir)
    
#     print(f"\n✓ Done! Panels saved to {output_dir}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Analyze & visualize how prompts change across attempts'
#     )
#     parser.add_argument(
#         'csv_path',
#         help='Path to CSV with columns: uid, gt, session, attempt, prompt, gen'
#     )
#     parser.add_argument(
#         'output_dir',
#         help='Output directory for generated panels'
#     )
    
#     args = parser.parse_args()
#     main(args.csv_path, args.output_dir)
