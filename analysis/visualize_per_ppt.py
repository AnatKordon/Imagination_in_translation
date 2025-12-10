# analysis/visualize_panels.py
import argparse, textwrap
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import re
import math

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # uses your project-level config.py

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# if there's a seed in the end of the filename - ignore it for matching purposes
SEED_TAG = re.compile(r"_seed\d+(?=\.\w+$)", re.IGNORECASE)
#if _gpt-image is in the name's ending

# --- caption + axes helpers ---
CAPTION_WIDTH = 40       # characters per line before wrapping
CAPTION_MAX_LINES = 9     # show up to N lines under each image
CAPTION_FONTSIZE = 8
CAPTION_YOFFSET = -0.16   # how far below the axes to draw the caption (negative = below)


#helper - remove's seed from name
def normalize_name(name: str) -> str:
    """
    Normalize filenames:
      - always remove `_seed1234` - This can be changed in the future if needed
      - optionally add `_gpt-image` before the extension, for example in my analysis of gpt-images in filename
    """
    name = name.strip()

    # Remove _seedNNNN if present
    name = SEED_TAG.sub("", name)

    return name

def load_all_logs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep the latest record per (uid, gt, attempt) in case of multiple slider moves
    # (use highest timestamp 'ts', then prefer lowest img_index if tied)
    if {"uid","gt","attempt","session"}.issubset(df.columns):
        df = (
            df.groupby(["uid","gt","attempt"], as_index=False, sort=False).tail(1)
        )
    return df

#find all images by this ppt - creates a dict mapping row and path by the uid
# a dictionary that maps generated image filenames (both the raw name and a “normalized” name with _seedNNN stripped) → full filesystem Path for that single UID’s folder.
#currently not in use
def build_uid_image_index(uid: str) -> Dict[str, Path]:
    root = Path(config.PARTICIPANTS_DIR) / uid
    print("PARTICIPANTS_DIR:", Path(config.PARTICIPANTS_DIR).resolve())
    print("GT_DIR:", Path(config.GT_DIR).resolve())
    index = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                # key_raw  = p.name # I don't need it as the key is always seedless
                key_norm = normalize_name(p.name)
                # index[key_raw] = p
                # don’t overwrite an existing raw key, but ensure the norm key exists
                index.setdefault(key_norm, p) # we get a dict where key is full path (seedless)
    # small debug
    print(f"Indexed {len(index)} files for uid={uid}. Example keys: {list(index.keys())[:3]}")
    return index


#reconstructing path by the direct path how it was saved - from csv to full path
def path_from_row(row) -> Path:
    filename = normalize_name(str(row["gen"])) # revoming seed tag if any

    uid      = str(row["uid"]).strip() # I changed the name of gpt-5 folder form the uid to this name, should look in the future if that causes some bugs...
    session  = int(row["session"]) # same as int(str(filename)[45:47]) - but more robust to use the csv value
    # print(f"uid: {uid}, session: {session}, filename: {filename}") # for debugging
    return (
        Path(config.PARTICIPANTS_DIR)
        / uid
        / "gen_images"
        / f"session_{session:02d}"
        / filename
    )

#extracting path from the just the 'gen' column - useful for shuffled data where uid/session do not match
#extracting the path from just the 'gen' column value according to it's structure of: uid_<uid>_session_<session>_gen_<filename>.png (example: 00aeccd632c742d48a9ffe94da201493_session01_attempt03_img01.png)
FILENAME_RE = re.compile(
    r"^(?P<uid>[^_]+)_session(?P<session>\d{2})_.*\.png$"
)

#not in use currently
def path_from_gen_col(row) -> Path:
    """Reconstruct the full path from the 'gen' column value."""
    # Extract the UID and session from the 'gen' string
    filename = normalize_name(str(row["gen"]).strip()) # revoming seed tag if any
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Invalid 'gen' format: {row['gen']}") # if pattern isn't matched
    uid = m.group("uid") #extract uid
    session = int(m.group("session"))
    
    return (
        Path(config.PARTICIPANTS_DIR)
        / uid
        / "gen_images"
        / f"session_{session:02d}"
        / filename
    )



def hide_axes(ax):
    """Remove ticks and spines but keep the axes alive for drawing text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def draw_caption(ax, text):
    """Place a multiline caption under the axes."""
    ax.text(
        0.5, CAPTION_YOFFSET, text,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=CAPTION_FONTSIZE,
        wrap=True,
    )

#opening image
def read_image(path: Optional[Path], box=(320, 320)) -> Image.Image:
    """Read image (or create a placeholder) and contain-fit into box."""
    if path is None or not path.exists():
        # placeholder
        w, h = box
        im = Image.new("RGB", box, (240, 240, 240))
        d = ImageDraw.Draw(im)
        msg = "missing"
        d.text((10, h // 2 - 7), msg, fill=(80, 80, 80))
        return im
    im = Image.open(path).convert("RGB")
    return ImageOps.contain(im, box)

#opening gt image
def read_gt(gt_name: str, gt_dir: Path, box=(320,320)) -> Image.Image:
    p = gt_dir / gt_name
    return read_image(p, box)

def wrap_lines(s: str, width: int = CAPTION_WIDTH, max_lines: int = CAPTION_MAX_LINES) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    wrapped = textwrap.fill(s.strip(), width=width)
    lines = wrapped.splitlines()
    return "\n".join(lines[:max_lines])

#this finally creates a panel for a single ppt
def panel_for_uid(uid: str, df_uid: pd.DataFrame, gt_list: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a per-UID filename -> path index once
    # uid_index = build_uid_image_index(uid)

    rows = len(gt_list)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.4, rows*4.6))
    if rows == 1:
        axes = np.array([axes])


    plt.subplots_adjust(hspace=1.10, wspace=0.06, top=0.96, bottom=0.04)

    for r, gt_name in enumerate(gt_list):

        # GT cell
        ax_gt = axes[r, 0]
        ax_gt.imshow(np.asarray(read_gt(gt_name, Path(config.GT_DIR))))
        hide_axes(ax_gt)                      # <- no ticks/spines
        ax_gt.set_title(f"GT: {gt_name}", fontsize=9)

        # Attempts 1..3
        for attempt in (1, 2, 3):
            ax = axes[r, attempt]
            row = (
                df_uid[(df_uid["gt"] == gt_name) & (df_uid["attempt"] == attempt)]
                .sort_values("ts")
                .tail(1)
            )

            if not row.empty:
                row = row.iloc[0]
                img_p = path_from_row(row)
                print(img_p)
                # img_p = resolve_gen_path_from_row(row, uid_index)
                ax.imshow(np.asarray(read_image(img_p)))
                hide_axes(ax)

                prompt_text = wrap_lines(row.get("prompt", ""))  # now uses higher width/lines
                draw_caption(ax, prompt_text)
            else:
                ax.imshow(np.asarray(read_image(None)))
                hide_axes(ax)
                draw_caption(ax, "no data")


    fig.suptitle(f"Participant {uid}", fontsize=12, y=0.995)
    out_path = out_dir / f"{uid}_panel.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main_uid(csv_path: Path, ge_list, out_dir: Path):
    df = load_all_logs(csv_path)

    # enforce dtypes we rely on
    for col in ["uid", "gt", "session", "attempt"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["attempt"] = df["attempt"].astype(int)

    
    uids = df["uid"].dropna().unique().tolist()

    print(f"Found {len(uids)} participants. GT order: {gt_list}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for uid in uids:
        df_uid = df[df["uid"] == uid]
        p = panel_for_uid(uid, df_uid, gt_list, out_dir) # generating a panel per uid
        outputs.append(p)
        # print(f"✓ wrote {p}")

    # print(f"\nDone. Panels saved in: {out_dir.resolve()}")

#############################################################################333

# Now constructing a panel based on gt image:
def panel_for_gt(gt_name: str, df: pd.DataFrame, out_dir: Path, uid_order: List[str] | None = None):
    """
    Build a panel for a single GT:
      - Top row: GT image spanning all columns
      - Then one row per participant: [uid | attempt1 | attempt2 | attempt3]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter rows for this GT and keep the latest record per (uid, attempt)
    df_gt = df[df["gt"] == gt_name].copy()
    if df_gt.empty:
        print(f"[panel_for_gt] No rows for GT '{gt_name}' — skipping.")
        return None

    # Ensure types we rely on
    df_gt["attempt"] = df_gt["attempt"].astype(int)

    # Decide participant order (left-to-right rows)
    if uid_order is None:
        uid_order = df_gt["uid"].dropna().unique().tolist()

    # Layout: 1 header row (GT) + N participant rows; 4 columns (uid + 3 attempts)
    n_rows = len(uid_order)
    n_cols = 4

    # Nice-ish sizing: each image ~3.4 wide, each row ~4.6 tall (like your other panel)
    fig_w = n_cols * 3.4
    fig_h = (1 + n_rows) * 4.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=1 + n_rows, ncols=n_cols,
        height_ratios=[1.2] + [1.0] * n_rows,
        width_ratios=[0.75, 1, 1, 1],   # narrower first column for the uid label
        hspace=1.10, wspace=0.06
    )

    # --- Top: GT spanning all columns ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(np.asarray(read_gt(gt_name, Path(config.GT_DIR))))
    hide_axes(ax_top)
    ax_top.set_title(f"GT: {gt_name}", fontsize=10)

    # --- Rows: one per participant ---
    for r, uid in enumerate(uid_order, start=1):
        # left label cell: UID text only
        ax_uid = fig.add_subplot(gs[r, 0])
        hide_axes(ax_uid)
        # compact uid label (hex can be long)
        uid_label = uid if len(uid) <= 20 else f"{uid[:10]}…{uid[-4:]}"
        ax_uid.text(0.5, 0.5, uid_label, ha="center", va="center", fontsize=9)

        # the participant's rows for this GT
        df_u = df_gt[df_gt["uid"] == uid]

        # attempts 1..3
        for attempt in (1, 2, 3):
            ax = fig.add_subplot(gs[r, attempt])
            row = (
                df_u[df_u["attempt"] == attempt]
                .sort_values("ts")
                .tail(1)
            )

            if not row.empty:
                row = row.iloc[0]
                img_p = path_from_row(row)
                ax.imshow(np.asarray(read_image(img_p)))
                hide_axes(ax)

                prompt_text = wrap_lines(row.get("prompt", ""))
                draw_caption(ax, prompt_text)
            else:
                # no row for this (uid, gt, attempt)
                ax.imshow(np.asarray(read_image(None)))
                hide_axes(ax)
                draw_caption(ax, "no data")

    # Title & export
    fig.suptitle(f"GT panel • {gt_name}", fontsize=12, y=0.995)
    out_path = out_dir / f"GT_{Path(gt_name).stem}_by_uid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main_gt_panels(csv_path: Path, gt_list: List[str], out_dir: Path):
    df = load_all_logs(csv_path)
    for col in ["uid", "gt", "session", "attempt"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["attempt"] = df["attempt"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    for gt_name in gt_list:
        p = panel_for_gt(gt_name, df, out_dir) # this is the function that generates a panel per gt
        if p:
            print(f"✓ wrote {p}")
    print(f"\nGT panels saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    gt_list = ['farm_h.jpg', 'fountain_l.jpg', 'garden_h.jpg', 'kitchen_l.jpg',
       'dam_l.jpg', 'conference_room_l.jpg', 'badlands_h.jpg',
       'bedroom_h.jpg']
    csv_path = config.PROCESSED_DIR / "participants_log_with_gpt_pilot_08092025_gpt-image-1_generation.csv" # "participants_log_with_gpt_pilot_08092025.csv"
    uid_out_dir = config.PANELS_DIR / "by_uid"
    gt_out_dir = config.PANELS_DIR  / "by_gt"
   
    main_uid(Path(csv_path), gt_list, Path(uid_out_dir))
    # main_gt_panels(Path(csv_path), gt_list, Path(gt_out_dir))