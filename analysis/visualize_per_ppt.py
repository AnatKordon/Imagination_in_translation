# analysis/visualize_panels.py
import argparse, textwrap
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # uses your project-level config.py

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def load_all_logs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep the latest record per (uid, gt, attempt) in case of multiple slider moves
    # (use highest timestamp 'ts', then prefer lowest img_index if tied)
    if {"uid","gt","attempt","session"}.issubset(df.columns):
        df = (
            df.groupby(["uid","gt","attempt"], as_index=False, sort=False).tail(1)
        )
    return df

#find all images by this ppt
def build_uid_image_index(uid: str) -> Dict[str, Path]:
    """Map generated image filename -> full path (search only this uid's folder)."""
    root = Path(config.PARTICIPANTS_DIR) / uid
    index = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                index[p.name] = p
    return index

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

def wrap_lines(s: str, width: int = 45, max_lines: int = 3) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    wrapped = textwrap.fill(s.strip(), width=width)
    # limit to max_lines
    lines = wrapped.splitlines()
    return "\n".join(lines[:max_lines])

def panel_for_uid(uid: str, df_uid: pd.DataFrame, gt_list: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_index = build_uid_image_index(uid)
    print(f"gen_index: {gen_index}")
    # canvas: rows = number of GTs (fixed order), cols = 4 (GT + attempts 1..3)
    rows = len(gt_list)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.4, rows*3.6))
    if rows == 1:  # matplotlib quirk: ensure 2D
        axes = np.array([axes])

    # consistent margin so xlabels (prompts) show below images
    plt.subplots_adjust(hspace=0.65, wspace=0.06, top=0.96, bottom=0.04)

    for r, gt_name in enumerate(gt_list):
        # left col: GT
        ax_gt = axes[r, 0]
        ax_gt.imshow(np.asarray(read_gt(gt_name, Path(config.GT_DIR))))
        ax_gt.set_axis_off()
        ax_gt.set_title(f"GT: {gt_name}", fontsize=9)

        # attempts 1..3
        for attempt in (1, 2, 3):
            ax = axes[r, attempt]
            row = (
                df_uid[(df_uid["gt"] == gt_name) & (df_uid["attempt"] == attempt)]
                .head(1) # the head is in case there are duplicates
            )
            if not row.empty:
                filename = row["gen"].iloc[0] if "gen" in row else None
                full_path_str = config.PARTICIPANTS_DIR / df_uid["uid"] / "gen_images" / f"session_{df_uid['session']:02d}" / filename    
                # gen_file = row["gen"].iloc[0] if "gen" in row else None
                print(f"gen_file: {full_path_str}")
                prompt = row["prompt"].iloc[0] if "prompt" in row else ""
                # neg    = row["negative_prompt"].iloc[0] if "negative_prompt" in row else ""
                img_p = gen_index.get(full_path_str) if isinstance(full_path_str, str) else None
                print(f"img_p: {img_p}")
                ax.imshow(np.asarray(read_image(img_p)))
                ax.set_axis_off()

                # prompt under image (and neg if present)
                prompt_text = wrap_lines(prompt, width=42, max_lines=3)
                # neg_text    = wrap_lines(f"avoid: {neg}", width=42, max_lines=1) if isinstance(neg, str) and neg.strip() else ""
                label = f"Attempt {attempt}\n{prompt_text}"
                # if neg_text:
                #     label += f"\n{neg_text}"
                ax.set_xlabel(label, fontsize=8)
                ax.xaxis.set_label_position('bottom')
            else:
                # no data for this attempt
                ax.imshow(np.asarray(read_image(None)))
                ax.set_axis_off()
                ax.set_xlabel(f"Attempt {attempt}\n(no data)", fontsize=8)
                ax.xaxis.set_label_position('bottom')

    fig.suptitle(f"Participant {uid}", fontsize=12, y=0.995)
    out_path = out_dir / f"{uid}_panel.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main(csv_path: Path, ge_list, out_dir: Path):
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
        p = panel_for_uid(uid, df_uid, gt_list, out_dir)
        outputs.append(p)
        print(f"✓ wrote {p}")

    print(f"\nDone. Panels saved in: {out_dir.resolve()}")

if __name__ == "__main__":
    gt_list = ['farm_h.jpg', 'fountain_l.jpg', 'garden_h.jpg', 'kitchen_l.jpg',
       'dam_l.jpg', 'conference_room_l.jpg', 'badlands_h.jpg',
       'bedroom_h.jpg']
    csv_path = config.PROCESSED_DIR / "08092025_pilot" / "participants_log_cleaned_pilot_08092025.csv"
    out_dir = config.PANELS_DIR / "pilot_08092025"

    main(Path(csv_path), gt_list, Path(out_dir))
