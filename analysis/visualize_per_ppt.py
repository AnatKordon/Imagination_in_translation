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

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config  # uses your project-level config.py

IMG_EXTS = {".png", ".jpg", ".jpeg"}


SEED_TAG = re.compile(r"_seed\d+(?=\.\w+$)", re.IGNORECASE)

#helper - remove's seed from name
def normalize_name(name: str) -> str:
    """Remove trailing `_seed123456` just before the extension."""
    return SEED_TAG.sub("", name).strip()

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
    root = Path(config.PARTICIPANTS_DIR) / uid
    print("PARTICIPANTS_DIR:", Path(config.PARTICIPANTS_DIR).resolve())
    print("GT_DIR:", Path(config.GT_DIR).resolve())
    index = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                key_raw  = p.name
                key_norm = normalize_name(p.name)
                index[key_raw] = p
                # don’t overwrite an existing raw key, but ensure the norm key exists
                index.setdefault(key_norm, p)
    # small debug
    print(f"Indexed {len(index)} files for uid={uid}. Example keys: {list(index.keys())[:3]}")
    return index

def build_gen_index(participants_root: Path, exts=("*.png","*.jpg","*.jpeg")):
    index = {}
    for pat in exts:
        for p in participants_root.rglob(pat):
            index[p.name] = p
    return index

#reconstructing path by the direct path how it was saved
def resolve_gen_path_from_row(row: pd.Series, uid_index: dict) -> Path | None:
    filename = str(row.get("gen", "")).strip()
    if not filename:
        return None

    uid_val     = str(row["uid"]).strip()
    session_val = int(row["session"])

    # 1) reconstructed path, raw filename
    recon = (
        Path(config.PARTICIPANTS_DIR)
        / uid_val
        / "gen_images"
        / f"session_{session_val:02d}"
        / filename
    )
    if recon.exists():
        return recon

    # 2) reconstructed path, normalized filename
    seedless = normalize_name(filename)
    if seedless != filename:
        recon2 = recon.with_name(seedless)
        if recon2.exists():
            return recon2

    # 3) look up in the per-UID index by raw or seedless keys
    return uid_index.get(filename) or uid_index.get(seedless)


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

    # Build a per-UID filename -> path index once
    uid_index = build_uid_image_index(uid)

    rows = len(gt_list)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.4, rows*3.6))
    if rows == 1:
        axes = np.array([axes])

    plt.subplots_adjust(hspace=0.65, wspace=0.06, top=0.96, bottom=0.04)

    for r, gt_name in enumerate(gt_list):
        # Left: GT
        ax_gt = axes[r, 0]
        ax_gt.imshow(np.asarray(read_gt(gt_name, Path(config.GT_DIR))))
        ax_gt.set_axis_off()
        ax_gt.set_title(f"GT: {gt_name}", fontsize=9)

        # Attempts 1..3
        for attempt in (1, 2, 3):
            ax = axes[r, attempt]
            # pick the latest record for this (uid, gt, attempt)
            row = (
                df_uid[(df_uid["gt"] == gt_name) & (df_uid["attempt"] == attempt)]
                .sort_values("ts")
                .tail(1)
            )

            if not row.empty:
                row = row.iloc[0]  # turn into Series
                img_p = resolve_gen_path_from_row(row, uid_index)

            #     # DEBUG prints to help you trace
            #     print(
            #     f"[DEBUG] uid={uid} gt={gt_name} attempt={attempt} "
            #     f"gen_csv={row.get('gen')} "
            #     f"seedless={normalize_name(str(row.get('gen','')))} "
            #     f"resolved={img_p}"
            # )

                ax.imshow(np.asarray(read_image(img_p)))
                ax.set_axis_off()

                # Prompt label
                prompt_text = wrap_lines(row.get("prompt", ""), width=42, max_lines=3)
                ax.set_xlabel(f"Attempt {attempt}\n{prompt_text}", fontsize=8)
                ax.xaxis.set_label_position('bottom')
            else:
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
