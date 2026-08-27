"""Contact sheets of Wilma's 2019 drawings: every drawing we use, per GT image and condition.

WHY THIS EXISTS
Every number in `analysis/notebooks/describing_vs_drawings_comparison.ipynb` reduces a
drawing to a count. This script puts the drawings themselves back on screen, so a bar can
be checked against the pages that produced it: 143 drawings across 5 ground-truth
photographs x 3 conditions, few enough to show ALL of them rather than a sample.

One sheet per ground-truth image. The conditions run down the page in memory order
(perception -> immediate -> delayed) as titled blocks, each wrapping its drawings over as
many rows of `--cols` as it needs - immediate has 15 drawings against perception's 6-7, and
giving every condition one long row would shrink every thumbnail to the widest condition's
column count. The GT photograph sits in the header, so the sheet is self-contained. Each
thumbnail is captioned
`uid - N` (participant id, validated object count), so an outlier dot in the notebook can
be traced to the page it came from.

Two flags on every thumbnail, both of them audit aids rather than results:
  - an amber frame = the page carries written text (`has_written_text`). Labelling was
    OPTIONAL in the 2019 study, so its presence varies between participants and
    conditions - see the header of nlp_analysis/drawings_descriptions.py.
  - a red frame = our describer flagged low `certainty` on that sheet. Those are the
    drawings worth reading by hand before trusting their counts.

SOURCE
`<processed>/wilmas_drawings_2019/combined/full_data/
 drawing_semantic_tag_image_validation_full_data_{arm}.csv`, written by
export_full_data.py after drawings_descriptions.py -> semantic_tagging.py ->
object_accuracy_detector.py. The `path` column points at the original JPEG, so the
thumbnails are the published drawings, not a re-render. `--arm` only changes the captions
and the frames (the two arms describe the SAME 143 pages, hinted or blind); the images and
their layout are identical.

USAGE
    python analysis/drawings_gallery.py                  # all 5 sheets, with_hint
    python analysis/drawings_gallery.py --arm no_hint
    python analysis/drawings_gallery.py --gt lighthouse_h.jpg --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from matplotlib.lines import Line2D

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config  # noqa: E402

ARMS = ("with_hint") #  THIS SHOULDN'T BE USED: "no_hint"

# The trials file speaks in `delay` slugs; these are the conditions we share with the 2019
# study, in memory order. The 2019 `category` arm has no counterpart here and is not in
# this CSV at all.
DELAY_ORDER = ["perc", "imm", "del"]
DELAY_LABELS = {"perc": "Perception", "imm": "Immediate memory", "del": "Delayed memory"}

# Frame colours. Both mark something to check, never something to conclude.
TEXT_COLOR, LOW_CERTAINTY_COLOR = "#e6a23c", "#d94f4f"
FRAME_LW = 2.2

THUMB_W, THUMB_H = 1.55, 1.72      # inches per thumbnail cell, caption included
HEADER_H = 2.1                     # inches reserved for the GT photo + title
DEFAULT_COLS = 8                   # drawings per row before a condition block wraps
BLOCK_TITLE_SIZE = 19              # condition headline - the sheet's main navigation cue


def data_path(arm: str) -> Path:
    return (config.ROOT / "Data" / "processed_data" / "wilmas_drawings_2019" / "combined"
            / "full_data" / f"drawing_semantic_tag_image_validation_full_data_{arm}.csv")


def load_drawings(arm: str = "with_hint") -> pd.DataFrame:
    """The 143 scored drawings, one row per page, with the JPEG path kept."""
    p = data_path(arm)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run analysis/nlp_analysis/drawings_descriptions.py for the "
            f"{arm!r} arm, then object_accuracy_detector.py and export_full_data.py")
    d = pd.read_csv(p)
    d["n_objects"] = pd.to_numeric(d.get("n_in_image_objects"), errors="coerce")
    missing = [q for q in d["path"] if not Path(q).exists()]
    if missing:
        print(f"  !! {len(missing)} drawing file(s) listed in the CSV are missing on disk, "
              f"e.g. {missing[0]}")
    return d


def _frame(ax, row) -> str | None:
    """Colour the cell border for the flags this drawing carries; None if it carries none."""
    color = None
    if str(row.get("certainty", "")).lower() == "low":
        color = LOW_CERTAINTY_COLOR
    elif bool(row.get("has_written_text", False)):
        color = TEXT_COLOR
    for side in ax.spines.values():
        side.set_visible(color is not None)
        if color is not None:
            side.set_color(color)
            side.set_linewidth(FRAME_LW)
    return color


def plot_gt_sheet(d: pd.DataFrame, gt: str, arm: str, out_dir: Path, dpi: int = 150,
                  cols: int = DEFAULT_COLS, save: bool = True):
    """One contact sheet: every drawing of `gt`, one titled block per condition.

    Each block wraps over ceil(n / cols) rows, so all three conditions share a thumbnail
    size no matter how unequal their Ns are.
    """
    sub = d[d["gt"] == gt]
    if sub.empty:
        raise ValueError(f"no drawings for {gt!r} - have: {sorted(d['gt'].unique())}")

    blocks = [(k, sub[sub["delay"] == k].sort_values("uid")) for k in DELAY_ORDER]
    blocks = [(k, rows, max(1, -(-len(rows) // cols))) for k, rows in blocks]   # ceil div
    total_rows = sum(nr for _, _, nr in blocks)

    fig_h = total_rows * THUMB_H + HEADER_H
    fig = plt.figure(figsize=(cols * THUMB_W + 1.0, fig_h))
    top = 1 - HEADER_H / fig_h
    gs = fig.add_gridspec(total_rows, cols, left=0.02, right=0.99, top=top, bottom=0.015,
                          wspace=0.06, hspace=0.42)

    r0 = 0
    for delay, rows, nr in blocks:
        for i in range(nr * cols):
            ax = fig.add_subplot(gs[r0 + i // cols, i % cols])
            ax.set_xticks([]); ax.set_yticks([])
            if i >= len(rows):
                ax.set_frame_on(False)          # pad the last row with blank cells
                continue
            row = rows.iloc[i]
            ax.imshow(imread(row["path"]))
            _frame(ax, row)
            n = row["n_objects"]
            ax.set_xlabel(f"{row['uid']} - {'' if pd.isna(n) else int(n)}", fontsize=6.5,
                          labelpad=1.5, color="#555555")
            if i == 0:                          # block title, over its first thumbnail
                ax.set_title(f"{DELAY_LABELS[delay]}   (n = {len(rows)})", loc="left",
                             fontsize=BLOCK_TITLE_SIZE, fontweight="bold", pad=10,
                             color="#222222")
        r0 += nr

    # Header: the ground-truth photograph, then the title block beside it.
    gt_path = config.GT_DIR / gt
    h = HEADER_H / fig_h
    # The header sits in the TOP of its band; the lower fifth is left clear for the first
    # block's title, which matplotlib draws above the grid's first row.
    gax = fig.add_axes((0.02, top + 0.30 * h, 0.115, h * 0.66))
    if gt_path.exists():
        gax.imshow(imread(gt_path))
    gax.set_xticks([]); gax.set_yticks([])
    for side in gax.spines.values():
        side.set_color("#333333")

    scene = str(sub["scene_label"].dropna().iloc[0]) if sub["scene_label"].notna().any() else gt
    fig.text(0.15, top + h * 0.88, f"{scene} - {gt}", fontsize=16, va="center")
    fig.text(0.15, top + h * 0.66,
             f"all {len(sub)} drawings (Bainbridge, Hall & Baker 2019), arm = {arm}",
             fontsize=10, va="center", color="#555555")
    fig.text(0.15, top + h * 0.48,
             "caption = participant id - validated object count",
             fontsize=8.5, va="center", color="#777777")
    fig.legend(handles=[Line2D([], [], color=TEXT_COLOR, lw=FRAME_LW, label="page carries written text"),
                        Line2D([], [], color=LOW_CERTAINTY_COLOR, lw=FRAME_LW,
                               label="describer flagged low certainty")],
               loc="upper right", bbox_to_anchor=(0.99, top + h * 0.94), frameon=False,
               fontsize=8.5)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"drawings_{Path(gt).stem}_{arm}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("saved:", out)
    return fig


def default_out_dir() -> Path:
    """`analysis/outputs/<DATASET>/drawings/` - a sibling of the aigen / nogen / plain trees.

    The sheets are the 2019 drawing corpus itself, not an output of one of our feedback arms,
    so they sit at the top of the dataset rather than under `combined/graphs/`.
    """
    return config.ROOT / "analysis" / "outputs" / config.DATASET / "drawings"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, default="with_hint")
    ap.add_argument("--gt", default=None, help="one GT filename; default = all five")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS,
                    help="drawings per row before a condition block wraps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = load_drawings(args.arm)
    out_dir = args.out or default_out_dir()
    gts = [args.gt] if args.gt else sorted(d["gt"].unique())
    print(f"{len(d)} drawings, arm = {args.arm}, {len(gts)} sheet(s) -> {out_dir}")
    for gt in gts:
        fig = plot_gt_sheet(d, gt, args.arm, out_dir, dpi=args.dpi, cols=args.cols)
        plt.close(fig)


if __name__ == "__main__":
    main()
