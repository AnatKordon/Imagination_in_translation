"""Shared plotting helpers for the notebooks in this folder.

Everything here was living as copy-pasted cells inside each notebook (the canonical copies
were in ``full_data_comparisons.ipynb`` cells 3, 5 and 7). This module holds the parts that
are *pure* - style constants, count derivation, bar decorations, loaders - so a new notebook
does not have to grow an eighth copy. Selection- and figure-level functions stay in the
notebooks: they close over a particular DataFrame and a particular question.

One deliberate change from the notebook originals: the ``COUNT_MODE`` global became an
explicit ``mode=`` argument. A module that silently reads a caller's global is a trap, and
the mode genuinely varies between figures.

The existing notebooks are untouched and do not import this - they keep their own cells.

Usage::

    import sys; sys.path.insert(0, str(Path(__file__).parent))   # or the notebook's folder
    from plot_helpers import *
    config = import_config()
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

__all__ = [
    # bootstrap
    "find_project_root", "import_config", "set_style",
    # constants
    "TASK_ORDER", "TASK_LABELS", "TASK_COLORS", "ATTEMPTS", "GEN_COLORS", "GEN_LABELS",
    "CATEGORIES", "VALIDATED_CATEGORIES", "DELAY_TO_TASK", "MEMORABILITY",
    "CAP_FACE", "CAP_LW", "CAP_LABEL", "UNVALIDATED_HATCH", "CI_COLOR",
    "BAR_LABEL_SIZE", "BAR_LABEL_COLOR", "BAR_LABEL_FMT",
    "Y_TICK_STEP", "Y_TARGET_TICKS", "GPT_DESC_COLOR", "GPT_DESC_LABEL",
    # counts
    "count_list", "is_validated", "cat_label", "prep_counts", "ppt_means",
    # axis / bar decoration
    "nice_step", "style_count_axis", "geom", "add_caps", "boot_ci", "add_ci",
    "draw_interval", "add_bar_labels", "figure_legend", "save_fig",
    # composed layouts
    "enumerate_tasks", "gens_present", "draw_bars", "draw_lines",
    # loaders / reshaping
    "load_gpt_ceiling", "load_labelme", "explode_items",
]


# ---------------------------------------------------------------------------
# Bootstrap - find config.py no matter where the notebook is opened from
# ---------------------------------------------------------------------------

def find_project_root(start=None):
    """Walk up from `start` (default: cwd) to the directory holding config.py."""
    root = Path(start or Path.cwd()).resolve()
    while not (root / "config.py").exists() and root != root.parent:
        root = root.parent
    if not (root / "config.py").exists():
        raise RuntimeError(f"no config.py above {start or Path.cwd()}")
    return root


def import_config(start=None):
    """Put the project root on sys.path and return the imported `config` module."""
    root = find_project_root(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import config
    return config


# ---------------------------------------------------------------------------
# Style + fixed color maps (per generation, reused in every figure)
# ---------------------------------------------------------------------------

def set_style(font_scale=1.6):
    """The shared look: whitegrid, large type sized for full-width figures."""
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "legend.title_fontsize": 17,
        "figure.titlesize": 26,
    })


TASK_ORDER = ["perception", "immediate", "delay"]
TASK_LABELS = {"perception": "Perception", "immediate": "Immediate memory",
               "delay": "Delayed memory"}
# Task is an ORDERED factor (how much access to the image: full -> brief -> delayed), so it
# gets a sequential navy ramp, dark to light. Generation is unordered, so it keeps the
# categorical Dark2 hues below. Keeping the two on different scales matters: they used to
# share Dark2, which made a green bar mean "perception" in one figure and "aigen" in the next.
TASK_COLORS = {"perception": "#0D2B45", "immediate": "#2E6E9E", "delay": "#7CB3D9"}
ATTEMPTS = [1, 2, 3]

# Dark2: green / orange / purple for the three feedback types, pink for the drawings -
# the same fixed mapping every notebook in this folder uses.
GEN_COLORS = {"aigen": "#1b9e77", "nogen": "#d95f02", "plain": "#7570b3", "drawing": "#e7298a"}
GEN_LABELS = {"aigen": "AI-gen (image)", "nogen": "No-gen (text)", "plain": "Plain (baseline)",
              "drawing": "Drawings (with hint)"}

# Semantic-tag categories: csv column -> nice axis label.
CATEGORIES = {
    "objects": "Objects",
    "stuff": "Stuff",
    "scene_category": "Scene category",
    "spatial_relations": "Spatial relations",
    "attr_color": "Color attributes",
    "adjectives": "Adjectives",
}

# Judged by object_accuracy_detector.py. Anything outside this set gets no cap and is drawn
# hatched, because a `false` there would be a matter of taste, not of the image.
VALIDATED_CATEGORIES = ("objects", "stuff", "scene_category", "spatial_relations", "attr_color")

# `delay` is the task column in the combined full-data tables.
DELAY_TO_TASK = {"perc": "perception", "imm": "immediate", "del": "delay"}

# The `_l` / `_h` suffix on the GT filenames is Bainbridge's memorability split. Only
# bedroom_l is low - which is exactly why it is the test case for the immediate-vs-delayed
# question, and also why that comparison rests on a single image.
MEMORABILITY = {
    "bedroom_l.jpg": "low",
    "living_room_h.jpg": "high",
    "conference_room_h.jpg": "high",
    "lighthouse_h.jpg": "high",
    "playground_h.jpg": "high",
}

# The cap is the SAME hue as its bar, hollow instead of solid - it is a part of that
# generation's total, not an extra series, so it must not introduce a new color.
CAP_FACE = "white"
CAP_LW = 1.6
CAP_LABEL = "Not in image (hallucinated)"
UNVALIDATED_HATCH = "//"
CI_COLOR = "#333333"

# The number printed over every bar: the mean VALIDATED count, i.e. the solid part only -
# never the hallucination cap and never the two summed.
BAR_LABEL_SIZE = 11
BAR_LABEL_COLOR = "#222222"
BAR_LABEL_FMT = "{:.1f}"

Y_TICK_STEP = None
Y_TARGET_TICKS = 8

GPT_DESC_COLOR = "#444444"
GPT_DESC_LABEL = "gpt-5.5_desc"


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

def count_list(cell):
    """Length of a stringified list cell; 0 for empty / NaN / unparseable.

    The raw tag columns are Python reprs (single quotes) while the validated ones are JSON,
    so literal_eval is used - it reads both.
    """
    if pd.isna(cell):
        return 0
    try:
        val = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return 0
    return len(val) if isinstance(val, (list, tuple)) else 0


def is_validated(category, mode="validated"):
    """True when this category carries a hallucination cap in the given mode."""
    return mode == "validated" and category in VALIDATED_CATEGORIES


def cat_label(category, mode="validated"):
    """Axis label, marking the unjudged category so it is never read as validated."""
    base = CATEGORIES[category]
    return base if is_validated(category, mode) or mode == "raw" else f"{base} (unvalidated)"


def prep_counts(d, name, mode="validated"):
    """Drop unjudged rows, add the task column, derive n_<cat> / h_<cat> per category.

    n_<cat> is what gets plotted as the solid bar (validated count in "validated" mode, raw
    tag count in "raw" mode); h_<cat> is the hollow hallucination cap stacked on top.
    """
    # Rows the judge could not score (API failure, missing image, ...) have NaN counts.
    # Dropping is safer than letting them average in as zeros - re-run the validator to fill.
    if "error" in d.columns and d["error"].notna().any():
        n_bad = int(d["error"].notna().sum())
        print(f"  !! {name}: dropping {n_bad}/{len(d)} row(s) with a validation error")
        d = d[d["error"].isna()].copy()
    d = d.copy()
    d["task"] = d["delay"].map(DELAY_TO_TASK)   # overrides any table's own task labels
    assert d["task"].notna().all(), f"{name}: unmapped delay values {set(d['delay'])}"
    d["attempt"] = d["attempt"].astype(int)
    for col in CATEGORIES:
        raw = d[col].apply(count_list)
        if is_validated(col, mode):
            d[f"n_{col}"] = pd.to_numeric(d[f"n_validated_{col}"], errors="coerce")
            d[f"h_{col}"] = pd.to_numeric(d[f"n_invalid_not_in_image_{col}"], errors="coerce")
            # The judge must have seen exactly the tags the tagger produced.
            extracted = pd.to_numeric(d[f"n_extracted_{col}"], errors="coerce")
            n_off = int((extracted != raw).sum())
            if n_off:
                print(f"  !! {name}/{col}: {n_off} row(s) where n_extracted != len(tag list)")
        else:
            d[f"n_{col}"] = raw
            d[f"h_{col}"] = 0
    return d


def ppt_means(data, keys, category):
    """Mean count + hallucination cap per key combination.

    Participant is the unit of analysis everywhere in these figures, so `keys` should include
    `uid`: this is the step that collapses a participant's 5 sessions into one point before
    any group mean or CI is taken.
    """
    cols = {f"n_{category}": "count", f"h_{category}": "cap"}
    return data.groupby(keys, as_index=False)[list(cols)].mean().rename(columns=cols)


# ---------------------------------------------------------------------------
# Axis + bar decoration
# ---------------------------------------------------------------------------

def nice_step(vmax, target=Y_TARGET_TICKS):
    """Round vmax/target up to the nearest 1, 2, 5 x 10^k."""
    if not vmax or vmax <= 0:
        return 1
    raw = vmax / target
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def style_count_axis(ax, vmax=None, step=Y_TICK_STEP):
    """y starts at 0; ticks every `step`, or a nice auto step from the panel range.

    When vmax is given the top is set explicitly. On a sharey figure that matters: set_ylim
    disables autoscale for the whole shared axis, so a taller panel drawn later would
    otherwise be clipped. Callers pass the max across every panel.
    """
    if vmax is not None and vmax > 0:
        ax.set_ylim(0, vmax * 1.05)
    else:
        ax.set_ylim(bottom=0)
    top = vmax if vmax is not None else ax.get_ylim()[1]
    ax.yaxis.set_major_locator(MultipleLocator(step or nice_step(top)))


def geom(series, dodge=True):
    """(offsets, width) of each bar slot, matching seaborn's categorical geometry.

    A total slot width of 0.8, split evenly between the hue levels when they are dodged, or
    used whole when the hue *is* the x variable (one bar per x, no dodge).
    """
    if not dodge:
        return {s: 0.0 for s in series}, 0.8
    n = len(series)
    width = 0.8 / n
    return {s: (i - (n - 1) / 2) * width for i, s in enumerate(series)}, width


def add_caps(ax, x_index, series, bottoms, caps, geom, color_of=None, hatch=False):
    """Stack the hollow 'not in image' segment on top of each bar.

    x_index : {x category -> integer position on the axis}
    bottoms : {(x, series) -> solid bar height}   caps: {(x, series) -> cap height}
    geom    : (offsets, width) from geom(), so caps land exactly on their bars.
    color_of: (x, series) -> edge color; defaults to the series' generation color.
    """
    offsets, width = geom
    color_of = color_of or (lambda xv, s: GEN_COLORS[s])
    for xv, i in x_index.items():
        for s in series:
            h = caps.get((xv, s), 0)
            if not h or np.isnan(h):
                continue
            ax.bar(i + offsets[s], h, bottom=bottoms.get((xv, s), 0),
                   width=width * 0.92, facecolor=CAP_FACE, edgecolor=color_of(xv, s),
                   linewidth=CAP_LW, hatch=UNVALIDATED_HATCH if hatch else None, zorder=2.5)


def boot_ci(values, n_boot=2000, ci=95, seed=0):
    """Percentile bootstrap CI of the mean, matching seaborn's errorbar=('ci', 95).

    Fixed seed so re-running the notebook does not jitter the whiskers.
    """
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    half = (100 - ci) / 2
    return np.percentile(means, half), np.percentile(means, 100 - half)


def add_ci(ax, x_index, series, data, x_col, hue_col, value_col, geom):
    """95% bootstrap CI on the mean of each bar; returns {(x, series) -> (lo, hi)} so the
    value labels can be placed clear of the whisker.

    Drawn on top of everything (zorder 5) so the hollow cap never hides the upper whisker,
    and in neutral ink so it reads as an annotation on the solid bar rather than as data.
    """
    offsets, _ = geom
    out = {}
    for xv, i in x_index.items():
        for s in series:
            vals = data[(data[x_col] == xv) & (data[hue_col] == s)][value_col]
            if len(vals) < 2:
                continue
            lo, hi = boot_ci(vals)
            if np.isnan(lo):
                continue
            m = vals.mean()
            out[(xv, s)] = (lo, hi)
            ax.errorbar(i + offsets[s], m, yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor=CI_COLOR, elinewidth=2.0, capsize=4, capthick=2.0, zorder=5)
    return out


def draw_interval(ax, x, lo, hi, width=0.10):
    """A lo-hi segment with end caps, drawn at x.

    Used instead of ax.errorbar wherever the interval is not symmetric about the point
    estimate - a bootstrap CI need not be, and errorbar's yerr cannot express that (it raises
    on a negative arm).
    """
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return
    ax.plot([x, x], [lo, hi], color=CI_COLOR, lw=2.2, zorder=5, solid_capstyle="butt")
    for y in (lo, hi):
        ax.plot([x - width / 2, x + width / 2], [y, y], color=CI_COLOR, lw=2.2, zorder=5)


def add_bar_labels(ax, x_index, series, means, cis, geom, headroom, fmt=BAR_LABEL_FMT):
    """Print the mean count over each bar, in small type.

    The number is the solid bar only - the hallucination cap is deliberately excluded, so the
    printed value is always "how much of this did the image support". It is placed above
    whatever is tallest at that bar (the stacked cap or the CI whisker) so nothing overlaps.
    """
    offsets, _ = geom
    for xv, i in x_index.items():
        for s in series:
            if (xv, s) not in means.index:
                continue
            m, cap = means.loc[(xv, s), "count"], means.loc[(xv, s), "cap"]
            if np.isnan(m):
                continue
            top = max(m + (0 if np.isnan(cap) else cap), cis.get((xv, s), (0, 0))[1])
            ax.text(i + offsets[s], top + 0.035 * headroom, fmt.format(m),
                    ha="center", va="bottom", fontsize=BAR_LABEL_SIZE, color=BAR_LABEL_COLOR,
                    zorder=6)


def figure_legend(fig, gens, marker="bar", labels=None, ncol=None, y=0.90,
                  caps=False, ci=False, gpt=False):
    """One legend for the whole figure, centered above the panels.

    gens   : the generations actually drawn, in draw order.
    labels : optional {gen -> label} override, for views where a generation is pinned to one
             attempt and the legend has to say so.
    marker : "bar" for a color patch, "o" for the line views.
    caps   : append the hollow 'not in image' patch.
    ci     : append the error-bar entry; may be a string, because the interval describes a
             different quantity in different views.
    gpt    : append the dashed gpt-5.5_desc reference-line entry.
    """
    labels = labels or {}
    text = lambda g: labels.get(g, GEN_LABELS.get(g, g))
    if marker == "bar":
        handles = [Patch(facecolor=GEN_COLORS[g], label=text(g)) for g in gens]
    else:
        handles = [Line2D([], [], color=GEN_COLORS[g], lw=3, marker="o", markersize=9,
                          label=text(g)) for g in gens]
    if caps:
        handles.append(Patch(facecolor=CAP_FACE, edgecolor="#444444", linewidth=CAP_LW,
                             label=CAP_LABEL))
    if ci:
        handles.append(Line2D([], [], color=CI_COLOR, lw=2.0, marker="_", markersize=10,
                              label=ci if isinstance(ci, str) else "95% CI (validated mean)"))
    if gpt:
        handles.append(Line2D([], [], color=GPT_DESC_COLOR, lw=2.4, ls="--",
                              label=GPT_DESC_LABEL))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(handles), frameon=False, columnspacing=2.2,
               handlelength=2.4, handletextpad=0.9)


def save_fig(fig, out):
    """Save with padding, so the y label / legend are never clipped at the edge."""
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.35)
    print("saved:", out)


# ---------------------------------------------------------------------------
# Loaders / reshaping
# ---------------------------------------------------------------------------

def load_gpt_ceiling(config, mode="validated", verbosity="medium"):
    """The GPT-with-the-image baseline: one description per GT image, same tagger, same judge.

    Returns the per-GT frame with n_<cat> / h_<cat> columns added, or None when the file has
    not been produced yet. The runs wrote these under both the newer "gpt-5.5_desc" and the
    older "gpt_ceiling" name; accept either, exactly as object_accuracy_detector.py does.
    """
    candidates = [
        config.COMBINED_PROCESSED_DIR / "gpt-5.5_desc"
        / f"gpt-5.5_desc_semantic_tag_image_validation_verbosity-{verbosity}.csv",
        config.COMBINED_PROCESSED_DIR / "gpt_ceiling"
        / f"gpt_ceiling_semantic_tag_image_validation_verbosity-{verbosity}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("no validated gpt-5.5_desc file - run: python analysis/nlp_analysis/"
              "object_accuracy_detector.py -c gpt-5.5_desc")
        return None
    g = pd.read_csv(path)
    if "error" in g.columns:
        g = g[g["error"].isna()].copy()
    for col in CATEGORIES:
        if is_validated(col, mode):
            g[f"n_{col}"] = pd.to_numeric(g[f"n_validated_{col}"], errors="coerce")
            g[f"h_{col}"] = pd.to_numeric(g[f"n_invalid_not_in_image_{col}"], errors="coerce")
        else:
            g[f"n_{col}"] = g[col].apply(count_list)
            g[f"h_{col}"] = 0
    print(f"loaded gpt-5.5_desc from {path.parent.name}/: {len(g)} descriptions, "
          f"{g['gt'].nunique()} GT images")
    return g


def load_labelme(config):
    """The LabelMe polygon annotations of the GT photos, as one tidy frame.

    Written by analysis/wilmas_labelme_gt_objects.py, one CSV per GT with columns
    gt / object / object_count. This is the objective inventory of what is actually in the
    image, in LabelMe's own vocabulary - which is NOT the tagger's vocabulary, so any
    comparison against participant tags needs an explicit alignment step.
    """
    base = (config.ROOT / "Data" / "other_datasets" / "wilmas_drawings_2019" / "LabelMe")
    frames = [pd.read_csv(p) for p in sorted(base.glob("*/*_object_counts.csv"))]
    if not frames:
        print(f"no LabelMe object_counts.csv under {base}")
        return None
    lm = pd.concat(frames, ignore_index=True)
    lm["object"] = lm["object"].str.strip().str.lower()
    print(f"loaded LabelMe: {len(lm)} annotated object types over {lm['gt'].nunique()} GT images")
    return lm


def explode_items(d, meta=("generation", "task", "gt", "uid", "session", "attempt")):
    """One row per tagged item, from the `item_evaluations` JSON column.

    Each element is {"id", "category", "item", "in_image"}, so this is the only source that
    carries item text, its category and the judge's verdict together - which is why every
    object-level analysis runs off it rather than off the list columns (whose raw/validated
    halves are Python-repr and JSON respectively).

    Returns columns: <meta...>, category, item, in_image.
    """
    rows = []
    for rec in d[list(meta) + ["item_evaluations"]].itertuples(index=False):
        cell = rec[-1]
        if pd.isna(cell):
            continue
        try:
            evals = json.loads(cell)
        except (ValueError, TypeError):
            continue
        head = rec[:-1]
        for e in evals:
            rows.append(head + (e.get("category"), str(e.get("item", "")).strip().lower(),
                                bool(e.get("in_image"))))
    out = pd.DataFrame(rows, columns=list(meta) + ["category", "item", "in_image"])
    print(f"exploded items: {len(out)} rows, {out['item'].nunique()} distinct item strings")
    return out


# ---------------------------------------------------------------------------
# Composed layouts - the two drawing calls every figure in the notebooks is built from
# ---------------------------------------------------------------------------

def enumerate_tasks(axes):
    """Pair a 3-panel axes row with TASK_ORDER - the layout used all through the notebooks."""
    return zip(np.atleast_1d(axes).ravel(), TASK_ORDER)


def gens_present(data, order=("aigen", "nogen", "plain", "drawing"), col="generation"):
    """The hue levels actually present in `data`, in canonical order - so a legend never shows
    an empty swatch and a missing arm reads as an absence rather than a blank bar."""
    have = set(data[col])
    return [g for g in order if g in have]


def draw_bars(ax, data, x_col, x_order, hue_col, hue_order, value_col,
              cap_col=None, palette=None, dots=True, values=True, fmt="{:.2f}"):
    """Grouped bars + participant dots + bootstrap CI + value labels, for ANY measure column.

    `value_col` is whatever is being plotted and `cap_col` is optional, so one call draws
    validated counts with a hallucination cap, a rate, or a similarity score. When hue_col ==
    x_col there is one bar per x and no dodge. Returns the height everything fits under, for
    the caller's y limit (which matters on sharey figures, where set_ylim kills autoscale).
    """
    palette = palette or GEN_COLORS
    dodge = hue_col != x_col
    g = geom(hue_order, dodge=dodge)
    x_index = {v: i for i, v in enumerate(x_order)}

    sns.barplot(data=data, x=x_col, y=value_col, hue=hue_col, order=x_order,
                hue_order=hue_order, palette=palette, errorbar=None, legend=False,
                ax=ax, zorder=2)

    plot = data.rename(columns={value_col: "count"})
    plot["cap"] = data[cap_col] if cap_col else 0.0
    means = plot.groupby([x_col] if not dodge else [x_col, hue_col])[["count", "cap"]].mean()
    if not dodge:
        means.index = pd.MultiIndex.from_arrays([means.index, means.index])

    if cap_col:
        add_caps(ax, x_index, hue_order, bottoms=means["count"].to_dict(),
                 caps=means["cap"].to_dict(), geom=g, color_of=lambda xv, s: palette[s])
    if dots:
        sns.stripplot(data=data, x=x_col, y=value_col, hue=hue_col, order=x_order,
                      hue_order=hue_order, palette=palette, dodge=dodge, jitter=0.16,
                      size=4.0, alpha=0.45, linewidth=0.4, edgecolor="white",
                      legend=False, ax=ax, zorder=3)
    cis = add_ci(ax, x_index, hue_order, data, x_col, hue_col, value_col, g)

    vmax = float(np.nanmax([data[value_col].max(), means.sum(axis=1).max()]))
    if values:
        add_bar_labels(ax, x_index, hue_order, means, cis, g, headroom=vmax, fmt=fmt)
        vmax *= 1.09     # room for the printed number above the tallest bar
    return vmax


def draw_lines(ax, data, x_col, x_order, hue_col, hue_order, value_col, palette=None,
               labels=None):
    """Mean +/- bootstrap CI as connected points - the view for an interaction, where the
    *slope* between two levels is what is being read rather than the heights."""
    palette = palette or GEN_COLORS
    for s in hue_order:
        xs, ms, los, his = [], [], [], []
        for i, xv in enumerate(x_order):
            v = data[(data[hue_col] == s) & (data[x_col] == xv)][value_col].dropna()
            if len(v) < 2:
                continue
            lo, hi = boot_ci(v)
            xs.append(i); ms.append(v.mean()); los.append(lo); his.append(hi)
        if not xs:
            continue
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                    color=palette[s], lw=3, marker="o", markersize=11, capsize=5,
                    capthick=2, elinewidth=2, label=(labels or {}).get(s, s), zorder=3)
    ax.set_xticks(range(len(x_order)))
