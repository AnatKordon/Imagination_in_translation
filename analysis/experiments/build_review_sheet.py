# Builds a human review set from the pilot runs: every tag the judges disagree
# about, with each judge's verdict side by side and a blank column for the
# reviewer's own call.
#
# Only disputed tags are included. A tag every run agreed on carries no
# information about which configuration is more accurate, and the point of the
# review is to produce a gold label that can score all of them.

from pathlib import Path
import argparse
import base64
import glob
import json
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import config  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis/outputs/experiments/validator"
KEY = ["gt", "uid", "session", "attempt", "category", "item"]

RUNS: list[tuple[str, str, str]] = []  # filled from argv: (file tag, column label, blurb)

CODES = {
    "V": ("valid", "The tag is fine: right category, said in the description, visible in the image."),
    "P": ("participant hallucination", "The participant described something the image does not show."),
    "M": ("model hallucination", "The extractor invented it; the description never said it."),
    "C": ("category misplacement", "Real enough, but it does not belong in this category."),
    "?": ("unsure", "Genuinely ambiguous - leave it to be excluded rather than guessed."),
}
SHORT = {"valid": "V", "participant_hallucination": "P",
         "model_hallucination": "M", "category_misplacement": "C"}


def newest(tag: str) -> pd.DataFrame:
    hits = sorted(glob.glob(str(OUT_DIR / f"pilot_tags_*{tag}_*.csv")), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"no pilot file for {tag}")
    d = pd.read_csv(hits[-1])
    for c in ["uid", "session", "attempt"]:
        d[c] = d[c].astype(str)
    return d.drop_duplicates(subset=KEY)


def build() -> tuple[pd.DataFrame, dict]:
    data = {lab: newest(tag) for tag, lab, _ in RUNS}
    labels = [lab for _, lab, _ in RUNS]
    first = labels[0]
    base = data[first][KEY + ["description", "old_in_image", "confidence", "note"]].copy()
    base = base.rename(columns={"confidence": "conf_A", "note": "note_A"})
    base["conf_first"] = base["conf_A"]
    if len(labels) > 1:
        base = base.merge(
            data[labels[1]][KEY + ["confidence"]].rename(columns={"confidence": "conf_second"}),
            on=KEY, how="left")
    for lab in labels:
        base = base.merge(
            data[lab][KEY + ["error_type"]].rename(columns={"error_type": lab}),
            on=KEY, how="left",
        )
    verdicts = base[labels]
    base["n_distinct"] = verdicts.apply(lambda r: r.dropna().nunique(), axis=1)
    # v1 only ever answered in_image, so it "disagrees" when its binary call and
    # the new valid/not-valid call point opposite ways.
    base["v1"] = base["old_in_image"].map({True: "valid", False: "REJECTED"})
    first = labels[0]
    v1_conflict = (
        ((base["old_in_image"] == False) & (base[first] == "valid"))     # noqa: E712
        | ((base["old_in_image"] == True) & (base[first] != "valid"))    # noqa: E712
    )
    base["why_listed"] = ""
    base.loc[base["n_distinct"] > 1, "why_listed"] = "judges disagree"
    base.loc[v1_conflict & (base["n_distinct"] > 1), "why_listed"] = "judges disagree + v1 conflict"
    base.loc[v1_conflict & (base["n_distinct"] <= 1), "why_listed"] = "v1 conflict"
    stats = compute_stats(base, labels, data)
    out = base[(base["n_distinct"] > 1) | v1_conflict].copy()
    for lab in labels:
        out[lab] = out[lab].map(SHORT).fillna("-")
    out["v1"] = out["old_in_image"].map({True: "V", False: "P?"}).fillna("-")
    out = out.sort_values(["gt", "category", "item"]).reset_index(drop=True)
    out.insert(0, "row_id", [f"{i+1:03d}" for i in range(len(out))])
    return out, stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def compute_stats(base: pd.DataFrame, labels: list[str], data: dict) -> dict:
    """Everything the reader needs to judge how far to trust the verdicts."""
    ERR = ["participant_hallucination", "category_misplacement", "model_hallucination"]
    st: dict = {"n_tags": len(base), "labels": labels, "runs": [], "pairs": [], "per_error": [],
                "by_category": [], "rates": []}
    for tag, lab, blurb in RUNS:
        d = data[lab]
        rows = d[["gt", "uid", "session", "attempt"]].drop_duplicates()
        st["runs"].append({
            "label": lab, "blurb": blurb, "rows": len(rows), "tags": len(d),
            "counts": {k: int(v) for k, v in d["error_type"].value_counts().items()},
        })
    # reliability across every pair of runs
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            m = base[base[a].notna() & base[b].notna()]
            if not len(m):
                continue
            either = m[(m[a] != "valid") | (m[b] != "valid")]
            both = int(((either[a] != "valid") & (either[b] != "valid")).sum())
            lo, hi = wilson(both, len(either))
            st["pairs"].append({
                "a": a, "b": b, "n": len(m),
                "disagree": round(100 * float((m[a] != m[b]).mean()), 2),
                "either": len(either), "both": both,
                "flagged_agree": round(100 * both / len(either), 1) if len(either) else None,
                "ci": [round(100 * lo, 1), round(100 * hi, 1)],
            })
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        m = base[base[a].notna() & base[b].notna()]
        for et in ERR:
            x, y = (m[a] == et), (m[b] == et)
            inter, union = int((x & y).sum()), int((x | y).sum())
            lo, hi = wilson(inter, union)
            st["per_error"].append({
                "error": et, "a": int(x.sum()), "b": int(y.sum()),
                "agree": round(100 * inter / union, 1) if union else None,
                "union": union, "ci": [round(100 * lo, 1), round(100 * hi, 1)],
            })
        m2 = m.copy()
        m2["bh"] = (m2["conf_first"] == "high") & (m2["conf_second"] == "high") \
            if "conf_second" in m2 else (m2["conf_first"] == "high")
        for lab_, sub in [("both high", m2[m2["bh"]]), ("either not high", m2[~m2["bh"]])]:
            if len(sub):
                st["rates"].append({"group": lab_, "n": len(sub),
                                    "disagree": round(100 * float((sub[a] != sub[b]).mean()), 2)})
        for cat, blk in m.groupby("category"):
            st["by_category"].append({
                "category": cat, "n": len(blk),
                "disagree": round(100 * float((blk[a] != blk[b]).mean()), 1),
            })
        st["by_category"].sort(key=lambda r: -r["disagree"])
        # the reported rate, per run and per source
        for src in ["participant", "gpt_ceiling"]:
            row = {"source": src}
            for lab in labels:
                d = data[lab]
                sub = d[d["source"] == src] if "source" in d else d
                scored = int((~sub["error_type"].isin(
                    ["category_misplacement", "model_hallucination"])).sum())
                ph = int((sub["error_type"] == "participant_hallucination").sum())
                row[lab] = round(100 * ph / scored, 2) if scored else None
            st["rates"].append(row)
    return st


def data_uri(name: str) -> str:
    p = Path(config.GT_DIR) / name
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


CSS = """
:root{
  --paper:#f6f7f8; --card:#ffffff; --ink:#14181c; --muted:#5c6670; --line:#dde1e5;
  --accent:#0f6e6e; --accent-soft:#e3efef;
  --v:#2f7d5d; --p:#b4622a; --m:#6b4fa8; --c:#3a6ea5; --v1:#8a929a;
  --chip-bg:#eef1f3;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#101418; --card:#171c21; --ink:#e6eaed; --muted:#98a2ab; --line:#283037;
    --accent:#4fb3ac; --accent-soft:#16302f;
    --v:#5cc08e; --p:#e0954f; --m:#a58ae0; --c:#6ea8e0; --v1:#79838c;
    --chip-bg:#1f262c;
  }
}
:root[data-theme="dark"]{
  --paper:#101418; --card:#171c21; --ink:#e6eaed; --muted:#98a2ab; --line:#283037;
  --accent:#4fb3ac; --accent-soft:#16302f;
  --v:#5cc08e; --p:#e0954f; --m:#a58ae0; --c:#6ea8e0; --v1:#79838c;
  --chip-bg:#1f262c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 96px}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  text-wrap:balance;margin:0;font-weight:600}
h1{font-size:2.1rem;letter-spacing:-.01em}
.sub{color:var(--muted);margin:6px 0 0;max-width:64ch}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin:0 0 8px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:20px 22px;margin:28px 0}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:14px}
.legend div{display:flex;gap:10px;align-items:flex-start}
.code{font-family:"IBM Plex Mono",monospace;font-weight:600;width:1.6rem;flex:none;
  text-align:center;border-radius:4px;padding:1px 0;font-size:.85rem}
.code.V{color:#fff;background:var(--v)} .code.P{color:#fff;background:var(--p)}
.code.M{color:#fff;background:var(--m)} .code.C{color:#fff;background:var(--c)}
.code.Q{color:var(--ink);background:var(--chip-bg)}
.legend b{font-weight:600;display:block;font-size:.92rem}
.legend span{color:var(--muted);font-size:.85rem;line-height:1.4}
section.trial{margin:44px 0 0;border-top:2px solid var(--ink);padding-top:20px}
.trial-head{display:grid;grid-template-columns:300px 1fr;gap:26px;align-items:start}
@media (max-width:820px){.trial-head{grid-template-columns:1fr}}
.imgbox{position:sticky;top:20px}
.imgbox img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.imgcap{font-family:"IBM Plex Mono",monospace;font-size:.75rem;color:var(--muted);margin-top:8px}
.desc{background:var(--accent-soft);border-left:3px solid var(--accent);
  padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:18px;font-size:.94rem}
.desc .who{font-family:"IBM Plex Mono",monospace;font-size:.7rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent);display:block;margin-bottom:4px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th{font-family:"IBM Plex Mono",monospace;font-size:.68rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);text-align:left;font-weight:500;
  padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:hover{background:var(--chip-bg)}
.rid{font-family:"IBM Plex Mono",monospace;color:var(--muted);font-size:.8rem;
  font-variant-numeric:tabular-nums}
.tag{font-family:"IBM Plex Mono",monospace;font-size:.86rem}
.cat{color:var(--muted);font-size:.78rem;font-family:"IBM Plex Mono",monospace}
.vs{text-align:center;font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:.82rem}
.vs.V{color:var(--v)} .vs.P{color:var(--p)} .vs.M{color:var(--m)} .vs.C{color:var(--c)}
.vs.o{color:var(--v1)}
.note{color:var(--muted);font-size:.8rem;max-width:34ch}
.blank{background:var(--chip-bg);border-radius:4px;min-width:2.4rem}
.split th.g{border-left:1px solid var(--line)} .split td.g{border-left:1px solid var(--line)}
footer{margin-top:56px;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:18px}
.vote{display:flex;gap:3px}
.vote button{font-family:"IBM Plex Mono",monospace;font-size:.78rem;font-weight:600;
  width:1.75rem;height:1.6rem;border:1px solid var(--line);background:var(--card);
  color:var(--muted);border-radius:4px;cursor:pointer;padding:0;line-height:1}
.vote button:hover{border-color:var(--accent);color:var(--ink)}
.vote button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.vote button[aria-pressed="true"]{color:#fff;border-color:transparent}
.vote button[aria-pressed="true"][data-v="V"]{background:var(--v)}
.vote button[aria-pressed="true"][data-v="P"]{background:var(--p)}
.vote button[aria-pressed="true"][data-v="M"]{background:var(--m)}
.vote button[aria-pressed="true"][data-v="C"]{background:var(--c)}
.vote button[aria-pressed="true"][data-v="?"]{background:var(--muted)}
tr.done{opacity:.55} tr.done:hover{opacity:1}
.bar{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--line);
  margin:0 -24px 0;padding:12px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar .grow{flex:1}
.prog{font-family:"IBM Plex Mono",monospace;font-size:.85rem;font-variant-numeric:tabular-nums}
.track{height:5px;background:var(--chip-bg);border-radius:3px;width:150px;overflow:hidden}
.track i{display:block;height:100%;background:var(--accent);width:0;transition:width .25s}
.bar button{font-family:"IBM Plex Sans",sans-serif;font-size:.82rem;padding:5px 12px;
  border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:5px;cursor:pointer}
.bar button:hover{border-color:var(--accent)}
.status{font-family:"IBM Plex Mono",monospace;font-size:.75rem;color:var(--muted)}
.status.ok{color:var(--v)} .status.warn{color:var(--p)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px;margin:22px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.stat h4{font-family:"IBM Plex Sans Condensed",sans-serif;font-size:.95rem;margin:0 0 10px;
  font-weight:600;letter-spacing:.01em}
.stat table{font-size:.82rem} .stat td,.stat th{padding:4px 6px}
.stat .num{text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.kpi{display:flex;gap:22px;flex-wrap:wrap;margin:4px 0 2px}
.kpi div{min-width:96px}
.kpi b{display:block;font-family:"IBM Plex Mono",monospace;font-size:1.5rem;line-height:1.1;
  font-variant-numeric:tabular-nums}
.kpi span{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.good{color:var(--v)} .warn{color:var(--p)} .bad{color:var(--m)}
.ci{color:var(--muted);font-size:.75rem}
.note-block{font-size:.85rem;color:var(--muted);margin-top:10px;line-height:1.45}
"""


APP_JS = r"""
(function () {
  var D = JSON.parse(document.getElementById('data').textContent);
  var V = JSON.parse(document.getElementById('verdicts').textContent || '{}');
  var LS = 'validator-review-' + D.stamp;
  var dirty = false, timer = null, publisher = null;

  try {
    var local = JSON.parse(localStorage.getItem(LS) || '{}');
    for (var k in local) { if (!(k in V)) { V[k] = local[k]; dirty = true; } }
  } catch (e) {}

  var esc = function (t) {
    return String(t == null ? '' : t).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };

  function setStatus(text, cls) {
    var el = document.getElementById('status');
    if (el) { el.textContent = text; el.className = 'status ' + (cls || ''); }
  }

  function progress() {
    var n = 0;
    for (var i = 0; i < D.rows.length; i++) { if (V[D.rows[i].row_id]) n++; }
    var pct = D.rows.length ? Math.round(100 * n / D.rows.length) : 0;
    document.getElementById('prog').textContent = n + ' / ' + D.rows.length;
    document.getElementById('fill').style.width = pct + '%';
  }

  function buildDoc() {
    var css = document.getElementById('css').textContent;
    var app = document.getElementById('app').textContent;
    var data = document.getElementById('data').textContent;
    return '<!doctype html><html lang="en"><head><meta charset="utf-8">'
      + '<meta name="viewport" content="width=device-width,initial-scale=1">'
      + '<title>Validator Verdict Review</title>'
      + '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      + 'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&'
      + 'family=IBM+Plex+Sans:wght@400;600&display=swap">'
      + '<style id="css">' + css + '<\/style></head><body><div id="root"></div>'
      + '<script id="data" type="application/json">' + data + '<\/script>'
      + '<script id="verdicts" type="application/json">' + JSON.stringify(V) + '<\/script>'
      + '<script id="app">' + app + '<\/script></body></html>';
  }

  function schedulePublish() {
    if (timer) clearTimeout(timer);
    setStatus('unsaved', 'warn');
    timer = setTimeout(function () {
      if (!publisher) { setStatus('saved in this browser only', 'warn'); return; }
      setStatus('saving...');
      publisher.publish(buildDoc()).then(function () {
        dirty = false;
        setStatus('saved', 'ok');
      }).catch(function (err) {
        // conflict means another view published first and every view reloads to
        // that version; anything else leaves the local copy as the record.
        setStatus(err && err.code === 'conflict' ? 'reloading' : 'saved locally only', 'warn');
      });
    }, 1200);
  }

  function choose(rowId, code) {
    if (V[rowId] === code) { delete V[rowId]; } else { V[rowId] = code; }
    try { localStorage.setItem(LS, JSON.stringify(V)); } catch (e) {}
    paint(rowId);
    progress();
    schedulePublish();
  }

  function paint(rowId) {
    var tr = document.querySelector('tr[data-row="' + rowId + '"]');
    if (!tr) return;
    var cur = V[rowId] || '';
    tr.classList.toggle('done', !!cur);
    var btns = tr.querySelectorAll('.vote button');
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute('aria-pressed', btns[i].getAttribute('data-v') === cur ? 'true' : 'false');
    }
  }

  function voteCell(rowId) {
    var out = '<td class="g"><div class="vote">';
    for (var i = 0; i < D.codes.length; i++) {
      var c = D.codes[i];
      out += '<button type="button" data-row="' + esc(rowId) + '" data-v="' + esc(c)
           + '" aria-pressed="false" title="' + esc(D.codeNames[c]) + '">' + esc(c) + '</button>';
    }
    return out + '</div></td>';
  }

  function pct(x) { return x === null || x === undefined ? '&mdash;' : x.toFixed(1) + '%'; }

  function statsPanel() {
    var S = D.stats;
    if (!S) return '';
    var o = ['<div class="panel"><h3>How far to trust these verdicts</h3><div class="stats">'];

    // headline reliability from the first pair of runs
    var p0 = S.pairs && S.pairs.length ? S.pairs[0] : null;
    if (p0) {
      var cls = p0.flagged_agree >= 65 ? 'good' : (p0.flagged_agree >= 45 ? 'warn' : 'bad');
      o.push('<div class="stat"><h4>Reproducibility, ' + esc(p0.a) + ' vs ' + esc(p0.b) + '</h4>'
        + '<div class="kpi">'
        + '<div><b>' + p0.disagree.toFixed(2) + '%</b><span>tags differ</span></div>'
        + '<div><b class="' + cls + '">' + pct(p0.flagged_agree) + '</b><span>agree on flagged</span></div>'
        + '</div><p class="note-block">Identical settings, so every difference is noise. '
        + p0.both + ' of ' + p0.either + ' flagged tags were flagged by both '
        + '<span class="ci">(95% CI ' + p0.ci[0] + '&ndash;' + p0.ci[1] + '%)</span>. '
        + 'The overall figure looks small only because most tags are unanimously valid.</p></div>');
    }

    // what each run found
    o.push('<div class="stat"><h4>What each run found</h4><table><tr><th>verdict</th>');
    for (var i = 0; i < S.runs.length; i++) o.push('<th class="num">' + esc(S.runs[i].label) + '</th>');
    o.push('</tr>');
    var kinds = ['valid', 'participant_hallucination', 'category_misplacement', 'model_hallucination'];
    for (var k = 0; k < kinds.length; k++) {
      o.push('<tr><td>' + esc(kinds[k].replace(/_/g, ' ')) + '</td>');
      for (var j = 0; j < S.runs.length; j++) {
        o.push('<td class="num">' + (S.runs[j].counts[kinds[k]] || 0) + '</td>');
      }
      o.push('</tr>');
    }
    o.push('<tr><td>rows / tags</td>');
    for (var q = 0; q < S.runs.length; q++) {
      o.push('<td class="num">' + S.runs[q].rows + ' / ' + S.runs[q].tags + '</td>');
    }
    o.push('</tr></table></div>');

    // per error type
    if (S.per_error && S.per_error.length) {
      o.push('<div class="stat"><h4>Reproducibility by error type</h4>'
        + '<table><tr><th>type</th><th class="num">run 1</th><th class="num">run 2</th>'
        + '<th class="num">agree</th></tr>');
      for (var e = 0; e < S.per_error.length; e++) {
        var r = S.per_error[e];
        var c2 = r.agree === null ? '' : (r.agree >= 65 ? 'good' : (r.agree >= 45 ? 'warn' : 'bad'));
        o.push('<tr><td>' + esc(r.error.replace(/_/g, ' ')) + '</td><td class="num">' + r.a
          + '</td><td class="num">' + r.b + '</td><td class="num ' + c2 + '">' + pct(r.agree)
          + '<br><span class="ci">' + r.ci[0] + '&ndash;' + r.ci[1] + '</span></td></tr>');
      }
      o.push('</table><p class="note-block">Jaccard: flagged by both, over flagged by either. '
        + 'Wide intervals are honest &mdash; these counts are small.</p></div>');
    }

    // confidence filter + reported rate
    var conf = [], rates = [];
    for (var z = 0; z < (S.rates || []).length; z++) {
      (S.rates[z].group ? conf : rates).push(S.rates[z]);
    }
    if (conf.length) {
      o.push('<div class="stat"><h4>Confidence as a filter</h4><table>'
        + '<tr><th>group</th><th class="num">tags</th><th class="num">differ</th></tr>');
      for (var c3 = 0; c3 < conf.length; c3++) {
        o.push('<tr><td>' + esc(conf[c3].group) + '</td><td class="num">' + conf[c3].n
          + '</td><td class="num">' + conf[c3].disagree.toFixed(2) + '%</td></tr>');
      }
      o.push('</table><p class="note-block">Tags both runs marked high confidence are far more '
        + 'stable, so confidence is a usable filter rather than decoration.</p></div>');
    }
    if (rates.length) {
      o.push('<div class="stat"><h4>Participant-hallucination rate</h4><table><tr><th>source</th>');
      for (var y = 0; y < S.labels.length; y++) o.push('<th class="num">' + esc(S.labels[y]) + '</th>');
      o.push('</tr>');
      for (var x = 0; x < rates.length; x++) {
        o.push('<tr><td>' + esc(rates[x].source.replace(/_/g, ' ')) + '</td>');
        for (var w = 0; w < S.labels.length; w++) {
          o.push('<td class="num">' + pct(rates[x][S.labels[w]]) + '</td>');
        }
        o.push('</tr>');
      }
      o.push('</table><p class="note-block">The number a paper would quote. GPT-ceiling '
        + 'descriptions were written with the image in view, so their rate should be zero.</p></div>');
    }

    // instability by category
    if (S.by_category && S.by_category.length) {
      o.push('<div class="stat"><h4>Where the instability sits</h4><table>'
        + '<tr><th>category</th><th class="num">tags</th><th class="num">differ</th></tr>');
      for (var b2 = 0; b2 < S.by_category.length; b2++) {
        var bc = S.by_category[b2];
        o.push('<tr><td>' + esc(bc.category) + '</td><td class="num">' + bc.n
          + '</td><td class="num">' + bc.disagree.toFixed(1) + '%</td></tr>');
      }
      o.push('</table></div>');
    }
    return o.join('') + '</div></div>';
  }

  function render() {
    var h = ['<div class="wrap">'];
    h.push('<div class="bar"><span class="prog" id="prog">0 / 0</span>'
      + '<span class="track"><i id="fill"></i></span>'
      + '<span class="status" id="status">loading</span><span class="grow"></span>'
      + '<button type="button" id="toggle">Hide reviewed</button>'
      + '<button type="button" id="copy">Copy as TSV</button></div>');
    h.push('<p class="eyebrow">semantic tag validator &middot; pilot adjudication</p>');
    h.push('<h1>Where the judges disagree</h1>');
    h.push('<p class="sub">' + D.rows.length + ' tags of ' + D.total + ' where the runs '
      + 'split, or where the new validator contradicts the original. Tags every run agreed on '
      + 'are omitted &mdash; they carry no information about which configuration is right. '
      + 'Click a code to record your call; it saves automatically.</p>');

    h.push('<div class="panel"><h3>Verdict codes</h3><div class="legend">');
    for (var i = 0; i < D.codes.length; i++) {
      var c = D.codes[i];
      h.push('<div><span class="code ' + (c === '?' ? 'Q' : c) + '">' + esc(c) + '</span>'
        + '<span><b>' + esc(D.codeNames[c]) + '</b><span>' + esc(D.codeHelp[c]) + '</span></span></div>');
    }
    h.push('</div>');
    h.push('<p class="sub" style="margin-top:12px">Judge each tag on its own merits, not '
      + 'by which run you agree with &mdash; an absolute label lets every configuration, '
      + 'including future ones, be scored against one standard. Where more than one fault '
      + 'applies use <b>C</b> &rarr; <b>M</b> &rarr; <b>P</b>: a tag in the wrong category was '
      + 'judged by the wrong test, and one the extractor invented says nothing about the '
      + 'participant.</p></div>');
    h.push(statsPanel());

    var groups = {};
    for (var r = 0; r < D.rows.length; r++) {
      (groups[D.rows[r].gt] = groups[D.rows[r].gt] || []).push(D.rows[r]);
    }
    Object.keys(groups).forEach(function (gt) {
      var rows = groups[gt];
      h.push('<section class="trial"><div class="trial-head">');
      h.push('<div class="imgbox"><img src="' + D.images[gt] + '" alt="' + esc(gt) + '">'
        + '<p class="imgcap">' + esc(gt) + ' &middot; ' + rows.length + ' disputed</p></div><div>');
      var lastDesc = null;
      rows.forEach(function (row, idx) {
        if (row.description !== lastDesc) {
          if (lastDesc !== null) h.push('</tbody></table></div>');
          h.push('<div class="desc"><span class="who">description shown to the judge</span>'
            + esc(row.description) + '</div>');
          h.push('<div class="scroll"><table class="split"><thead><tr><th>#</th><th>tag</th>'
            + '<th>category</th><th class="g">v1</th>' + D.labels.map(function(l){return '<th>'+esc(l)+'</th>';}).join('')
            + '<th class="g">your call</th><th>note (run A)</th></tr></thead><tbody>');
          lastDesc = row.description;
        }
        h.push('<tr data-row="' + esc(row.row_id) + '"><td class="rid">' + esc(row.row_id)
          + '</td><td class="tag">' + esc(row.item) + '</td><td class="cat">'
          + esc(row.category) + '</td>');
        ['v1'].concat(D.labels).forEach(function (lab) {
          var v = row[lab] || '-';
          var cls = lab === 'v1' ? 'o' : ('VPMC'.indexOf(v) >= 0 ? v : '');
          h.push('<td class="vs ' + cls + (lab === 'v1' ? ' g' : '') + '">' + esc(v) + '</td>');
        });
        h.push(voteCell(row.row_id));
        h.push('<td class="note">' + esc(row.note) + '</td></tr>');
      });
      h.push('</tbody></table></div></div></div></section>');
    });

    h.push('<footer>v1 = the original validator, which only answered <em>is it in the image</em>'
      + ' &mdash; its <span class="vs o">P?</span> means it rejected a tag without saying whose '
      + 'error it was. A/B/C = gpt-5.4-mini at medium, high, and high repeated; D = gpt-5.5 at '
      + 'medium. B and C dropped the largest rows by exceeding the output-token cap, shown as '
      + '&ldquo;-&rdquo;.</footer></div>');
    document.getElementById('root').innerHTML = h.join('');

    for (var j = 0; j < D.rows.length; j++) paint(D.rows[j].row_id);
    progress();

    document.getElementById('root').addEventListener('click', function (ev) {
      var b = ev.target.closest ? ev.target.closest('.vote button') : null;
      if (b) choose(b.getAttribute('data-row'), b.getAttribute('data-v'));
    });
    document.getElementById('toggle').addEventListener('click', function () {
      var on = this.getAttribute('data-on') === '1';
      this.setAttribute('data-on', on ? '0' : '1');
      this.textContent = on ? 'Hide reviewed' : 'Show all';
      var trs = document.querySelectorAll('tr[data-row]');
      for (var i = 0; i < trs.length; i++) {
        var has = !!V[trs[i].getAttribute('data-row')];
        trs[i].style.display = (!on && has) ? 'none' : '';
      }
    });
    document.getElementById('copy').addEventListener('click', function () {
      var lines = ['row_id\tgt\tcategory\titem\tv1\t'+D.labels.join('\t')+'\tmy_verdict'];
      for (var i = 0; i < D.rows.length; i++) {
        var r = D.rows[i];
        var cells=[r.row_id,r.gt,r.category,r.item,r.v1];
        for (var q=0;q<D.labels.length;q++) cells.push(r[D.labels[q]]);
        cells.push(V[r.row_id]||'');
        lines.push(cells.join('\t'));
      }
      var txt = lines.join('\n');
      if (navigator.clipboard) {
        navigator.clipboard.writeText(txt).then(function () {
          setStatus('copied ' + D.rows.length + ' rows', 'ok');
        });
      }
    });
  }

  render();
  setStatus(dirty ? 'unsaved local edits restored' : 'ready');

  if (window.claude && window.claude.use) {
    window.claude.use('artifact').then(function (a) {
      publisher = a;
      setStatus(a ? (dirty ? 'unsaved local edits restored' : 'ready') : 'this browser only',
                a ? '' : 'warn');
    }).catch(function () { setStatus('this browser only', 'warn'); });
  } else {
    setStatus('this browser only', 'warn');
  }
})();
"""


def render(df: pd.DataFrame, stamp: str, stats: dict) -> str:
    """Emit a data-driven page: all markup is built at runtime from the JSON blocks."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "row_id": r["row_id"], "gt": r["gt"], "category": r["category"],
            "item": str(r["item"]), "description": str(r["description"]),
            "v1": r["v1"], **{lab: r[lab] for _, lab, _ in RUNS},
            "note": "" if pd.isna(r["note_A"]) else str(r["note_A"]),
        })
    payload = {
        "stamp": stamp,
        "stats": stats,
        "rows": rows,
        "total": 647,
        "images": {gt: data_uri(gt) for gt in df["gt"].unique()},
        "labels": [lab for _, lab, _ in RUNS],
        "codes": list(CODES.keys()),
        "codeNames": {k: v[0] for k, v in CODES.items()},
        "codeHelp": {k: v[1] for k, v in CODES.items()},
    }
    return (
        '<title>Validator Verdict Review</title>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&'
        'family=IBM+Plex+Sans:wght@400;600&display=swap">\n'
        f'<style id="css">{CSS}</style>\n'
        '<div id="root"></div>\n'
        '<script id="data" type="application/json">'
        + json.dumps(payload).replace("</", "<\\/") +
        '</script>\n'
        '<script id="verdicts" type="application/json">{}</script>\n'
        f'<script id="app">{APP_JS}</script>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="run tags, e.g. _K _L")
    ap.add_argument("--labels", nargs="+", required=True, help="column label per run")
    ap.add_argument("--blurbs", nargs="+", required=True, help="one-line description per run")
    args = ap.parse_args()
    if not (len(args.runs) == len(args.labels) == len(args.blurbs)):
        raise SystemExit("--runs, --labels and --blurbs must be the same length")
    RUNS.extend(zip(args.runs, args.labels, args.blurbs))

    df, stats = build()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"review_disputed_{stamp}.csv"
    labels = [lab for _, lab, _ in RUNS]
    cols = (["row_id", "gt", "category", "item", "v1"] + labels
            + ["conf_A", "note_A", "why_listed", "description", "uid", "session", "attempt"])
    out = df[cols].copy()
    out.insert(5 + len(labels), "my_verdict", "")
    out.to_csv(csv_path, index=False)

    html_path = OUT_DIR / f"review_disputed_{stamp}.html"
    html_path.write_text(render(df, stamp, stats), encoding="utf-8")
    print(f"{len(df)} disputed tags")
    print(f"  csv  {csv_path}")
    print(f"  html {html_path}")


if __name__ == "__main__":
    main()
