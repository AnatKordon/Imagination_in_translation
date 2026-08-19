# Data log

One entry per data-processing run: what came in, what came out, what it cost.

**Why this file exists:** nothing in the pipeline records what state the data is
in. Answering "have these participants been processed?" otherwise means diffing
raw folder names against report CSVs by hand. Add an entry whenever you run the
outlier pipeline or a paid downstream step. Procedure lives in
[`../PIPELINE.md`](../PIPELINE.md).

Format: newest first. Backups referenced as `backup_DDMMYY/` inside each
condition's `processed_data/` and `outputs/.../outliers/` folder.

---

## 2026-08-19 — downstream chain: image generation + similarity (steps 4–5)

Collection complete (25 valid participants in every cell as of 2026-08-18), so
the downstream chain was run in one pass. **Steps 4, 5 and 6 only — object
validation (`object_accuracy_detector.py`) deliberately NOT run**, pending a
change to the accuracy metric that will require re-running it across all
conditions and participants, old and new.

### Step 4 — offline image generation (PAID)

`generate_images_by_prompt.py`, nogen + plain. **288 images generated,
0 failures.** Parameters unchanged and matching the online worker:
`gpt-image-2`, 1024x1024, quality medium, PNG (~1.7–2.3 MB each).

| condition | kept rows | reused from disk | generated | excluded (skipped) |
|---|---|---|---|---|
| plain_del | 120 | 96 | 24 | 30 |
| plain_perc | 121 | 121 | 0 | 4 |
| plain_imm | 114 | 114 | 0 | 16 |
| nogen_imm | 345 | 288 | 57 | 90 |
| nogen_del | 315 | 222 | 93 | 225 |
| nogen_perc | 351 | 333 | 18 + 96 | 159 |

Throughput was ~55–60 s/image and degraded over long runs (2.3 it/s early,
~43 s/it late), consistent with server-side throttling. Budget ~1 h per 60
images, not the ~25 s/image an early sample suggests.

**Interruption and recovery.** The batch run covering plain + nogen was killed
partway through `nogen_perc` (last condition, largest batch) — 96 of its 114
images had been written but `trials_final.csv` had not, since the script writes
that only at the end of a condition. Re-running the single condition resumed
from disk, generated the remaining 18, and cost nothing extra. **The resumable
design works as documented**: PNGs already on disk are never re-billed. The
correct recovery action is simply to re-run the unfinished condition.

**Result: all 9 conditions consistent** — `trials_final.csv` row count equals
`trials_final_pregen.csv` in every offline-gen condition, with **0 placeholder
`gen` values** anywhere.

### Step 5 — similarity (free, local GPU)

`add_similarity_scores.py`, all 9 conditions, ~8 min total.

| condition | rows | missing CLIP/VGG | `clip_self_prev` populated |
|---|---|---|---|
| aigen_perc | 351 | 0 | 234 |
| aigen_imm | 357 | 0 | 238 |
| aigen_del | 318 | 0 | 212 |
| nogen_perc | 351 | 0 | 234 |
| nogen_imm | 345 | 0 | 230 |
| nogen_del | 315 | 0 | 210 |
| plain_perc | 121 | 0 | 0 |
| plain_imm | 114 | 0 | 0 |
| plain_del | 120 | 0 | 0 |

`trials_final_sim.csv` row counts match `trials_final.csv` exactly everywhere,
and no row is missing a GT-based distance. `clip_self_prev_similarity` is empty
for all three plain conditions by construction — one attempt per session means
no preceding attempt exists.

**Token truncation, for the record (not currently an issue).** 1,005 of 2,392
rows (42%) have prompts over CLIP's 77-token limit, distributed very unevenly:
223/351 in nogen_perc and 208/351 in aigen_perc vs 10/120 in plain_del. This
affects **only** `clip_vis_text_similarity`, since image-image distances do not
use the text encoder. **That column is not used in the current analyses**, so
this is inert — but if prompt-image alignment is ever brought into an analysis,
the truncation is condition-correlated and would need controlling via
`token_num`.

### Step 6 — semantic tagging (PAID, resumable)

`semantic_tagging.py --condition all`, model `gpt-5.5`. **411 new rows** tagged;
existing rows skipped by the resume-by-key logic and not re-billed.

| condition | target | already tagged | newly tagged |
|---|---|---|---|
| aigen_perc | 351 | 279 | 72 |
| aigen_imm | 357 | 327 | 30 |
| aigen_del | 318 | 297 | 21 |
| nogen_perc | 351 | 237 | 114 |
| nogen_imm | 345 | 288 | 57 |
| nogen_del | 315 | 222 | 93 |
| plain_perc | 121 | 121 | 0 |
| plain_imm | 114 | 114 | 0 |
| plain_del | 120 | 96 | 24 |

### Data integrity check — zero orphans

Before running the chain, all three downstream outputs were checked for rows
present in the output but **no longer** in the kept set (which would happen if a
previously-valid participant had since been excluded). **Zero orphans across all
9 conditions and all 3 output files.** Every row ever generated, scored or
tagged is still kept data, so the whole chain is purely additive — nothing needs
deleting and no stale row can contaminate the tables.

### Outstanding

- **Object validation not run** — awaiting the metric change, then a full
  re-run across all conditions and participants (old and new). This is the only
  step in the chain that is deliberately behind.
- Step 7 (`aggregate_conditions.py`, `export_full_data.py`, `computing_RDMS.py`,
  `visualize_per_ppt.py`, notebooks) not yet run.

---

## 2026-08-18 — final two participants; all 9 conditions at 25

**Backup:** `backup_180826/` in every condition's `processed_data/` and
`outputs/.../outliers/`. Captures the post-stage-2 state of this day's run.

**What came in:** 2 new JATOS folders, filling the two cells left short on
2026-08-17.

| condition | study_result | uid | status |
|---|---|---|---|
| aigen_perc | 1228370 | `5uqtavqc1786970836674` | full |
| nogen_del | 1228157 | `7jxjg1at1786959618225` | full |

**Ran:** `consensus` + `build_trials_final` for `aigen_perc` and `nogen_del`
only. Stages 1–2 had already been run separately at 12:45 that day, so
`all_trials.csv` and the structural reports were current; the missing piece was
AI scoring. The other 7 conditions had no new data and were not touched.

### Result — all nine conditions at exactly 25 valid participants

| condition | ppts | valid | removed | short | digit-span | AI | usable sessions |
|---|---|---|---|---|---|---|---|
| aigen_perc | 33 | 25 | 8 | 4 | 0 | 34 | 125/164 |
| aigen_imm | 27 | 25 | 2 | 10 | 0 | 4 | 121/135 |
| aigen_del | 28 | 25 | 3 | 16 | 16 | 9 | 106/138 |
| nogen_perc | 34 | 25 | 9 | 12 | 0 | 33 | 126/170 |
| nogen_imm | 29 | 25 | 4 | 20 | 0 | 7 | 118/145 |
| nogen_del | 37 | 25 | 12 | 31 | 47 | 11 | 115/185 |
| plain_perc | 25 | 25 | 0 | 2 | 0 | 2 | 121/125 |
| plain_imm | 26 | 25 | 1 | 9 | 0 | 5 | 116/130 |
| plain_del | 30 | 25 | 5 | 11 | 17 | 2 | 124/150 |
| **total** | **269** | **225** | **44** | 115 | 80 | 107 | **1072/1342** |

Gate columns are dropped *sessions* and overlap. Exclusion rate 16.4%.
**Recruitment target met: 25 valid participants in every cell.**

### The two new participants

- `5uqtavqc1786970836674` (aigen_perc) — 1 attempt flagged by the AI judges,
  costing it session 1. **4 usable, kept.** Cleared the 3-session floor with one
  to spare; two more flagged attempts would have excluded it and left the cell
  short again.
- `7jxjg1at1786959618225` (nogen_del) — clean, 0 flagged, **5/5 usable, kept.**
  Also passed digit span, the gate that had cost this cell participants in three
  consecutive rounds.

### AI judges

**Spend: $0.10** (aigen_perc $0.052, nogen_del $0.053). Only 15 API calls per
judge per condition; 475 and 539 unique prompts came from cache.

### Process note — the diff-by-report check has a blind spot

The usual "which folders are new" check (raw `study_result_*` on disk vs. the
`study_result` column of `outlier_report_participants.csv`) returned **zero new
folders** this day, because stages 1–2 had already been run and the reports
already listed both. Their prompts were nonetheless completely unscored — 15
trials each, 0 rows in `ai_suspicion_scores.csv` — so both were silently passing
the AI gate by default.

**Membership in the outlier report does not mean AI-scored.** The reliable
signal is the mtime of `ai_usage/ai_suspicion_scores.csv` against
`all_trials.csv`: if the scores file is older, rescoring is owed. Or check
directly:

```python
import pandas as pd, config
for c in config.CONDITIONS:
    p = config.paths_for(c)
    at = pd.read_csv(p.processed_dir / "all_trials.csv")
    sc = pd.read_csv(p.analysis_dir / "outliers" / "ai_usage" / "ai_suspicion_scores.csv")
    missing = set(at.uid) - set(sc.uid)
    print(c, f"{len(missing)} unscored participants", sorted(missing) if missing else "")
```

### Staleness as of this entry

Unchanged in shape — steps 4–7 still not run. `trials_final.csv` is current for
the aigen conditions (351 / 357 / 318); for nogen and plain the current file is
`trials_final_pregen.csv` (351 / 345 / 315, 121 / 114 / 120) and
`trials_final.csv` still holds the pre-2026-08-16 set. `trials_final_sim.csv`
and everything in `nlp_analysis/` are stale in all nine conditions.

**Next:** collection is complete, so the full downstream chain (steps 4–7 in
[`../PIPELINE.md`](../PIPELINE.md)) can now be run in one clean pass.

---

## 2026-08-17 — second top-up round, 5 of 9 conditions

**Backup:** `backup_170826/` in every condition's `processed_data/` and
`outputs/.../outliers/` (including `ai_usage/` and `digit_span/` subfolders).
Holds the 2026-08-16 end state.

**What came in:** 9 new JATOS folders, targeted at the cells left short by the
previous round.

| condition | new folders | study_result IDs |
|---|---|---|
| aigen_perc | 2 | 1227965, 1227972 |
| aigen_del | 1 | 1227966 |
| nogen_perc | 1 | 1227949 |
| nogen_del | 4 | 1227961, 1227976, 1228007, 1228024 |
| plain_del | 1 | 1227962 |

aigen_imm, nogen_imm, plain_perc, plain_imm unchanged.

**Ran:** stages 1–3 (`outlier_pipeline.run` → `consensus` →
`build_trials_final`). Steps 4–7 deliberately **not** run — deferred until
collection is complete, then the whole chain gets one clean pass.

### Result

| condition | ppts | valid | removed | short | digit-span | AI | usable sessions |
|---|---|---|---|---|---|---|---|
| aigen_perc | 32 | 24 | 8 | 4 | 0 | 33 | 121/159 |
| aigen_imm | 27 | 25 | 2 | 10 | 0 | 4 | 121/135 |
| aigen_del | 28 | 25 | 3 | 16 | 16 | 9 | 106/138 |
| nogen_perc | 34 | 25 | 9 | 12 | 0 | 33 | 126/170 |
| nogen_imm | 29 | 25 | 4 | 20 | 0 | 7 | 118/145 |
| nogen_del | 36 | 24 | 12 | 31 | 47 | 11 | 110/180 |
| plain_perc | 25 | 25 | 0 | 2 | 0 | 2 | 121/125 |
| plain_imm | 26 | 25 | 1 | 9 | 0 | 5 | 116/130 |
| plain_del | 30 | 25 | 5 | 11 | 17 | 2 | 124/150 |
| **total** | **267** | **223** | **44** | 115 | 80 | 106 | **1063/1332** |

Gate columns are dropped *sessions* and overlap (a session can fail more than
one gate). Exclusion rate 16.5%.

**Seven of nine conditions now sit at exactly 25 valid participants.** Two are
short by one, each for a recurring reason:

- **aigen_perc — 24.** AI suspicion. Third round in which this cell loses
  participants to the AI gate; suspected trials 56 → 59.
- **nogen_del — 24.** Digit span, `digitspan_dropped` 43 → 47. Third
  consecutive round this cell underdelivers on the same gate. Its 4 new folders
  yielded 3 valid.

### Worked example: why aigen_perc reads 25 mid-run and 24 after

Stage 2 reported 25 valid; the post-`consensus` rebuild reported 24. Not a
discrepancy — it is exactly what the rebuild step exists to catch, and it is
worth understanding because it will recur every round.

`outlier_pipeline.run` applies the AI gate using whatever
`ai_suspicion_scores.csv` is already on disk. New participants' prompts are
absent from it, and an unscored attempt counts as not-suspected, so new arrivals
pass the AI gate by default at stage 2. Only after `consensus` scores them and
`build_trials_final` runs again are the exclusions real.

Here, new participant `3twgxzvt1786899783149` had 5 structurally complete
sessions and passed the word-count gate on all 5. Scoring flagged 3 individual
attempts:

| session | attempt | gpt | gemini | claude | judges ≥80 |
|---|---|---|---|---|---|
| 2 | 2 | 92 | 85 | 75 | 2 of 3 |
| 4 | 2 | 82 | 90 | 82 | 3 of 3 |
| 5 | 3 | 84 | 85 | 72 | 2 of 3 |

Because **one flagged attempt drops its whole session**, 3 flagged attempts out
of 15 destroyed 3 of 5 sessions → 2 usable → below `MIN_USABLE_SESSIONS = 3` →
participant excluded. The other new folder,
`y6v3kv8b1786900818776`, was clean at 5/5. Net +1 valid (23 → 24), and the same
+3 that moved `ai_dropped` from 30 to 33. No pre-existing participant changed
status — the judge cache means old prompts are never re-scored, so old verdicts
cannot move.

Note that 2 of the 3 decisive attempts rest on gpt+gemini agreeing while claude
sat below threshold (75, 72). The swing-vote asymmetry recorded in the
2026-08-16 entry decided a participant here.

### AI judges

All three judges scored all 3,185 trials: zero NA scores, no errors, guard never
fired. **Spend: $0.41.**

| condition | cost |
|---|---|
| nogen_del | $0.19 |
| aigen_perc | $0.10 |
| nogen_perc | $0.05 |
| aigen_del | $0.05 |
| plain_del | $0.02 |
| aigen_imm / nogen_imm / plain_perc / plain_imm | $0.00 (no new data) |

### Staleness as of this entry

Same shape as 2026-08-16 — steps 4–7 not run.

| conditions | current file | stale |
|---|---|---|
| aigen (×3) | `trials_final.csv` (339 / 357 / 318) | `trials_final_sim.csv` + both `nlp_analysis/` files |
| nogen (×3), plain (×3) | `trials_final_pregen.csv` (351 / 345 / 300, 121 / 114 / 120) | `trials_final.csv` (237 / 288 / 222, 121 / 114 / 96) + all 3 downstream files |

plain_perc and plain_imm happen to match by coincidence — no new participants,
so nothing moved. Use the corrected freshness check in
[`../PIPELINE.md`](../PIPELINE.md#staleness--the-main-hazard); comparing against
`trials_final.csv` gives a false OK for nogen/plain.

### Notes

- The 3 `partial` folders are the same previously-reconstructed ones from
  2026-08-16 (2 in aigen_perc, 1 in aigen_del), not new problems.
- **Open:** 2 participants still needed — 1 for aigen_perc, 1 for nogen_del.

---

## 2026-08-16 — new participants added to 7 of 9 conditions

**Backup:** `backup_160826/` in every condition's `processed_data/` and
`outputs/.../outliers/` (including `ai_usage/` and `digit_span/` subfolders).

**What came in:** 33 new JATOS folders. Counts matched the previous run's
exclusions almost exactly, condition by condition — these were replacement
recruits for previously excluded participants.

| condition | new folders | previously excluded |
|---|---|---|
| aigen_perc | 5 | 5 |
| aigen_imm | 2 | 2 |
| aigen_del | 2 | 2 |
| nogen_perc | 8 | 8 |
| nogen_imm | 4 | 4 |
| nogen_del | 7 | 7 |
| plain_perc | 0 | 0 |
| plain_imm | 0 | 1 |
| plain_del | 5 | 4 |

**Ran:** stages 1–3 only (`outlier_pipeline.run` → `consensus` →
`build_trials_final`). Steps 4–7 deliberately **not** run.

**Staleness as of this entry** — reference row count vs. downstream files:

| conditions | reference | state |
|---|---|---|
| aigen (×3) | `trials_final.csv` is current | `trials_final_sim.csv` + both `nlp_analysis/` files stale |
| nogen (×3), plain_del | `trials_final_pregen.csv` is current | `trials_final.csv` **also stale** (awaiting image generation) + all 3 downstream files stale |
| plain_perc, plain_imm | — | consistent; no new participants, nothing changed |

Note the nogen/plain case: `trials_final.csv` does not update until
`generate_images_by_prompt.py` runs, so for those conditions the whole
downstream chain including `trials_final.csv` is still the pre-2026-08-16 set.
A naive freshness check against `trials_final.csv` reports a false OK there —
use the corrected check in [`../PIPELINE.md`](../PIPELINE.md#staleness--the-main-hazard).

### Before → after

| condition | ppts | excluded | kept | (was) | usable sessions | (was) |
|---|---|---|---|---|---|---|
| aigen_perc | 30 | 7 | 23 | 20 | 114 | 96 |
| aigen_imm | 27 | 2 | 25 | 23 | 121 | 111 |
| aigen_del | 27 | 3 | 24 | 23 | 103 | 99 |
| nogen_perc | 33 | 9 | 24 | 17 | 123 | 87 |
| nogen_imm | 29 | 4 | 25 | 21 | 118 | 99 |
| nogen_del | 32 | 11 | 21 | 18 | 97 | 82 |
| plain_perc | 25 | 0 | 25 | 25 | 121 | 121 |
| plain_imm | 26 | 1 | 25 | 25 | 116 | 116 |
| plain_del | 29 | 5 | 24 | 20 | 119 | 99 |
| **total** | **258** | **42** | **216** | **192** | **1032** | **910** |

Exclusion rate 14.7% → 16.3%.

### Baseline before this run (for reference)

225 participants, 33 excluded (14.7%), 192 kept; 1122 sessions, 910 usable
(18.9% dropped). Gate attribution over the 212 dropped sessions (gates overlap):
AI-suspicion 80, short answers 100, digit-span 52 (of 368 delay sessions),
incomplete 1.

### Four cells still short of 25

- **nogen_del (21)** — digit span. `digitspan_dropped` 27 → 43. Second
  consecutive failure to reach 25; new recruits fail the same gate as the old.
- **aigen_perc (23), nogen_perc (24)** — AI suspicion. `ai_dropped` 23 → 30 and
  27 → 31. The perception conditions carry the heaviest AI-suspected counts
  (56/446 and 62/495, vs 6/405 for aigen_imm). Replacing perception
  participants keeps yielding more flagged ones.
- **plain_del (24)** — mixed, mostly digit span (12 → 17).

### AI judges

All three judges scored all 3,060 trials: zero NA scores, per-model call counts
matched, the all-judges-failed guard never fired, no errors in the log.

Flag rates at the ≥80 threshold differ substantially — gpt 11.2%, gemini 7.2%,
claude 3.8% (means 30.5 / 23.4 / 35.8; pairwise correlations 0.44–0.51). Of 457
trials flagged by at least one judge, only 53 were flagged by all three and 289
by exactly one (dropped by the 2-of-3 rule); 168 trials ended up `ai_suspected`.
**gpt is effectively the swing vote in most 2-of-3 decisions** — worth
remembering if the perception conditions' exclusion rate becomes a write-up
issue.

**Spend: $1.71 total** (549 calls; a cold run would have been ~3,000).

| condition | cost |
|---|---|
| nogen_perc | $0.43 |
| nogen_del | $0.35 |
| aigen_perc | $0.27 |
| nogen_imm | $0.21 |
| aigen_imm | $0.10 |
| aigen_del | $0.10 |
| plain_perc / plain_imm / plain_del | $0.00 (all cached) |

### Notes

- Three folders came through as `partial`, all reconstructed from `data.txt`
  rather than lost: `study_result_1167890` and `study_result_1168393`
  (aigen_perc, missing `participants.csv`), `study_result_1174054` (aigen_del,
  missing `participants.csv`; `trials.csv` + `digit_span.csv` reconstructed).
- `study_result_1174054` / uid `8ko1thjh1781766796078` passes every gate and is
  **kept**, but contributes only 3 sessions instead of 5 — sitting exactly at
  the `MIN_USABLE_SESSIONS = 3` floor. Its session 4 has digit-span data but
  never reached three attempts.

---

## Before 2026-08-16

Not logged — this file starts here. The `backup_160826/` folders hold the state
as of the previous run; `git log` covers code changes.
