# Analysis pipeline — the runbook

**Start here.** This is the map: what runs, in what order, what it costs, and
which file feeds which. Per-stage mechanism lives in the package READMEs linked
below; the record of what has actually been run lives in
[`docs/data_log.md`](docs/data_log.md).

- [`analysis/outlier_pipeline/README.md`](analysis/outlier_pipeline/README.md) — exclusions, gates, AI-usage judges
- [`analysis/nlp_analysis/README.md`](analysis/nlp_analysis/README.md) — semantic tagging, object validation
- [`similarity/README.md`](similarity/README.md) — CLIP / VGG / LPIPS scores
- [`analysis/README.md`](analysis/README.md) — everything else in `analysis/`

---

## The one-paragraph version

Nine conditions (3 generation types × 3 memory tasks). Raw JATOS exports land in
`Data/participants_data/Full_experiment/<gen>/<task>/`. The outlier pipeline
aggregates them and applies four exclusion gates to produce `trials_final.csv`.
Everything downstream — similarity scores, semantic tags, object validation,
notebooks — is built **from** `trials_final.csv`, in a fixed order, and **none of
it rebuilds itself when `trials_final.csv` changes.** That last point is the one
that bites; see [Staleness](#staleness-the-main-hazard).

## Data flow

```
Data/participants_data/<DATASET>/<gen>/<task>/jatos_results_files_*/
  study_result_*/comp-result_*/files/{participants,trials,digit_span}.csv + images
        │
        │  (1) outlier_pipeline.run  — structure check + reconstruct + aggregate
        ▼
Data/processed_data/<DATASET>/<gen>/<task>/
  all_trials.csv, all_participants.csv, all_digit_span.csv, summary_by_uid.csv
        │
        │  (2) ai_usage_suspicion.consensus   [PAID, cached]
        │  (3) build_trials_final             — applies the 4 gates
        ▼
  trials_final.csv                    (aigen: done here)
  trials_final_pregen.csv             (nogen/plain: placeholder `gen`)
        │
        │  (4) generate_images_by_prompt      [PAID] — nogen/plain only
        ▼
  trials_final.csv                    (all conditions)
        │
        ├── (5) add_similarity_scores  ──────────►  trials_final_sim.csv
        │                                            │
        ├── (6) semantic_tagging       [PAID] ─────► nlp_analysis/trials_final_semantic_tags.csv
        │            │                                        │
        │            └── object_accuracy_detector [PAID] ────► nlp_analysis/trials_final_semantic_tag_image_validation.csv
        │                                                     + ..._executive_summary.csv
        ▼
  (7) notebooks / aggregate_conditions / export_full_data / computing_RDMS / visualize_per_ppt
```

`<DATASET>` is `Full_experiment` (set in `condition_maps.yaml`). A **condition
slug** is `<gen>_<task>` — `aigen_perc`, `nogen_imm`, `plain_del`, … — and the
folder path is just the slug split on `_`. `config.paths_for(slug)` resolves all
three trees (participants / processed / outputs); `config.load([slugs], sim=)`
concatenates analysis tables across conditions.

---

## Runbook: a new batch of participants arrived

This is the common case and the reason this file exists. Steps 1–3 are the
"outlier process"; 4–7 are the downstream rebuild. Per condition, or loop over
all nine.

### 0. Check what is actually new

Nothing tracks this automatically. Diff the raw folders against the last
processed report:

```python
# study_result_* on disk vs. the study_result column of the last outlier report
import csv, glob, os
raw  = {os.path.basename(p) for p in glob.glob(
        "Data/participants_data/Full_experiment/<gen>/<task>/jatos_results_files_*/study_result_*")}
seen = {r["study_result"] for r in csv.DictReader(open(
        "analysis/outputs/Full_experiment/<gen>/<task>/outliers/outlier_report_participants.csv"))}
print(len(raw - seen), "new")
```

### 1. Back up before overwriting

Steps 2–3 overwrite `trials_final*.csv` and every report in `outliers/`. Copy
them to a dated subfolder first:

```bash
pd=Data/processed_data/Full_experiment/<gen>/<task>
od=analysis/outputs/Full_experiment/<gen>/<task>/outliers
mkdir -p $pd/backup_DDMMYY $od/backup_DDMMYY
find $pd -maxdepth 1 -name '*.csv' -exec cp -p {} $pd/backup_DDMMYY/ \;
find $od -maxdepth 1 -name '*.csv' -exec cp -p {} $od/backup_DDMMYY/ \;
```

**Backups are safe inside `processed_data/` and `outputs/`** — the only
recursive globs in the codebase are `aggregate.py`'s `pdir.glob("**/trials.csv")`
and they run on the *participants* tree. **Never put a backup anywhere under
`Data/participants_data/`** — a stray `trials.csv` there gets aggregated and
silently double-counts rows.

Then add an entry to [`docs/data_log.md`](docs/data_log.md).

### 2. Outlier pipeline — free, idempotent

```bash
python -m analysis.outlier_pipeline.run --condition aigen_perc   # or no flag for all 9
```

Structure check → reconstruction from `data.txt` (fallback only) → aggregation →
exclusion gates. Writes `all_*.csv`, `summary_by_uid.csv`,
`trials_final(_pregen).csv`, and the reports under `outliers/`.

### 3. AI-usage judges — PAID, cached, then rebuild

```bash
python -m analysis.outlier_pipeline.ai_usage_suspicion.consensus --condition aigen_perc
python -m analysis.outlier_pipeline.build_trials_final          --condition aigen_perc
```

**The rebuild is not optional.** Step 2 applies the AI gate using whatever
`ai_suspicion_scores.csv` already exists, so new participants' prompts are
unscored and default to "not AI". Only after `consensus` + a second
`build_trials_final` are the exclusions correct. The tell that you forgot: the
`ai_dropped=` counts are identical to the previous run.

Cached by (model + rubric hash + prompt hash) in `_judge_cache.json`, so a rerun
only bills genuinely new prompts. `--limit N` caps unique prompts;
`--report-only` rebuilds CSVs with no API calls.

**Sanity check the judges** — `score_dataframe` swallows per-row exceptions as
`pd.NA`, so a partly-dead judge still exits 0:

```python
import pandas as pd
d = pd.read_csv("analysis/outputs/Full_experiment/<gen>/<task>/outliers/ai_usage/ai_suspicion_scores.csv")
print({j: int(d[f"{j}_score"].isna().sum()) for j in ("gpt", "gemini", "claude")})  # want all 0
```
Also confirm the three per-model call counts in the printed spend table match.

### 4. Offline image generation — PAID, nogen + plain only

```bash
python analysis/generate_images_by_prompt.py nogen_perc
```

Reads `trials_final_pregen.csv`, generates one PNG per kept row, writes
`trials_final.csv`. Only calls the API for rows that survived the gates;
excluded rows get a deterministic filename but no image. Resumable — re-scans
disk and only generates what is missing. **aigen skips this step** (its images
were saved during the session).

### 5. Similarity — free (local GPU), full rebuild

```bash
python analysis/add_similarity_scores.py --condition aigen_perc
```

`trials_final.csv` → `trials_final_sim.csv`. Not incremental: recomputes every
row. No API cost, just GPU time. See [`similarity/README.md`](similarity/README.md).

### 6. Semantic tagging + object validation — PAID, resumable

```bash
python analysis/nlp_analysis/semantic_tagging.py        --condition all
python analysis/nlp_analysis/object_accuracy_detector.py --condition all
```

Both resume by row key: rows already in the output are never re-tagged or
re-billed, so after new participants arrive these only cost the new rows. See
[`analysis/nlp_analysis/README.md`](analysis/nlp_analysis/README.md).

### 7. Cross-condition exports and figures

```bash
python analysis/aggregate_conditions.py --all --sim   # <gen>_by_task, <task>_by_gen, all_conditions
python analysis/export_full_data.py                   # combined/full_data/ — for R / LMM
python analysis/computing_RDMS.py
python analysis/visualize_per_ppt.py --condition aigen_perc
```
Then re-run the notebooks in `analysis/notebooks/`.

---

## Staleness — the main hazard

**Nothing downstream of `trials_final.csv` invalidates itself, and nothing
warns you.** Rerun the outlier pipeline and `trials_final_sim.csv` plus every
file in `nlp_analysis/` keeps its *old* row count and its *old* participant set.
The notebooks read `trials_final_sim.csv`, not `trials_final.csv` — so an
analysis will happily report the previous N with no error anywhere.

**The trap:** for nogen and plain, `trials_final.csv` is itself downstream —
step 3 writes only `trials_final_pregen.csv`, and `trials_final.csv` doesn't
move until step 4 generates the images. So comparing the downstream files to
`trials_final.csv` gives a **false OK** for those conditions: everything is
uniformly old together. The reference row count is `trials_final_pregen.csv`
where it exists, `trials_final.csv` otherwise.

```python
import pandas as pd, config

DOWNSTREAM = ["trials_final.csv", "trials_final_sim.csv",
              "nlp_analysis/trials_final_semantic_tags.csv",
              "nlp_analysis/trials_final_semantic_tag_image_validation.csv"]

for c in config.CONDITIONS:
    d = config.paths_for(c).processed_dir
    # nogen/plain: the pipeline's real output is pregen, not trials_final
    ref_file = "trials_final_pregen.csv" if (d / "trials_final_pregen.csv").exists() else "trials_final.csv"
    ref = len(pd.read_csv(d / ref_file))
    counts = {f: len(pd.read_csv(d / f)) for f in DOWNSTREAM if (d / f).exists()}
    stale = [f for f, n in counts.items() if n != ref]
    print(f"{c:12} {ref_file:24} ref={ref:4}", "OK" if not stale else f"STALE: {stale}")
```

Equal row counts against the reference is the invariant. Dated `backup_*/`
folders give you the previous state to diff against.

---

## Cost and idempotency

| step | script | cost | rerun behaviour |
|---|---|---|---|
| 1 structure + aggregate + gates | `outlier_pipeline.run` | free | idempotent, full rebuild |
| 2 AI judges | `ai_usage_suspicion.consensus` | **paid** ~$0.10–0.45/condition | cached per prompt; only new prompts billed |
| 3 rebuild gates | `build_trials_final` | free | idempotent |
| 4 offline images | `generate_images_by_prompt` | **paid** per image | resumable; only missing PNGs, only kept rows |
| 5 similarity | `add_similarity_scores` | free (GPU) | **full recompute every time** |
| 6 semantic tags | `semantic_tagging` | **paid** | resumable by row key |
| 6 object validation | `object_accuracy_detector` | **paid** | resumable by row key |
| 7 exports/figures | various | free | idempotent |

Paid steps need `.env` at project root with `OPENAI_API_KEY`, `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`.

---

## CSV catalogue

### `Data/processed_data/<DATASET>/<gen>/<task>/`

| file | written by | one row is | read by |
|---|---|---|---|
| `all_trials.csv` | `aggregate.py` | one attempt, all participants, pre-exclusion | gates, AI judges, `generate_images_by_prompt` |
| `all_participants.csv` | `aggregate.py` | one participant (demographics, session meta) | reports |
| `all_digit_span.csv` | `aggregate.py` | one digit-span try (delay conditions only) | digit-span gate + metrics |
| `summary_by_uid.csv` | `aggregate.py` | one participant, counts per session | eyeballing |
| `trials_final_pregen.csv` | `build_trials_final` | kept attempt, `gen` still placeholder (nogen/plain only) | `generate_images_by_prompt` |
| **`trials_final.csv`** | `build_trials_final` (aigen) / `generate_images_by_prompt` (nogen, plain) | **kept attempt of a usable session of a non-excluded participant** | everything downstream |
| `trials_final_sim.csv` | `add_similarity_scores` | kept attempt + similarity columns | notebooks, RDMs, panels, `export_full_data` |
| `nlp_analysis/trials_final_semantic_tags.csv` | `semantic_tagging` | kept attempt + semantic fields | validation, notebooks |
| `nlp_analysis/trials_final_semantic_tag_image_validation.csv` | `object_accuracy_detector` | kept attempt + per-object image validation | notebooks, `export_full_data` |
| `nlp_analysis/..._executive_summary.csv` | `object_accuracy_detector` | aggregate validation summary | reporting |

### `analysis/outputs/<DATASET>/<gen>/<task>/outliers/`

| file | one row is |
|---|---|
| `outlier_report_summary.csv` | full/partial/unusable counts for the condition |
| `outlier_report_participants.csv` | one JATOS folder: status, missing files, what was reconstructed |
| `exclusion_report_sessions.csv` | one (uid, session): every gate column + `usable` |
| `exclusion_report_participants.csv` | one uid: sessions passing each gate, `usable_sessions`, `excluded` |
| `ai_usage/ai_suspicion_scores.csv` | one trial: 3 judge scores + explanations, `n_judges_flagged`, `ai_suspected` |
| `ai_usage/ai_suspicion_summary.csv` | condition-level counts |
| `ai_usage/ai_suspicion_by_participant.csv` | one uid: flagged/excluded attempts and sessions |
| `ai_usage/_judge_cache.json` | judge-call cache — **do not delete**, it is the money |
| `digit_span/digit_span_performance.csv` | one (uid, session): exact-match mean, positional accuracy, digit recall |
| `digit_span/digit_span_try_counts.csv` | one (uid, session): number of tries |

### `Data/processed_data/<DATASET>/combined/`

Cross-condition bundles from `aggregate_conditions.py`
(`<gen>_by_task.csv`, `<task>_by_gen.csv`, `all_conditions.csv`) and
`export_full_data.py` (`combined/full_data/*_full_data.csv`, for R/LMM).

---

## Exclusion gates (summary)

A session is **usable** only if it passes all four; a participant is **excluded
entirely** below `MIN_USABLE_SESSIONS = 3` usable sessions.

| gate | rule |
|---|---|
| `is_full_session` | has the condition's expected attempts (`attempts` in `condition_maps.yaml`: 3 aigen/nogen, 1 plain) |
| `is_short_session` | no attempt under `MIN_WORDS = 8` words (pure word count, no LLM) |
| `is_digitspan_failed` | delay only: `exact_match_mean >= 0.15` over `>= 15` tries; **missing record = failed** |
| `is_ai_session` | no attempt with `ai_suspected` (≥2 of 3 judges ≥80). **One flagged attempt drops the whole session.** |

Full detail, constants, and per-condition expectations:
[`analysis/outlier_pipeline/README.md`](analysis/outlier_pipeline/README.md).

---

## Conventions

- **Never write to `Data/participants_data/`.** It is the scientific source of
  truth. `generate_images_by_prompt` adds PNGs to the JATOS `files/` folders but
  never rewrites a raw `trials.csv`.
- **Notebooks read `trials_final.csv` / `trials_final_sim.csv`**, never
  `all_trials.csv` and never a `_pregen` file.
- **Add a condition** by adding a row to `CONDITIONS` in `condition_maps.yaml`;
  paths and expectations follow from the slug. Nothing hardcodes condition names.
- **`CURRENT_CONDITION`** in the YAML only drives legacy single-condition
  globals. Prefer `--condition` flags and `config.paths_for(slug)`.
