# Outlier pipeline

Turns raw JATOS exports into a clean `trials_final.csv` per condition, by
classifying participant folders, aggregating their CSVs, and applying four
exclusion gates (structure, short answers, digit-span performance, AI-usage
suspicion). Everything is keyed by a condition slug (e.g. `aigen_perc`,
`aigen_imm`) resolved through `config.paths_for(slug)`.

## Folder map (per condition)

| location | contents |
|---|---|
| `Data/<experiment>/jatos_results_files_*/` | raw JATOS export: `study_result_*/comp-result_*/files/` with `participants.csv`, `trials.csv`, (`digit_span.csv`), images, and a `data.txt` fallback log |
| `Data/processed_data/<condition>/` | aggregated CSVs: `all_trials.csv`, `all_participants.csv`, `all_digit_span.csv`, `summary_by_uid.csv`, **`trials_final.csv`** |
| `analysis/outputs/<condition>/outliers/` | all reports: `outlier_report_*.csv`, `exclusion_report_*.csv`, `digit_span/`, `ai_usage/` |

`<condition>` maps to a nested path like `Full_experiment/aigen/perc` (see
`condition_maps.yaml`; `config.paths_for` resolves it).

## The stages, in order

### 1. Structure check + reconstruction (`report.py`, `structure_check.py`, `convert_data_txt.py`)
Walks every `study_result_*/comp-result_*` folder and classifies it
**full / partial / unusable** based on required files (`participants.csv`,
`trials.csv`, plus `digit_span.csv` for delay conditions) and the presence of
images. Reconstruction from `data.txt` is a **fallback only**: it is attempted
only when required CSVs are missing AND a `data.txt` exists in that folder
(it is a raw concatenation of JSON objects — no separators — parsed with
`json.JSONDecoder.raw_decode` in a loop). Folders whose CSVs are present are
never touched, and a folder missing both its CSVs and `data.txt` stays
partial/unusable.
Writes `outliers/outlier_report_summary.csv` (full/partial/unusable counts) and
`outliers/outlier_report_participants.csv` (one row per folder: status, missing
files, what was reconstructed, full-session counts).

### 2. Aggregation (`analysis/aggregate.py` — outside this package)
Concatenates every participant's `trials.csv` / `participants.csv` /
`digit_span.csv` into the condition-level `all_trials.csv`,
`all_participants.csv`, `all_digit_span.csv` (+ `summary_by_uid.csv`),
tagging each row with its JATOS folder for provenance.

### 3. AI-usage suspicion (`ai_usage_suspicion/`) — separate, paid step
Three LLM judges (GPT `gpt-5.4-mini`, Gemini `gemini-2.5-flash`, Claude
`claude-haiku-4-5`) score every unique prompt 0-100 for "was this pasted from a
chatbot rather than typed by the participant", using one shared rubric
(`common.SYSTEM_PROMPT`, calibrated against a labeled gold set — see
`prompt_variants.py` / `eval_prompts.py` / `eval_data/`). A trial is
**`ai_suspected`** when **>= 2 of 3 judges score >= 80**
(`common.MIN_AGREEMENT` / `common.THRESHOLD`).

```bash
python -m analysis.outlier_pipeline.ai_usage_suspicion.consensus --condition aigen_imm
```

- **Cost**: ~3 calls per unique prompt (~$1.30 / 370 prompts). NOT run by the
  main pipeline — run it manually when `all_trials.csv` changes.
- **Resumable**: every judge result is checkpointed to `_judge_cache.json`
  (keyed by model + rubric hash + prompt hash); an interrupted run re-run with
  the same command only pays for what is missing. Changing the rubric
  invalidates the cache automatically.
- `--limit N` scores only the first N unique prompts (cost guard);
  `--report-only` rebuilds the derived CSVs from saved scores with no API calls.

Writes to `outliers/ai_usage/`:
- `ai_suspicion_scores.csv` — per trial: uid/session/attempt/prompt, each judge's
  score + one-sentence explanation, `n_judges_flagged`, `ai_suspected`
- `ai_suspicion_summary.csv` — trials scored / suspected, sessions implicated
- `ai_suspicion_by_participant.csv` — per uid: flagged attempts, excluded
  attempts (whole flagged sessions), flagged sessions

### 4. Exclusions -> trials_final (`build_trials_final.py`, `exclusions.py`)
Builds a per-session gate table and keeps only clean data. A session is
**usable** only if it passes ALL gates:

| gate | source | rule |
|---|---|---|
| `is_full_session` | `session_summary.py` | has all `REQUIRED_ATTEMPTS = 3` attempts |
| `is_short_session` | `prompt_quality.py` | no attempt under `MIN_WORDS = 8` words |
| `is_digitspan_failed` | `digit_span_metrics.py` (delay conditions only) | session recall `exact_match_mean >= 0.15` over `>= 15` tries; missing record = failed |
| `is_ai_session` | `outliers/ai_usage/ai_suspicion_scores.csv` | no attempt with `ai_suspected` — **one flagged attempt drops the whole session (all 3 attempts)** |

A participant is **excluded entirely** when they have fewer than
`MIN_USABLE_SESSIONS = 3` usable sessions. `trials_final.csv` keeps only usable
sessions of non-excluded participants.

**How the short-prompt gate is scored** (`prompt_quality.py`): each attempt's
`prompt` is whitespace-split and its words counted; an attempt with fewer than
`MIN_WORDS = 8` words is a short attempt. The session gate is then *any*: one
short attempt anywhere in the session marks the whole session
`is_short_session` and it is dropped. There is no LLM or semantic judgment
here — it is a pure word-count floor for prompts too short to be a real
description.

**How digit-span pass/fail is decided** (`digit_span_metrics.py`, delay
conditions only): the input is the aggregated `all_digit_span.csv` (one row per
digit-span try, with `presented_sequence` and `participant_response`).
Responses are cleaned (numeric strings normalized, e.g. `"123.0"` -> `"123"`),
then each try gets `is_exact_match` = 1 only when the response equals the
presented sequence exactly. Per (uid, session), `session_pass_fail` computes
`exact_match_mean` (share of perfectly recalled tries) and `num_tries`; the
session passes (`session_ok`) only when **`exact_match_mean >= 0.15` AND
`num_tries >= 15`**. In `exclusions.session_table` a session with no digit-span
record at all is treated as **failed** (recall cannot be verified). Two softer
accuracy measures are computed for reporting only (not gating): positional
accuracy (digits correct in the right position) and digit recall (correct
digits regardless of order); they appear in
`outliers/digit_span/digit_span_performance.csv`, with per-session try counts
in `digit_span_try_counts.csv` and a per-participant accuracy-by-length plot in
`digit_span_accuracy_by_length.png`.

The AI gate is **optional-input**: if `ai_suspicion_scores.csv` doesn't exist,
`build_trials_final` prints a skip warning and applies the other gates only
(same pattern as digit-span for non-delay conditions). Free and idempotent —
rerun any time.

Writes: `trials_final.csv` (processed_data),
`outliers/exclusion_report_sessions.csv` (every session, every gate column,
`usable`), `outliers/exclusion_report_participants.csv` (per uid: sessions
passing each gate — `full_sessions_structural`, `good_wordcount_sessions`,
`good_digitspan_sessions`, `good_nonai_sessions` — plus `usable_sessions`,
`excluded`), and for delay conditions `outliers/digit_span/` performance
reports + accuracy plot.

## How to run

```bash
# Full pipeline (stages 1, 2, 4) for one condition or all:
python -m analysis.outlier_pipeline.run --condition aigen_perc
python -m analysis.outlier_pipeline.run

# AI scoring (stage 3) — manual, paid, per condition, BEFORE the build if you
# want the AI gate applied:
python -m analysis.outlier_pipeline.ai_usage_suspicion.consensus --condition aigen_perc

# Rebuild trials_final.csv + exclusion reports only (free, no API):
python -m analysis.outlier_pipeline.build_trials_final --condition aigen_perc
```

Typical order for a new/updated condition:
**run (or aggregate) -> consensus -> build_trials_final.** Console output of the
build shows the impact per gate:
`sessions usable: N/M (full=..., short_answer_dropped=..., digitspan_dropped=..., ai_dropped=...)`.

## Constants worth knowing

| constant | value | where |
|---|---|---|
| `REQUIRED_ATTEMPTS` | 3 | `session_summary.py` |
| `MIN_WORDS` | 8 | `prompt_quality.py` |
| `MIN_ACCURACY`, `MIN_TRIALS` | 0.15, 15 | `digit_span_metrics.py` |
| `THRESHOLD`, `MIN_AGREEMENT` | 80, 2-of-3 | `ai_usage_suspicion/common.py` |
| `MIN_USABLE_SESSIONS` | 3 | `exclusions.py` |
| `EXPECTED_SESSIONS` | 5 | `report.py` |

## AI-judge calibration harness (`ai_usage_suspicion/`)
Not part of the pipeline — tooling used to tune the judge rubric, kept for
future recalibration:
- `eval_data/gpt_descs.csv` / `human_descs.csv` — labeled gold set (GPT-5 vs
  real participant descriptions of the same 5 scenes)
- `prompt_variants.py` — candidate rubrics (`baseline` = pre-calibration prompt,
  frozen; `v1` = shipping prompt)
- `eval_prompts.py` — scores gold set variants with all 3 judges (dev samples +
  held-out test split, cached)
- `export_results.py` — dumps per-prompt scores + rubric text to
  `eval_data/results_report.md` / `results_scores.csv` (no API calls)
- `calibration_probe.py` — quick hand-labeled sanity probe

API keys: `.env` at project root must define `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (loaded by `common.py` via dotenv).
