# NLP analysis

Turns each kept description into structured semantic content, then checks that
content against the ground-truth image. Two paid LLM stages, both resumable,
both keyed by condition slug through `config.paths_for(slug)`.

Runs **after** the outlier pipeline: input is `trials_final.csv`, so only kept
attempts of usable sessions of non-excluded participants are ever tagged or
billed. Pipeline order: [`../../PIPELINE.md`](../../PIPELINE.md).

```
trials_final.csv
      │  semantic_tagging.py                    [PAID, resumable]
      ▼
nlp_analysis/trials_final_semantic_tags.csv
      │  object_accuracy_detector.py            [PAID, resumable]
      ▼
nlp_analysis/trials_final_semantic_tag_image_validation.csv
              + ..._executive_summary.csv
```

## 1. Semantic tagging (`semantic_tagging.py`)

Extracts six categories from each `prompt`, as JSON, expanded into adjacent
DataFrame columns. All values are arrays of lowercase strings, `[]` when absent,
never null, deduplicated within a category.

| field | what it holds |
|---|---|
| `objects` | "things" — bounded, cohesive entities, incl. room/architectural features and explicitly named parts |
| `stuff` | matter and visual phenomena without cohesive bounds (sky, grass, light) |
| `scene_category` | the scene type as explicitly stated ("bedroom", "room") |
| `spatial_relations` | explicitly described spatial arrangement |
| `attr_color` | explicit color phrases only |
| `adjectives` | modifier-like descriptions, paired with their referent ("large table") |

The rubric is deliberately strict and literal: **nothing is inferred.** Likely
objects, hidden objects, implied context, and common-sense completions are not
extracted, and explicit negations ("no clouds", "without a door") suppress
extraction of that entity in every category. Compound names stay whole
("coffee table" is one object) unless the participant describes a part relation.
This matters when interpreting counts — they measure what was *said*, not what
was *there*.

```bash
python analysis/nlp_analysis/semantic_tagging.py --condition all
python analysis/nlp_analysis/semantic_tagging.py --condition aigen_perc nogen_imm
```

- Model: `DEFAULT_MODEL = "gpt-5.5"` — chosen after comparing `gpt-5.4-mini`,
  `gpt-5.5`, `claude-sonnet-5`, `claude-opus-4-8`; gpt-5.5 adhered to the rubric
  best. `gpt-5.4-mini` is the fast/cheap fallback.
- Output: `<processed_dir>/nlp_analysis/trials_final_semantic_tags.csv`
- **Resumable by row key**: rows already present in the output are never
  re-tagged or re-billed, and progress is checkpointed every
  `CHECKPOINT_EVERY` rows. After new participants arrive this only costs the new
  rows. `MAX_NEW_ROWS` caps a single run.

Setting `RUN_EXPERIMENT` switches the script into model-comparison mode: it
samples `EXPERIMENT_N` rows, tags them with every model in `EXPERIMENT_MODELS`,
and writes a by-field comparison via `compare_model_experiments.py`. That path
writes to `analysis/outputs/experiments/`, not to any condition — it never
touches pipeline data.

## 2. Object / image validation (`object_accuracy_detector.py`)

A vision judge sees the **ground-truth image** plus the tags extracted from the
participant's description of it, and answers one strictly-scoped question per
tag: `in_image` true/false — does the image visually support this claim? It is a
verifier, not a re-tagger; it cannot add tags, only confirm or reject.

This is what separates *what the participant said* from *what was actually
there* — the accuracy half of the semantic analysis.

```bash
python analysis/nlp_analysis/object_accuracy_detector.py --condition all
```

- Model: `DEFAULT_MODEL = "gpt-5.4-mini"`, `VERBOSITY = "low"`, structured
  output enforced by `TAG_IMAGE_VALIDATION_SCHEMA`.
- Output: `trials_final_semantic_tag_image_validation.csv` beside the tags, plus
  `..._executive_summary.csv`.
- **Resumable** the same way — already-validated rows are skipped.
- `--condition gpt-5.5_desc` validates the **GPT ceiling baseline** instead of a
  participant condition: GPT's own descriptions of the same ground-truth images,
  tagged and validated identically, giving an upper bound to compare human
  performance against. It lives in
  `Data/processed_data/Full_experiment/combined/gpt-5.5_desc/` and uses
  verbosity-suffixed filenames (`..._verbosity-medium.csv`). The older
  `gpt_ceiling/` folder is the first run of the same thing, still read as a
  fallback.

## Supporting scripts

| script | purpose |
|---|---|
| `gpt_image_desc_api.py` (in `analysis/`) | generates the GPT baseline descriptions of the GT images, in `trials_final.csv`'s column scheme |
| `drawings_descriptions.py` | describes the 2019 Wilma drawings, with and without a category hint, into separate trees combined per arm |
| `compare_model_experiments.py` | builds the by-field model comparison from `semantic_tags__*.csv` in the experiments folder |
| `qa_additions_in_attempts.py` | QA of what participants added between attempts |
| `dsg_tagging.py` | alternative DSG-style tagging → `ppt_w_gpt_dsg_tagging.csv` |
| `pos_tagging.py`, `demo_pos.py`, `brysbaert/` | POS tagging and Brysbaert concreteness norms |
| `correct_text.py` | spelling/typo normalisation |

Several of these still point at hardcoded absolute paths or legacy pilot CSVs
rather than `config.paths_for` — check the top of the file before assuming a
script is condition-aware.

## Notebooks

`comparing_conditions.ipynb`, `within_gen_semantic_counts.ipynb`,
`hit_rate_analyses.ipynb`, `feedback_vs_no_feedback.ipynb`,
`corrected_objects_analysis.ipynb`, `describing_vs_drawings_comparison.ipynb`
read the two CSVs above via `SEM_FILE` / `VAL_FILE` constants near the top.

**They do not check freshness.** If the outlier pipeline has been rerun since
the last tagging pass, these files still hold the previous participant set and
the notebooks will report the old N without warning. Run the staleness check in
[`../../PIPELINE.md`](../../PIPELINE.md) first.

## Cost control

Both stages need `OPENAI_API_KEY` in `.env` at the project root. Both resume by
row key, so the safe habit is: run, interrupt freely, re-run — you only ever pay
for untagged rows. To estimate before spending, diff the row count of
`trials_final.csv` against the existing output.
