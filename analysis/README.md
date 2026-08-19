# analysis/

Everything downstream of the raw JATOS exports. Start with
[`../PIPELINE.md`](../PIPELINE.md) for the run order and the CSV catalogue; this
file is just a map of what lives here.

## Packages

| package | what it does |
|---|---|
| [`outlier_pipeline/`](outlier_pipeline/README.md) | structure check, aggregation, exclusion gates, AI-usage judges → `trials_final.csv` |
| [`nlp_analysis/`](nlp_analysis/README.md) | semantic tagging + ground-truth image validation |
| [`../similarity/`](../similarity/README.md) | CLIP / VGG / LPIPS model wrappers (project root, not here) |

## Pipeline scripts

| script | in → out |
|---|---|
| `aggregate.py` | per-participant CSVs → `all_trials.csv`, `all_participants.csv`, `all_digit_span.csv`, `summary_by_uid.csv`. Called by `outlier_pipeline.run`; rarely run alone. |
| `generate_images_by_prompt.py` | `trials_final_pregen.csv` → PNGs + `trials_final.csv`. **nogen/plain only**, PAID, resumable, kept rows only. Never rewrites a raw `trials.csv`. |
| `add_similarity_scores.py` | `trials_final.csv` → `trials_final_sim.csv`. Free (GPU), full recompute. |
| `gpt_image_desc_api.py` | GPT descriptions of the GT images — the ceiling baseline, in `trials_final.csv`'s column scheme. |

## Cross-condition outputs

| script | produces |
|---|---|
| `aggregate_conditions.py` | `<gen>_by_task.csv`, `<task>_by_gen.csv`, `all_conditions.csv` in `processed_data/<DATASET>/combined/`. `--sim` uses `trials_final_sim`. For in-memory work use `config.load([slugs], sim=True)` instead. |
| `export_full_data.py` | `combined/full_data/*_full_data.csv` — all 9 conditions with `generation` + `delay` columns, for R / LMM. Digit span combined across generations (delay conditions only). |
| `computing_RDMS.py` | representational dissimilarity matrices from `trials_final_sim.csv`. |
| `visualize_per_ppt.py` | per-participant GT-vs-generation panels. Also exports `path_from_row_jatos`, which reconstructs a generated image's path from the `gen` column — used by `add_similarity_scores.py`. |

## Outputs live outside the code

Results go to `analysis/outputs/<DATASET>/<gen>/<task>/` (reports, panels, RDMs,
graphs) and `analysis/outputs/<DATASET>/combined/`. Canonical CSVs go to
`Data/processed_data/...`. Nothing writes results next to the scripts.

## Other files

`notebooks/` holds the analysis notebooks; `digit_span/` and `experiments/`
hold sub-analyses; `flux_gen.py`, `gpt_gen-old.py`, `shuffle.py`,
`aggregate_gpt*.py`, `wilmas_labelme_gt_objects.py` are one-off or legacy
helpers. `data_preparation.ipynb` and `plotting.ipynb` are working notebooks,
not part of the pipeline.

Before reusing any of the one-off scripts, check the top of the file — several
predate `config.paths_for` and still carry hardcoded absolute paths or point at
legacy pilot CSVs.

## Conventions

- Address conditions by **slug** (`aigen_perc`) and resolve paths with
  `config.paths_for(slug)`. Don't hardcode folder names.
- `CURRENT_CONDITION` in `condition_maps.yaml` only feeds the legacy
  single-condition globals in `config.py`; prefer `--condition` flags.
- Analysis code reads `trials_final.csv` / `trials_final_sim.csv` — never
  `all_trials.csv`, never a `_pregen` file.
