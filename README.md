---
title: Imagination In Translation
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: An interactive experiment
license: mit
---

# Imagination in translation

An interactive experiment studying the gap between semantic and visual
representations in humans and AI.

## Project description

The project investigates whether people can convey a mental image to an AI
through language alone. Participants describe a ground-truth image in words; an
image model generates an image from that description. Depending on condition,
participants then see feedback and refine their description across further
attempts. All descriptions, generated images, similarity scores and model
parameters are logged for later analysis of language-to-vision alignment.

## Experimental design

Nine conditions — 3 generation types × 3 memory tasks — declared in
[`condition_maps.yaml`](condition_maps.yaml) and addressed by **slug**
(`<gen>_<task>`, e.g. `aigen_perc`).

| generation | feedback | attempts/session | saves images during session |
|---|---|---|---|
| `aigen` | generated image | 3 | yes |
| `nogen` | text | 3 | no — generated post-hoc |
| `plain` | none | 1 | no |

| task | when the description is produced |
|---|---|
| `perc` | perception — image visible |
| `imm` | immediate memory |
| `del` | delayed memory (with digit-span distractor) |

## Documentation

| doc | covers |
|---|---|
| **[`PIPELINE.md`](PIPELINE.md)** | **start here** — run order, CSV catalogue, costs, the staleness hazard |
| [`docs/data_log.md`](docs/data_log.md) | what has actually been run: dates, Ns, exclusions, spend |
| [`analysis/README.md`](analysis/README.md) | map of the analysis scripts |
| [`analysis/outlier_pipeline/README.md`](analysis/outlier_pipeline/README.md) | exclusion gates, AI-usage judges |
| [`analysis/nlp_analysis/README.md`](analysis/nlp_analysis/README.md) | semantic tagging, ground-truth image validation |
| [`similarity/README.md`](similarity/README.md) | CLIP / VGG16 / LPIPS similarity scores |

Returning after a break, or new participant data has arrived? Go to the
"new batch of participants arrived" runbook in [`PIPELINE.md`](PIPELINE.md).

## Repository layout

```
app.py, src/            Streamlit experiment app
config.py               condition slugs -> paths (paths_for, spec_for, load)
condition_maps.yaml     what conditions exist + canonical filenames
Data/
  participants_data/    raw JATOS exports — source of truth, never written to
  processed_data/       aggregated + analysis CSVs
GT_images/              ground-truth stimuli
analysis/               pipeline, NLP analysis, notebooks
  outputs/              reports, panels, RDMs, figures
similarity/             CLIP / VGG / LPIPS wrappers
```

## Setup

Python 3.11 or 3.12.

```bash
git clone https://github.com/AnatKordon/Imagination_in_translation.git
cd Imagination_in_translation
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

Create a `.env` in the project root. The analysis pipeline's paid stages need:

```bash
OPENAI_API_KEY=...      # semantic tagging, object validation, one AI-usage judge
GEMINI_API_KEY=...      # AI-usage judge
ANTHROPIC_API_KEY=...   # AI-usage judge
```

Run the experiment app:

```bash
streamlit run app.py
```

## Analysis quick start

```bash
# 1-3: outlier pipeline (stage 2 is paid, and the rebuild after it is required)
python -m analysis.outlier_pipeline.run
python -m analysis.outlier_pipeline.ai_usage_suspicion.consensus --condition aigen_perc
python -m analysis.outlier_pipeline.build_trials_final           --condition aigen_perc

# 4-6: downstream (per condition)
python analysis/generate_images_by_prompt.py nogen_perc     # nogen/plain only, paid
python analysis/add_similarity_scores.py --condition aigen_perc
python analysis/nlp_analysis/semantic_tagging.py         --condition all   # paid
python analysis/nlp_analysis/object_accuracy_detector.py --condition all   # paid
```

Read [`PIPELINE.md`](PIPELINE.md) before running any of this — the ordering
constraints and the silent-staleness problem are documented there.

## License

MIT — see [`LICENSE`](LICENSE).

## Authors

Anat Korol Gordon, Itai Peleg, Maayan Shirizly, Nataliya Kalanova,
Sivan Flomen, Yaniv Kopelman

## Contact

Questions, suggestions or bug reports: **anat.korol@gmail.com** (Anat Korol Gordon).
