# Superseded notebooks

Kept as a record of earlier analyses. None of these should be used for current results.

Two separate reasons they are here:

**Pre-date validator v2.** They read `n_in_image_*` / `n_not_in_image_*`, which answer only
*is this tag visible in the image*. Validator v2 separates that from whether the tag was
filed in the right category and whether the participant actually said it, so counts here
include extraction artefacts and hallucination figures overstate participant error.

**Run on pilot datasets.** `corrected_objects_analysis`, `feedback_vs_no_feedback` and
`hit_rate_analyses` read `pilot_2` tables (`ppt_trials_w_object_validation.csv`), not the
full experiment. Those files never had v2 columns, so there is nothing to migrate.

| notebook | reads | superseded by |
|---|---|---|
| `old_exploratory_hypotheses.ipynb` | full-experiment combined export | `../exploratory_hypotheses.ipynb` |
| `corrected_objects_analysis.ipynb` | pilot_2 datasets | `../full_data_comparisons.ipynb` |
| `feedback_vs_no_feedback.ipynb` | pilot_2 feedback/no-feedback | `../full_data_comparisons.ipynb` |
| `hit_rate_analyses.ipynb` | pilot_2 perception | `../cross_gen_validated_semantic_counts.ipynb` |

Current column meanings, for reference when reading old output:

| column | meaning |
|---|---|
| `n_extracted` | every tag the extractor produced |
| `n_in_image` | visible in the ground-truth image |
| `n_valid` | visible **and** right category **and** actually said |
| `n_scored` | `n_extracted - model_hallucination - category_misplacement` |

`n_scored == n_valid + n_participant_hallucination`
