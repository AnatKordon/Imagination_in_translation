# ⚠️ STALE — this folder was NOT updated on 2026-08-24

**33 rows here. The with_hint arm has 61. Do not compare the two on raw counts.**

## What happened

On 2026-08-24 the source folder `Data/other_datasets/wilmas_drawings_2019/perception_drawings`
was replaced with the complete release. Perception went from 16 subjects to 25, and the
drawings of our five GT scenes went from 33 to 61.

Those 28 new drawings were described into `with_hint/draw_perc` **only**, by decision
(Anat, 2026-08-24: "I decided to leave no_hint behind").

|                     | rows | subjects |
|---------------------|------|----------|
| `with_hint/draw_perc` | 61 | 25 |
| `no_hint/draw_perc` (here) | 33 | 16 |

`draw_imm` (75) and `draw_del` (35) are NOT affected — they are complete in both arms.
This is a perception-only problem.

## Why it matters

This arm exists for one purpose: to measure how much the scene hint inflates object
counts, by describing the same drawings blind. That only works if the two arms hold the
**same drawings**. They no longer do for perception.

Subtracting the arms' perception totals now mixes the hint's effect with a sample almost
twice the size. It would show a large hint effect that is mostly just extra participants.

## What to do

- **Comparing arms?** Restrict to the 33 `original_name` values present in both. An inner
  join on `original_name` does this correctly.
- **Only need hinted data?** Ignore this folder; use `with_hint/`.
- **Want the arms balanced again?** Run:

  ```
  python analysis/nlp_analysis/drawings_descriptions.py -c draw_perc --no-scene-hint
  ```

  The resume logic fills exactly the 28 missing rows (it will not re-describe the 33) and
  both arms end at 61. Then delete this file.

## Also stale because of this

`combined/full_data/drawing_semantic_tag_image_validation_full_data_no_hint.csv` still
holds the old 143 rows and has not been re-exported. Note that image validation has not
been run for the 28 new drawings in either arm.
