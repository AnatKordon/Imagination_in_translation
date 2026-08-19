# Similarity

Image and text embedding models used to score how close a participant's
generated image is to the ground-truth image they were describing. Pure model
wrappers — no pipeline logic, no paths, no condition awareness. The driver is
[`analysis/add_similarity_scores.py`](../analysis/add_similarity_scores.py).

Everything here runs **locally on GPU**. No API keys, no cost.

## Modules

| module | model | provides |
|---|---|---|
| `CLIP_similarity.py` | CLIP (512-d) | `get_clip_visual_embedding(path)`, `get_clip_text_embedding(text)` → `(emb, real_token_num)`, `cosine_similarity(a, b)` |
| `vgg_similarity.py` | VGG16 / ImageNet, layer `Classifier_4` (fc7) | `VGGEmbedder.get_embedding(img_path)`, `compute_similarity_score(e1, e2)` → `(similarity, scaled_similarity, cosine_distance)` |
| `LPIPS_similarity.py` | LPIPS (VGG backbone) | `compute_lpips_score(gt_path, gen_path)` → perceptual distance |
| `SGPT_embedder.py` | `Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit` | sentence embeddings for prompt-similarity work; **not** used by `add_similarity_scores.py` |

`compute_similarity_score` returns three things: raw cosine similarity in
[-1, 1], a `scaled_similarity` mapped to [0, 100] (this is the number that was
shown to participants in the app), and cosine distance. The analysis CSVs store
**distances** for image-image comparisons and **similarities** for the rest.

## What lands in `trials_final_sim.csv`

Written by `add_similarity_scores.py`, one row per kept attempt:

| column | meaning | direction |
|---|---|---|
| `clip_cosine_distance` | CLIP GT-image vs generated image | lower = more similar |
| `vgg_fc7_distance` | VGG16 fc7 GT-image vs generated image | lower = more similar |
| `clip_vis_text_similarity` | generated image vs the participant's prompt (CLIP cross-modal) | higher = better aligned |
| `clip_self_prev_similarity` | generated image at attempt *n* vs attempt *n-1*, same (uid, gt) | higher = less change between attempts |
| `token_num` | real CLIP token count of the prompt | — |

Three things to know about these columns:

- **`clip_self_prev_similarity` is a refinement measure, not a GT measure.** It
  is stored on the *later* attempt; the first attempt of each target is NaN by
  construction. It captures how much the image moved as the participant revised.
- **Rows with no ground truth get `pd.NA`** for the GT-based columns. The
  self-similarity loop still runs, because the generated embedding exists.
- **Prompts are truncated at 77 tokens** by CLIP's tokenizer. `token_num`
  records the true length so you can find and handle truncated prompts.

LPIPS is computed in the loop but currently not written to the CSV — the
assignment lines are commented out in `add_similarity_scores.py`. Uncomment
there if you want it.

## Running

```bash
python analysis/add_similarity_scores.py --condition aigen_perc
```

`trials_final.csv` → `trials_final_sim.csv`, per condition.

**Not incremental.** Every row is recomputed on every run, so new participants
mean a full rebuild of the condition. That is fine — it is GPU time, not money —
but it is not instant, and it means `trials_final_sim.csv` is always exactly as
fresh as its last run and never partially updated.

The generated-image path per row comes from `path_from_row_jatos(row,
paths.participants_dir)`, which reconstructs the JATOS `files/` path from the
`gen` column. So similarity **cannot** run for nogen/plain until
`generate_images_by_prompt.py` has produced the PNGs and written real filenames
into `trials_final.csv`. See [`../PIPELINE.md`](../PIPELINE.md).
