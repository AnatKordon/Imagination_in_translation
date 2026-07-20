"""Offline gen-image generation for the `nogen` conditions.

The nogen (verbal-feedback, no-generation) data was collected without producing
an AI image per prompt: every raw trials.csv has gen == "VERBAL_FEEDBACK_NO_GEN".
This script generates one image per prompt using parameters IDENTICAL to the
online Cloudflare worker, saves each PNG into the correct JATOS `files/` folder
using the aigen naming convention, and records the filename in the *derived*
all_trials.csv so it flows into the rest of the pipeline.

Design guarantees:
  * Raw per-participant trials.csv are NEVER written (scientific source of truth).
    Only the derived processed_data/.../all_trials.csv gen column is overwritten.
  * Idempotent / resumable: re-scans the tree each run and only calls the API for
    PNGs missing on disk, so participants added later are filled on a re-run.

Only kept data is generated: the API is called solely for (uid, session) pairs
that survived the outlier pipeline, i.e. that are present in trials_final.csv.
Excluded participants/sessions still get their deterministic filename written to
all_trials.csv, but no image is paid for.

Pipeline order (gen step must run AFTER aggregation + exclusions):
    aggregate.py          ->  all_trials.csv          (gen = placeholder, all participants)
    outlier_pipeline.run  ->  trials_final_pregen.csv (gen = placeholder, kept rows only)
    this script           ->  PNGs into files/, fills gen in all_trials.csv, and
                              writes trials_final.csv = pregen + real gen filenames

So trials_final.csv is only ever created once the images exist — notebooks keep
reading trials_final.csv and never see a placeholder gen.

Usage:
    python analysis/generate_images_by_prompt.py                # default: the 3 nogen conditions
    python analysis/generate_images_by_prompt.py plain          # all 3 plain conditions
    python analysis/generate_images_by_prompt.py plain_perc plain_imm plain_del
"""

import os
import sys
import base64
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# --- CONFIGURATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default set to process when no condition is given on the command line.
# Each condition has its own JATOS tree + all_trials.csv + trials_final.csv.
DEFAULT_CONDITIONS = ["nogen_imm", "nogen_perc", "nogen_del"]

# Generation parameters — MUST match the online Cloudflare worker exactly:
#   model "gpt-image-2", size "1024x1024", quality "medium", n 1, output_format "png".
GEN_MODEL = "gpt-image-2"
GEN_SIZE = "1024x1024"
GEN_QUALITY = "medium"
GEN_OUTPUT_FORMAT = "png"

SAVE_EVERY = 5  # periodic interim save of all_trials.csv


def build_gpt_image_filename(uid: str, session: int, attempt: int) -> str:
    """Constructs filename: uv0vccs51773064211294_session01_attempt01.png"""
    return f"{uid}_session{session:02d}_attempt{attempt:02d}.png"


def build_out_path(row, participants_dir: Path, filename: str) -> Path:
    """Reconstruct the PNG target inside the raw JATOS tree.

    participants_dir already IS the globbed jatos_results_files_* directory, so:
      <jatos_dir>/study_result_<ID>/comp-result_<ID>/files/<filename>
    """
    study_result = str(row["study_result"]).strip()
    comp_result = str(row["comp_result"]).strip()
    return participants_dir / study_result / comp_result / "files" / filename


def generate_gpt_image_from_prompt(prompt: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = client.images.generate(
        model=GEN_MODEL,
        prompt=prompt,
        size=GEN_SIZE,
        quality=GEN_QUALITY,
        n=1,
        output_format=GEN_OUTPUT_FORMAT,
    )

    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    with open(out_path, "wb") as f:
        f.write(img_bytes)

    return getattr(resp.data[0], "revised_prompt", None)


def kept_path_for(paths) -> Path | None:
    """The post-exclusion table that defines what to generate.

    Normally trials_final_pregen.csv (written by the outlier pipeline). Older
    conditions whose trials_final.csv was built before the pregen hand-off existed
    fall back to it, so re-runs there stay resumable. None if neither exists.
    """
    pregen = paths.csv("trials_final_pregen")
    if pregen.exists():
        return pregen
    final = paths.csv("trials_final")
    return final if final.exists() else None


def write_trials_final(kept_df: pd.DataFrame, all_trials: pd.DataFrame, out_path: Path) -> None:
    """Copy the freshly filled gen / revised_prompt columns onto the kept rows.

    Column order and row set come from the pregen table; only the generation
    columns are updated, keyed on (uid, session, attempt).
    """
    keys = ["uid", "session", "attempt"]
    src = all_trials.set_index(keys)
    idx = pd.MultiIndex.from_frame(kept_df[keys])
    for col in ("gen", "revised_prompt"):
        if col in src.columns:
            kept_df[col] = src[col].reindex(idx).to_numpy()
    kept_df.to_csv(out_path, index=False)


def process_condition(condition: str):
    # aigen images were produced online and their filenames come from JATOS —
    # regenerating would overwrite real data with new images. Never touch them.
    if not config.spec_for(condition).offline_gen:
        print(f"⏭️  {condition}: online-generated condition, images already exist; skipping")
        return

    paths = config.paths_for(condition)
    participants_dir = paths.participants_dir
    all_trials_path = paths.processed_dir / "all_trials.csv"
    kept_path = kept_path_for(paths)

    if participants_dir is None or not Path(participants_dir).exists():
        print(f"⏭️  {condition}: participants dir not found ({participants_dir}); skipping")
        return
    if not all_trials_path.exists():
        print(f"⏭️  {condition}: {all_trials_path} not found — run aggregate.py first; skipping")
        return
    if kept_path is None:
        print(f"⏭️  {condition}: no {config.FILES['trials_final_pregen']} — run the outlier "
              f"pipeline (python -m analysis.outlier_pipeline.run --condition {condition}) "
              f"first; skipping")
        return

    print(f"\n📂 {condition}: reading {all_trials_path}")
    df = pd.read_csv(all_trials_path).copy()
    kept_df = pd.read_csv(kept_path)
    kept_pairs = {(str(u), int(s)) for u, s in zip(kept_df["uid"], kept_df["session"])}
    print(f"🚀 {condition}: {len(df)} rows, {len(kept_df)} kept rows / "
          f"{len(kept_pairs)} kept uid+session pairs in {kept_path.name}")

    generated = skipped = excluded = failed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=condition):
        # Deterministic filename — always recorded, even when the PNG already exists
        # or the row is excluded, so the column stays consistent across all_trials.
        filename = build_gpt_image_filename(row["uid"], int(row["session"]), int(row["attempt"]))
        df.at[idx, "gen"] = filename

        # Never pay the API for participants/sessions dropped by the outlier pipeline.
        if (str(row["uid"]), int(row["session"])) not in kept_pairs:
            excluded += 1
            continue

        try:
            out_path = build_out_path(row, Path(participants_dir), filename)

            # Resume logic: never pay the API to regenerate an image already on disk.
            if out_path.exists():
                skipped += 1
                continue

            prompt = str(row["prompt"])
            revised = generate_gpt_image_from_prompt(prompt, out_path)
            df.at[idx, "revised_prompt"] = revised
            generated += 1

            if generated % SAVE_EVERY == 0:
                df.to_csv(all_trials_path, index=False)

        except Exception as e:
            failed += 1
            print(f"❌ {condition} error at index {idx}: {e}")
            continue

    df.to_csv(all_trials_path, index=False)
    print(f"✅ {condition}: {generated} generated, {skipped} already on disk, "
          f"{excluded} excluded (not kept), {failed} failed. Updated {all_trials_path.name}")

    # The kept rows now have real filenames -> promote them to the analysis table.
    if failed:
        print(f"⚠️  {condition}: {failed} rows failed — re-run to fill them before "
              f"trusting {paths.csv('trials_final').name}")
    write_trials_final(kept_df, df, paths.csv("trials_final"))
    print(f"📝 {condition}: wrote {paths.csv('trials_final')} ({len(kept_df)} rows)")


def resolve_conditions(args) -> list:
    """Accept condition slugs ('plain_perc') and generation groups ('plain')."""
    if not args:
        return DEFAULT_CONDITIONS
    conditions = []
    for a in args:
        if a in config.GROUPS_BY_GEN:          # 'aigen' | 'nogen' | 'plain'
            conditions.extend(config.GROUPS_BY_GEN[a])
        elif a in config.CONDITIONS:
            conditions.append(a)
        else:
            raise SystemExit(f"unknown condition/group: {a}")
    return conditions


def main():
    for condition in resolve_conditions(sys.argv[1:]):
        process_condition(condition)


if __name__ == "__main__":
    main()
