import os
import base64
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
CSV_PATH = Path("/mnt/hdd/anatkorol/Imagination_in_translation/Data/processed_data/comparing_conditions/3_conditions_with_digit_span.csv")
OUT_DIR = Path("/mnt/hdd/anatkorol/Imagination_in_translation/analysis/experiments/model_comparison")
N_ROWS = 10
MODELS = ["gpt-image-1", "gpt-image-1.5", "gpt-image-2"]
SEED = 42


def generate_image(prompt: str, model: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"    [skip] {out_path.name} already exists")
        return

    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    with open(out_path, "wb") as f:
        f.write(img_bytes)


def build_figure(sampled: pd.DataFrame) -> None:
    n_rows = len(sampled)
    n_models = len(MODELS)

    # Layout: prompt col (narrower) + one col per model
    col_widths = [3.5] + [4.5] * n_models
    row_height = 5.0

    fig_width = sum(col_widths)
    fig_height = row_height * n_rows + 0.8  # +0.8 for header

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Column header row
    header_y = 1.0 - (0.4 / fig_height)
    header_xs = []
    cumulative = 0
    for w in col_widths:
        header_xs.append((cumulative + w / 2) / fig_width)
        cumulative += w

    header_labels = ["Prompt"] + MODELS
    for hx, label in zip(header_xs, header_labels):
        fig.text(hx, header_y, label, ha="center", va="center",
                 fontsize=13, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none"))

    for row_i, (_, row) in enumerate(sampled.iterrows()):
        prompt = str(row["prompt"])

        # --- Prompt cell ---
        left = 0 / fig_width
        bottom = 1.0 - (0.8 + (row_i + 1) * row_height) / fig_height
        width = col_widths[0] / fig_width
        height = row_height / fig_height

        ax_txt = fig.add_axes([left, bottom, width, height])
        ax_txt.set_facecolor("#f9f9f9")
        ax_txt.set_xticks([])
        ax_txt.set_yticks([])
        for spine in ax_txt.spines.values():
            spine.set_edgecolor("#cccccc")

        wrapped = textwrap.fill(prompt, width=38)
        ax_txt.text(0.05, 0.97, wrapped,
                    va="top", ha="left",
                    fontsize=7.5,
                    transform=ax_txt.transAxes,
                    clip_on=True)

        ax_txt.text(0.05, 0.02, f"row {row_i + 1}",
                    va="bottom", ha="left",
                    fontsize=7, color="#888888",
                    transform=ax_txt.transAxes)

        # --- Image cells ---
        for col_i, model in enumerate(MODELS):
            left = sum(col_widths[:col_i + 1]) / fig_width
            bottom = 1.0 - (0.8 + (row_i + 1) * row_height) / fig_height
            width = col_widths[col_i + 1] / fig_width
            height = row_height / fig_height

            ax_img = fig.add_axes([left, bottom, width, height])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_edgecolor("#cccccc")

            model_slug = model.replace(".", "-")
            img_path = OUT_DIR / model_slug / f"row_{row_i:02d}.png"

            if img_path.exists():
                img = Image.open(img_path)
                ax_img.imshow(img, aspect="auto")
            else:
                ax_img.set_facecolor("#fff0f0")
                ax_img.text(0.5, 0.5, "generation\nfailed",
                            ha="center", va="center",
                            color="#cc0000", fontsize=10,
                            transform=ax_img.transAxes)

    out_path = OUT_DIR / "comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✅  Figure saved → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    sampled = df.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)
    sampled.to_csv(OUT_DIR / "sampled_rows.csv", index=False)
    print(f"Sampled {N_ROWS} rows  (seed={SEED})\n")

    for row_i, (_, row) in enumerate(sampled.iterrows()):
        prompt = str(row["prompt"])
        short = prompt[:60].replace("\n", " ")
        print(f"Row {row_i + 1:02d}: {short}…")

        for model in MODELS:
            model_slug = model.replace(".", "-")
            out_path = OUT_DIR / model_slug / f"row_{row_i:02d}.png"
            print(f"  → [{model}]", end=" ", flush=True)
            try:
                generate_image(prompt, model, out_path)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")

    print("\nBuilding comparison figure…")
    build_figure(sampled)


if __name__ == "__main__":
    main()
