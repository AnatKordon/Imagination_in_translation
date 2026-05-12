"""
Prompt Evolution Analysis & Visualization
===========================================
Analyze how participants modify descriptions across attempts and visualize changes.

Usage:
    python qa_additions_in_attempts.py <csv_path> <output_folder>
    
Example:
    python qa_additions_in_attempts.py \
        Data/processed_data/comparing_conditions/3_conditions_with_digit_span.csv \
        analysis/prompt_evolution

Output:
    - output_folder/per_participant/: One PNG per participant
    - output_folder/per_gt_image/: One PNG per ground truth image
"""

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Import spaCy for tokenization
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg"}
SEED_TAG = re.compile(r"_seed\d+(?=\.\w+$)", re.IGNORECASE)

# Text rendering constants
TEXT_FONTSIZE = 9
TEXT_LINE_HEIGHT = 1.4
TEXT_MAX_WIDTH = 60  # characters per line
TEXT_MAX_LINES = 40  # max lines to display

# Colors for changes (RGB tuples 0-1 scale for matplotlib, 0-255 for PIL)
COLOR_ADD = (76, 175, 80)      # Green
COLOR_REMOVE = (244, 67, 54)   # Red
COLOR_MODIFY = (255, 193, 7)   # Yellow
COLOR_NEUTRAL = (50, 50, 50)   # Dark grey for unchanged text

# PIL font (will load if available)
try:
    TEXT_FONT_PIL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    TEXT_FONT_PIL_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
except:
    TEXT_FONT_PIL = ImageFont.load_default()
    TEXT_FONT_PIL_BOLD = ImageFont.load_default()

# ============================================================================
# IMAGE UTILITIES (reused from visualize_per_ppt.py)
# ============================================================================

def normalize_name(name: str) -> str:
    """Remove seed tag from filename."""
    name = name.strip()
    name = SEED_TAG.sub("", name)
    return name

def read_image(path: Optional[Path], box=(320, 320)) -> Image.Image:
    """Read image or create placeholder, fit to box."""
    if path is None or not path.exists():
        w, h = box
        im = Image.new("RGB", box, (240, 240, 240))
        d = ImageDraw.Draw(im)
        d.text((10, h // 2 - 7), "missing", fill=(80, 80, 80))
        return im
    im = Image.open(path).convert("RGB")
    return ImageOps.contain(im, box)

def path_from_row(row) -> Path:
    """Reconstruct full path from CSV row."""
    filename = normalize_name(str(row["gen"]))
    uid = str(row["uid"]).strip()
    session = int(row["session"])
    return (
        Path(config.PARTICIPANTS_DIR)
        / uid
        / "gen_images"
        / f"session_{session:02d}"
        / filename
    )

def hide_axes(ax):
    """Remove ticks and spines from axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# ============================================================================
# TEXT DIFF & TOKENIZATION
# ============================================================================

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    if nlp:
        doc = nlp(text.lower())
        return [token.text for token in doc]
    else:
        # Fallback: simple split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

def count_tokens(text: str) -> int:
    """Count tokens in text using spaCy or fallback."""
    if nlp:
        return len(nlp(text))
    else:
        return len(tokenize_text(text))

def compare_prompts(prompt_1: str, prompt_2: str) -> Dict[str, Any]:
    """
    Compare two prompts and identify additions, removals, modifications.
    
    Returns:
        Dict with:
        - 'tokens_1', 'tokens_2': token counts
        - 'words_1', 'words_2': word counts
        - 'changes': list of (word, change_type) tuples
        - 'metrics': dict with counts of additions, removals, modifications
    """
    words_1 = prompt_1.split()
    words_2 = prompt_2.split()
    
    # Use SequenceMatcher for word-level diff
    matcher = SequenceMatcher(None, words_1, words_2)
    opcodes = matcher.get_opcodes()
    
    changes = []  # list of (word, change_type)
    added_count = 0
    removed_count = 0
    modified_count = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for word in words_1[i1:i2]:
                changes.append((word, 'equal'))
        elif tag == 'replace':
            # Words that are replaced
            for word in words_1[i1:i2]:
                changes.append((word, 'remove'))
                removed_count += 1
            for word in words_2[j1:j2]:
                changes.append((word, 'add'))
                added_count += 1
        elif tag == 'delete':
            for word in words_1[i1:i2]:
                changes.append((word, 'remove'))
                removed_count += 1
        elif tag == 'insert':
            for word in words_2[j1:j2]:
                changes.append((word, 'add'))
                added_count += 1
    
    # Calculate metrics
    total_words_2 = len(words_2) if words_2 else 1
    total_words_1 = len(words_1) if words_1 else 1
    
    metrics = {
        'words_1': len(words_1),
        'words_2': len(words_2),
        'tokens_1': count_tokens(prompt_1),
        'tokens_2': count_tokens(prompt_2),
        'added': added_count,
        'removed': removed_count,
        'pct_added': (added_count / total_words_2) * 100 if total_words_2 > 0 else 0,
        'pct_removed': (removed_count / total_words_1) * 100 if total_words_1 > 0 else 0,
        'pct_changed': ((added_count + removed_count) / total_words_2) * 100 if total_words_2 > 0 else 0,
        'tokens_changed': added_count + removed_count,
    }
    
    # Position analysis: where do changes occur?
    position_data = analyze_change_positions(changes)
    metrics.update(position_data)
    
    return {
        'changes': changes,
        'metrics': metrics,
    }

def analyze_change_positions(changes: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Analyze where in the text changes occur."""
    total = len(changes)
    if total == 0:
        return {'change_positions': 'N/A'}
    
    q_size = total // 4
    positions = {
        'beginning': 0,      # 0-25%
        'early_mid': 0,      # 25-50%
        'late_mid': 0,       # 50-75%
        'end': 0,            # 75-100%
    }
    
    for i, (word, change_type) in enumerate(changes):
        if change_type != 'equal':
            pct = (i / total) * 100
            if pct < 25:
                positions['beginning'] += 1
            elif pct < 50:
                positions['early_mid'] += 1
            elif pct < 75:
                positions['late_mid'] += 1
            else:
                positions['end'] += 1
    
    return {'change_positions': positions}

# ============================================================================
# VISUALIZATION: METRICS TABLE
# ============================================================================

def render_metrics_table(metrics_1_to_2: Dict, metrics_2_to_3: Dict) -> Image.Image:
    """
    Create a PIL Image with metrics table for transitions.
    
    Shows:
        - Word count, token count
        - % added, % removed, % changed
        - Tokens changed
        - Change positions
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Build table data
    headers = ['Transition', 'Words', 'Tokens', '% Added', '% Removed', '% Changed', 'Tokens Δ', 'Position']
    
    row_1_2 = [
        'Attempt 1→2',
        f"{metrics_1_to_2['metrics']['words_1']}→{metrics_1_to_2['metrics']['words_2']}",
        f"{metrics_1_to_2['metrics']['tokens_1']}→{metrics_1_to_2['metrics']['tokens_2']}",
        f"{metrics_1_to_2['metrics']['pct_added']:.1f}%",
        f"{metrics_1_to_2['metrics']['pct_removed']:.1f}%",
        f"{metrics_1_to_2['metrics']['pct_changed']:.1f}%",
        f"{metrics_1_to_2['metrics']['tokens_changed']}",
        _format_position(metrics_1_to_2['metrics']['change_positions']),
    ]
    
    row_2_3 = [
        'Attempt 2→3',
        f"{metrics_2_to_3['metrics']['words_1']}→{metrics_2_to_3['metrics']['words_2']}",
        f"{metrics_2_to_3['metrics']['tokens_1']}→{metrics_2_to_3['metrics']['tokens_2']}",
        f"{metrics_2_to_3['metrics']['pct_added']:.1f}%",
        f"{metrics_2_to_3['metrics']['pct_removed']:.1f}%",
        f"{metrics_2_to_3['metrics']['pct_changed']:.1f}%",
        f"{metrics_2_to_3['metrics']['tokens_changed']}",
        _format_position(metrics_2_to_3['metrics']['change_positions']),
    ]
    
    table = ax.table(
        cellText=[row_1_2, row_2_3],
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.2],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Convert to PIL image
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(image_from_plot)
    plt.close(fig)
    
    return img

def _format_position(positions_dict: Dict) -> str:
    """Format position dict to readable string."""
    if isinstance(positions_dict, str):
        return positions_dict
    return (f"B:{positions_dict['beginning']} EM:{positions_dict['early_mid']} "
            f"LM:{positions_dict['late_mid']} E:{positions_dict['end']}")

# ============================================================================
# VISUALIZATION: COLORED PROMPT TEXT
# ============================================================================

def render_prompt_with_changes(changes: List[Tuple[str, str]], width: int = TEXT_MAX_WIDTH) -> Image.Image:
    """
    Render prompt text with color-coded changes.
    - Green: additions (bold)
    - Red: removals (bold)
    - Yellow: modifications (bold)
    - Black: unchanged
    
    Returns PIL Image.
    """
    # Wrap text into lines
    lines = _wrap_with_changes(changes, width)
    
    # Calculate image size
    line_height = int(TEXT_FONTSIZE * TEXT_LINE_HEIGHT)
    img_height = len(lines) * line_height + 20
    img_width = width * 7  # rough estimate (char width ~7px)
    
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    y_pos = 10
    for line_content in lines:  # list of (text, color, bold) tuples
        x_pos = 10
        for text, color, is_bold in line_content:
            font = TEXT_FONT_PIL_BOLD if is_bold else TEXT_FONT_PIL
            draw.text((x_pos, y_pos), text, fill=color, font=font)
            x_pos += len(text) * 6  # approximate char width
        y_pos += line_height
    
    return img

def _wrap_with_changes(changes: List[Tuple[str, str]], width: int) -> List[List[Tuple[str, Tuple, bool]]]:
    """
    Wrap text preserving change information.
    Returns list of lines, each line is list of (text, color, is_bold).
    """
    lines = []
    current_line = []
    current_length = 0
    
    for word, change_type in changes:
        # Determine color and bold
        if change_type == 'equal':
            color = COLOR_NEUTRAL
            is_bold = False
        elif change_type == 'add':
            color = COLOR_ADD
            is_bold = True
        elif change_type == 'remove':
            color = COLOR_REMOVE
            is_bold = True
        else:  # modify
            color = COLOR_MODIFY
            is_bold = True
        
        word_with_space = word + ' '
        
        # Check if word fits in current line
        if current_length + len(word_with_space) > width:
            if current_line:
                lines.append(current_line)
                current_line = []
                current_length = 0
        
        current_line.append((word_with_space, color, is_bold))
        current_length += len(word_with_space)
    
    if current_line:
        lines.append(current_line)
    
    return lines

# ============================================================================
# PANEL GENERATION: ATTEMPT COMPARISON
# ============================================================================

def create_attempt_comparison_panel(
    uid: str, gt: str, session: int, df_subset: pd.DataFrame,
) -> Optional[Image.Image]:
    """
    Create a single panel showing 3 attempts with metrics.
    Layout: [Metrics Table] / [Attempt1 text + img | Attempt2 text + img | Attempt3 text + img]
    """
    # Get attempts 1, 2, 3
    attempts = {}
    for attempt in [1, 2, 3]:
        row = df_subset[df_subset['attempt'] == attempt]
        if not row.empty:
            attempts[attempt] = row.iloc[0]
        else:
            attempts[attempt] = None
    
    # Skip if any attempt missing
    if any(attempts[i] is None for i in [1, 2, 3]):
        return None
    
    # Extract prompts
    prompt_1 = str(attempts[1]['prompt'])
    prompt_2 = str(attempts[2]['prompt'])
    prompt_3 = str(attempts[3]['prompt'])
    
    # Compare 1->2 and 2->3
    comp_1_2 = compare_prompts(prompt_1, prompt_2)
    comp_2_3 = compare_prompts(prompt_2, prompt_3)
    
    # Render components
    metrics_img = render_metrics_table(comp_1_2, comp_2_3)
    
    # For each attempt, render text with changes
    text_img_1 = render_prompt_with_changes(
        [(w, 'equal') for w in prompt_1.split()], TEXT_MAX_WIDTH
    )
    text_img_2 = render_prompt_with_changes(comp_1_2['changes'], TEXT_MAX_WIDTH)
    text_img_3 = render_prompt_with_changes(comp_2_3['changes'], TEXT_MAX_WIDTH)
    
    # Load images
    img_1 = read_image(path_from_row(attempts[1]), box=(280, 280))
    img_2 = read_image(path_from_row(attempts[2]), box=(280, 280))
    img_3 = read_image(path_from_row(attempts[3]), box=(280, 280))
    
    # Build composite layout using matplotlib
    fig = plt.figure(figsize=(15, 12))
    
    # Top: metrics
    ax_metrics = plt.subplot(2, 1, 1)
    ax_metrics.imshow(metrics_img)
    ax_metrics.axis('off')
    
    # Bottom: 3 columns for attempts
    for col, (text_img, gen_img, attempt_num) in enumerate([
        (text_img_1, img_1, 1),
        (text_img_2, img_2, 2),
        (text_img_3, img_3, 3),
    ]):
        ax_text = plt.subplot(4, 3, 4 + col)
        ax_text.imshow(text_img)
        ax_text.axis('off')
        ax_text.set_title(f'Attempt {attempt_num} Text', fontsize=10, weight='bold')
        
        ax_img = plt.subplot(4, 3, 7 + col)
        ax_img.imshow(np.asarray(gen_img))
        ax_img.axis('off')
        ax_img.set_title(f'Attempt {attempt_num} Image', fontsize=10)
    
    fig.suptitle(f'{uid} | {gt} | Session {session}', fontsize=12, weight='bold')
    plt.tight_layout()
    
    # Convert to PIL image
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    result = Image.fromarray(image_from_plot)
    plt.close(fig)
    
    return result

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def generate_per_participant_panels(csv_path: Path, out_dir: Path):
    """
    For each participant: generate one PNG per (gt, session).
    Output: out_dir/per_participant/{uid}/{uid}_{gt_stem}_{session}.png
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['uid', 'gt', 'session', 'attempt', 'prompt', 'gen']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Needs: {required_cols}")
    
    df['attempt'] = df['attempt'].astype(int)
    
    per_ppt_dir = out_dir / 'per_participant'
    per_ppt_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by uid
    for uid in df['uid'].unique():
        if pd.isna(uid):
            continue
        
        df_uid = df[df['uid'] == uid]
        uid_dir = per_ppt_dir / str(uid)
        uid_dir.mkdir(exist_ok=True)
        
        # For each (gt, session) combination
        for (gt, session), df_combo in df_uid.groupby(['gt', 'session']):
            gt_stem = Path(gt).stem
            
            panel_img = create_attempt_comparison_panel(uid, gt, session, df_combo)
            if panel_img:
                out_path = uid_dir / f'{uid}_{gt_stem}_{session}.png'
                panel_img.save(out_path, dpi=(150, 150))
                print(f"✓ Saved {out_path}")
            else:
                print(f"⊘ Skipped {uid} | {gt} | Session {session} (missing attempts)")

def generate_per_gt_panels(csv_path: Path, out_dir: Path):
    """
    For each GT image: generate one PNG per session showing all participants.
    Output: out_dir/per_gt_image/{gt_stem}/{gt_stem}_{session}.png
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['uid', 'gt', 'session', 'attempt', 'prompt', 'gen']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Needs: {required_cols}")
    
    df['attempt'] = df['attempt'].astype(int)
    
    per_gt_dir = out_dir / 'per_gt_image'
    per_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by (gt, session)
    for (gt, session), df_combo in df.groupby(['gt', 'session']):
        gt_stem = Path(gt).stem
        gt_dir = per_gt_dir / gt_stem
        gt_dir.mkdir(exist_ok=True)
        
        # Create grid showing all participants for this GT & session
        participants = df_combo['uid'].unique()
        n_rows = len(participants)
        
        fig = plt.figure(figsize=(16, n_rows * 5))
        
        for row_idx, uid in enumerate(participants):
            df_uid_gt = df_combo[df_combo['uid'] == uid]
            
            panel_img = create_attempt_comparison_panel(uid, gt, session, df_uid_gt)
            if panel_img:
                ax = plt.subplot(n_rows, 1, row_idx + 1)
                ax.imshow(panel_img)
                ax.axis('off')
        
        fig.suptitle(f'GT: {gt} | Session {session}', fontsize=14, weight='bold', y=0.995)
        plt.tight_layout()
        
        out_path = gt_dir / f'{gt_stem}_session{session}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved {out_path}")

def main(csv_path: str, output_dir: str):
    """Main entry point."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    print(f"Loading CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("Generating per-participant panels...")
    generate_per_participant_panels(csv_path, output_dir)
    
    print("\nGenerating per-GT panels...")
    generate_per_gt_panels(csv_path, output_dir)
    
    print(f"\n✓ Done! Panels saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze & visualize how prompts change across attempts'
    )
    parser.add_argument(
        'csv_path',
        help='Path to CSV with columns: uid, gt, session, attempt, prompt, gen'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for generated panels'
    )
    
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
