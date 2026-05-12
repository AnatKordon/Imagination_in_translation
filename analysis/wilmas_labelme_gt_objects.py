# Here we will extract wilma's full list of object per gt image, and also the label me annotations.

import pandas as pd
import sys
from pathlib import Path
import os
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image, ImageDraw,  ImageFont


# Make sure we can import config.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from config import GT_DIR
print(GT_DIR)
#print the content of the GT_DIR to check if it's correct
GT_NAMES = os.listdir(GT_DIR)

#wilmas data per our gt images with the xml file containing all objects that appear in the image and the coordinates of the polygons of each object. We will save a csv file with the count of each object per gt image, and also an image with all the masks/annotations drawn on top of the original image. This will be useful for the analysis and also for the paper to show what objects appear in each gt image.
MAIN_PATH = "/mnt/hdd/anatkorol/Imagination_in_translation/Data/other_datasets/wilmas_drawings_2019/LabelMe"

def find_single_file_recursive(folder, extensions):
    matches = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(extensions):
                matches.append(os.path.join(root, f))

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 file with {extensions} in {folder}, found {len(matches)}:\n{matches}"
        )

    return matches[0]
def extract_objects_and_polygons(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    polygons = []

    for obj in root.findall(".//object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue

        object_name = name_el.text.strip().lower()
        objects.append(object_name)

        pts = []
        for pt in obj.findall(".//pt"):
            x = pt.findtext("x")
            y = pt.findtext("y")
            if x is not None and y is not None:
                pts.append((int(float(x)), int(float(y))))

        if pts:
            polygons.append((object_name, pts))

    return objects, polygons

for gt in GT_NAMES:
    folder_name = gt.replace(".jpg", "")
    folder_path = os.path.join(MAIN_PATH, folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Output paths
    csv_path = os.path.join(folder_path, f"{folder_name}_object_counts.csv")
    labeled_img_path = os.path.join(folder_path, f"{folder_name}_all_masks_labeled.jpg")

    # Delete previous outputs if they exist
    for old_file in [csv_path, labeled_img_path]:
        if os.path.exists(old_file):
            os.remove(old_file)

    image_path = find_single_file_recursive(folder_path, (".jpg", ".jpeg", ".png"))
    xml_path = find_single_file_recursive(folder_path, (".xml",))

    objects, polygons = extract_objects_and_polygons(xml_path)
    counts = Counter(objects)

    # 1. Save separate CSV per GT image
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gt", "object", "object_count"])
        writer.writeheader()

        for object_name, count in sorted(counts.items()):
            writer.writerow({
                "gt": gt,
                "object": object_name,
                "object_count": count
            })

    # 2. Save image with all masks/annotations drawn

    # Convert to RGBA for transparency support
    img = Image.open(image_path).convert("RGBA")

    # Create overlay for transparent labels
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except:
        font = ImageFont.load_default()

    for object_name, pts in polygons:
        if len(pts) > 1:
            # Draw polygon on main image (still strong)
            ImageDraw.Draw(img).line(pts + [pts[0]], fill="red", width=3)

            x, y = pts[0]

            bbox = draw.textbbox((x, y), object_name, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            padding = 4

            # Semi-transparent black (alpha=120 out of 255)
            draw.rectangle(
                [x, y, x + text_w + padding*2, y + text_h + padding*2],
                fill=(0, 0, 0, 120)
            )

            draw.text(
                (x + padding, y + padding),
                object_name,
                fill=(255, 255, 255, 255),
                font=font
            )

    # Combine overlay with image
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB for saving
    img = img.convert("RGB")

    img.save(labeled_img_path)


    print(f"Done: {gt}")
    print(f"  CSV: {csv_path}")
    print(f"  Image: {labeled_img_path}")