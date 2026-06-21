"""Classify a single study_result/comp-result folder as full / partial / unusable."""
from pathlib import Path

from .convert_data_txt import TYPE_TO_CSV, parse_data_txt, write_reconstructed_csvs

REQUIRED_BASE = ["participants.csv", "trials.csv"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _required_files(is_del: bool) -> list[str]:
    return REQUIRED_BASE + (["digit_span.csv"] if is_del else [])


def _has_images(files_dir: Path) -> bool:
    return files_dir.exists() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in files_dir.iterdir())


def classify_participant(comp_result_dir: Path, is_del: bool) -> dict:
    """Classify one comp-result folder, reconstructing CSVs from data.txt if needed."""
    required = _required_files(is_del)
    files_dir = comp_result_dir / "files"
    missing = [f for f in required if not (files_dir / f).exists()]
    reconstructed: set[str] = set()

    if not missing and _has_images(files_dir):
        status = "full"
    else:
        data_txt = comp_result_dir / "data.txt"
        if data_txt.exists():
            rows_by_type = parse_data_txt(data_txt)
            write_reconstructed_csvs(files_dir, rows_by_type)
            # "reconstructed" reflects which CSVs originate from data.txt, whether
            # written just now or by a previous run (write is skipped if already present).
            reconstructed = {
                TYPE_TO_CSV[t] for t in rows_by_type
                if TYPE_TO_CSV.get(t) and (files_dir / TYPE_TO_CSV[t]).exists()
            }
            missing = [f for f in required if not (files_dir / f).exists()]
            if not missing and _has_images(files_dir):
                status = "full"
            elif (files_dir / "trials.csv").exists():
                status = "partial"
            else:
                status = "unusable"
        else:
            status = "unusable"

    return {
        "study_result": comp_result_dir.parent.name,
        "comp_result": comp_result_dir.name,
        "status": status,
        "missing": missing,
        "reconstructed": reconstructed,
        "files_dir": files_dir,
    }
