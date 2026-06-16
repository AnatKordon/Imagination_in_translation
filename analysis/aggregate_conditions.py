"""Cross-condition aggregation for the full experiment.

The two realistic comparisons, each producing a self-describing filename that
says what is held fixed *by* what varies:

  * one generation across the 3 tasks   -> "<gen>_by_task.csv"
        save_generation("aigen")   # aigen_perc + aigen_imm + aigen_del -> aigen_by_task.csv
  * one task across the 3 generations    -> "<task>_by_gen.csv"
        save_task("perc")          # aigen_perc + nogen_perc + plain_perc -> perc_by_gen.csv

Plus the full grid -> "all_conditions.csv" (save_all()).

For purely in-memory work (no file written) use the config helper directly::

    import config
    df = config.load(["aigen_perc", "nogen_imm", "plain_del"], sim=True)

CLI::

    python analysis/aggregate_conditions.py                 # all bundles
    python analysis/aggregate_conditions.py --generation aigen
    python analysis/aggregate_conditions.py --task perc --sim

All outputs land in Data/processed_data/Full_experiment/combined/.
"""
from pathlib import Path
import argparse
import sys

# Make sure we can import config.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

GENERATIONS = ["aigen", "nogen", "plain"]
TASKS = ["perc", "imm", "del"]


def _save(conditions, out_name: str, sim: bool):
    """Concatenate `conditions` and write to combined/ (skips if nothing exists)."""
    out = config.load(conditions, sim=sim)
    if out.empty:
        print(f"nothing to write for {out_name} (no source CSVs found yet)")
        return out
    config.COMBINED_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.COMBINED_PROCESSED_DIR / out_name
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({out.shape[0]} rows, conditions={sorted(out['condition'].unique())})")
    return out


def save_generation(gen: str, sim: bool = False):
    """Fixed generation, varying task -> <gen>_by_task.csv."""
    conds = [c for c in config.CONDITIONS if c.split("_")[0] == gen]
    suffix = "_sim" if sim else ""
    return _save(conds, f"{gen}_by_task{suffix}.csv", sim)


def save_task(task: str, sim: bool = False):
    """Fixed task, varying generation -> <task>_by_gen.csv."""
    conds = [c for c in config.CONDITIONS if c.split("_")[1] == task]
    suffix = "_sim" if sim else ""
    return _save(conds, f"{task}_by_gen{suffix}.csv", sim)


def save_all(sim: bool = False):
    """Full 3x3 grid -> all_conditions.csv."""
    suffix = "_sim" if sim else ""
    return _save(config.CONDITIONS, f"all_conditions{suffix}.csv", sim)


def save_standard_bundles(sim: bool = False):
    """All the recurring combos: full grid + each generation + each task."""
    save_all(sim=sim)
    for gen in GENERATIONS:
        save_generation(gen, sim=sim)
    for task in TASKS:
        save_task(task, sim=sim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--generation", choices=GENERATIONS,
                   help="Save one generation across the 3 tasks (<gen>_by_task.csv).")
    g.add_argument("--task", choices=TASKS,
                   help="Save one task across the 3 generations (<task>_by_gen.csv).")
    g.add_argument("--all", action="store_true", help="Save the full grid only.")
    ap.add_argument("--sim", action="store_true",
                    help="Use the with-similarity table (trials_final_sim).")
    args = ap.parse_args()

    if args.generation:
        save_generation(args.generation, sim=args.sim)
    elif args.task:
        save_task(args.task, sim=args.sim)
    elif args.all:
        save_all(sim=args.sim)
    else:
        save_standard_bundles(sim=args.sim)


if __name__ == "__main__":
    main()
