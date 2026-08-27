# Measures the judge's run-to-run reliability from two runs of the SAME
# configuration over the same rows. Every difference between them is noise:
# nothing changed but the sampling of the model's own reasoning.
#
# Overall agreement is a misleading headline here, because ~92% of tags are
# unanimously valid and inflate it. What matters is agreement on the tags that
# carry information - the ones either run flagged - and whether the aggregate
# rates a study would report survive a re-run even when individual labels do not.

from pathlib import Path
import argparse
import glob
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/outputs/experiments/validator"
KEY = ["gt", "uid", "session", "attempt", "category", "item"]
ERRORS = ["participant_hallucination", "category_misplacement", "model_hallucination"]


def load(tag: str) -> pd.DataFrame:
    hits = sorted(glob.glob(str(OUT_DIR / f"pilot_tags_*{tag}_*.csv")), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"no pilot file for {tag}")
    d = pd.read_csv(hits[-1])
    for c in ["uid", "session", "attempt"]:
        d[c] = d[c].astype(str)
    d.attrs["name"] = Path(hits[-1]).name
    return d.drop_duplicates(subset=KEY)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - behaves sanely for the small counts here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="_G")
    ap.add_argument("--b", default="_H")
    ap.add_argument("--vs-old", default="_A", help="an earlier run to contrast against")
    S = "/tmp/claude-1004/-mnt-hdd-anatkorol-Imagination-in-translation/4c9c2c38-5b20-40cd-9092-d5bc0ddc66c6/scratchpad"
    ap.add_argument("--log-a", default=f"{S}/runG.log")
    ap.add_argument("--log-b", default=f"{S}/runH.log")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    m = A[KEY + ["error_type", "confidence", "source"]].rename(
        columns={"error_type": "r1", "confidence": "c1"}
    ).merge(
        B[KEY + ["error_type", "confidence"]].rename(columns={"error_type": "r2", "confidence": "c2"}),
        on=KEY, how="inner",
    )
    n = len(m)

    # Row accounting first: a skipped row silently removes all of its tags from
    # the comparison, and the rows that fail are the tag-dense ones, so a small
    # row loss can be a large and non-random data loss.
    def rows_of(d):
        return set(map(tuple, d[["gt", "uid", "session", "attempt"]].drop_duplicates().values))
    r1, r2 = rows_of(A), rows_of(B)
    union = r1 | r2
    tags_by_row = A.groupby(["gt", "uid", "session", "attempt"]).size().to_dict()

    print("=" * 84)
    print("ROW COMPLETION")
    print("=" * 84)
    print(f"   run 1 completed {len(r1)} rows, {len(A)} tags")
    print(f"   run 2 completed {len(r2)} rows, {len(B)} tags")
    print(f"   rows in both:   {len(r1 & r2)}")
    for lab, missing in [("run 1 skipped", union - r1), ("run 2 skipped", union - r2)]:
        if missing:
            lost_tags = sum(tags_by_row.get(k, 0) for k in missing)
            print(f"   {lab}: {len(missing)} row(s), ~{lost_tags} tags")
            for k in sorted(missing, key=lambda x: -tags_by_row.get(x, 0)):
                print(f"       {k[0]:22} uid={k[1][:12]:12} ~{tags_by_row.get(k, 0):3} tags")
        else:
            print(f"   {lab}: none")
    for tag, log in [(args.a, args.log_a), (args.b, args.log_b)]:
        if log and Path(log).exists():
            errs = [l.strip() for l in Path(log).read_text(errors="ignore").splitlines()
                    if "ERROR:" in l]
            print(f"   {tag} error lines in log: {len(errs)}")
            for e in errs[:6]:
                print(f"       {e[:110]}")
    print()

    print("=" * 84)
    print("RELIABILITY OF THE CURRENT VALIDATOR  (two identical runs)")
    print("=" * 84)
    print(f"  run 1: {A.attrs['name']}")
    print(f"  run 2: {B.attrs['name']}")
    print(f"  tags judged by both: {n}   (run1 produced {len(A)}, run2 {len(B)})\n")

    dis = int((m.r1 != m.r2).sum())
    lo, hi = wilson(dis, n)
    print(f"1. RAW AGREEMENT")
    print(f"   identical verdict on {n - dis}/{n} tags = {(n-dis)/n:.2%}")
    print(f"   disagreement {dis}/{n} = {dis/n:.2%}  [95% CI {lo:.2%}-{hi:.2%}]")
    print("   NOTE: inflated by the large unanimous-valid majority; see (3).\n")

    print("2. WHAT EACH RUN FOUND")
    tab = pd.DataFrame({"run1": m.r1.value_counts(), "run2": m.r2.value_counts()}).fillna(0).astype(int)
    tab["delta"] = tab.run2 - tab.run1
    print(tab.to_string(), "\n")

    print("3. AGREEMENT ON THE TAGS THAT CARRY INFORMATION")
    either = m[(m.r1 != "valid") | (m.r2 != "valid")]
    both = int(((either.r1 != "valid") & (either.r2 != "valid")).sum())
    same = int((either.r1 == either.r2).sum())
    if len(either):
        blo, bhi = wilson(both, len(either))
        print(f"   flagged by either run: {len(either)}")
        print(f"     flagged by both:     {both} = {both/len(either):.1%}  [95% CI {blo:.1%}-{bhi:.1%}]")
        print(f"     same error type:     {same} = {same/len(either):.1%}")
        print(f"     only one run:        {len(either)-both} = {(len(either)-both)/len(either):.1%}  <- coin flips\n")

    print("4. PER ERROR TYPE  (Jaccard = both / either)")
    for et in ERRORS:
        a_, b_ = (m.r1 == et), (m.r2 == et)
        inter, union = int((a_ & b_).sum()), int((a_ | b_).sum())
        if union:
            jlo, jhi = wilson(inter, union)
            print(f"   {et:26} run1={a_.sum():3} run2={b_.sum():3}  "
                  f"agree {inter}/{union} = {inter/union:5.1%}  [CI {jlo:.0%}-{jhi:.0%}]")
        else:
            print(f"   {et:26} never fired in either run")
    print()

    print("5. DOES THE REPORTED RATE SURVIVE?  (what a paper would quote)")
    for label, sub in [("all tags", m), ("participants only", m[m.source != "gpt_ceiling"]),
                       ("gpt ceiling only", m[m.source == "gpt_ceiling"])]:
        if not len(sub):
            continue
        out = []
        for col in ["r1", "r2"]:
            scored = int((~sub[col].isin(["category_misplacement", "model_hallucination"])).sum())
            ph = int((sub[col] == "participant_hallucination").sum())
            out.append(ph / scored if scored else 0.0)
        print(f"   {label:20} run1={out[0]:6.2%}  run2={out[1]:6.2%}  "
              f"swing={abs(out[0]-out[1])*100:4.2f} pp")
    print()

    print("6. CONFIDENCE AS A FILTER")
    m["both_high"] = (m.c1 == "high") & (m.c2 == "high")
    for label, sub in [("both runs high", m[m.both_high]), ("either not high", m[~m.both_high])]:
        if len(sub):
            d2 = int((sub.r1 != sub.r2).sum())
            print(f"   {label:18} n={len(sub):4}  disagree {d2:3} = {d2/len(sub):6.2%}")
    kept = int(m.both_high.sum())
    print(f"   restricting to both-high keeps {kept}/{n} = {kept/n:.1%} of tags\n")

    print("7. WHERE THE INSTABILITY LIVES")
    rows = []
    for cat, blk in m.groupby("category"):
        d2 = int((blk.r1 != blk.r2).sum())
        rows.append({"category": cat, "n": len(blk), "disagree": d2,
                     "pct": round(100 * d2 / len(blk), 1)})
    print(pd.DataFrame(rows).sort_values("pct", ascending=False).to_string(index=False), "\n")

    print("8. EVERY DISAGREEMENT")
    for _, r in m[m.r1 != m.r2].iterrows():
        print(f"   [{r['category']:17}] {r['item']!r:36} {r.r1:25} vs {r.r2:25} conf {r.c1}/{r.c2}")

    try:
        old = load(args.vs_old)
        j = m.merge(old[KEY + ["error_type"]].rename(columns={"error_type": "old"}), on=KEY, how="inner")
        print(f"\n9. EFFECT OF THE PROMPT REVISIONS  (vs {old.attrs['name']}, n={len(j)})")
        for col, lab in [("r1", "current run1"), ("r2", "current run2")]:
            print(f"   {lab} differs from the older prompt on "
                  f"{int((j[col]!=j['old']).sum())}/{len(j)} = {(j[col]!=j['old']).mean():.2%}")
        print("   (compare against the run-to-run noise in (1): a prompt change is only")
        print("    demonstrably real if it moves more tags than noise alone does)")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
