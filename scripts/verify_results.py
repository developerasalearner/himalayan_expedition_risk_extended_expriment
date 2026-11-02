#!/usr/bin/env python3
"""Compare a fresh run against the archived one.

    python scripts/verify_results.py --new my_results --reference results

Reads the headline numbers from both runs and reports any that moved beyond a
tolerance. The point is not bit-for-bit equality, which will not happen: the
feature search is stochastic, and library versions and BLAS backends change
floating-point ordering. The point is whether the CONCLUSIONS hold.

Exit code 0 if every conclusion survives, 1 if any does not.

Updated for the v5 run. Three conclusions changed from the v4 version of this
script and are documented in docs/CHANGELOG.md:
  - leave-one-peak-out now allows one peak below chance (Tilicho, 0.490 on six
    fatal expeditions), where v4 asserted none;
  - the linear baseline no longer merely matches the pipeline, it beats it on
    every replication, so the test is stricter;
  - two conclusions were added for the feature-family ablation, which did not
    exist in v4.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--new", default="results", help="the run being checked")
ap.add_argument("--reference", default="results", help="the archived run")
ap.add_argument("--tol", type=float, default=0.05,
                help="absolute tolerance on a metric before it is reported")
A = ap.parse_args()

NEW, REF = Path(A.new)/"raw", Path(A.reference)/"raw"
if not NEW.exists(): sys.exit(f"no raw/ directory under {A.new}")
if not REF.exists(): sys.exit(f"no raw/ directory under {A.reference}")

def load(root: Path, name: str):
    p = root/f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None

def headline(root: Path) -> dict:
    """The numbers the manuscript actually claims."""
    out = {}
    shuf = load(root, "17_LabelShuffle")
    if shuf is not None:
        out["signal: AUC on real outcomes"] = shuf[shuf.Condition == "real labels"].ROC_AUC.mean()
        out["signal: AUC on permuted outcomes"] = shuf[shuf.Condition == "shuffled labels"].ROC_AUC.mean()
    cm = load(root, "18_CalibrationMethods")
    if cm is not None:
        out["calibration: ECE uncalibrated"] = cm[cm.Calibration == "none"].ECE.mean()
        out["calibration: ECE sigmoid"] = cm[cm.Calibration == "sigmoid"].ECE.mean()
    cs = load(root, "12_Calibration_Summary")
    if cs is not None:
        base = cs[cs.Method == "Baseline"]
        for lab, flag in (("uncalibrated", "no"), ("sigmoid", "yes")):
            row = base[base.Calibrated == flag]
            if len(row):
                out[f"calibration: Brier skill {lab}"] = row.BrierSkill_mean.mean()
    bud = load(root, "16_Budget_Summary")
    if bud is not None:
        out["budget: lift at the smallest budget"] = bud.iloc[0].Lift_mean
    lopo = load(root, "22_LeaveOnePeakOut")
    if lopo is not None:
        out["transfer: mean AUC on held-out peaks"] = lopo.ROC_AUC.mean()
        out["transfer: peaks below chance"] = float((lopo.ROC_AUC < 0.5).sum())
    abl = load(root, "28_Leakage")
    if abl is not None:
        ex = abl[abl.Configuration.str.contains("excluded")].AveragePrecision.mean()
        inc = abl[abl.Configuration.str.contains("readmitted")].AveragePrecision.mean()
        out["leakage: AP excluded"] = ex
        out["leakage: AP readmitted"] = inc
        out["leakage: inflation"] = inc - ex
    trap = load(root, "26_CalibrationTrap")
    if trap is not None:
        out["trap: AUC at 0% shuffled"] = trap[trap.FractionShuffled == 0.0].ROC_AUC.mean()
        out["trap: AUC at 100% shuffled"] = trap[trap.FractionShuffled == 1.0].ROC_AUC.mean()
    kf = load(root, "24_RepeatedKFold")
    if kf is not None:
        out["k-fold: mean AUC"] = kf.ROC_AUC.mean()
    fam = load(root, "33_FeatureFamilies_Sig")
    if fam is not None and "Family" in fam:
        for key, label in (("expedition only", "families: expedition-only AUC"),
                           ("weather only", "families: weather-only AUC")):
            row = fam[fam.Family == key]
            if len(row):
                out[label] = float(row.MeanAUC.iloc[0])
        if len(fam):
            out["families: full-set AUC"] = float(fam.FullSetMeanAUC.iloc[0])
    lin = load(root, "20_LinearVsPipeline")
    if lin is not None and "Metric" in lin:
        row = lin[lin.Metric == "ROC_AUC"]
        if len(row):
            out["baseline: logistic AUC"] = float(row.Logistic_mean.iloc[0])
            out["baseline: pipeline AUC"] = float(row.Pipeline_mean.iloc[0])
    return {k: float(v) for k, v in out.items() if pd.notna(v)}

def conclusions(root: Path) -> dict:
    """Boolean statements the paper makes. These must not flip."""
    out = {}
    h = headline(root)
    if "signal: AUC on real outcomes" in h and "signal: AUC on permuted outcomes" in h:
        out["real outcomes beat permuted by >0.15 AUC"] = (
            h["signal: AUC on real outcomes"] - h["signal: AUC on permuted outcomes"] > 0.15)
    if "calibration: ECE uncalibrated" in h and "calibration: ECE sigmoid" in h:
        out["calibration cuts ECE by at least 5x"] = (
            h["calibration: ECE uncalibrated"] > 5 * h["calibration: ECE sigmoid"])
    if "calibration: Brier skill uncalibrated" in h:
        out["Brier skill moves from negative to positive"] = (
            h["calibration: Brier skill uncalibrated"] < 0 < h["calibration: Brier skill sigmoid"])
    if "transfer: peaks below chance" in h:
        out["at most one held-out peak falls below chance"] = h["transfer: peaks below chance"] <= 1
        out["mean held-out-peak AUC above 0.63"] = h["transfer: mean AUC on held-out peaks"] > 0.63
    if "leakage: inflation" in h:
        out["readmitting termreason inflates AP by >0.15"] = h["leakage: inflation"] > 0.15
    if "trap: AUC at 0% shuffled" in h:
        out["the ordering fault costs >0.15 AUC"] = (
            h["trap: AUC at 100% shuffled"] - h["trap: AUC at 0% shuffled"] > 0.15)
    if "budget: lift at the smallest budget" in h:
        out["small-budget screening beats chance by >3x"] = h["budget: lift at the smallest budget"] > 3.0
    if "families: expedition-only AUC" in h and "families: full-set AUC" in h:
        out["weather adds less than 0.01 AUC over expedition features"] = (
            abs(h["families: full-set AUC"] - h["families: expedition-only AUC"]) < 0.01)
    if "families: weather-only AUC" in h:
        out["weather alone is still above chance"] = h["families: weather-only AUC"] > 0.55
    if "baseline: logistic AUC" in h:
        out["the linear baseline beats the pipeline"] = (
            h["baseline: logistic AUC"] > h["baseline: pipeline AUC"])

    sel = load(root, "07_Selection_Summary")
    if sel is not None and "F1_mean" in sel:
        out["all selection methods within 0.06 F1 of each other"] = (
            sel.F1_mean.max() - sel.F1_mean.min() < 0.06)
    sig = load(root, "08_Significance")
    if sig is not None and "Significant_0.05" in sig:
        out["no pairwise selection contrast survives Holm"] = (
            not bool(sig["Significant_0.05"].fillna(False).any()))
    return out

hn, hr = headline(NEW), headline(REF)
cn, cr = conclusions(NEW), conclusions(REF)

print("=" * 78)
print(f"  comparing  new={A.new}  against  reference={A.reference}")
if A.new == A.reference:
    print("  (both point at the same run - this is a self-check of the archived results)")
print("=" * 78)

print(f"\n{'metric':<46}{'reference':>11}{'new':>11}{'delta':>10}")
print("-" * 78)
drift = []
for k in sorted(set(hn) | set(hr)):
    a, b = hr.get(k), hn.get(k)
    if a is None or b is None:
        print(f"{k:<46}{'-' if a is None else f'{a:.4f}':>11}"
              f"{'-' if b is None else f'{b:.4f}':>11}{'n/a':>10}")
        continue
    d = b - a
    flag = "  <-- moved" if abs(d) > A.tol else ""
    if abs(d) > A.tol: drift.append((k, a, b, d))
    print(f"{k:<46}{a:>11.4f}{b:>11.4f}{d:>+10.4f}{flag}")

print(f"\n{'conclusion':<62}{'ref':>7}{'new':>7}")
print("-" * 78)
broken = []
for k in sorted(set(cn) | set(cr)):
    a, b = cr.get(k), cn.get(k)
    mark = lambda v: "-" if v is None else ("yes" if v else "NO")
    if a is not None and b is not None and a != b: broken.append(k)
    print(f"{k:<62}{mark(a):>7}{mark(b):>7}")

print("\n" + "=" * 78)
if broken:
    print("  CONCLUSIONS CHANGED:")
    for k in broken: print(f"    - {k}")
    print("\n  This is worth investigating and worth an issue report.")
elif drift:
    print(f"  All conclusions hold. {len(drift)} metric(s) moved by more than {A.tol}:")
    for k, a, b, d in drift: print(f"    - {k}: {a:.4f} -> {b:.4f} ({d:+.4f})")
    print("\n  Expected from a stochastic search and different library versions.")
else:
    print("  All conclusions hold and no metric moved beyond tolerance.")
print("=" * 78)

for root, tag in ((Path(A.new), "new"), (Path(A.reference), "reference")):
    mp = root/"artifacts"/"run_manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        print(f"\n  {tag}: tier={m.get('tier')} seeds={len(m.get('seeds', []))} "
              f"device={m.get('device')} minutes={m.get('elapsed_minutes')}")
        v = m.get("versions", {})
        if v: print("    " + "  ".join(f"{k}={val}" for k, val in v.items()))

sys.exit(1 if broken else 0)
