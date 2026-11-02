#!/usr/bin/env python3
"""
Regenerate Table 4 of the manuscript on your own machine.

The calibration-trap stage in master_experiment_v5.py records discrimination
only. Table 4 also needs the calibration measures (ECE, MCE, Brier, Brier
skill), because the paper's argument is that those measures stay satisfactory
while discrimination collapses. This script recomputes the whole table.

It re-uses the data stage and the modelling helpers from
master_experiment_v5.py, so it needs no new configuration and produces numbers
identical to the main run.

    cd ~/Desktop/himalayan
    source .venv/bin/activate
    python trap_calibration_table.py

Runtime is a few minutes: the data stage dominates, the 80 model fits do not.
Output: trap_with_calibration.csv, plus the Table 4 rows printed to screen.

NOTE (fixed 2 November): the first version of this script passed --tier smoke
to keep the data stage quick. The tier also sets the seed list, so it silently
ran 2 seeds instead of 10. It now passes --tier full. The tier has no effect on
the data stage, so this costs nothing.
"""
import sys, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "master_experiment_v5.py")
if not os.path.exists(SCRIPT):
    sys.exit("put this file next to master_experiment_v5.py")

# run master_experiment_v5 only as far as the modelling helpers
sys.argv = ["master_experiment_v5.py", "--tier", "full", "--outdir",
            os.path.join(HERE, "_trap_tmp")]
LINES = open(SCRIPT).read().split("\n")
cut_data = [i for i, l in enumerate(LINES) if "for k,v in DATA_STATS.items()" in l][0] + 1
cut_opt  = [i for i, l in enumerate(LINES) if l.startswith("#  OPTIMISERS")][0]
G = {"__name__": "__main__"}
exec(compile("\n".join(LINES[:cut_opt]), SCRIPT, "exec"), G)

split, fit_model, calib_report = G["split"], G["fit_model"], G["calib_report"]
SEEDS = G["SEEDS"]
FRACS = [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

rows = []
for s in SEEDS:
    Xtr, Xte, ytr, yte = split(s)
    order = np.argsort(Xtr.index.values)          # restore archive order
    Xs, ys = Xtr.iloc[order], ytr.iloc[order]
    n = len(ys)
    for fr in FRACS:
        idx = np.arange(n); k = int(round(fr * n))
        if k > 1:
            r = np.random.RandomState(s)
            pick = r.choice(n, size=k, replace=False)
            sh = pick.copy(); r.shuffle(sh); idx[pick] = idx[sh]
        m = fit_model(Xs.iloc[idx], ys.iloc[idx], s, calibrate=True, legacy=True)
        rep = calib_report(yte, m.predict_proba(Xte)[:, 1])
        rows.append(dict(FractionShuffled=fr, Seed=s,
                         ROC_AUC=rep["ROC_AUC"], AveragePrecision=rep["AveragePrecision"],
                         ECE=rep["ECE"], MCE=rep["MCE"], Brier=rep["Brier"],
                         BrierSkillScore=rep["BrierSkillScore"],
                         MeanPredicted=rep["MeanPredicted"], BaseRate=rep["BaseRate"]))
    print(f"   seed {s} done", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, "trap_with_calibration.csv"), index=False)

cols = ["ROC_AUC", "AveragePrecision", "ECE", "MCE", "Brier", "BrierSkillScore", "MeanPredicted"]
summary = d.groupby("FractionShuffled")[cols].agg(["mean", "std"])
print("\n=== Table 4: mean and standard deviation over the %d seeds ===" % d.Seed.nunique())
print(summary.round(4).to_string())
print("\nMCE is reported but is unstable at this prediction range; its spread "
      "across seeds can exceed its mean, so it is not usable as a check.")

lo = d[d.FractionShuffled == 0.0].sort_values("Seed")["ROC_AUC"].values
hi = d[d.FractionShuffled == 1.0].sort_values("Seed")["ROC_AUC"].values
from scipy import stats
print("\ncost of the fault      %.4f AUC" % (hi.mean() - lo.mean()))
print("seeds harmed           %d of %d" % ((lo < hi).sum(), len(lo)))
print("paired Wilcoxon p      %.5f" % stats.wilcoxon(lo, hi).pvalue)
print("worst unshuffled AUC   %.4f" % lo.min())
print("\nwritten: trap_with_calibration.csv")
