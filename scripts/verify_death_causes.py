#!/usr/bin/env python3
"""
Verify the death-date and cause-of-death counts quoted in Section 8.1.

These come from the member table of the expedition archive, which the main
pipeline does not read: master_experiment_v5.py works from exped.csv only.
This script reproduces the four numbers in the Discussion so they are not
unsupported assertions.

    python verify_death_causes.py
"""
import glob, os, sys
import pandas as pd

try:
    import kagglehub
except ImportError:
    sys.exit("pip install kagglehub")

EP = kagglehub.dataset_download("siddharth0935/himalayan-expeditions")
WP = kagglehub.dataset_download("bonesclarke26/historic-weather-data-for-himalayan-peaks")

M = pd.read_csv(os.path.join(EP, "members.csv"), low_memory=False)
peaks = {os.path.basename(f)[:-8].upper() for f in glob.glob(os.path.join(WP, "*.parquet"))}

m = M[M.peakid.isin(peaks)]
is_true = lambda s: s.astype(str).str.upper().isin(["TRUE", "Y", "1"])
dead = m[is_true(m.death)]
member_deaths = dead[~is_true(dead.hired)]

dd = pd.to_datetime(member_deaths["deathdate"], errors="coerce")
print("member deaths on the 27 peaks        %d   (paper: 681)" % len(member_deaths))
print("  with a usable death date           %d  (%.1f%%)   (paper: 674, 99.0%%)"
      % (dd.notna().sum(), 100 * dd.notna().mean()))
print("  dated and inside 1980+ coverage    %d   (paper: 580)"
      % (dd >= pd.Timestamp("1980-01-01")).sum())
print("\ncause of death, counts quoted in Section 8.1:")
print(member_deaths["deathtype"].value_counts().head(8).to_string())
