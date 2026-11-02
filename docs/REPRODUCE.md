# Reproducing the study

The archived run in `results/` was produced by a single command. This document
covers setup, monitoring, resuming, and what to expect.

---

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS: install libomp first

**XGBoost will not import on macOS without it.** This is the most common
failure.

```bash
brew install libomp
python -c "import xgboost; print('xgboost', xgboost.__version__)"
```

### Data access

See [`../data/README.md`](../data/README.md). Confirm it works before a long
run:

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('siddharth0935/himalayan-expeditions'))"
```

---

## 2. Smoke test first

```bash
python scripts/master_experiment_v5.py --tier smoke --outdir smoke_results
```

About ten minutes. It exercises all 23 stages at tiny budgets and ends with
`COMPLETE`. **The numbers are meaningless** — only the plumbing is being
checked.

---

## 3. The full run

```bash
nohup python scripts/master_experiment_v5.py --tier full --outdir results > run.log 2>&1 &
```

On a MacBook Air M2 this took **194 minutes**. Budget longer on older hardware.

Stop the machine sleeping:

```bash
caffeinate -i -w $(pgrep -f master_experiment_v5.py)      # macOS
```

### Tiers

| Tier | Runs | Search budget | Wall clock |
|---|---|---|---|
| `smoke` | 2 | tiny | ~10 min |
| `standard` | 5 | P=20, T=30 | ~1.5 h |
| `full` | 10 | P=20, T=30 | ~3.3 h |

`full` is what the paper reports. Note that the tier also sets the seed list,
so a `smoke` run produces two seeds rather than ten.

---

## 4. Watching progress

```bash
cat results/STATUS.txt            # checklist of all 23 stages with elapsed time
tail -F run.log                   # live detail; Ctrl-C stops watching, not the run
pgrep -f master_experiment_v5.py  # still alive?
```

Use `tail -F`, not `-f`, if the log does not exist yet. All output is flushed,
so what you see is live. Feature selection is the long stage, roughly 140
optimiser runs; if the log looks frozen, check that the `[step/total]` counter
is still climbing.

`STATUS.txt` looks like:

```
Himalayan master experiment  |  tier=full  device=cpu
elapsed  142.6 min

  [x] data
  [x] master figure
  [>] feature selection
  [ ] imbalance
  ...
```

To stop it: `pkill -f master_experiment_v5.py`, then `pkill caffeinate`.

---

## 5. If it stops

Run the same command again. Every unit of work is checkpointed under
`results/checkpoints/`, so finished work is skipped. At most one optimiser run
is lost. Note that `>` truncates the log, so use `>>` if you want to keep the
earlier output.

---

## 6. Rebuilding only the dataset figure

```bash
python scripts/regenerate_figure1.py \
    --coords results/artifacts/peak_coordinates_verified_final.csv \
    --outdir results
```

Replaces `results/figures/fig01_dataset_overview.{png,pdf}` in place and
touches nothing else. Takes 3 to 5 minutes, because it rebuilds the expedition
metadata.

**The archived `fig01` was built from the unverified template and should be
regenerated before the figure is used.** The run wrote
`peak_coordinates_VERIFY.csv` with approximate coordinates, all marked
`VERIFIED,NO`, and used it, because no verified file was present in that
output directory. Running the command above with the verified file replaces it.

**Use `peak_coordinates_verified_final.csv`, not the `_VERIFY.csv` template.**
The script writes the template with approximate coordinates on every run, all
marked `VERIFIED,NO`, and will use it if no verified file is supplied. The
verified file in `results/artifacts/` was checked peak by peak against the
Himalayan Database; among other corrections it places Lhotse Shar at its own
position rather than Lhotse main's.

---

## 7. Checking your run against ours

```bash
python scripts/verify_results.py --new results --reference results
```

Point `--new` at your run and `--reference` at the archived one. It re-derives
thirteen boolean conclusions from the stored tables and exits 1 if any flips.

**Expect small differences.** The feature search is stochastic, and although
seeds are fixed, different library versions and BLAS backends change
floating-point ordering. Conclusions are stable; third decimals are not. If a
*conclusion* flips, that is worth an issue report.

One measure is known to be unstable on this data and should not alarm you: the
maximum calibration error. Almost every prediction falls below 0.2, so only one
or two equal-width bins are occupied and the maximum is taken over very few
cells. Across the ten partitions its standard deviation exceeds its mean.

---

## 8. The two supplementary tables

```bash
python scripts/trap_calibration_table.py
python scripts/verify_death_causes.py
```

The first recomputes the calibration-trap table including the reliability
measures the main run does not store, and writes
`trap_with_calibration.csv`. The archived copy is
`results/raw/32_CalibrationTrap_WithCalibration.csv`. A few minutes; the data
stage dominates.

The second reads the archive's member table, which the main pipeline never
touches, and reproduces the death-date and cause-of-death counts quoted in the
Discussion. Under a minute.

---

## 9. What gets written

```
results/
  Himalayan_MasterResults.xlsx   34 sheets
  figures/                        14 figures, PNG + PDF, 300 dpi
  raw/                            one CSV per sheet
  artifacts/                      selected features, convergence traces, manifest,
                                  per-peak exceedance thresholds, coordinates
  checkpoints/                    resume state (git-ignored)
  STATUS.txt
```

The manifest records library versions, seeds, elapsed time, dataset counts,
data provenance, the exceedance percentile and every peak's thresholds.
