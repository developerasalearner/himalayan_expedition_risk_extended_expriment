# What changed between the v4 and v5 runs

The v4 run was archived in an earlier version of this repository. It was
replaced because an audit against the raw archives found four defects, three of
which changed the numbers. This file records all of them, because the earlier
results were public and the difference should be traceable.

The v5 script is `scripts/master_experiment_v5.py`. Its header carries the same
list in short form.

---

## 1. Counters measured hours, not days

The weather archive is **hourly**: 24 rows per calendar day, timestamped UTC.
Everest alone has 394,488 rows for 16,437 days. The v4 pipeline took its
expedition windows directly over those rows, so four features named `_days`
were counting hours, and `snow_max` was the largest hourly fall rather than the
largest daily one.

**v5** collapses each peak's series to one row per calendar day before any
window is taken.

## 2. Two features were identically zero for all 9,457 expeditions

| Feature | v4 threshold | Why it never fired |
|---|---|---|
| `heavy_snow_days` | snowfall > 10 | snowfall is in **centimetres**, so this meant 10 cm within one hour; the largest observed hourly fall is 7.21 cm |
| `extreme_wind_days` | wind > 70 km/h | across **443,799 peak-days** the daily maximum 100 m wind never reaches 70 km/h; the largest observed is 58.5 |

Absolute thresholds borrowed from mountaineering experience are unreachable in
a reanalysis product that averages over kilometres of relief.

**v5** replaces all three fixed-threshold counters with exceedances of **each
peak's own 95th-percentile climatology**, computed over its full 1980–2024
record. An adverse day is one that is unusual *for that mountain*. The
thresholds are written to `results/artifacts/run_manifest.json`.

## 3. Raw counts were confounded with expedition duration

Counts grow with window length by construction. In v4 they correlated with
duration at r = 0.65 to 0.72, so a model reading them was partly reading how
long the team stayed.

**v5** expresses the counters as the **share of days** in the window. Three of
the four are now effectively independent of duration (r = 0.04, −0.06, −0.03).
The fourth, the solid-precipitation rate, retains r = 0.38, which is physical
rather than artefactual and is reported as such.

## 4. Four engineered flags were zero in every row

`oxygen_used`, `oxygen_free_ascent`, `is_standard_route` and
`is_commercial_route`.

This Kaggle release stores the archive's Y/N flags as **booleans**. The v4
parser tested `str(v).strip().upper() == "Y"`, which scores `True` as `"TRUE"`
and returns 0.

**This did not invalidate any v4 result.** The underlying signal still reached
the models through the raw `o2used`, `stdrte` and `comrte` columns, and a
constant column cannot change a tree prediction. Correct prevalences are
34.9%, 64.0%, 59.6% and 59.8%.

It did, however, produce an instructive artefact. Six constant columns reached
the feature search and were "selected" in 41 to 50% of the 140 method-seed
runs, which is the rate expected from a random binary mask. The optimisers
could not distinguish a zero-information column from an informative one. After
the repair, selection frequency tracks informativeness: mean temperature 76%,
team size 74%, calendar year 66%, the exceedance rates 55 to 62%.

## 5. Imputation leaked across every split

`build_table()` filled numeric gaps with the median of the **whole** 9,457-row
table, and the split happened afterwards. Test values therefore informed the
training imputation, in every protocol including leave-one-peak-out and the
temporal eras. About a quarter of rows have a missing duration, so this was not
a negligible amount of information.

**v5** moves imputation inside `screen()`, which every protocol routes through,
so medians come from the training side of each split, fold, era and held-out
peak only.

## 6. Smaller corrections

- A **zero-variance screen** now runs ahead of the correlation screen.
- `storm_days` is renamed `solid_precip_rate`. WMO codes at or above 70 denote
  solid precipitation, not storms, and the feature fires on 62% of days.
- The GPU probe no longer reports `cuda` on a CPU-only machine. XGBoost warns
  rather than raising when asked for an absent GPU, so the v4 manifest recorded
  a GPU that was never used.
- The calibration-trap stage now records the full set of calibration measures,
  not only discrimination.
- A **feature-family ablation** was added, which did not exist in v4.

---

## What this changed in the results

| | v4 | v5 |
|---|---|---|
| Constant columns reaching the model | 6 | **0** |
| Columns after screening | 47 | **44** |
| Cost of the calibration fault | 0.17 AUC | **0.199 AUC** |
| Friedman across selection methods | p = 0.236 | **p = 0.0024**, but no pairwise contrast survives Holm |
| Leave-one-peak-out below chance | 0 of 12 | **1 of 12** (Tilicho, 0.490 on six fatal expeditions) |
| Budget lift at 0.5% | 5.10× | **4.83×** |
| Does weather help? | **not tested** | **no**: +0.0008 AUC, p = 0.94 |

Dataset composition is unchanged: 11,425 source expeditions, 9,457 retained,
504 positives, 5.33%, 6,619 training and 2,838 test rows.

Three conclusions in `scripts/verify_results.py` were revised accordingly:
leave-one-peak-out now permits one peak below chance; the linear baseline test
became stricter, since logistic regression now beats the pipeline rather than
merely matching it; and two conclusions were added for the feature-family
ablation.

---

## A note on how these were found

All four defects were found by auditing the pipeline against the raw archives
rather than by reading the code. The first three were visible only by computing
the distribution of each engineered feature; the fourth only by checking the
dtype of the source columns. A feature that is constant for every row will not
raise an error, will not fail a test that checks shapes, and will be selected
at chance rate by a search that cannot tell it apart from noise.
