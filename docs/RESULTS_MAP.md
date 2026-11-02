# Every claim in the paper, and where it comes from

Each row names a claim, the number, and the sheet in
`results/Himalayan_MasterResults.xlsx` or the file in `results/raw/` that
produced it. Sheet names and CSV names match.

Section numbers refer to the submitted manuscript.

---

## Section 3 — Data

| Claim | Value | Source |
|---|---|---|
| Expeditions in the source archive | 11,425 | `02_Dataset` |
| Peaks with a matching weather series | 27 | `02_Dataset` |
| Expeditions retained | 9,457 (82.8%) | `02_Dataset` |
| No usable weather | 1,452 (15.4%) | `02_Dataset` |
| — of which dates incomplete | 1,213 (12.8%) | derived; see `scripts/master_experiment_v5.py` data stage |
| — of which entirely pre-1980 | 239 (2.5%) | as above |
| Fatality rate, dates-missing group | 4.8% | as above |
| Fatality rate, pre-1980 group | 22.6% | as above |
| Fatality rate, group with weather | 4.9% | as above |
| Fatal expeditions | 504 (5.33%) | `02_Dataset` |
| Member deaths | 681 | `03_PeakSummary`, Deaths column |
| Negative-to-positive ratio | 17.8 : 1 | `02_Dataset` |
| Encoded columns, then after screening | 70 → 44 | `02_Dataset` |
| Train / test rows and positives | 6,619 / 353, 2,838 / 151 | `02_Dataset` |
| Duplicate feature rows | 112 | `02_Dataset` |
| Per-peak table | all 27 | `03_PeakSummary` |
| Predictor dictionary | 25 rows | `04_FeatureDictionary` |
| Everest + Ama Dablam + Cho Oyu share | 56% | `03_PeakSummary` |
| Daily max 100 m wind never reaches 70 km/h | max 58.5 | derived from the weather archive |
| Per-peak 95th-percentile thresholds | all 27 | `artifacts/run_manifest.json`, `peak_extreme_thresholds` |
| Figure 1 | — | `figures/fig01_dataset_overview.pdf` |

## Section 4 — Methodology

| Claim | Value | Source |
|---|---|---|
| Configuration: seeds, population, iterations, thresholds | — | `01_Config` |
| Final operating threshold | 0.11 | `01_Config` |
| Selected feature subsets, every method and seed | 140 rows | `29_SelectedFeatures` |
| Reference for every method compared | — | `31_Citations` |

## Section 6 — The calibration failure

| Claim | Value | Source |
|---|---|---|
| AUC by fraction of rows shuffled | 0.514 → 0.713 | `26_CalibrationTrap` |
| Cost of the fault | 0.1985 AUC, 10/10 seeds, p = 0.002 | `26_CalibrationTrap` |
| Worst single unshuffled run | 0.4199 | `26_CalibrationTrap` |
| Abrupt transition between 10% and 25% | — | `26_CalibrationTrap`, `figures/fig12_calibration_trap.pdf` |
| **Table 4**, calibration measures under the fault | ECE 0.0100 vs 0.0041 | `raw/32_CalibrationTrap_WithCalibration.csv` |
| MCE unstable, SD exceeds mean | 0.027 ± 0.063 | as above |
| Train/test ratio is not the cause | 0.7213 at 70/30, 0.7259 at 80/20 | diagnostic run, reported in text |
| Pooling is not the cause | 0.5444 pooled vs 0.5518 per fold | diagnostic run |
| Duplicates are not the cause | 0.0017 | diagnostic run |

> Table 4 is **not** produced by the main run. Regenerate it with
> `python scripts/trap_calibration_table.py`.

## Section 7 — Results

| Claim | Value | Source |
|---|---|---|
| Real vs permuted outcomes | 0.7161 vs 0.4989 | `17_LabelShuffle` |
| Average precision, real vs permuted | 0.1404 vs 0.0553 | `17_LabelShuffle` |
| **Table 5**, reported configuration | 0.6922 calibrated | `12_Calibration_Summary`, `13_Final` |
| Uncalibrated vs calibrated ECE | 0.0804 → 0.0043 | `12_Calibration_Summary` |
| Brier skill | −0.506 → +0.023 | `12_Calibration_Summary` |
| Repeated k-fold | 0.7261 ± 0.021 | `24_RepeatedKFold` |
| Isotonic vs sigmoid | 0.0040 / 0.689 vs 0.0043 / 0.692 | `18_CalibrationMethods` |
| **Table 6**, feature families | +0.0008, p = 0.94 | `32_FeatureFamilies`, `33_FeatureFamilies_Sig` |
| Weather alone | 0.6096 | `32_FeatureFamilies` |
| Year removal costs | 0.0157, p = 0.027 | `33_FeatureFamilies_Sig` |
| **Table 7**, conventional baselines | — | `05_Baselines` |
| Logistic vs pipeline | 0.7485 vs 0.6922, 10/10, p = 0.002 | `20_LinearVsPipeline`, `19_LinearBaseline` |
| **Table 8**, selection methods | GA 0.1755 best by F1 | `07_Selection_Summary` |
| Friedman | p = 0.0024 | `08_Significance` |
| No pairwise contrast survives Holm | — | `08_Significance` |
| AUC ordering inverts | Embedded 0.6936, no-selection 0.6909 | `07_Selection_Summary` |
| Selection frequency tracks informativeness | 76% / 74% / 66% / 55–62% | `29_SelectedFeatures`, `30_FeatureStability` |
| Imbalance treatments | SMOTE family ECE ~0.070 | `10_Imbalance_Summary`, `12_Calibration_Summary` |
| Undersampling degenerate | recall 1.000, ECE 0.405, BSS −3.39 | `12_Calibration_Summary` |
| **Table 9**, review budgets | 4.83× at 0.5% | `16_Budget_Summary` |
| Honest vs test-tuned threshold | 0.1508 vs 0.1745 | `14_HonestThreshold` |
| **Table 10**, temporal transfer | 0.604 / 0.665 / 0.659 | `21_Temporal` |
| **Table 10**, leave-one-peak-out | mean 0.6727, Tilicho 0.4901 | `22_LeaveOnePeakOut` |
| Leakage, termreason readmitted | AP 0.133 → 0.380 | `28_Leakage` |
| Archival ablation | 0.692 vs 0.699 | `25_Archival` |

## Section 8 — Discussion

| Claim | Value | Source |
|---|---|---|
| Death dates available | 674 of 681, 99.0% | `scripts/verify_death_causes.py` |
| Dated and within weather coverage | 580 | as above |
| Causes: falls, avalanches, AMS | 226, 168, 91 | as above |

> These read the archive's `members.csv`, which the main pipeline does not
> touch. Run the script to reproduce them.

## Section 9 — Limitations

| Claim | Value | Source |
|---|---|---|
| Hyperparameter sensitivity, reduced budget | — | `27_Sensitivity` |
| Tuning grid | depth 3, 100 trees, lr 0.05 | `23_Tuning` |

---

## Figures

| Figure | File | Used in |
|---|---|---|
| Fig. 1 | `fig01_dataset_overview` | §3.8 |
| Fig. 2 | `fig12_calibration_trap` | §6.4 |
| Fig. 3 | `fig07_label_shuffle` | §7.1 |
| Fig. 4 | `fig05_curves` | §7.2 |
| Fig. 5 | `fig02_selection_distribution` | §7.5 |
| Fig. 6 | `fig04_calibration_metrics` | §7.6 |
| Fig. 7 | `fig06_budget` | §7.7 |
| — | `fig03_convergence` | supplementary |
| — | `fig08_temporal` | supplementary |
| — | `fig09_leave_one_peak_out` | supplementary, duplicates Table 10 |
| — | `fig10_sensitivity` | supplementary |
| — | `fig11_leakage` | supplementary |
| — | `fig13_archival` | supplementary |
| — | `fig14_feature_stability` | supplementary |

## Sheet index

`00_README` inside the workbook lists all 34 sheets with a one-line
description of each.
