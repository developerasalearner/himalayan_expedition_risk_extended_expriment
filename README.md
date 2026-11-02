# Fatality risk screening for Himalayan expeditions

Reproduction package for the manuscript by **Chinmaya Kumar Palo** and
**Himansu Sekhar Behera** (Veer Surendra Sai University of Technology, Burla).

> *A silent failure in cross-validated probability calibration for grouped
> environmental data, with an application to fatality risk in Himalayan
> expeditions*

Everything reported in the paper comes from **one run of one script**. That run
is archived in [`results/`](results/), and you can reproduce it with a single
command.

---

## What this study does

Fatal outcomes on Himalayan climbs are rare: **504 of 9,457 expeditions** across
**27 peaks** record a member death, a rate of **5.33%**, or about **17.8 : 1**.
That rarity makes the task easy to get wrong. A model that always answers "no"
is correct 94.7% of the time and useless.

The pipeline links expedition records to daily weather over each recorded
interval, engineers climate and party descriptors, excludes every field that
encodes the outcome, searches for a useful feature subset, corrects the model's
probabilities after training, and chooses an operating point from a review
budget rather than a formula.

**This is a retrospective study.** Weather is summarised from base camp to
termination, and that endpoint is unknown while a climb is under way. Several
expedition descriptors are also recorded after the fact. Nothing here is a
real-time warning system.

---

## Headline findings

Two of the three principal findings are negative. They are reported as findings.

| Finding | Evidence |
|---|---|
| The records carry a real signal | ROC-AUC **0.7161** against **0.4989** on permuted outcomes |
| **Weather adds nothing** over expedition features | **+0.0008** AUC, Wilcoxon **p = 0.94**, 5 wins in 10 |
| Weather alone is still above chance | **0.6096** on twelve meteorological predictors |
| **Logistic regression beats the pipeline** | 0.7485 vs 0.6922 AUC, **10 of 10** runs, p = 0.002 |
| **A calibration fault costs 0.199 AUC** | grouped row order gives **0.514**, shuffled gives **0.713**, all 10 runs |
| Standard calibration checks do not see it | ECE **0.0100** under the fault; only discrimination reveals it |
| Calibration still matters | ECE **0.0804 → 0.0043**, Brier skill **−0.506 → +0.023** |
| Budget screening works | the riskiest **0.5%** are fatal **4.83×** more often than chance |
| Post-outcome fields inflate results | average precision **0.133 → 0.380** on readmitting one column |
| Feature selection did *not* matter | no pairwise contrast among 14 strategies survives Holm correction |
| It transfers to unseen mountains, weakly | mean AUC **0.673** across 12 held-out peaks; 1 of 12 below chance |
| It transfers across decades | pre-1990 → 0.604; pre-2000 → 0.665; pre-2010 → 0.659 |

On a threshold-free measure the three conventional selectors and the
no-selection control outrank **all eleven** metaheuristics. The apparent
advantage of metaheuristic selection is an artefact of tuning the threshold on
the test partition.

---

## Reproduce it

```bash
# clone, or use GitHub's "Download ZIP" button and unzip
git clone https://github.com/developerasalearner/himalayan-expedition-risk.git
cd himalayan-expedition-risk

python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# macOS only - XGBoost will not import without this
brew install libomp

python scripts/master_experiment_v5.py --tier smoke --outdir smoke_results   # ~10 min
python scripts/master_experiment_v5.py --tier full  --outdir results         # ~3.3 h
```

Data is downloaded automatically; see [`data/README.md`](data/README.md) for
the Kaggle token and a manual fallback.

Full instructions, progress monitoring and resume behaviour:
**[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**

### Checking your run against ours

```bash
python scripts/verify_results.py --new results --reference results
```

Thirteen boolean conclusions the paper makes are re-derived from the stored
tables. The script exits 1 if any of them flips. Metric drift in the third
decimal is expected and reported separately; a flipped conclusion is not.

### Two supplementary tables

```bash
python scripts/trap_calibration_table.py    # Table 4: calibration measures under the fault
python scripts/verify_death_causes.py       # the death-date and cause counts in Section 8.1
```

Both are quick. The first recomputes the calibration-trap table with the
reliability measures the main run does not record; the second reads the
archive's member table, which the main pipeline never touches.

---

## What is in here

```
scripts/
  master_experiment_v5.py      the whole study - 23 stages, one command
  regenerate_figure1.py        rebuilds the dataset overview figure alone
  verify_results.py            re-derives 13 conclusions from the stored tables
  trap_calibration_table.py    supplementary Table 4
  verify_death_causes.py       death-date and cause-of-death counts
results/                       the archived run behind every number in the paper
  Himalayan_MasterResults.xlsx   34 sheets, one per table
  figures/                     14 figures, PNG and PDF at 300 dpi
  raw/                         one CSV per sheet, for scripted checking
  artifacts/                   selected features, convergence traces, run manifest,
                               per-peak exceedance thresholds, peak coordinates
  run.log                      the full stage-by-stage log of the archived run
docs/
  REPRODUCE.md                 step by step, including macOS specifics
  RESULTS_MAP.md               every paper claim mapped to a sheet or figure
  METHODS_NOTES.md             design decisions and the calibration fault in full
  CHANGELOG.md                 what changed between the v4 and v5 runs, and why
data/README.md                 how to obtain the two source datasets
```

**Start with [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md)** if you are checking a
specific number in the paper. It points each claim at the sheet that produced it.

---

## How the evaluation was done

- **Ten independent runs.** A seed fixes the train/test split, the search and
  model initialisation together, so variation across runs reflects end-to-end
  uncertainty rather than search noise alone.
- **Paired significance testing** with Wilcoxon signed-rank and Holm correction,
  plus Friedman across all methods.
- **The test partition is never touched during feature selection.** The search
  scores candidates on an inner split of the training partition.
- **All preprocessing is fitted on the training side only**, including median
  imputation. Every protocol routes through one screening function, so random
  splits, temporal eras, grouped folds and repeated cross-validation each get
  medians computed from their own training data and nothing else.
- **Operating points are reported at fixed review budgets**, because the right
  threshold depends on how many reports a human can read, not on the data.
- **Transfer is tested explicitly** by holding out whole mountains and whole
  eras, not only by random splitting.

### The exclusion rule

Expedition archives are compiled after the fact, so several fields describe how
a trip ended. These are removed before any modelling:

`mdeaths`, `hdeaths`, `death_rate`, `severe_outcome`, `smtmembers`,
`success1`–`success4`, and `termreason`.

`termreason` is the subject of a deliberate ablation. Readmitting it alone lifts
average precision from 0.133 to 0.380. **Those higher numbers are not better
performance** — the classifier gains partial sight of its own label. The gap is
reported precisely so that a shift of that size is not mistaken for progress.

---

## A reproducibility trap worth knowing about

`CalibratedClassifierCV` splits its **internal** folds without shuffling when
given a plain integer `cv`. If training rows arrive grouped, by mountain or
sorted by date, which is the natural order of this archive, each fold model is
trained on data that systematically omits a mountain or an era. Those models
disagree, their calibrated outputs are averaged, and ranking collapses.

Measured cost on this data: **0.514 versus 0.713 AUC**, in all ten runs,
p = 0.002. The transition is abrupt: still broken at 10% of rows shuffled,
repaired at 25%.

**The reason it is dangerous is that the usual checks miss it.** Under the fault
the expected calibration error is 0.0100 and the Brier score is within two
thousandths of the working model. Only discrimination reveals the failure.

The fix, applied throughout:

```python
cv = StratifiedKFold(3, shuffle=True, random_state=seed)
CalibratedClassifierCV(base, method="sigmoid", cv=cv)
```

The diagnostic, which costs nothing: compute discrimination before and after
calibration. A working calibration map is a monotone rescaling and cannot change
the ordering, so the two must agree to within noise.

Sheet `26_CalibrationTrap`, the supplementary table
`raw/32_CalibrationTrap_WithCalibration.csv` and
`figures/fig12_calibration_trap.png` characterise it as a dose-response curve.

---

## Environment of the archived run

Apple MacBook Air (M2), macOS, **CPU only**. Ten runs completed in **194
minutes**. No GPU is required or used. Exact package versions are in
`results/artifacts/run_manifest.json` and pinned in `requirements.txt`.

---

## Limitations

- One archive, 27 peaks, one region. No external replication.
- Retrospective by construction; not a forecasting system.
- The outcome is expedition-level, so a fatality is attributed to a window
  averaging 28 days. Death dates exist for 99% of the member deaths and were
  not used. This is the most likely reason the weather block contributes
  nothing, and it cuts against the paper's own negative result.
- 151 positives in the test partition limits how finely anything can be
  measured.
- 15.4% of records have no usable weather. Two distinct causes: 1,213 lack
  usable dates, 239 predate the 1980 start of the weather record. The second
  group is fatal in 22.6% of cases against 4.9% elsewhere, so the missingness
  flag carries era information.
- 112 of 9,457 rows are exact duplicates. Removing them changes ROC-AUC by
  −0.0017, so they are retained and reported.
- The calibration fault is demonstrated on one dataset.
- The archived `results/figures/fig01_dataset_overview` was rendered from the
  auto-generated coordinate template rather than the verified coordinate file,
  which is also in `results/artifacts/`. Regenerate it with `make figure`
  before reusing the figure. Coordinates are used for plotting only and enter
  no model.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper rather than this
repository alone.

## Licence

Code is MIT (see [`LICENSE`](LICENSE)). The two source datasets are **not**
redistributed here and carry their own terms; see
[`data/README.md`](data/README.md).

## Disclosure

In accordance with publisher policy on AI-generated content, the authors
disclose that Claude (Anthropic) assisted in drafting and editing the manuscript
text and in refactoring and documenting this repository. All data processing,
experimental design decisions, numerical results and figures are the authors'
own work, and the authors take full responsibility for the content.
