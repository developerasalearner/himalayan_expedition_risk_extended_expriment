# Design decisions, and why

Choices that are not obvious from reading the code, and the reasoning behind
them. For what changed since the previous archived run, see
[`CHANGELOG.md`](CHANGELOG.md).

---

## Why accuracy is never reported

At a 5.33% positive rate a classifier that answers "no fatality" every time is
correct 94.7% of the time. Accuracy appears in this repository only where its
uselessness is the point being made. Discrimination is reported as ROC-AUC and
average precision, and average precision is the more informative of the two
here because it gives no credit for correctly dismissing abundant negatives.

## Why calibration, and why it is reported separately from ranking

A screening decision combines a forecast with a cost and a capacity
constraint, and only a calibrated probability can enter that combination.
Boosted trees fitted under positive-class weighting produce scores that rank
adequately but are not probabilities: the weighting inflates them towards the
positive class by construction. Before calibration this model scores worse than
forecasting the historical rate for every expedition (Brier skill −0.506).

Ranking and reliability are different properties and are reported separately
throughout. The calibration failure documented in the paper is precisely a
procedure that preserves one and destroys the other.

## The calibration failure, in full

`CalibratedClassifierCV` given a plain integer `cv` draws its internal folds as
contiguous blocks of the training matrix, stratified on the label only. Nothing
constrains their composition in the predictor space.

Our rows arrive grouped by mountain and ordered by date, which is how any
archive assembled peak by peak arrives. Two things then go wrong at once:

1. Each of the K base models is fitted on K−1 blocks that systematically omit a
   mountain or an era, so the models disagree about which expeditions are
   dangerous. Averaging disagreeing orderings moves the result towards chance.
2. Each calibration map is fitted on the held-out block, whose base rate and
   score distribution differ substantially from the pooled data. Fatality rate
   across these peaks ranges from 0.47% to 15.8%.

The asymmetry follows. Each component map is roughly right about frequencies in
its own block, so their average still lands near the correct overall rate and
aggregate calibration survives. What does not survive is the ordering of
individual cases.

Stratification makes this worse, not better, by guaranteeing that the one
property a practitioner is likely to check — the class balance of each fold —
looks correct.

**The fix:**

```python
r = np.random.RandomState(seed)
idx = np.arange(len(ytr)); r.shuffle(idx)
Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]
cv = StratifiedKFold(3, shuffle=True, random_state=seed)
CalibratedClassifierCV(base, method="sigmoid", cv=cv)
```

**The diagnostic**, which costs nothing and we recommend as routine: compute
discrimination before and after calibration on the same held-out data. A
working calibration map is a monotone rescaling of scores and cannot change
their order, so the two must agree to within sampling noise. Here they are
0.716 and 0.692 when correct, and 0.514 when not.

The faulty path is retained in exactly one place, the deliberate trap study,
where its cost is measured across all ten seeds rather than asserted from one
observation.

## Why exceedances against each peak's own climatology

Absolute thresholds do not work in this product. Across 443,799 peak-days the
daily maximum 100 m wind never reaches 70 km/h. A reanalysis grid cell averages
over kilometres of relief and cannot represent summit-ridge conditions.

Defining an adverse day as one exceeding that peak's own 95th percentile makes
the feature mean "unusual for this mountain", which is the hazard-relevant
notion, and removes the cross-peak climatology differences that an absolute
threshold would confound with peak identity. Everest's snow threshold is
5.11 cm/day; Ama Dablam's is 2.66.

## Why rates rather than counts

Counts grow with window length by construction, and in this archive they
correlated with expedition duration at r = 0.65 to 0.72. A model reading them
would partly be reading how long the team stayed on the mountain. As shares of
the window, three of the four are effectively independent of duration.

## Why in-loss weighting rather than resampling

Every SMOTE variant leaves the expected calibration error near 0.070 after the
identical calibration step, against 0.0043 for in-loss weighting. Synthetic
oversampling distorts the score distribution in a way post-hoc calibration does
not repair. Random undersampling achieves recall 1.000 at precision equal to
the base rate: it flags everything.

Every configuration is therefore audited for degeneracy, and the audit result
is reported alongside its metrics.

## Why all preprocessing is fitted inside one function

`screen()` performs median imputation, the zero-variance screen and the
collinearity screen, in that order, from the training side only. Every
protocol in the script routes through it: random splits, temporal eras,
grouped leave-one-peak-out folds and repeated cross-validation. Putting it in
one place is the reason the imputation leak described in the changelog could
be fixed by changing one function rather than six call sites.

## Why the exclusion rule is strict

Expedition archives are compiled after the fact. Nine fields are removed
because they are recorded after the outcome or determined by it.

A tenth, `termreason`, is the interesting case. It records why an expedition
ended and its categories include accident involving death or injury, so it is
partly a restatement of the label. It also encodes genuinely predictive
information about non-fatal failure modes. Retaining it is defensible if a
model may use everything the archive contains; excluding it is required if the
model must be usable before an expedition has ended. We take the second view
and quantify the first: readmitting the field lifts average precision from
0.133 to 0.380.

Three further fields — ascent claimed, ascent disputed, primary reference —
are archival rather than physical, and they track era: a primary reference
exists for 6.3% of pre-1990 attempts against 2.3% afterwards. They are retained
with an ablation, which changes discrimination by 0.007.

## Why the headline selector was fixed in advance

`SELECTOR = "FOA_GWO"` is set before the comparison runs and is not changed
afterwards. The paper reports that it ranks second of fourteen by F1 and
seventh by AUC, and that no pairwise contrast against it survives Holm
correction. Choosing the configuration after seeing which performed best would
have made every reported figure optimistic.

## Why thresholds are reported three ways

The right threshold depends on how many reports a human can read, not on the
data. An F1-optimal threshold tuned on the test partition is optimistic, and we
measure by how much: F1 0.1508 when the threshold is chosen on a held-out
calibration slice, against 0.1745 when tuned on the test data.

Every comparative F1 in the selection and imbalance tables uses the tuned form
and is therefore an upper bound. The budget analysis uses no threshold at all,
only the ranking, which is why it is the operating point we recommend.

## Why ten seeds

A single partition contains 151 test positives, which is a noisy basis for
comparison. The seed governs the split, the stochastic components of the search
and model initialisation jointly, so the ten replicates are independent end to
end rather than ten evaluations of one fitted model.

Ten paired replicates afford limited power. Non-significant results here mean
an absence of detectable difference, not evidence of equivalence, and the
paper says so.

## Known instability: the maximum calibration error

Almost every prediction falls below 0.2, so only one or two equal-width bins
are occupied and the maximum is taken over very few, sometimes sparsely
populated, cells. Across ten partitions its standard deviation exceeds its
mean. It is reported for completeness and relied on for nothing.
