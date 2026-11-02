#!/usr/bin/env python3
# =====================================================================================
#  HIMALAYAN EXPEDITION FATALITY RISK - MASTER EXPERIMENT (v5)
#
#  v5 CHANGELOG - all corrections, each verified against the raw Kaggle
#  archives. Items 1-3 change the reported numbers; 4-5 do not.
#
#   1. The weather archive is HOURLY (24 rows per day, UTC). v4 windowed over
#      those rows directly, so heavy_snow_days, extreme_wind_days,
#      rapid_warming_days and storm_days counted HOURS, and snow_max was the
#      largest hourly fall rather than the largest daily one. Now collapsed to
#      one row per calendar day before any window is taken.
#   2. Two counters were identically zero for all 9,457 expeditions. Snowfall
#      is in CENTIMETRES, so ">10" meant 10 cm within a single hour, which
#      never happens; and across 443,799 peak-days the daily maximum 100 m
#      wind never reaches 70 km/h (largest observed 58.5). Both are replaced
#      by exceedances of each peak's own 95th-percentile climatology.
#   3. Raw counts correlated with exposure duration at r = 0.66 to 0.72, so
#      they partly measured expedition length. Now expressed as the share of
#      days in the window.
#   4. The Y/N archive flags are stored as BOOLEANS in this release. The v4
#      parser tested for the string "Y", so oxygen_used, oxygen_free_ascent,
#      is_standard_route and is_commercial_route were zero in every row. The
#      underlying signal still reached the models through the raw o2used,
#      stdrte and comrte columns, so no v4 result was invalidated by this.
#   5. A zero-variance screen now runs ahead of the correlation screen, and the
#      GPU probe no longer reports "cuda" on a CPU-only Mac.
#   6. IMPUTATION LEAKAGE. v4 filled numeric gaps with the median of the whole
#      9,457-row table and split afterwards, so test values informed training
#      imputation - in every protocol, including leave-one-peak-out and the
#      temporal eras. The manuscript claimed training-set medians. Imputation
#      now happens inside screen(), from the training side of each split only.
#      23.5% of rows have a missing totdays, so this was not a trivial amount
#      of information.
#   7. NEW EXPERIMENT - feature-family ablation. Weather features discriminate
#      individually at AUC 0.50-0.56, while team size reaches 0.69 and year
#      0.67. Whether weather adds anything jointly was never tested, and it is
#      the first question a hazards referee will ask about a paper with
#      "weather-aware" in the title. Sheets 32 and 33.
#
#  One script. One run. Every number in the manuscript comes from here.
#
#  Runs on macOS/Linux from a terminal. Designed to survive being left overnight:
#  it checkpoints after every unit of work, writes a STATUS file you can read at
#  any time, and resumes exactly where it stopped if interrupted.
#
#  IMPORTANT - this version applies the calibration repair everywhere. The
#  post-hoc calibration tool splits its internal folds WITHOUT shuffling; when
#  training rows arrive grouped by mountain or sorted by date, ranking ability
#  collapses by about 0.17 AUC. Every model here is fitted with shuffled rows and
#  a shuffled internal split. The faulty path is retained ONLY inside Study J,
#  which characterises the trap deliberately.
#
#  Usage:  python master_experiment.py            (full run, ~10h)
#          python master_experiment.py --tier smoke   (~10 min, validates setup)
#          python master_experiment.py --status        (print progress and exit)
# =====================================================================================
from __future__ import annotations
import argparse, glob, json, os, platform, random, sys, time, warnings, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupKFold, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

# ---------------------------------------------------------------- CLI and tiers
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tier", default="full", choices=["smoke", "standard", "full"])
ap.add_argument("--status", action="store_true", help="print progress and exit")
ap.add_argument("--outdir", default="results")
ARGS = ap.parse_args()

TIERS = {"smoke":    dict(seeds=2,  pop=6,  iters=3,  sens_seeds=1, sens_pop=4,  sens_iters=2,  abl=1, tune=1),
         "standard": dict(seeds=5,  pop=20, iters=30, sens_seeds=3, sens_pop=10, sens_iters=15, abl=3, tune=2),
         "full":     dict(seeds=10, pop=20, iters=30, sens_seeds=3, sens_pop=10, sens_iters=15, abl=5, tune=3)}
T = TIERS[ARGS.tier]
SEEDS       = [42 + 7 * i for i in range(T["seeds"])]
POP, ITERS  = T["pop"], T["iters"]
SENS_SEEDS  = SEEDS[:T["sens_seeds"]]
SENS_POP, SENS_ITERS = T["sens_pop"], T["sens_iters"]
ABL_SEEDS   = SEEDS[:T["abl"]]
TUNE_SEEDS  = SEEDS[:T["tune"]]

ROOT = Path(ARGS.outdir).resolve()
FIG, RAW, ART, CKPT, LOGD = ROOT/"figures", ROOT/"raw", ROOT/"artifacts", ROOT/"checkpoints", ROOT/"logs"
for d in (ROOT, FIG, RAW, ART, CKPT, LOGD): d.mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "STATUS.txt"

if ARGS.status:
    print(STATUS.read_text() if STATUS.exists() else "no run has started yet")
    sys.exit(0)

# ---------------------------------------------------------------- constants
CORR_THRESHOLD, TEST_SIZE, INNER_VAL = 0.90, 0.30, 0.20
SURROGATE   = dict(n_estimators=50, max_depth=6, learning_rate=0.1)
FINAL_XGB   = dict(n_estimators=100, max_depth=4, learning_rate=0.1)
FITNESS_W   = dict(recall=0.5, f1=0.3, precision=0.2)
LAMBDA_SIZE = 0.02
CALIB_METHOD, CALIB_CV, ECE_BINS = "sigmoid", 3, 10
THR_F1 = dict(low=0.10, high=0.30, points=41)
THR_W  = dict(low=0.10, high=0.20, points=41)
FINAL_THRESHOLD, THRESHOLD_VAL_SIZE = 0.110, 0.25
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
TEMPORAL_CUTOFFS = [1990, 2000, 2010]
LOPO_MIN_N, LOPO_MIN_POS = 50, 5
ARCHIVAL = ["claimed", "disputed", "primref"]
OUTCOME_FIELDS = ['mdeaths','death_rate','severe_outcome','hdeaths','smtmembers',
                  'success1','success2','success3','success4']
TERMREASON = 'termreason'
CLIMATE = ["temp_mean","temp_max","temp_min","temp_std","snow_total","snow_max",
           "snow_extreme_rate","wind_mean","wind_max","wind_extreme_rate",
           "pressure_mean","pressure_min","warming_extreme_rate","solid_precip_rate"]
# v5: the three fixed-threshold counters were replaced. See the header changelog.
EXTREME_PCTL = 95        # per-peak percentile defining an extreme day
WEATHER_SOURCE = dict(
    expeditions="kaggle:siddharth0935/himalayan-expeditions (The Himalayan Database)",
    weather="kaggle:bonesclarke26/historic-weather-data-for-himalayan-peaks v6",
    weather_upstream="Open-Meteo Historical Weather API (https://open-meteo.com)",
    weather_licence="CC BY 4.0",
    weather_granularity="hourly, UTC (GMT+0); collapsed to daily before use",
    snowfall_unit="cm")
TUNING_GRID = dict(max_depth=[3,4,6], n_estimators=[100,300],
                   learning_rate=[0.05,0.1], subsample=[0.8,1.0])

N_THREADS = max(1, (os.cpu_count() or 4))
def resolve_device():
    """XGBoost does not raise when asked for cuda on a CPU-only machine: it warns
    and falls back. Catching exceptions alone therefore reported "cuda" on an M2
    Mac, which is how the v4 manifest came to claim a GPU that was never used."""
    import warnings as _w, io as _io, contextlib as _c
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            buf = _io.StringIO()
            with _c.redirect_stderr(buf):
                m = XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", verbosity=1)
                m.fit(np.zeros((4,2)), np.array([0,1,0,1])); m.predict(np.zeros((2,2)))
            text = " ".join(str(c.message) for c in caught) + buf.getvalue()
        return "cpu" if ("No visible GPU" in text or "changed from GPU to CPU" in text) else "cuda"
    except Exception:
        return "cpu"
DEVICE = resolve_device()

def set_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s); random.seed(s); np.random.seed(s)

# ---------------------------------------------------------------- progress
T0 = time.time()
STAGES = ["data","master figure","baselines","feature selection","imbalance",
          "calibration","final model","sensitivity","leakage ablation",
          "honest threshold","budget cutoffs","label shuffle","calibration methods",
          "linear baseline","temporal split","leave-one-peak-out","tuning",
          "repeated k-fold","archival columns","calibration trap","feature families","figures","workbook"]
_done = []
def stage(name, note=""):
    _done.append(name)
    el = (time.time()-T0)/60
    lines = [f"Himalayan master experiment  |  tier={ARGS.tier}  device={DEVICE}",
             f"started  {datetime.fromtimestamp(T0).strftime('%Y-%m-%d %H:%M:%S')}",
             f"elapsed  {el:.1f} min", ""]
    for s in STAGES:
        mark = "[x]" if s in _done else ("[>]" if s == name else "[ ]")
        lines.append(f"  {mark} {s}")
    lines += ["", f"last: {name} {note}"]
    STATUS.write_text("\n".join(lines))
    print(f"\n{'='*78}\n  {name.upper()}  ({el:.1f} min elapsed)\n{'='*78}", flush=True)

def ck_load(n):
    p = CKPT/f"{n}.jsonl"
    if not p.exists(): return {}
    out={}
    for line in p.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try: r=json.loads(line); out[r["_k"]]=r
        except json.JSONDecodeError: continue
    return out
def ck_add(n,k,rec):
    r=dict(rec); r["_k"]=k
    with open(CKPT/f"{n}.jsonl","a") as fh: fh.write(json.dumps(r,default=float)+"\n")

# =====================================================================================
#  DATA
# =====================================================================================
stage("data")
try:
    import kagglehub
except ImportError:
    os.system(f"{sys.executable} -m pip install -q kagglehub"); import kagglehub

EP = kagglehub.dataset_download("siddharth0935/himalayan-expeditions")
WP = kagglehub.dataset_download("bonesclarke26/historic-weather-data-for-himalayan-peaks")
print("  expeditions:", EP, "\n  weather    :", WP, flush=True)

E = pd.read_csv(os.path.join(EP,"exped.csv"), low_memory=False)
for c in ["bcdate","smtdate","termdate"]:
    E[c] = pd.to_datetime(E[c], errors="coerce").dt.date
try:
    PEAKS_META = pd.read_csv(os.path.join(EP,"peaks.csv"), low_memory=False)
except Exception:
    PEAKS_META = pd.DataFrame()

# The archive is HOURLY (24 rows per calendar day, timestamped UTC). v4 windowed
# straight over those hourly rows, so every counter named "_days" was really
# counting hours. Collapse to one row per day first.
W, THRESH = {}, {}
for f in sorted(glob.glob(os.path.join(WP,"*.parquet"))):
    k = os.path.basename(f).replace(".parquet","").upper()
    d = pd.read_parquet(f, columns=["date","temperature_2m","snowfall",
                                    "wind_speed_100m","surface_pressure","weather_code"])
    d["date"] = pd.to_datetime(d["date"]).dt.date
    D = (d.groupby("date")
           .agg(t_mean=("temperature_2m","mean"), t_max=("temperature_2m","max"),
                t_min=("temperature_2m","min"),   snow=("snowfall","sum"),
                w_max=("wind_speed_100m","max"),  w_mean=("wind_speed_100m","mean"),
                p_mean=("surface_pressure","mean"), p_min=("surface_pressure","min"),
                code=("weather_code","max"))
           .reset_index().sort_values("date").reset_index(drop=True))
    D["dT"] = D["t_mean"].diff()
    # An "extreme" day is defined against THIS peak's own 1980-2024 climatology.
    # Absolute thresholds cannot work here: across 443,799 peak-days the daily
    # maximum 100 m wind never once reaches 70 km/h, so v4's extreme_wind_days
    # was identically zero, and so was heavy_snow_days at 10 cm/hour.
    THRESH[k] = dict(snow=float(np.percentile(D["snow"], EXTREME_PCTL)),
                     wind=float(np.percentile(D["w_max"], EXTREME_PCTL)),
                     warm=float(np.percentile(D["dT"].dropna(), EXTREME_PCTL)))
    W[k] = D

kept = E[E.peakid.isin(set(W))].copy().reset_index(drop=True)
def _interval(r):
    s,e = r["bcdate"], r["termdate"]
    if pd.isna(e) and not pd.isna(r.get("smtdate")): e = r["smtdate"]
    if pd.isna(e) and not pd.isna(s) and not pd.isna(r.get("totdays")):
        e = s + timedelta(days=int(r["totdays"]))
    return s, e
_b = kept.apply(_interval, axis=1, result_type="expand"); kept["s"], kept["e"] = _b[0], _b[1]

EMPTY = {c: np.nan for c in CLIMATE}
_sum = [None]*len(kept)
WEATHER_SPAN = {}
for pk, g in kept.groupby("peakid"):
    w = W.get(pk)
    if w is None:
        for i in g.index: _sum[i] = dict(EMPTY)
        continue
    WEATHER_SPAN[pk] = (w["date"].min(), w["date"].max(), len(w))
    dts = w["date"].to_numpy(); th = THRESH[pk]
    for i, r in g.iterrows():
        if pd.isna(r["s"]) or pd.isna(r["e"]): _sum[i] = dict(EMPTY); continue
        win = w.loc[(dts>=r["s"]) & (dts<=r["e"])]
        if win.empty: _sum[i] = dict(EMPTY); continue
        n = len(win)
        # Counters are expressed as the SHARE of days in the window, not a raw
        # count. v4's raw counts correlated with exposure duration at r = 0.72,
        # so they partly measured how long the expedition lasted.
        _sum[i] = {"temp_mean":win.t_mean.mean(),"temp_max":win.t_max.max(),
                   "temp_min":win.t_min.min(),"temp_std":win.t_mean.std(),
                   "snow_total":win.snow.sum(),"snow_max":win.snow.max(),
                   "snow_extreme_rate":float((win.snow>th["snow"]).sum())/n,
                   "wind_mean":win.w_mean.mean(),"wind_max":win.w_max.max(),
                   "wind_extreme_rate":float((win.w_max>th["wind"]).sum())/n,
                   "pressure_mean":win.p_mean.mean(),"pressure_min":win.p_min.min(),
                   "warming_extreme_rate":float((win.dT>th["warm"]).sum())/n,
                   "solid_precip_rate":float((win.code>=70).sum())/n}

CLIM = pd.DataFrame(_sum)
full = pd.concat([kept, CLIM], axis=1)
# This Kaggle release stores the archive's Y/N flags as BOOLEANS, not as the
# letters Y and N. Testing str(v)=="Y" therefore scored True as "TRUE" and
# returned 0 for every row, which silently produced four all-zero columns
# (oxygen_used, oxygen_free_ascent, is_standard_route, is_commercial_route).
_TRUE = {"Y","YES","TRUE","T","1","1.0"}
_yes = lambda s: s.apply(lambda v: 1 if str(v).strip().upper() in _TRUE else 0)
_team = full.totmembers.replace(0,np.nan); _days = full.totdays.replace(0,np.nan)
full["team_size"]=full.totmembers; full["death_rate"]=full.mdeaths/_team
full["summit_rate"]=full.smtmembers/_team
full["oxygen_used"]=_yes(full.o2used); full["oxygen_free_ascent"]=_yes(full.o2none)
full["exposure_days"]=full.totdays; full["elevation_risk"]=full.highpoint
full["camp_density"]=full.camps/_days; full["technical_difficulty"]=full.rope/_days
full["is_standard_route"]=_yes(full.stdrte); full["is_commercial_route"]=_yes(full.comrte)
full=pd.concat([full,pd.get_dummies(full.season,prefix="season")],axis=1)
full["fatality"]=(full.mdeaths>0).astype(int)
full["severe_outcome"]=(full.death_rate.notna()&(full.death_rate>0.20)).astype(int)

META = pd.DataFrame({"peakid": full["peakid"].values,
                     "bc_year": [d.year if not pd.isna(d) else np.nan for d in kept["bcdate"]],
                     "year": pd.to_numeric(full.get("year"), errors="coerce").values
                              if "year" in full else np.nan,
                     "interval_missing": CLIM["temp_mean"].isna().astype(int).values,
                     "totmembers": full["totmembers"].values,
                     "mdeaths": full["mdeaths"].values,
                     "fatality": full["fatality"].values})
META["year_final"] = META["bc_year"].fillna(META["year"])

def build_table(exclude_termreason=True, drop_archival=False):
    drop = [c for c in OUTCOME_FIELDS if c in full.columns]
    if exclude_termreason and TERMREASON in full.columns: drop.append(TERMREASON)
    if drop_archival: drop += [c for c in ARCHIVAL if c in full.columns]
    npred = ['expid','peakid','primid','chksum','route1','route2','route3','route4','leaders',
             'sponsor','nation','host','countries','approach','agency','campsites','accidents',
             'achievment','othersmts','termnote','ascent1','ascent2','ascent3','ascent4',
             'bcdate','smtdate','termdate','s','e','smttime','smtdays','summit_rate']
    cl = full.drop(columns=drop + [c for c in npred if c in full.columns])
    # v5: NO imputation here. v4 filled every numeric gap with the median of the
    # WHOLE table and only then split, so test-partition values informed the
    # training imputation - and the manuscript claimed training-set medians,
    # which was simply not what the code did. Missing values are now left in
    # place and imputed inside screen(), from training medians only, separately
    # for every split, fold, era and held-out peak.
    for c in CLIMATE:
        if c in cl: cl[f"{c}_missing"]=cl[c].isna().astype(int)
    t = cl.copy()
    for c in t.select_dtypes(exclude=["int64","float64","int32","float32"]).columns:
        t[c] = LabelEncoder().fit_transform(t[c].astype(str))
    return t

TABLE = build_table()
X_ALL, Y_ALL = TABLE.drop(columns=["fatality"]), TABLE["fatality"]
PEAKID = META["peakid"]

DEAD_COLUMNS = []
def screen(Xtr, Xte):
    """Training-median imputation, then zero-variance, then collinearity.

    Every generalisation protocol in this script routes through here, so this is
    the one place that has to be right: random splits, temporal eras, grouped
    leave-one-peak-out folds and repeated k-fold all get medians computed from
    their own training side and nothing else.

    v4 had no variance screen, so six constant columns reached the models and
    the feature search. A constant column cannot change a tree prediction, but
    the optimisers still 'selected' those six in 41-50% of runs - chance rate
    for a random binary mask - which is itself evidence of how flat the fitness
    landscape is."""
    med = Xtr.median(numeric_only=True)
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    if Xtr.isna().any().any() or Xte.isna().any().any():
        # a column entirely absent on the training side has no median
        Xtr, Xte = Xtr.fillna(0), Xte.fillna(0)
    dead = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) <= 1]
    if dead:
        Xtr, Xte = Xtr.drop(columns=dead), Xte.drop(columns=dead)
        if not DEAD_COLUMNS: DEAD_COLUMNS.extend(dead)
    corr = Xtr.corr().abs()
    up = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    d = [c for c in up.columns if (up[c] > CORR_THRESHOLD).any()]
    return Xtr.drop(columns=d), Xte.drop(columns=d), dead + d

def split(seed, table=None):
    t = TABLE if table is None else table
    X, y = t.drop(columns=["fatality"]), t["fatality"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    Xtr, Xte, _ = screen(Xtr, Xte)
    return Xtr, Xte, ytr, yte

_a, _b2, _c, _d = split(SEEDS[0])
DATA_STATS = dict(expeditions_total=len(E), expeditions_retained=len(kept),
                  weather_peaks=len(W), peaks_in_data=int(kept.peakid.nunique()),
                  pct_no_interval=round(float(CLIM["temp_mean"].isna().mean()*100),4),
                  positives=int(full.fatality.sum()),
                  positive_rate_pct=round(float(full.fatality.mean()*100),4),
                  imbalance_ratio=round(float((1-full.fatality.mean())/full.fatality.mean()),3),
                  encoded_columns=int(X_ALL.shape[1]),
                  columns_after_screen=int(_a.shape[1]),
                  train_rows=len(_a), test_rows=len(_b2),
                  train_positives=int(_c.sum()), test_positives=int(_d.sum()),
                  duplicate_rows=int(X_ALL.duplicated(keep=False).sum()))
for k,v in DATA_STATS.items(): print(f"   {k:28s} {v}", flush=True)

# =====================================================================================
#  MODELLING - the calibration repair lives here and is applied everywhere
# =====================================================================================
def spw(y):
    y = np.asarray(y); p = int((y==1).sum()); n = int((y==0).sum())
    return n/p if p else 1.0

def make_xgb(y, seed, weighted=True, **kw):
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
             device=DEVICE, verbosity=0, random_state=seed, n_jobs=N_THREADS)
    if weighted: p["scale_pos_weight"] = spw(y)
    p.update(kw); return XGBClassifier(**p)

def fit_model(Xtr, ytr, seed, calibrate=True, method=CALIB_METHOD, legacy=False, **kw):
    """THE REPAIR.

    CalibratedClassifierCV given a plain integer cv splits its internal folds
    WITHOUT shuffling. When training rows arrive grouped by mountain or ordered
    by date - which they do here - those folds become unrepresentative clusters,
    three different correction curves get averaged, and ranking ability drops by
    about 0.17 AUC. Averaging differing monotone maps is not monotone.

    So: shuffle the rows, and hand the calibrator a shuffled StratifiedKFold.
    legacy=True reproduces the faulty behaviour, used only by Study J."""
    if not legacy:
        r = np.random.RandomState(seed)
        idx = np.arange(len(ytr)); r.shuffle(idx)
        Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]
    base = make_xgb(ytr, seed, **kw)
    if not calibrate:
        m = base
    else:
        cv = CALIB_CV if legacy else StratifiedKFold(CALIB_CV, shuffle=True, random_state=seed)
        m = CalibratedClassifierCV(base, method=method, cv=cv)
    set_seeds(seed); m.fit(Xtr, ytr)
    return m

def minority(y, pred):
    return dict(Accuracy=accuracy_score(y,pred),
                Precision=precision_score(y,pred,zero_division=0),
                Recall=recall_score(y,pred,zero_division=0),
                F1=f1_score(y,pred,zero_division=0))

def ece(y, p, n_bins=ECE_BINS, adaptive=False):
    y=np.asarray(y,float); p=np.asarray(p,float)
    edges = np.unique(np.quantile(p,np.linspace(0,1,n_bins+1))) if adaptive \
            else np.linspace(0,1,n_bins+1)
    if len(edges)<3: return np.nan, np.nan
    e=m=0.0; N=len(y)
    for lo,hi in zip(edges[:-1],edges[1:]):
        msk=(p>lo)&(p<=hi) if lo>edges[0] else (p>=lo)&(p<=hi)
        if msk.sum()==0: continue
        gap=abs(y[msk].mean()-p[msk].mean()); e+=(msk.sum()/N)*gap; m=max(m,gap)
    return float(e), float(m)

def calib_report(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    base=float(y.mean()); br=brier_score_loss(y,p); ref=base*(1-base)
    e,mx=ece(y,p); ae,amx=ece(y,p,adaptive=True)
    return dict(Brier=br, BrierSkillScore=1-br/ref if ref>0 else np.nan,
                ECE=e, MCE=mx, AdaptiveECE=ae, AdaptiveMCE=amx,
                LogLoss=log_loss(y,p,labels=[0,1]), MeanPredicted=float(p.mean()),
                BaseRate=base, ROC_AUC=roc_auc_score(y,p),
                AveragePrecision=average_precision_score(y,p))

def thr_f1(y,p,g=THR_F1):
    best=(0.5,-1,0,0)
    for t in np.linspace(g["low"],g["high"],g["points"]):
        pr=(p>=t).astype(int); f=f1_score(y,pr,zero_division=0)
        if f>best[1]: best=(float(t),float(f),float(precision_score(y,pr,zero_division=0)),
                            float(recall_score(y,pr,zero_division=0)))
    return best

def thr_weighted(y,p,g=THR_W,wr=0.6,wp=0.4):
    best=(0.5,-1,0,0)
    for t in np.linspace(g["low"],g["high"],g["points"]):
        pr=(p>=t).astype(int)
        pre=precision_score(y,pr,zero_division=0); rec=recall_score(y,pr,zero_division=0)
        o=wr*rec+wp*pre
        if o>best[1]: best=(float(t),float(o),float(pre),float(rec))
    return best

def degeneracy(y,pred,base):
    y=np.asarray(y); pred=np.asarray(pred); ppr=float(pred.mean())
    rec=recall_score(y,pred,zero_division=0); pre=precision_score(y,pred,zero_division=0)
    acc=accuracy_score(y,pred); flags=[]
    if ppr>=0.90: flags.append("flags almost everything")
    if ppr<=0.001: flags.append("flags nothing")
    if rec>=0.95 and pre<=base*1.15: flags.append("recall~1 at base-rate precision")
    if acc<base*2 and ppr>0.5: flags.append("accuracy below trivial floor")
    return dict(PredictedPositiveRate=ppr, Degenerate=bool(flags),
                DegeneracyReason="; ".join(flags))

class Evaluator:
    """Scores one binary feature mask. Caches by mask, so repeated candidates cost
    nothing and ModelFits records genuine work."""
    def __init__(self,Xtr,Xva,ytr,yva,seed,weights=None,lam=LAMBDA_SIZE,floor=0.0,sur=None):
        self.Xtr,self.Xva,self.ytr,self.yva=Xtr,Xva,ytr,yva
        self.seed=seed; self.w=weights or FITNESS_W; self.lam=lam; self.floor=floor
        self.sur=sur or SURROGATE; self.spw=spw(ytr); self.cache={}
        self.n_req=self.n_fit=0; self.secs=0.0
    def raw(self,mask):
        self.n_req+=1
        if mask.sum()==0: return 0.0
        k=mask.tobytes()
        if k in self.cache: return self.cache[k]
        cols=np.where(mask==1)[0]
        m=XGBClassifier(objective="binary:logistic",eval_metric="logloss",tree_method="hist",
                        device=DEVICE,verbosity=0,random_state=self.seed,n_jobs=N_THREADS,
                        scale_pos_weight=self.spw,**self.sur)
        t=time.time(); m.fit(self.Xtr.iloc[:,cols],self.ytr)
        pred=m.predict(self.Xva.iloc[:,cols]); self.secs+=time.time()-t; self.n_fit+=1
        p=precision_score(self.yva,pred,zero_division=0)
        r=recall_score(self.yva,pred,zero_division=0)
        f=f1_score(self.yva,pred,zero_division=0)
        s=self.w["recall"]*r+self.w["f1"]*f+self.w["precision"]*p
        if self.floor>0 and p<self.floor: s*= (p/self.floor)
        self.cache[k]=s; return s
    def __call__(self,mask):
        if mask.sum()==0: return 0.0
        return self.raw(mask)-self.lam*(mask.sum()/len(mask))

# =====================================================================================
#  OPTIMISERS
# =====================================================================================
def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-60,60)))
def _init(p,n): return np.random.randint(0,2,(p,n))
def _best(P,S): i=int(np.argmax(S)); return P[i].copy(), float(S[i])

def opt_GA(f,n,pop,it,mut=0.02):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for _ in range(it):
        kids=[b.copy()]
        while len(kids)<pop:
            a1,a2=random.sample(list(P),2); cut=np.random.randint(1,max(n-1,2))
            c=np.concatenate([a1[:cut],a2[cut:]]); fl=np.random.rand(n)<mut; c[fl]=1-c[fl]
            kids.append(c)
        P=np.array(kids); S=np.array([f(p) for p in P])
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_PSO(f,n,pop,it,w=0.7,c1=1.5,c2=1.5):
    P=_init(pop,n); V=np.random.uniform(-1,1,(pop,n)); pb=P.copy()
    pbs=np.array([f(p) for p in P]); b,bs=_best(pb,pbs); h=[]
    for _ in range(it):
        for i in range(pop):
            r1,r2=np.random.rand(n),np.random.rand(n)
            V[i]=w*V[i]+c1*r1*(pb[i]-P[i])+c2*r2*(b-P[i])
            P[i]=(_sig(V[i])>np.random.rand(n)).astype(int); s=f(P[i])
            if s>pbs[i]:
                pb[i],pbs[i]=P[i].copy(),s
                if s>bs: b,bs=P[i].copy(),s
        h.append(bs)
    return dict(mask=b,history=h)

def _gwo_step(P,S,f,n,pop,t,it):
    o=np.argsort(S); a_,b_,d_=P[o[-1]],P[o[-2]],P[o[-3]]; a=2-2*t/it; Q=P.copy()
    for i in range(pop):
        pos=[]
        for L in (a_,b_,d_):
            r1,r2=np.random.rand(2); A,C=2*a*r1-a,2*r2
            pos.append(L-A*np.abs(C*L-P[i]))
        Q[i]=(_sig(sum(pos)/3.0)>np.random.rand(n)).astype(int)
    return Q, np.array([f(q) for q in Q])

def opt_GWO(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        P,S=_gwo_step(P,S,f,n,pop,t,it)
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_EVO(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for _ in range(it):
        ctr=P.mean(axis=0)
        for i in range(pop):
            nb=P[random.randint(0,pop-1)]; tau=np.random.rand(4)
            c=P[i]+(tau[0]*b-tau[1]*ctr)+(tau[2]*b-tau[3]*nb)
            c=(_sig(c)>0.5).astype(int); s=f(c)
            if s>S[i]:
                P[i],S[i]=c,s
                if s>bs: b,bs=c.copy(),s
        h.append(bs)
    return dict(mask=b,history=h)

def _foa_step(P,f,n,pop):
    sd=[]
    for tr in P:
        s=tr.copy(); j=np.random.randint(n); s[j]=1-s[j]; sd.append(s)
    pool=np.vstack((P,np.array(sd))); ps=np.array([f(p) for p in pool])
    keep=np.argsort(ps)[-pop:]; return pool[keep], ps[keep]

def opt_FOA(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for _ in range(it):
        P,S=_foa_step(P,f,n,pop)
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_FOA_GWO(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        P,S=_foa_step(P,f,n,pop)
        if S.max()>bs: b,bs=_best(P,S)
        P,S=_gwo_step(P,S,f,n,pop,t,it)
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_GWO_FOA(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        P,S=_gwo_step(P,S,f,n,pop,t,it)
        if S.max()>bs: b,bs=_best(P,S)
        P,S=_foa_step(P,f,n,pop)
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_BWOA(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        a=2-2*t/it
        for i in range(pop):
            A=2*a*np.random.rand()-a; C=2*np.random.rand()
            if np.random.rand()<0.5:
                if abs(A)<1: step=b-A*np.abs(C*b-P[i])
                else:
                    rnd=P[np.random.randint(pop)]; step=rnd-A*np.abs(C*rnd-P[i])
            else:
                l=np.random.uniform(-1,1)
                step=np.abs(b-P[i])*np.exp(l)*np.cos(2*np.pi*l)+b
            P[i]=(_sig(step)>np.random.rand(n)).astype(int)
        S=np.array([f(p) for p in P])
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_BHHO(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        Eo=2*np.random.rand()-1; Ee=2*Eo*(1-t/it)
        for i in range(pop):
            if abs(Ee)>=1:
                rnd=P[np.random.randint(pop)]
                step=rnd-np.random.rand()*np.abs(rnd-2*np.random.rand()*P[i])
            else:
                mean=P.mean(axis=0)
                step=(b-P[i])-Ee*np.abs(2*(1-np.random.rand())*b-P[i]) \
                     if np.random.rand()>=0.5 else b-Ee*np.abs(b-mean)
            P[i]=(_sig(step)>np.random.rand(n)).astype(int)
        S=np.array([f(p) for p in P])
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

def opt_BSSA(f,n,pop,it):
    P=_init(pop,n); S=np.array([f(p) for p in P]); b,bs=_best(P,S); h=[]
    for t in range(it):
        c1=2*np.exp(-((4*(t+1)/it)**2))
        for i in range(pop):
            if i==0:
                c2,c3=np.random.rand(n),np.random.rand(n)
                step=np.where(c3<0.5,b+c1*c2,b-c1*c2)
            else: step=(P[i]+P[i-1])/2.0
            P[i]=(_sig(step)>np.random.rand(n)).astype(int)
        S=np.array([f(p) for p in P])
        if S.max()>bs: b,bs=_best(P,S)
        h.append(bs)
    return dict(mask=b,history=h)

OPTS = {"GA":opt_GA,"PSO":opt_PSO,"GWO":opt_GWO,"EVO":opt_EVO,"FOA":opt_FOA,
        "FOA_GWO":opt_FOA_GWO,"GWO_FOA":opt_GWO_FOA,
        "BWOA":opt_BWOA,"BHHO":opt_BHHO,"BSSA":opt_BSSA}

CITATIONS = {
 "GA":"Holland, Adaptation in Natural and Artificial Systems, 1975",
 "PSO":"Kennedy & Eberhart, ICNN 1995",
 "GWO":"Mirjalili, Mirjalili & Lewis, Adv Eng Softw 69:46-61, 2014",
 "EVO":"Evolutionary centroid variant (this work; retained as a failure case)",
 "FOA":"Ghaemi & Feizi-Derakhshi, Expert Syst Appl 41(15):6676-6687, 2014",
 "FOA_GWO":"Proposed sequential hybrid (this work)",
 "GWO_FOA":"Reversed-order control (this work)",
 "BWOA":"Mirjalili & Lewis, Adv Eng Softw 95:51-67, 2016",
 "BHHO":"Heidari et al., Future Gener Comput Syst 97:849-872, 2019",
 "BSSA":"Mirjalili et al., Adv Eng Softw 114:163-191, 2017",
 "Embedded":"L1 pre-filter with greedy backward elimination (this work)",
 "MutualInfo":"Mutual-information filter baseline",
 "RFE":"Guyon et al., Mach Learn 46:389-422, 2002",
 "AllFeatures":"No selection (control)"}

def sel_embedded(Xtr,ytr,seed,cv=5):
    sc=StandardScaler().fit_transform(Xtr)
    try:
        from sklearn import __version__ as v; mj,mn=(int(x) for x in v.split(".")[:2])
        lr=(LogisticRegression(l1_ratio=1.0,solver="liblinear",class_weight="balanced",
                               max_iter=1000,random_state=seed) if (mj,mn)>=(1,8)
            else LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",
                                    max_iter=1000,random_state=seed))
    except Exception:
        lr=LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",
                              max_iter=1000,random_state=seed)
    lr.fit(sc,ytr)
    feats=list(Xtr.columns[lr.coef_.ravel()!=0]) or list(Xtr.columns)
    def score(fs):
        skf=StratifiedKFold(cv,shuffle=True,random_state=seed); out=[]
        for tr,va in skf.split(Xtr[fs],ytr):
            m=make_xgb(ytr.iloc[tr],seed,**FINAL_XGB); m.fit(Xtr[fs].iloc[tr],ytr.iloc[tr])
            pr=m.predict(Xtr[fs].iloc[va])
            out.append(FITNESS_W["recall"]*recall_score(ytr.iloc[va],pr,zero_division=0)
                      +FITNESS_W["f1"]*f1_score(ytr.iloc[va],pr,zero_division=0)
                      +FITNESS_W["precision"]*precision_score(ytr.iloc[va],pr,zero_division=0))
        return float(np.mean(out))
    cur,best=list(feats),score(feats); imp=True
    while imp and len(cur)>1:
        imp=False
        tri=[(f,score([x for x in cur if x!=f])) for f in cur]
        f,s=max(tri,key=lambda z:z[1])
        if s>best: cur.remove(f); best=s; imp=True
    return cur

def sel_mi(Xtr,ytr,seed,k):
    mi=mutual_info_classif(Xtr,ytr,random_state=seed)
    return list(Xtr.columns[np.argsort(mi)[::-1][:k]])
def sel_rfe(Xtr,ytr,seed,k):
    return list(Xtr.columns[RFE(make_xgb(ytr,seed,n_estimators=50,max_depth=4,
                                         learning_rate=0.1),
                                n_features_to_select=k,step=0.2).fit(Xtr,ytr).support_])

# =====================================================================================
#  MASTER DATASET FIGURE
#  Built entirely from the data. Coordinates are written to a CSV you verify
#  before publication - the map panel renders only if that file is complete.
# =====================================================================================
stage("master figure")
sns.set_theme(style="whitegrid", context="paper")

APPROX_COORDS = {  # VERIFY BEFORE PUBLICATION - approximate, for plotting only
 "EVER":(27.9881,86.9250),"LHOT":(27.9619,86.9331),"MAKA":(27.8897,87.0889),
 "CHOY":(28.0942,86.6608),"KANG":(27.7025,88.1475),"MANA":(28.5497,84.5597),
 "ANN1":(28.5956,83.8203),"DHA1":(28.6967,83.4875),"AMAD":(27.8617,86.8611),
 "PUMO":(28.0069,86.8281),"NUPT":(27.9667,86.8889),"BARU":(27.8722,86.9806),
 "HIML":(28.6350,84.1697),"LANG":(28.2558,85.5169),"TILI":(28.6961,83.8494),
 "GAUR":(27.9597,86.3364),"JANU":(27.6817,88.0447),"PUTH":(28.7444,83.1472),
 "TUKU":(28.7433,83.4675),"KGUR":(28.5069,84.4286),"ANN2":(28.5358,84.1225),
 "ANN3":(28.5850,83.9917),"ANN4":(28.5375,84.0819),"ANNS":(28.5178,83.8081),
 "LSHR":(27.9614,86.9331),"MAK2":(27.8889,87.0778),"LUNR":(28.9000,86.3500)}

def master_dataset_figure():
    m = META.copy()
    g = (m.groupby("peakid")
           .agg(Expeditions=("fatality","size"), Fatal=("fatality","sum"),
                Deaths=("mdeaths","sum"), Members=("totmembers","sum"),
                FirstYear=("year_final","min"), LastYear=("year_final","max"))
           .reset_index())
    g["FatalityRate"] = (g.Fatal/g.Expeditions*100).round(2)
    g = g.sort_values("Expeditions", ascending=False)
    g.to_csv(RAW/"peak_summary.csv", index=False)

    coord = pd.DataFrame([dict(peakid=p, lat=APPROX_COORDS.get(p,(np.nan,np.nan))[0],
                               lon=APPROX_COORDS.get(p,(np.nan,np.nan))[1],
                               VERIFIED="NO") for p in g.peakid])
    cpath = ART/"peak_coordinates_VERIFY.csv"
    if not cpath.exists(): coord.to_csv(cpath, index=False)
    coord = pd.read_csv(cpath)
    have = coord.dropna(subset=["lat","lon"])

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.28)

    # (a) map or ranked bars
    ax = fig.add_subplot(gs[0, :2])
    if len(have) >= 5:
        d = g.merge(have, on="peakid")
        sc = ax.scatter(d.lon, d.lat, s=np.sqrt(d.Expeditions)*14,
                        c=d.FatalityRate, cmap="YlOrRd", edgecolor="black", linewidth=.6,
                        alpha=.9, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r.peakid, (r.lon, r.lat), fontsize=7,
                        xytext=(4,4), textcoords="offset points")
        plt.colorbar(sc, ax=ax, label="Fatality rate (%)", shrink=.85)
        ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
        ax.set_title("(a) Study peaks: circle area is expedition count,\n"
                     "colour is the share of expeditions with a member death")
    else:
        d = g.head(20)
        b = ax.barh(d.peakid[::-1], d.Expeditions[::-1],
                    color=plt.cm.YlOrRd(d.FatalityRate[::-1]/max(d.FatalityRate.max(),1)))
        ax.set_xlabel("Expeditions recorded"); ax.set_title(
            "(a) Expeditions per peak, shaded by fatality rate\n"
            "(fill artifacts/peak_coordinates_VERIFY.csv to get the map instead)")

    # (b) fatality rate per peak
    ax = fig.add_subplot(gs[0, 2])
    d = g.sort_values("FatalityRate", ascending=True).tail(15)
    ax.barh(d.peakid, d.FatalityRate, color="tab:red", alpha=.8)
    ax.axvline(float(m.fatality.mean()*100), ls="--", c="black", lw=1,
               label=f"overall {m.fatality.mean()*100:.2f}%")
    ax.set_xlabel("Fatality rate (%)"); ax.legend(fontsize=7)
    ax.set_title("(b) Deadliest peaks by rate")

    # (c) expeditions and fatality rate over time
    ax = fig.add_subplot(gs[1, :2])
    yr = m.dropna(subset=["year_final"]).copy()
    yr["decade"] = (yr.year_final//10*10).astype(int)
    dd = yr.groupby("decade").agg(n=("fatality","size"), rate=("fatality","mean")).reset_index()
    ax.bar(dd.decade, dd.n, width=7, color="steelblue", alpha=.75, label="expeditions")
    ax.set_ylabel("Expeditions", color="steelblue"); ax.set_xlabel("Decade")
    a2 = ax.twinx(); a2.plot(dd.decade, dd.rate*100, color="crimson", marker="o", lw=2,
                             label="fatality rate")
    a2.set_ylabel("Fatality rate (%)", color="crimson")
    ax.set_title("(c) Activity grew while the fatality rate fell - the distribution shift\n"
                 "that makes chronological evaluation harder than random splitting")

    # (d) class imbalance
    ax = fig.add_subplot(gs[1, 2])
    n_pos = int(m.fatality.sum()); n_neg = len(m)-n_pos
    ax.bar(["non-fatal","fatal"], [n_neg,n_pos], color=["lightsteelblue","crimson"])
    for i,v in enumerate([n_neg,n_pos]):
        ax.text(i, v, f"{v:,}\n({v/len(m)*100:.2f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Expeditions")
    ax.set_title(f"(d) Class imbalance\n{n_neg/max(n_pos,1):.1f} : 1")

    # (e) weather coverage
    ax = fig.add_subplot(gs[2, 0])
    cov = m.groupby("peakid")["interval_missing"].mean().sort_values()
    ax.hist(cov*100, bins=12, color="seagreen", alpha=.85)
    ax.set_xlabel("% of expeditions with no usable weather interval")
    ax.set_ylabel("Peaks")
    ax.set_title(f"(e) Weather coverage\noverall {m.interval_missing.mean()*100:.1f}% unusable")

    # (f) team size against fatality
    ax = fig.add_subplot(gs[2, 1])
    q = m.copy(); q["bin"] = pd.cut(q.totmembers, [0,3,6,10,15,25,200],
                                    labels=["1-3","4-6","7-10","11-15","16-25","26+"])
    r = q.groupby("bin", observed=True).agg(rate=("fatality","mean"), n=("fatality","size")).reset_index()
    ax.bar(r["bin"].astype(str), r.rate*100, color="darkorange", alpha=.85)
    for i,(v,nn) in enumerate(zip(r.rate*100, r.n)):
        ax.text(i, v, f"n={nn}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Team size"); ax.set_ylabel("Fatality rate (%)")
    ax.set_title("(f) Larger parties, higher rate")

    # (g) seasons
    ax = fig.add_subplot(gs[2, 2])
    if "season" in full.columns:
        s = pd.DataFrame({"season": full["season"].astype(str), "fatality": full["fatality"]})
        rr = s.groupby("season").agg(rate=("fatality","mean"), n=("fatality","size")).reset_index()
        rr = rr[rr.n >= 30].sort_values("rate")
        ax.bar(rr.season, rr.rate*100, color="mediumpurple", alpha=.85)
        ax.set_ylabel("Fatality rate (%)"); ax.tick_params(axis="x", rotation=30)
        ax.set_title("(g) By season")

    fig.suptitle(f"Himalayan expedition dataset: {len(m):,} expeditions on "
                 f"{m.peakid.nunique()} peaks, {n_pos} with a member death "
                 f"({m.fatality.mean()*100:.2f}%)", fontsize=13, y=0.985)
    for ext in ("png","pdf"): fig.savefig(FIG/f"fig01_dataset_overview.{ext}",
                                          dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("   fig01_dataset_overview written", flush=True)
    print(f"   coordinates template: {cpath}", flush=True)
    print(f"   map panel active: {len(have)>=5} ({len(have)} peaks with coordinates)", flush=True)
    return g

PEAK_SUMMARY = master_dataset_figure()

# =====================================================================================
#  STUDIES
# =====================================================================================
def eval_subset(feats, Xtr, Xte, ytr, yte, seed):
    fs=[f for f in feats if f in Xtr.columns]
    if not fs: return dict(Accuracy=np.nan,Precision=np.nan,Recall=np.nan,F1=np.nan,
                           NumFeatures=0,Degenerate=True,DegeneracyReason="empty",
                           PredictedPositiveRate=np.nan,ROC_AUC=np.nan,AveragePrecision=np.nan)
    m=fit_model(Xtr[fs],ytr,seed,calibrate=False)
    pred=m.predict(Xte[fs]); pr=m.predict_proba(Xte[fs])[:,1]
    o=minority(yte,pred); o.update(degeneracy(yte,pred,float(yte.mean())))
    o["ROC_AUC"]=roc_auc_score(yte,pr); o["AveragePrecision"]=average_precision_score(yte,pr)
    o["NumFeatures"]=len(fs); return o

# ---- Table I -------------------------------------------------------------------
stage("baselines")
_rows=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s)
    for nm,m in {"Decision Tree":DecisionTreeClassifier(class_weight="balanced",random_state=s),
                 "Random Forest":RandomForestClassifier(n_estimators=100,class_weight="balanced",
                                                        random_state=s,n_jobs=N_THREADS),
                 "XGBoost":make_xgb(ytr,s),
                 "k-NN":KNeighborsClassifier(n_neighbors=5)}.items():
        set_seeds(s); m.fit(Xtr,ytr); pred=m.predict(Xte)
        cm=confusion_matrix(yte,pred)
        _rows.append(dict(Model=nm,Seed=s,**{k:round(v,4) for k,v in minority(yte,pred).items()},
                          TruePositives=int(cm[1,1]),TestPositives=int(yte.sum())))
BASE=pd.DataFrame(_rows)
print(BASE.groupby("Model")[["Accuracy","Precision","Recall","F1"]].mean().round(4).to_string(),flush=True)

# ---- feature selection ---------------------------------------------------------
stage("feature selection")
done=ck_load("selection"); SEL=[]; FEATS={}; HIST={}
tot=len(SEEDS)*(len(OPTS)+4); step=0
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); n=Xtr.shape[1]
    Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=INNER_VAL,stratify=ytr,random_state=s)
    for nm,fn in OPTS.items():
        step+=1; k=f"{nm}|{s}"
        if k in done:
            SEL.append(done[k]); FEATS.setdefault(nm,{})[s]=done[k]["Features"]
            HIST.setdefault(nm,{})[s]=done[k].get("History",[]); continue
        set_seeds(s); ev=Evaluator(Xi,Xv,yi,yv,s); t=time.time()
        r=fn(ev,n,POP,ITERS); wall=time.time()-t
        fts=list(Xtr.columns[np.where(r["mask"]==1)[0]])
        rec=dict(Method=nm,Seed=s,Features=fts,History=r["history"],
                 SearchSeconds=round(wall,3),FitnessRequests=ev.n_req,ModelFits=ev.n_fit,
                 CacheHits=ev.n_req-ev.n_fit,FitSeconds=round(ev.secs,3),
                 FinalFitness=float(r["history"][-1]) if r["history"] else np.nan)
        rec.update(eval_subset(fts,Xtr,Xte,ytr,yte,s))
        SEL.append(rec); ck_add("selection",k,rec)
        FEATS.setdefault(nm,{})[s]=fts; HIST.setdefault(nm,{})[s]=r["history"]
        print(f"   [{step}/{tot}] {nm:9s} seed={s} F1={rec['F1']:.4f} |s|={rec['NumFeatures']:3d} "
              f"{wall:6.1f}s fits={ev.n_fit} | {(time.time()-T0)/60:.0f}m",flush=True)
    for nm in ["Embedded","MutualInfo","RFE","AllFeatures"]:
        step+=1; k=f"{nm}|{s}"
        if k in done:
            SEL.append(done[k]); FEATS.setdefault(nm,{})[s]=done[k]["Features"]; continue
        set_seeds(s); t=time.time()
        if nm=="Embedded": fts=sel_embedded(Xtr,ytr,s)
        elif nm=="AllFeatures": fts=list(Xtr.columns)
        else:
            kk=int(np.median([len(v) for v in FEATS.get("FOA_GWO",{s:[0]*34}).values()])) or 34
            kk=max(2,min(kk,Xtr.shape[1]))
            fts=sel_mi(Xtr,ytr,s,kk) if nm=="MutualInfo" else sel_rfe(Xtr,ytr,s,kk)
        rec=dict(Method=nm,Seed=s,Features=fts,History=[],SearchSeconds=round(time.time()-t,3),
                 FitnessRequests=np.nan,ModelFits=np.nan,CacheHits=np.nan,FitSeconds=np.nan,
                 FinalFitness=np.nan)
        rec.update(eval_subset(fts,Xtr,Xte,ytr,yte,s))
        SEL.append(rec); ck_add("selection",k,rec); FEATS.setdefault(nm,{})[s]=fts
        print(f"   [{step}/{tot}] {nm:9s} seed={s} F1={rec['F1']:.4f} |s|={rec['NumFeatures']:3d}",flush=True)
SELDF=pd.DataFrame(SEL)

def summarise(df,mets=("F1","Precision","Recall","Accuracy","ROC_AUC","AveragePrecision",
                       "NumFeatures","SearchSeconds"),by="Method"):
    out=[]
    for m_,g in df.groupby(by):
        r={by:m_,"Runs":len(g),
           "DegenerateRuns":int(g["Degenerate"].sum()) if "Degenerate" in g else 0}
        for k in mets:
            if k not in g: continue
            v=pd.to_numeric(g[k],errors="coerce").dropna().values
            if not len(v): continue
            mu=float(np.mean(v)); sd=float(np.std(v,ddof=1)) if len(v)>1 else 0.0
            h=(stats.t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))) if len(v)>1 else 0.0
            r[f"{k}_mean"]=round(mu,4); r[f"{k}_std"]=round(sd,4)
            r[f"{k}_ci95_low"]=round(mu-h,4); r[f"{k}_ci95_high"]=round(mu+h,4)
            r[f"{k}_median"]=round(float(np.median(v)),4)
        out.append(r)
    o=pd.DataFrame(out)
    return o.sort_values("F1_mean",ascending=False).reset_index(drop=True) if "F1_mean" in o else o

def holm(p):
    idx=np.argsort(p); adj=np.empty(len(p)); run=0.0
    for rank,i in enumerate(idx):
        run=max(run,(len(p)-rank)*p[i]); adj[i]=min(run,1.0)
    return adj

def paired(df,ref="FOA_GWO",met="F1"):
    piv=df.pivot_table(index="Seed",columns="Method",values=met)
    if ref not in piv.columns: return pd.DataFrame(),{}
    rows=[]
    for m_ in piv.columns:
        if m_==ref: continue
        pr=piv[[ref,m_]].dropna()
        if len(pr)<3:
            rows.append(dict(Comparison=f"{ref} vs {m_}",n=len(pr),Wilcoxon_p=np.nan,
                             Note="too few paired runs")); continue
        a,b=pr[ref].values,pr[m_].values
        try: st,p=stats.wilcoxon(a,b)
        except ValueError: st,p=np.nan,1.0
        d=a-b
        rows.append(dict(Comparison=f"{ref} vs {m_}",n=len(pr),
                         Reference_mean=round(float(a.mean()),4),Other_mean=round(float(b.mean()),4),
                         MedianDiff=round(float(np.median(d)),4),Wins=int((d>0).sum()),
                         Ties=int((d==0).sum()),Losses=int((d<0).sum()),
                         Wilcoxon_p=round(float(p),5),Note=""))
    o=pd.DataFrame(rows)
    if len(o) and o.Wilcoxon_p.notna().any():
        m_=o.Wilcoxon_p.notna()
        o.loc[m_,"Holm_p"]=np.round(holm(o.loc[m_,"Wilcoxon_p"].values),5)
        o["Significant_0.05"]=o["Holm_p"]<0.05
    fr={}
    comp=piv.dropna(axis=1,how="any")
    if comp.shape[1]>=3 and comp.shape[0]>=3:
        try:
            st,p=stats.friedmanchisquare(*[comp[c].values for c in comp.columns])
            fr=dict(statistic=round(float(st),4),p_value=round(float(p),6),
                    methods=list(comp.columns),n_seeds=int(comp.shape[0]))
        except Exception as e: fr=dict(error=str(e))
    return o,fr

SELSUM=summarise(SELDF); SIG,FRIED=paired(SELDF)
print(SELSUM[["Method","F1_mean","F1_std","NumFeatures_mean","DegenerateRuns"]].to_string(index=False),flush=True)
print("friedman:",FRIED,flush=True)

SELECTOR="FOA_GWO"
def subset_for(seed,Xtr,sel=SELECTOR):
    f=[x for x in FEATS.get(sel,{}).get(seed,[]) if x in Xtr.columns]
    return f or list(Xtr.columns)

SAMPLERS={"SMOTE":SMOTE,"ADASYN":ADASYN,"SMOTEENN":SMOTEENN,
          "SMOTETomek":SMOTETomek,"UnderSampling":RandomUnderSampler}
def resample(meth,X,y,seed):
    if meth not in SAMPLERS: return X,y
    Xr,yr=SAMPLERS[meth](random_state=seed).fit_resample(X,y)
    return pd.DataFrame(Xr,columns=X.columns), pd.Series(yr)

# ---- imbalance + calibration ---------------------------------------------------
stage("imbalance")
IMB=[];CAL=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); f=subset_for(s,Xtr); A,B=Xtr[f],Xte[f]; base=float(yte.mean())
    for meth in ["Baseline"]+list(SAMPLERS):
        set_seeds(s); Xr,yr=resample(meth,A,ytr,s)
        for cal in (False,True):
            m=fit_model(Xr,yr,s,calibrate=cal); p=m.predict_proba(B)[:,1]
            rep=calib_report(yte,p)
            tw,_,pw,rw=thr_weighted(yte,p); pw_pred=(p>=tw).astype(int)
            tn,fp,fn,tp=confusion_matrix(yte,pw_pred).ravel()
            CAL.append(dict(Method=meth,Seed=s,Calibrated="yes" if cal else "no",
                            Threshold=round(tw,4),Accuracy=round(accuracy_score(yte,pw_pred),4),
                            Precision=round(pw,4),Recall=round(rw,4),
                            F1=round(f1_score(yte,pw_pred,zero_division=0),4),
                            TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
                            **{k:(round(v,6) if isinstance(v,float) else v) for k,v in rep.items()}))
            if cal:
                tf,f1f,pf,rf=thr_f1(yte,p); pred=(p>=tf).astype(int)
                row=dict(Method=meth,Seed=s,Threshold=round(tf,4),F1=round(f1f,4),
                         Precision=round(pf,4),Recall=round(rf,4),
                         Accuracy=round(accuracy_score(yte,pred),4),
                         TrainRows=int(len(yr)),NumFeatures=len(f))
                row.update(degeneracy(yte,pred,base)); IMB.append(row)
    print(f"   seed {s} done",flush=True)
IMBDF=pd.DataFrame(IMB); CALDF=pd.DataFrame(CAL)
IMBSUM=summarise(IMBDF,("F1","Precision","Recall","Accuracy","Threshold","TrainRows"))
stage("calibration")
CALSUM=(CALDF.groupby(["Method","Calibrated"])
        .agg(Runs=("Seed","nunique"),ECE_mean=("ECE","mean"),ECE_std=("ECE","std"),
             AdaptiveECE_mean=("AdaptiveECE","mean"),MCE_mean=("MCE","mean"),
             Brier_mean=("Brier","mean"),BrierSkill_mean=("BrierSkillScore","mean"),
             LogLoss_mean=("LogLoss","mean"),F1_mean=("F1","mean"),
             Precision_mean=("Precision","mean"),Recall_mean=("Recall","mean"),
             ROC_AUC_mean=("ROC_AUC","mean"),AP_mean=("AveragePrecision","mean"))
        .round(5).reset_index())
print(CALSUM[CALSUM.Method=="Baseline"].to_string(index=False),flush=True)

# ---- final configuration -------------------------------------------------------
stage("final model")
FIN=[];CURVES={}
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); f=subset_for(s,Xtr)
    m=fit_model(Xtr[f],ytr,s,calibrate=True); p=m.predict_proba(Xte[f])[:,1]
    pred=(p>=FINAL_THRESHOLD).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    rep=calib_report(yte,p)
    FIN.append(dict(Seed=s,Threshold=FINAL_THRESHOLD,NumFeatures=len(f),
                    Accuracy=round(accuracy_score(yte,pred),4),
                    Precision=round(precision_score(yte,pred,zero_division=0),4),
                    Recall=round(recall_score(yte,pred,zero_division=0),4),
                    F1=round(f1_score(yte,pred,zero_division=0),4),
                    TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
                    **{k:(round(v,6) if isinstance(v,float) else v) for k,v in rep.items()}))
    CURVES[s]=dict(y=np.asarray(yte),p=p)
FINDF=pd.DataFrame(FIN)
print(FINDF[["Seed","Precision","Recall","F1","ROC_AUC","AveragePrecision","ECE"]].to_string(index=False),flush=True)

# ---- sensitivity ---------------------------------------------------------------
stage("sensitivity")
done=ck_load("sens"); SENS=[]
grid=([("weights",w,LAMBDA_SIZE,0.0) for w in
       [(.5,.3,.2),(.6,.3,.1),(.4,.3,.3),(.7,.2,.1),(.34,.33,.33),(.3,.3,.4),
        (.2,.3,.5),(.5,.5,.0),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]]
     +[("lambda",(.5,.3,.2),l,0.0) for l in [0.,.005,.01,.02,.05,.10,.20]]
     +[("precision_floor",(.5,.3,.2),LAMBDA_SIZE,fl) for fl in [0.,.10,.15,.20]])
tot=len(grid)*len(SENS_SEEDS); st=0
for kind,w,lam,fl in grid:
    for s in SENS_SEEDS:
        st+=1; k=f"{kind}|{w}|{lam}|{fl}|{s}"
        if k in done: SENS.append(done[k]); continue
        Xtr,Xte,ytr,yte=split(s)
        Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=INNER_VAL,stratify=ytr,random_state=s)
        set_seeds(s)
        ev=Evaluator(Xi,Xv,yi,yv,s,weights=dict(recall=w[0],f1=w[1],precision=w[2]),
                     lam=lam,floor=fl)
        r=opt_FOA_GWO(ev,Xtr.shape[1],SENS_POP,SENS_ITERS)
        fts=list(Xtr.columns[np.where(r["mask"]==1)[0]])
        rec=dict(Study=kind,w_recall=w[0],w_f1=w[1],w_precision=w[2],Lambda=lam,
                 PrecisionFloor=fl,Seed=s,ModelFits=ev.n_fit)
        rec.update(eval_subset(fts,Xtr,Xte,ytr,yte,s))
        SENS.append(rec); ck_add("sens",k,rec)
        if st%6==0: print(f"   sens {st}/{tot}",flush=True)
SENSDF=pd.DataFrame(SENS)

# ---- leakage ablation ----------------------------------------------------------
stage("leakage ablation")
done=ck_load("abl"); ABL=[]
T_in=build_table(exclude_termreason=False); T_ex=TABLE
for s in ABL_SEEDS:
    for lab,tb in (("termreason excluded",T_ex),("termreason readmitted",T_in)):
        k=f"{lab}|{s}"
        if k in done: ABL.append(done[k]); continue
        Xtr,Xte,ytr,yte=split(s,tb)
        Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=INNER_VAL,stratify=ytr,random_state=s)
        set_seeds(s); ev=Evaluator(Xi,Xv,yi,yv,s)
        r=opt_FOA_GWO(ev,Xtr.shape[1],POP,ITERS)
        fts=list(Xtr.columns[np.where(r["mask"]==1)[0]])
        m=fit_model(Xtr[fts],ytr,s,calibrate=True); p=m.predict_proba(Xte[fts])[:,1]
        pred=(p>=FINAL_THRESHOLD).astype(int)
        rec=dict(Configuration=lab,Seed=s,NumFeatures=len(fts),
                 TermreasonSelected=bool(TERMREASON in fts),ScreenedColumns=int(Xtr.shape[1]),
                 Accuracy=round(accuracy_score(yte,pred),4),
                 Precision=round(precision_score(yte,pred,zero_division=0),4),
                 Recall=round(recall_score(yte,pred,zero_division=0),4),
                 F1=round(f1_score(yte,pred,zero_division=0),4),
                 ROC_AUC=round(roc_auc_score(yte,p),4),
                 AveragePrecision=round(average_precision_score(yte,p),4))
        ABL.append(rec); ck_add("abl",k,rec)
        print(f"   {lab:22s} seed={s} AUC={rec['ROC_AUC']:.4f} AP={rec['AveragePrecision']:.4f}",flush=True)
ABLDF=pd.DataFrame(ABL)

# ---- honest threshold ----------------------------------------------------------
stage("honest threshold")
HON=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); f=subset_for(s,Xtr); A,B=Xtr[f],Xte[f]
    Xi,Xv,yi,yv=train_test_split(A,ytr,test_size=THRESHOLD_VAL_SIZE,stratify=ytr,random_state=s)
    mv=fit_model(Xi,yi,s,calibrate=True); pv=mv.predict_proba(Xv)[:,1]
    tau,_,_,_=thr_f1(yv,pv)
    mf=fit_model(A,ytr,s,calibrate=True); pt=mf.predict_proba(B)[:,1]
    ph=(pt>=tau).astype(int)
    tt,ft,pt_,rt=thr_f1(yte,pt)
    tn,fp,fn,tp=confusion_matrix(yte,ph).ravel()
    HON.append(dict(Seed=s,NumFeatures=len(f),Tau_validation=round(tau,4),
                    Tau_testtuned=round(tt,4),
                    F1_honest=round(f1_score(yte,ph,zero_division=0),4),F1_testtuned=round(ft,4),
                    Precision_honest=round(precision_score(yte,ph,zero_division=0),4),
                    Recall_honest=round(recall_score(yte,ph,zero_division=0),4),
                    TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn)))
HONDF=pd.DataFrame(HON)
HONDF["OptimismBias_F1"]=(HONDF.F1_testtuned-HONDF.F1_honest).round(4)
print(f"   honest F1 {HONDF.F1_honest.mean():.4f} vs test-tuned {HONDF.F1_testtuned.mean():.4f} "
      f"(inflation {HONDF.OptimismBias_F1.mean():+.4f})",flush=True)

# ---- budget cut-offs -----------------------------------------------------------
stage("budget cutoffs")
BUD=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); f=subset_for(s,Xtr)
    m=fit_model(Xtr[f],ytr,s,calibrate=True); p=m.predict_proba(Xte[f])[:,1]
    order=np.argsort(p)[::-1]; y=np.asarray(yte); tot=int(y.sum())
    for fr in BUDGETS:
        k=max(1,int(round(fr*len(y)))); sel=order[:k]; tp=int(y[sel].sum())
        BUD.append(dict(Seed=s,BudgetFraction=fr,Flagged=k,TruePositives=tp,
                        FalseAlarms=k-tp,Recall=round(tp/tot,4),Precision=round(tp/k,4),
                        Lift=round((tp/k)/float(y.mean()),3),
                        ScoreAtCutoff=round(float(p[order[k-1]]),5),TestPositives=tot))
BUDDF=pd.DataFrame(BUD)
BUDSUM=(BUDDF.groupby("BudgetFraction").agg(Flagged=("Flagged","mean"),
        TruePositives=("TruePositives","mean"),FalseAlarms=("FalseAlarms","mean"),
        Recall_mean=("Recall","mean"),Recall_sd=("Recall","std"),
        Precision_mean=("Precision","mean"),Precision_sd=("Precision","std"),
        Lift_mean=("Lift","mean")).round(4).reset_index())
print(BUDSUM.to_string(index=False),flush=True)

# ---- label shuffle floor -------------------------------------------------------
stage("label shuffle")
SHUF=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s)
    m=fit_model(Xtr,ytr,s,calibrate=True); p=m.predict_proba(Xte)[:,1]
    SHUF.append(dict(Seed=s,Condition="real labels",Shuffle=0,
                     ROC_AUC=round(roc_auc_score(yte,p),4),
                     AveragePrecision=round(average_precision_score(yte,p),4)))
    for i in range(3):
        rng=np.random.RandomState(s*100+i)
        ys=pd.Series(rng.permutation(ytr.values),index=ytr.index)
        ms=fit_model(Xtr,ys,s,calibrate=True); ps=ms.predict_proba(Xte)[:,1]
        SHUF.append(dict(Seed=s,Condition="shuffled labels",Shuffle=i+1,
                         ROC_AUC=round(roc_auc_score(yte,ps),4),
                         AveragePrecision=round(average_precision_score(yte,ps),4)))
SHUFDF=pd.DataFrame(SHUF)
print(SHUFDF.groupby("Condition")[["ROC_AUC","AveragePrecision"]].mean().round(4).to_string(),flush=True)

# ---- calibration methods -------------------------------------------------------
stage("calibration methods")
CM=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s); f=subset_for(s,Xtr)
    for meth in ["none","sigmoid","isotonic"]:
        m=fit_model(Xtr[f],ytr,s,calibrate=(meth!="none"),
                    method=("sigmoid" if meth=="none" else meth))
        p=m.predict_proba(Xte[f])[:,1]
        CM.append(dict(Seed=s,Calibration=meth,
                       **{k:(round(v,6) if isinstance(v,float) else v)
                          for k,v in calib_report(yte,p).items()}))
CMDF=pd.DataFrame(CM)
print(CMDF.groupby("Calibration")[["ECE","Brier","ROC_AUC"]].mean().round(5).to_string(),flush=True)

# ---- linear baseline -----------------------------------------------------------
stage("linear baseline")
LIN=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s)
    sc=StandardScaler().fit(Xtr); set_seeds(s)
    r=np.random.RandomState(s); idx=np.arange(len(ytr)); r.shuffle(idx)
    m=CalibratedClassifierCV(LogisticRegression(class_weight="balanced",max_iter=2000,
                                                random_state=s),
                             method=CALIB_METHOD,
                             cv=StratifiedKFold(CALIB_CV,shuffle=True,random_state=s))
    m.fit(sc.transform(Xtr)[idx],ytr.iloc[idx])
    p=m.predict_proba(sc.transform(Xte))[:,1]
    rep=calib_report(yte,p); tau,f1v,pv2,rv=thr_f1(yte,p)
    LIN.append(dict(Seed=s,Model="Calibrated logistic regression",Threshold=round(tau,4),
                    F1=round(f1v,4),Precision=round(pv2,4),Recall=round(rv,4),
                    ROC_AUC=round(rep["ROC_AUC"],4),
                    AveragePrecision=round(rep["AveragePrecision"],4),
                    ECE=round(rep["ECE"],5),Brier=round(rep["Brier"],5)))
LINDF=pd.DataFrame(LIN)
_c=LINDF.set_index("Seed"); _f=FINDF.set_index("Seed"); _i=_c.index.intersection(_f.index)
LINCMP=[]
for met in ["ROC_AUC","AveragePrecision","F1"]:
    a,b=_c.loc[_i,met],_f.loc[_i,met]
    try: w=stats.wilcoxon(a,b); pv=round(float(w.pvalue),5)
    except ValueError: pv=np.nan
    LINCMP.append(dict(Metric=met,Logistic_mean=round(float(a.mean()),4),
                       Pipeline_mean=round(float(b.mean()),4),
                       Difference=round(float(a.mean()-b.mean()),4),
                       LogisticWins=int((a>b).sum()),n=len(_i),Wilcoxon_p=pv))
LINCMPDF=pd.DataFrame(LINCMP)
print(LINCMPDF.to_string(index=False),flush=True)

# ---- temporal split ------------------------------------------------------------
stage("temporal split")
YR=pd.to_numeric(META["year_final"],errors="coerce")
TMP=[]
for cut in TEMPORAL_CUTOFFS:
    trm,tem=(YR<=cut).values,(YR>cut).values
    if trm.sum()<500 or tem.sum()<200: continue
    X,y=TABLE.drop(columns=["fatality"]),TABLE["fatality"]
    Xtr,Xte,ytr,yte=X[trm],X[tem],y[trm],y[tem]
    if ytr.sum()<20 or yte.sum()<20: continue
    Xtr,Xte,_=screen(Xtr,Xte)
    Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=INNER_VAL,stratify=ytr,random_state=SEEDS[0])
    set_seeds(SEEDS[0]); ev=Evaluator(Xi,Xv,yi,yv,SEEDS[0])
    r=opt_FOA_GWO(ev,Xtr.shape[1],POP,ITERS)
    hyb=list(Xtr.columns[np.where(r["mask"]==1)[0]])
    for lab,cols in [("hybrid subset",hyb),("all features",list(Xtr.columns))]:
        m=fit_model(Xtr[cols],ytr,SEEDS[0],calibrate=True); p=m.predict_proba(Xte[cols])[:,1]
        tau,f1v,pv2,rv=thr_f1(yte,p)
        TMP.append(dict(CutoffYear=cut,Subset=lab,NumFeatures=len(cols),
                        TrainRows=int(trm.sum()),TestRows=int(tem.sum()),
                        TrainPositives=int(ytr.sum()),TestPositives=int(yte.sum()),
                        TrainPosRate=round(float(ytr.mean()),4),
                        TestPosRate=round(float(yte.mean()),4),Threshold=round(tau,4),
                        F1=round(f1v,4),Precision=round(pv2,4),Recall=round(rv,4),
                        ROC_AUC=round(roc_auc_score(yte,p),4),
                        AveragePrecision=round(average_precision_score(yte,p),4),
                        NoSkill=round(float(yte.mean()),4)))
        print(f"   <= {cut} ({lab}) AUC={TMP[-1]['ROC_AUC']:.4f} AP={TMP[-1]['AveragePrecision']:.4f}",flush=True)
TMPDF=pd.DataFrame(TMP)

# ---- leave one peak out --------------------------------------------------------
stage("leave-one-peak-out")
X,y=TABLE.drop(columns=["fatality"]),TABLE["fatality"]
pk=PEAKID.values; cnt=pd.Series(pk).value_counts()
elig=[p for p in cnt.index if cnt[p]>=LOPO_MIN_N and int(y[pk==p].sum())>=LOPO_MIN_POS]
LOPO=[]
for p_ in elig:
    te=(pk==p_); tr=~te; ytr,yte=y[tr],y[te]
    if ytr.sum()<20: continue
    Xtr,Xte,_=screen(X[tr],X[te])
    m=fit_model(Xtr,ytr,SEEDS[0],calibrate=True); pr=m.predict_proba(Xte)[:,1]
    try: auc=roc_auc_score(yte,pr)
    except ValueError: auc=np.nan
    tau,f1v,pv2,rv=thr_f1(yte,pr)
    LOPO.append(dict(HeldOutPeak=p_,TestRows=int(te.sum()),TestPositives=int(yte.sum()),
                     TestPosRate=round(float(yte.mean()),4),
                     ROC_AUC=round(auc,4) if auc==auc else np.nan,
                     AveragePrecision=round(average_precision_score(yte,pr),4),
                     NoSkill=round(float(yte.mean()),4),
                     Lift=round(average_precision_score(yte,pr)/max(float(yte.mean()),1e-9),2),
                     F1=round(f1v,4),Precision=round(pv2,4),Recall=round(rv,4)))
    print(f"   {p_:5s} n={int(te.sum()):5d} pos={int(yte.sum()):3d} AUC={LOPO[-1]['ROC_AUC']}",flush=True)
LOPODF=pd.DataFrame(LOPO).sort_values("TestRows",ascending=False).reset_index(drop=True)

# ---- classifier tuning ---------------------------------------------------------
stage("tuning")
from itertools import product
combos=[dict(zip(TUNING_GRID,v)) for v in product(*TUNING_GRID.values())]
TUNE=[]
for s in TUNE_SEEDS:
    Xtr,Xte,ytr,yte=split(s); best=None; bs=-1
    for cfg in combos:
        set_seeds(s); m=make_xgb(ytr,s,**cfg)
        sc=cross_val_score(m,Xtr,ytr,cv=StratifiedKFold(3,shuffle=True,random_state=s),
                           scoring="average_precision")
        TUNE.append(dict(Seed=s,**cfg,CV_AveragePrecision=round(float(sc.mean()),5),
                         CV_std=round(float(sc.std()),5),Selected=False))
        if sc.mean()>bs: best,bs=cfg,sc.mean()
    for r_ in TUNE:
        if r_["Seed"]==s and all(r_[k]==best[k] for k in best): r_["Selected"]=True
    m=fit_model(Xtr,ytr,s,calibrate=True,**best); p=m.predict_proba(Xte)[:,1]
    tau,f1v,pv2,rv=thr_f1(yte,p)
    TUNE.append(dict(Seed=s,**best,CV_AveragePrecision=round(float(bs),5),CV_std=np.nan,
                     Selected="TEST",Test_ROC_AUC=round(roc_auc_score(yte,p),4),
                     Test_AveragePrecision=round(average_precision_score(yte,p),4),
                     Test_F1=round(f1v,4)))
    print(f"   seed {s} best={best} CV_AP={bs:.4f} test_AP={TUNE[-1]['Test_AveragePrecision']:.4f}",flush=True)
TUNEDF=pd.DataFrame(TUNE)

# ---- repeated k-fold -----------------------------------------------------------
stage("repeated k-fold")
KF=[]
for i,(tr,te) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=2,
                                                   random_state=SEEDS[0]).split(X,y),1):
    Xtr,Xte,_=screen(X.iloc[tr],X.iloc[te]); ytr,yte=y.iloc[tr],y.iloc[te]
    m=fit_model(Xtr,ytr,SEEDS[0]+i,calibrate=True); p=m.predict_proba(Xte)[:,1]
    rep=calib_report(yte,p); tau,f1v,pv2,rv=thr_f1(yte,p)
    KF.append(dict(Fold=i,TestRows=len(yte),TestPositives=int(yte.sum()),
                   Threshold=round(tau,4),F1=round(f1v,4),Precision=round(pv2,4),
                   Recall=round(rv,4),ROC_AUC=round(rep["ROC_AUC"],4),
                   AveragePrecision=round(rep["AveragePrecision"],4),
                   ECE=round(rep["ECE"],5),Brier=round(rep["Brier"],5)))
KFDF=pd.DataFrame(KF)
print(f"   k-fold AUC {KFDF.ROC_AUC.mean():.4f} +/- {KFDF.ROC_AUC.std():.4f}",flush=True)

# ---- archival columns ----------------------------------------------------------
stage("archival columns")
done=ck_load("arch"); ARCH=[]
T_wo=build_table(drop_archival=True)
for s in SEEDS:
    for lab,tb in (("with archival columns",TABLE),("without archival columns",T_wo)):
        k=f"{lab}|{s}"
        if k in done: ARCH.append(done[k]); continue
        Xtr,Xte,ytr,yte=split(s,tb)
        Xi,Xv,yi,yv=train_test_split(Xtr,ytr,test_size=INNER_VAL,stratify=ytr,random_state=s)
        set_seeds(s); ev=Evaluator(Xi,Xv,yi,yv,s)
        r=opt_FOA_GWO(ev,Xtr.shape[1],POP,ITERS)
        fts=list(Xtr.columns[np.where(r["mask"]==1)[0]])
        m=fit_model(Xtr[fts],ytr,s,calibrate=True); p=m.predict_proba(Xte[fts])[:,1]
        tau,f1v,pv2,rv=thr_f1(yte,p)
        rec=dict(Configuration=lab,Seed=s,ScreenedColumns=int(Xtr.shape[1]),
                 NumFeatures=len(fts),Threshold=round(tau,4),F1=round(f1v,4),
                 Precision=round(pv2,4),Recall=round(rv,4),
                 ROC_AUC=round(roc_auc_score(yte,p),4),
                 AveragePrecision=round(average_precision_score(yte,p),4))
        ARCH.append(rec); ck_add("arch",k,rec)
        print(f"   {lab:26s} seed={s} AUC={rec['ROC_AUC']:.4f}",flush=True)
ARCHDF=pd.DataFrame(ARCH)

# ---- the calibration trap ------------------------------------------------------
stage("calibration trap")
TRAP=[]; FRACS=[0.0,0.02,0.05,0.10,0.25,0.50,0.75,1.0]
for s in SEEDS:
    Xtr,Xte,ytr,yte=split(s)
    order=np.argsort(Xtr.index.values); Xs,ys=Xtr.iloc[order],ytr.iloc[order]; n=len(ys)
    for fr in FRACS:
        idx=np.arange(n); k=int(round(fr*n))
        if k>1:
            r=np.random.RandomState(s); pick=r.choice(n,size=k,replace=False)
            sh=pick.copy(); r.shuffle(sh); idx[pick]=idx[sh]
        m=fit_model(Xs.iloc[idx],ys.iloc[idx],s,calibrate=True,legacy=True)
        p=m.predict_proba(Xte)[:,1]
        _tr=calib_report(yte,p)
        TRAP.append(dict(FractionShuffled=fr,Seed=s,
                         ROC_AUC=round(_tr["ROC_AUC"],4),
                         AveragePrecision=round(_tr["AveragePrecision"],4),
                         ECE=round(_tr["ECE"],5),MCE=round(_tr["MCE"],5),
                         Brier=round(_tr["Brier"],5),
                         BrierSkillScore=round(_tr["BrierSkillScore"],5),
                         MeanPredicted=round(_tr["MeanPredicted"],5)))
    print(f"   seed {s}: "+"  ".join(
        f"{fr:.2f}->{[t for t in TRAP if t['Seed']==s and t['FractionShuffled']==fr][0]['ROC_AUC']:.3f}"
        for fr in FRACS),flush=True)
TRAPDF=pd.DataFrame(TRAP)

# =====================================================================================
#  FEATURE-FAMILY ABLATION
#
#  The question a hazards reviewer asks first: does the weather actually add
#  anything? Single-feature discrimination says the strongest predictors are
#  team size (AUC 0.69) and year (0.67), while the best weather feature reaches
#  only 0.56. That does not settle it - weather could still contribute jointly -
#  so it is settled here, by fitting each family on its own under identical
#  splits. "year removed" variants test how much of the signal is era.
# =====================================================================================
stage("feature families")
WEATHER_COLS = set(CLIMATE) | {f"{c}_missing" for c in CLIMATE}
FAM=[]
for s_ in SEEDS:
    Xtr,Xte,ytr,yte=split(s_)
    exped=[c for c in Xtr.columns if c not in WEATHER_COLS]
    wx   =[c for c in Xtr.columns if c in WEATHER_COLS]
    fams={"expedition only":exped,
          "weather only":wx,
          "both (full set)":list(Xtr.columns),
          "both, year removed":[c for c in Xtr.columns if c!="year"],
          "expedition only, year removed":[c for c in exped if c!="year"]}
    for lab,cols in fams.items():
        if not cols: continue
        m=fit_model(Xtr[cols],ytr,s_,calibrate=True); p=m.predict_proba(Xte[cols])[:,1]
        rep=calib_report(yte,p)
        FAM.append(dict(Family=lab,Seed=s_,NumFeatures=len(cols),
                        ROC_AUC=round(rep["ROC_AUC"],4),
                        AveragePrecision=round(rep["AveragePrecision"],4),
                        BrierSkillScore=round(rep["BrierSkillScore"],4),
                        ECE=round(rep["ECE"],5)))
    print(f"   seed {s_} done ({len(exped)} expedition, {len(wx)} weather)",flush=True)
FAMDF=pd.DataFrame(FAM)
FAMSUM=(FAMDF.groupby("Family")
        .agg(NumFeatures=("NumFeatures","first"),
             AUC_mean=("ROC_AUC","mean"),AUC_sd=("ROC_AUC","std"),
             AP_mean=("AveragePrecision","mean"),
             BSS_mean=("BrierSkillScore","mean"),ECE_mean=("ECE","mean"))
        .round(4).reset_index().sort_values("AUC_mean",ascending=False))
# paired tests against the full set, on matched seeds
_full=FAMDF[FAMDF.Family=="both (full set)"].set_index("Seed")["ROC_AUC"]
_rows=[]
for fam in FAMDF.Family.unique():
    if fam=="both (full set)": continue
    a=FAMDF[FAMDF.Family==fam].set_index("Seed")["ROC_AUC"]
    i=a.index.intersection(_full.index)
    try: pv=float(stats.wilcoxon(a[i],_full[i]).pvalue)
    except Exception: pv=np.nan
    _rows.append(dict(Family=fam,MeanAUC=round(float(a[i].mean()),4),
                      FullSetMeanAUC=round(float(_full[i].mean()),4),
                      Delta=round(float(a[i].mean()-_full[i].mean()),4),
                      WinsVsFull=int((a[i]>_full[i]).sum()),N=len(i),
                      Wilcoxon_p=round(pv,5) if pv==pv else np.nan))
FAMSIG=pd.DataFrame(_rows)
print(FAMSUM.to_string(index=False),flush=True)
print(FAMSIG.to_string(index=False),flush=True)

# =====================================================================================
#  FIGURES
# =====================================================================================
stage("figures")
PAL="colorblind"
def save(fig,name):
    for ext in ("png","pdf"): fig.savefig(FIG/f"{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig); print("   ",name,flush=True)

o=SELDF.groupby("Method")["F1"].mean().sort_values(ascending=False).index.tolist()
fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(data=SELDF,x="Method",y="F1",order=o,ax=ax[0],palette=PAL)
sns.stripplot(data=SELDF,x="Method",y="F1",order=o,ax=ax[0],color="k",size=3,alpha=.6)
ax[0].set_title(f"Minority F1 across {SELDF.Seed.nunique()} independent runs")
ax[0].tick_params(axis="x",rotation=60)
sns.boxplot(data=SELDF,x="Method",y="NumFeatures",order=o,ax=ax[1],palette=PAL)
ax[1].set_title("Selected subset size"); ax[1].tick_params(axis="x",rotation=60)
fig.tight_layout(); save(fig,"fig02_selection_distribution")

ks=[k for k in ["FOA_GWO","GWO_FOA","FOA","GWO"] if k in HIST]
if ks:
    fig,ax=plt.subplots(figsize=(8,5))
    for k in ks:
        runs=[h for h in HIST[k].values() if h]
        if not runs: continue
        L=min(len(r) for r in runs); M=np.array([r[:L] for r in runs],float)
        mu=M.mean(0); sd=M.std(0,ddof=1) if M.shape[0]>1 else np.zeros(L)
        xx=np.arange(1,L+1); ax.plot(xx,mu,lw=2,label=k); ax.fill_between(xx,mu-sd,mu+sd,alpha=.15)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best-so-far fitness")
    ax.set_title("Convergence: proposed order, its reverse, and its components")
    ax.legend(); fig.tight_layout(); save(fig,"fig03_convergence")

fig,ax=plt.subplots(1,3,figsize=(16,4.5))
for i,m_ in enumerate(["ECE","Brier","AdaptiveECE"]):
    sns.barplot(data=CALDF,x="Method",y=m_,hue="Calibrated",ax=ax[i],palette=PAL,errorbar="sd")
    ax[i].set_title(m_); ax[i].tick_params(axis="x",rotation=45)
fig.tight_layout(); save(fig,"fig04_calibration_metrics")

grid=np.linspace(0,1,101); tp_,pr_,au,ap_=[],[],[],[]
for d in CURVES.values():
    yy,pp=d["y"],d["p"]; fpr,tpr,_=roc_curve(yy,pp); tp_.append(np.interp(grid,fpr,tpr))
    pcv,rcv,_=precision_recall_curve(yy,pp); pr_.append(np.interp(grid,rcv[::-1],pcv[::-1]))
    au.append(roc_auc_score(yy,pp)); ap_.append(average_precision_score(yy,pp))
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
Tm=np.array(tp_); ax[0].plot(grid,Tm.mean(0),lw=2,
    label=f"AUC {np.mean(au):.3f} ± {np.std(au,ddof=1):.3f}")
ax[0].fill_between(grid,Tm.mean(0)-Tm.std(0),Tm.mean(0)+Tm.std(0),alpha=.2)
ax[0].plot([0,1],[0,1],"--",c="grey"); ax[0].legend(loc="lower right")
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC across runs")
Pm=np.array(pr_); ax[1].plot(grid,Pm.mean(0),lw=2,
    label=f"AP {np.mean(ap_):.3f} ± {np.std(ap_,ddof=1):.3f}")
ax[1].fill_between(grid,Pm.mean(0)-Pm.std(0),Pm.mean(0)+Pm.std(0),alpha=.2)
bl=np.mean([d["y"].mean() for d in CURVES.values()])
ax[1].axhline(bl,ls="--",c="grey",label=f"No skill {bl:.3f}")
ax[1].legend(); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-recall across runs")
fig.tight_layout(); save(fig,"fig05_curves")

g=BUDSUM.copy(); fig,ax=plt.subplots(1,2,figsize=(13,4.8)); xx=g.BudgetFraction*100
ax[0].errorbar(xx,g.Recall_mean,yerr=g.Recall_sd.fillna(0),marker="o",capsize=3,label="Recall")
ax[0].errorbar(xx,g.Precision_mean,yerr=g.Precision_sd.fillna(0),marker="s",capsize=3,label="Precision")
ax[0].set_xscale("log"); ax[0].set_xlabel("Review budget (% flagged)"); ax[0].legend()
ax[0].set_title("What a fixed review budget buys")
ax[1].plot(xx,g.Lift_mean,marker="o",color="tab:green")
ax[1].axhline(1,ls="--",c="grey",label="no better than random")
ax[1].set_xscale("log"); ax[1].set_xlabel("Review budget (%)"); ax[1].set_ylabel("Lift")
ax[1].legend(); ax[1].set_title("Enrichment over random selection")
fig.tight_layout(); save(fig,"fig06_budget")

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
for i,m_ in enumerate(["ROC_AUC","AveragePrecision"]):
    sns.boxplot(data=SHUFDF,x="Condition",y=m_,ax=ax[i],palette=PAL)
    sns.stripplot(data=SHUFDF,x="Condition",y=m_,ax=ax[i],color="k",size=3,alpha=.6)
    if m_=="ROC_AUC": ax[i].axhline(.5,ls="--",c="grey")
    ax[i].set_title(f"{m_}: real vs shuffled outcomes")
fig.tight_layout(); save(fig,"fig07_label_shuffle")

if not TMPDF.empty:
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for i,m_ in enumerate(["ROC_AUC","AveragePrecision"]):
        sns.barplot(data=TMPDF,x="CutoffYear",y=m_,hue="Subset",ax=ax[i],palette=PAL)
        if m_=="ROC_AUC": ax[i].axhline(.5,ls="--",c="grey")
        ax[i].set_title(m_)
    fig.suptitle("Temporal split: train before the cut-off, test after")
    fig.tight_layout(); save(fig,"fig08_temporal")

if not LOPODF.empty:
    d=LOPODF.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC")
    fig,ax=plt.subplots(figsize=(8,max(4,.42*len(d))))
    ax.barh(d.HeldOutPeak,d.ROC_AUC,
            color=["tab:red" if v<.5 else "tab:blue" for v in d.ROC_AUC])
    ax.axvline(.5,ls="--",c="grey",label="no better than random")
    ax.set_xlabel("ROC-AUC on the held-out mountain"); ax.legend()
    ax.set_title("Transfer to a mountain never seen in training")
    fig.tight_layout(); save(fig,"fig09_leave_one_peak_out")

if not SENSDF.empty:
    fig,ax=plt.subplots(1,3,figsize=(16,4.6))
    w=SENSDF[SENSDF.Study=="weights"].copy()
    if not w.empty:
        w["Weights"]=w.apply(lambda r:f"{r.w_recall:.2f}/{r.w_f1:.2f}/{r.w_precision:.2f}",axis=1)
        sns.barplot(data=w,x="Weights",y="F1",ax=ax[0],palette=PAL,errorbar="sd")
        ax[0].set_title("Fitness weights"); ax[0].tick_params(axis="x",rotation=70)
    l=SENSDF[SENSDF.Study=="lambda"]
    if not l.empty:
        gg=l.groupby("Lambda").agg(F1=("F1","mean"),sd=("F1","std"),k=("NumFeatures","mean")).reset_index()
        ax[1].errorbar(gg.Lambda,gg.F1,yerr=gg.sd.fillna(0),marker="o",capsize=3)
        a2=ax[1].twinx(); a2.plot(gg.Lambda,gg.k,ls="--",c="tab:orange",marker="s")
        a2.set_ylabel("mean subset size",color="tab:orange")
        ax[1].set_xlabel("lambda"); ax[1].set_ylabel("F1"); ax[1].set_title("Size penalty")
    fl=SENSDF[SENSDF.Study=="precision_floor"]
    if not fl.empty:
        sns.barplot(data=fl,x="PrecisionFloor",y="Precision",ax=ax[2],palette=PAL,errorbar="sd")
        ax[2].set_title("Precision floor")
    fig.tight_layout(); save(fig,"fig10_sensitivity")

if not ABLDF.empty:
    m_=ABLDF.melt(id_vars=["Configuration","Seed"],
                  value_vars=["ROC_AUC","AveragePrecision","F1"],
                  var_name="Metric",value_name="Value")
    fig,ax=plt.subplots(figsize=(8,4.8))
    sns.barplot(data=m_,x="Metric",y="Value",hue="Configuration",ax=ax,palette=PAL,errorbar="sd")
    ax.set_title("Leakage ablation: higher is NOT better")
    fig.tight_layout(); save(fig,"fig11_leakage")

g=TRAPDF.groupby("FractionShuffled").agg(m=("ROC_AUC","mean"),s=("ROC_AUC","std")).reset_index()
fig,ax=plt.subplots(figsize=(7.5,4.8))
ax.errorbar(g.FractionShuffled*100,g.m,yerr=g.s.fillna(0),marker="o",capsize=3,lw=2)
ax.axhline(g.m.iloc[-1],ls="--",c="grey",label="fully shuffled")
ax.set_xlabel("Percentage of training rows shuffled"); ax.set_ylabel("ROC-AUC")
ax.set_title("A calibration trap: how much row ordering breaks post-hoc calibration")
ax.legend(); fig.tight_layout(); save(fig,"fig12_calibration_trap")

if not ARCHDF.empty:
    m_=ARCHDF.melt(id_vars=["Configuration","Seed"],
                   value_vars=["ROC_AUC","AveragePrecision","F1"],
                   var_name="Metric",value_name="Value")
    fig,ax=plt.subplots(figsize=(8,4.6))
    sns.barplot(data=m_,x="Metric",y="Value",hue="Configuration",ax=ax,palette=PAL,errorbar="sd")
    ax.set_title("Effect of removing archival cataloguing columns")
    fig.tight_layout(); save(fig,"fig13_archival")

stab={}
for f_ in FEATS.get("FOA_GWO",{}).values():
    for x in f_: stab[x]=stab.get(x,0)+1
nr=max(len(FEATS.get("FOA_GWO",{})),1)
STABDF=(pd.DataFrame([dict(Feature=k,TimesSelected=v,SelectionFrequency=round(v/nr,3))
                      for k,v in stab.items()])
        .sort_values("TimesSelected",ascending=False).reset_index(drop=True))
if not STABDF.empty:
    s_=STABDF.head(25)
    fig,ax=plt.subplots(figsize=(7.5,max(4,.32*len(s_))))
    sns.barplot(x="SelectionFrequency",y="Feature",data=s_,ax=ax,palette=PAL)
    ax.axvline(.5,ls="--",c="grey"); ax.set_xlabel(f"Selection frequency across {nr} runs")
    ax.set_title("Feature-selection stability of the proposed hybrid")
    fig.tight_layout(); save(fig,"fig14_feature_stability")

# =====================================================================================
#  WORKBOOK, ARTIFACTS, ARCHIVE
# =====================================================================================
stage("workbook")
FEATDICT=[
 ("temp_mean","climate","Mean 2 m temperature over the recorded interval","degC"),
 ("temp_max","climate","Maximum 2 m temperature","degC"),
 ("temp_min","climate","Minimum 2 m temperature","degC"),
 ("temp_std","climate","Standard deviation of daily mean temperature","degC"),
 ("snow_total","climate","Total snowfall over the interval","cm"),
 ("snow_max","climate","Largest single-day snowfall","cm"),
 ("snow_extreme_rate","exceedance","Share of days above this peak's 95th-percentile daily snowfall","0-1"),
 ("wind_mean","climate","Mean 100 m wind speed","km/h"),
 ("wind_max","climate","Maximum daily-maximum 100 m wind speed","km/h"),
 ("wind_extreme_rate","exceedance","Share of days above this peak's 95th-percentile daily maximum wind","0-1"),
 ("pressure_mean","climate","Mean surface pressure","hPa"),
 ("pressure_min","climate","Minimum surface pressure","hPa"),
 ("warming_extreme_rate","exceedance","Share of days whose day-over-day warming exceeds this peak's 95th percentile","0-1"),
 ("solid_precip_rate","exceedance","Share of days whose WMO code indicates solid precipitation (code >= 70)","0-1"),
 ("team_size","vulnerability","Total expedition members","count"),
 ("exposure_days","vulnerability","Recorded expedition duration","days"),
 ("camp_density","vulnerability","Camps established per day","camps/day"),
 ("technical_difficulty","vulnerability","Fixed rope laid per day","m/day"),
 ("elevation_risk","vulnerability","Highest point reached","m"),
 ("oxygen_used","vulnerability","Supplementary oxygen used","0/1"),
 ("oxygen_free_ascent","vulnerability","No supplementary oxygen","0/1"),
 ("is_standard_route","vulnerability","Standard route attempted","0/1"),
 ("is_commercial_route","vulnerability","Commercial expedition","0/1"),
 ("season_*","vulnerability","One-hot indicators for climbing season","0/1"),
 ("<climate>_missing","missingness","1 when no usable weather interval existed","0/1"),
 ("fatality","target","1 when a member death is recorded","0/1")]

CONFIG=pd.DataFrame([
 dict(Parameter="tier",Value=ARGS.tier,Comment="compute budget preset"),
 dict(Parameter="independent runs (seeds)",Value=str(SEEDS),Comment="seed sets split and search"),
 dict(Parameter="population size P",Value=POP,Comment=""),
 dict(Parameter="iterations T",Value=ITERS,Comment="fixed budget, no early stopping"),
 dict(Parameter="GWO control parameter a",Value="2 -> 0, linear in t/T",Comment=""),
 dict(Parameter="GA mutation rate",Value=0.02,Comment=""),
 dict(Parameter="PSO inertia / c1 / c2",Value="0.7 / 1.5 / 1.5",Comment=""),
 dict(Parameter="binarisation",Value="sigmoid transfer, stochastic threshold",Comment=""),
 dict(Parameter="fitness weights (R/F1/P)",
      Value=f"{FITNESS_W['recall']}/{FITNESS_W['f1']}/{FITNESS_W['precision']}",Comment=""),
 dict(Parameter="lambda (size penalty)",Value=LAMBDA_SIZE,Comment=""),
 dict(Parameter="surrogate",Value=str(SURROGATE),Comment=""),
 dict(Parameter="inner validation split",Value=INNER_VAL,Comment="test never touched in search"),
 dict(Parameter="correlation screen",Value=CORR_THRESHOLD,Comment="absolute Pearson"),
 dict(Parameter="test size",Value=TEST_SIZE,Comment="stratified"),
 dict(Parameter="calibration",Value=f"{CALIB_METHOD}, cv={CALIB_CV}, SHUFFLED inner folds",
      Comment="the repair; see sheet 22"),
 dict(Parameter="final threshold",Value=FINAL_THRESHOLD,Comment=""),
 dict(Parameter="sensitivity budget",Value=f"P={SENS_POP}, T={SENS_ITERS}, seeds={SENS_SEEDS}",
      Comment="reduced; not comparable to the main study"),
 dict(Parameter="device",Value=DEVICE,Comment=""),
 dict(Parameter="python",Value=platform.python_version(),Comment=""),
 dict(Parameter="platform",Value=platform.platform(),Comment=""),
 dict(Parameter="numpy/pandas/sklearn/xgboost",
      Value=f"{np.__version__}/{pd.__version__}/{__import__('sklearn').__version__}/"
            f"{__import__('xgboost').__version__}",Comment=""),
 dict(Parameter="finished (UTC)",Value=datetime.now(timezone.utc).isoformat(),Comment="")])

README=pd.DataFrame([
 ("00_README","this sheet"),("01_Config","every parameter used"),
 ("02_Dataset","record counts, peaks, imbalance, screening"),
 ("03_PeakSummary","per-mountain expeditions, deaths, rate"),
 ("04_FeatureDictionary","every engineered feature defined"),
 ("05_Baselines","untuned classifiers across runs"),
 ("06_Selection_PerRun","one row per method and run"),
 ("07_Selection_Summary","mean, sd, 95% CI per method"),
 ("08_Significance","paired Wilcoxon with Holm correction; Friedman"),
 ("09_Imbalance_PerRun","resampling treatments"),
 ("10_Imbalance_Summary","aggregated"),
 ("11_Calibration_PerRun","ECE, Brier, skill, log loss"),
 ("12_Calibration_Summary","calibrated vs uncalibrated"),
 ("13_Final","the evaluated configuration across runs"),
 ("14_HonestThreshold","cut-off from validation; test-tuning bias"),
 ("15_Budget_PerRun","review-capacity operating points"),
 ("16_Budget_Summary","the operational table for the manuscript"),
 ("17_LabelShuffle","what pure noise scores"),
 ("18_CalibrationMethods","none vs sigmoid vs isotonic"),
 ("19_LinearBaseline","calibrated logistic regression"),
 ("20_LinearVsPipeline","paired comparison"),
 ("21_Temporal","train on the past, test on the future"),
 ("22_LeaveOnePeakOut","transfer to an unseen mountain"),
 ("23_Tuning","does the ranking survive tuning"),
 ("24_RepeatedKFold","5-fold x 2 on the final configuration"),
 ("25_Archival","without the cataloguing columns"),
 ("26_CalibrationTrap","AUC against fraction of rows shuffled"),
 ("27_Sensitivity","fitness weights, lambda, precision floor"),
 ("28_Leakage","termreason ablation"),
 ("29_SelectedFeatures","subset chosen per method per run"),
 ("30_FeatureStability","how often each feature survives"),
 ("31_Citations","reference for every method compared"),
 ("32_FeatureFamilies","weather-only vs expedition-only vs both, per seed"),
 ("33_FeatureFamilies_Sig","paired Wilcoxon of each family against the full set")],
 columns=["Sheet","Contents"])

featrows=[dict(Method=m_,Seed=s_,NumFeatures=len(f_),Features="; ".join(f_))
          for m_,d_ in FEATS.items() for s_,f_ in d_.items()]
FEATDF=pd.DataFrame(featrows)
FRIEDDF=pd.DataFrame([FRIED]) if FRIED else pd.DataFrame()
SIGSHEET=pd.concat([SIG,pd.DataFrame([{}]),FRIEDDF],ignore_index=True) if not SIG.empty else FRIEDDF

SHEETS=[("00_README",README),("01_Config",CONFIG),
 ("02_Dataset",pd.DataFrame([dict(Item=k,Value=v) for k,v in DATA_STATS.items()])),
 ("03_PeakSummary",PEAK_SUMMARY),
 ("04_FeatureDictionary",pd.DataFrame(FEATDICT,columns=["Feature","Family","Definition","Unit"])),
 ("05_Baselines",BASE),("06_Selection_PerRun",SELDF.drop(columns=["History"],errors="ignore")),
 ("07_Selection_Summary",SELSUM),("08_Significance",SIGSHEET),
 ("09_Imbalance_PerRun",IMBDF),("10_Imbalance_Summary",IMBSUM),
 ("11_Calibration_PerRun",CALDF),("12_Calibration_Summary",CALSUM),
 ("13_Final",FINDF),("14_HonestThreshold",HONDF),("15_Budget_PerRun",BUDDF),
 ("16_Budget_Summary",BUDSUM),("17_LabelShuffle",SHUFDF),("18_CalibrationMethods",CMDF),
 ("19_LinearBaseline",LINDF),("20_LinearVsPipeline",LINCMPDF),("21_Temporal",TMPDF),
 ("22_LeaveOnePeakOut",LOPODF),("23_Tuning",TUNEDF),("24_RepeatedKFold",KFDF),
 ("25_Archival",ARCHDF),("26_CalibrationTrap",TRAPDF),("27_Sensitivity",SENSDF),
 ("28_Leakage",ABLDF),("29_SelectedFeatures",FEATDF),("30_FeatureStability",STABDF),
 ("31_Citations",pd.DataFrame([dict(Method=k,Reference=v) for k,v in CITATIONS.items()])),
 ("32_FeatureFamilies",pd.concat([FAMSUM,pd.DataFrame([{}]),FAMDF],ignore_index=True)),
 ("33_FeatureFamilies_Sig",FAMSIG)]

XLSX=ROOT/"Himalayan_MasterResults.xlsx"
try:
    import xlsxwriter; ENG="xlsxwriter"
except ImportError:
    ENG="openpyxl"
with pd.ExcelWriter(XLSX,engine=ENG) as xl:
    for nm,d_ in SHEETS:
        if d_ is None or len(d_)==0: d_=pd.DataFrame({"note":["not produced in this run"]})
        d_=d_.copy()
        for c in d_.columns:
            if d_[c].dtype==object:
                d_[c]=d_[c].apply(lambda v: json.dumps(v) if isinstance(v,(list,dict)) else v)
        d_.to_excel(xl,sheet_name=nm[:31],index=False)
print(f"   workbook: {XLSX}",flush=True)

for nm,d_ in SHEETS:
    if d_ is not None and len(d_): d_.to_csv(RAW/f"{nm}.csv",index=False)
json.dump({"metadata":{"tier":ARGS.tier,"seeds":SEEDS,"pop":POP,"iters":ITERS,
                       "termreason_excluded":True,"calibration_repair":True},
           "features":{m_:{str(s_):f_ for s_,f_ in d_.items()} for m_,d_ in FEATS.items()}},
          open(ART/"selected_features.json","w"),indent=2)
json.dump({m_:{str(s_):list(map(float,h)) for s_,h in d_.items()} for m_,d_ in HIST.items()},
          open(ART/"convergence_histories.json","w"),indent=2)
json.dump({"tier":ARGS.tier,"seeds":SEEDS,"device":DEVICE,
           "sources":WEATHER_SOURCE,"extreme_percentile":EXTREME_PCTL,
           "dead_columns_dropped":DEAD_COLUMNS,
           "peak_extreme_thresholds":THRESH,
           "elapsed_minutes":round((time.time()-T0)/60,2),"dataset":DATA_STATS,
           "friedman":FRIED,"versions":{"python":platform.python_version(),
           "numpy":np.__version__,"pandas":pd.__version__,
           "sklearn":__import__("sklearn").__version__,
           "xgboost":__import__("xgboost").__version__}},
          open(ART/"run_manifest.json","w"),indent=2,default=str)

ZIP=ROOT.parent/f"{ROOT.name}_bundle.zip"
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    z.write(XLSX,XLSX.name)
    for sub in ("figures","raw","artifacts"):
        for f_ in sorted((ROOT/sub).glob("*")): z.write(f_,f"{sub}/{f_.name}")

el=(time.time()-T0)/60
STATUS.write_text(f"COMPLETE\ntier={ARGS.tier}\nelapsed {el:.1f} min\n"
                  f"workbook {XLSX}\nbundle {ZIP}\n")
print(f"\n{'='*78}\n  COMPLETE in {el:.1f} min ({el/60:.1f} h)")
print(f"  workbook : {XLSX}")
print(f"  bundle   : {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB)")
print(f"  figures  : {len(list(FIG.glob('*.png')))} png")
print("="*78,flush=True)
print("\nHEADLINE")
print(SELSUM[["Method","F1_mean","F1_std","NumFeatures_mean","DegenerateRuns"]].to_string(index=False))
print(f"\nSignal check: real {SHUFDF[SHUFDF.Condition=='real labels'].ROC_AUC.mean():.4f} "
      f"vs shuffled {SHUFDF[SHUFDF.Condition=='shuffled labels'].ROC_AUC.mean():.4f}")
print(f"Calibration: ECE {CMDF[CMDF.Calibration=='none'].ECE.mean():.4f} -> "
      f"{CMDF[CMDF.Calibration=='sigmoid'].ECE.mean():.4f}")
if not LOPODF.empty:
    print(f"Unseen mountains: mean AUC {LOPODF.ROC_AUC.mean():.4f}, "
          f"{int((LOPODF.ROC_AUC<0.5).sum())} of {len(LOPODF)} below 0.5")
print("\nVERIFY artifacts/peak_coordinates_VERIFY.csv before using the map panel.")
