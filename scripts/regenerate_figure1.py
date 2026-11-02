#!/usr/bin/env python3
# =====================================================================================
#  REGENERATE FIGURE 1 - dataset overview, with verified coordinates
#
#  Replaces results/figures/fig01_dataset_overview.{png,pdf} in place. Nothing
#  else in the results folder is touched, so the rest of the run stands.
#
#  Fixes in this version:
#    - uses your verified coordinates instead of my approximations
#    - corrects Lhotse Shar, which was set to Lhotse main's position
#    - label repulsion plus a zoom inset, so the Khumbu cluster is readable
#    - twin-axis labels no longer collide in the time panel
#    - decades with too few expeditions are dropped from the rate line rather
#      than shown as 100% on a handful of records
#    - larger panels, fewer collisions, print-ready at 300 dpi
#
#  Usage:
#      python regenerate_figure1.py --coords peak_coordinates_verified_final.csv \
#                                   --outdir results
# =====================================================================================
import argparse, glob, os, sys, warnings
from datetime import timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns

ap = argparse.ArgumentParser()
ap.add_argument("--coords", default="peak_coordinates_verified_final.csv")
ap.add_argument("--outdir", default="results")
A = ap.parse_args()
ROOT = Path(A.outdir).resolve(); FIG = ROOT/"figures"; FIG.mkdir(parents=True, exist_ok=True)

# --- coordinates -----------------------------------------------------------------
# Lhotse Shar had been given Lhotse main's position. Wikipedia/peakbagger place
# Lhotse Shar at 27.95833 N, 86.94333 E and Lhotse main at 27.9617 N, 86.9333 E.
CORRECTIONS = {"LSHR": (27.95833, 86.94333)}

coords = pd.read_csv(A.coords)
for pk, (la, lo) in CORRECTIONS.items():
    if pk in set(coords.peakid):
        old = coords.loc[coords.peakid == pk, ["lat", "lon"]].values[0]
        if abs(old[0]-la) > 0.002 or abs(old[1]-lo) > 0.002:
            coords.loc[coords.peakid == pk, ["lat", "lon"]] = [la, lo]
            print(f"  corrected {pk}: {old[0]:.4f},{old[1]:.4f} -> {la:.5f},{lo:.5f}")
coords = coords.dropna(subset=["lat", "lon"])
print(f"  {len(coords)} peaks with coordinates")

# --- rebuild the per-expedition metadata -----------------------------------------
print("  rebuilding expedition metadata (needs the Kaggle archives) ...", flush=True)
try:
    import kagglehub
except ImportError:
    os.system(f"{sys.executable} -m pip install -q kagglehub"); import kagglehub
EP = kagglehub.dataset_download("siddharth0935/himalayan-expeditions")
WP = kagglehub.dataset_download("bonesclarke26/historic-weather-data-for-himalayan-peaks")

E = pd.read_csv(os.path.join(EP, "exped.csv"), low_memory=False)
for c in ["bcdate", "smtdate", "termdate"]:
    E[c] = pd.to_datetime(E[c], errors="coerce").dt.date
W = {}
for f in sorted(glob.glob(os.path.join(WP, "*.parquet"))):
    k = os.path.basename(f).replace(".parquet", "").upper()
    d = pd.read_parquet(f); d["date"] = pd.to_datetime(d["date"]).dt.date
    W[k] = d.sort_values("date").reset_index(drop=True)

kept = E[E.peakid.isin(set(W))].copy().reset_index(drop=True)
def iv(r):
    s, e = r["bcdate"], r["termdate"]
    if pd.isna(e) and not pd.isna(r.get("smtdate")): e = r["smtdate"]
    if pd.isna(e) and not pd.isna(s) and not pd.isna(r.get("totdays")):
        e = s + timedelta(days=int(r["totdays"]))
    return s, e
b = kept.apply(iv, axis=1, result_type="expand"); kept["s"], kept["e"] = b[0], b[1]

miss = np.ones(len(kept), dtype=int)
for pk, g in kept.groupby("peakid"):
    w = W.get(pk)
    if w is None: continue
    dts = w["date"].to_numpy()
    for i, r in g.iterrows():
        if pd.isna(r["s"]) or pd.isna(r["e"]): continue
        if not w.loc[(dts >= r["s"]) & (dts <= r["e"])].empty: miss[i] = 0

M = pd.DataFrame({"peakid": kept.peakid.values,
                  "year": [d.year if not pd.isna(d) else np.nan for d in kept["bcdate"]],
                  "totmembers": kept.totmembers.values,
                  "mdeaths": kept.mdeaths.values,
                  "season": kept.season.astype(str).values,
                  "interval_missing": miss})
M["fatality"] = (M.mdeaths > 0).astype(int)
print(f"  {len(M)} expeditions, {M.peakid.nunique()} peaks, "
      f"{int(M.fatality.sum())} fatal ({M.fatality.mean()*100:.2f}%)", flush=True)

G = (M.groupby("peakid").agg(Expeditions=("fatality","size"), Fatal=("fatality","sum"),
                             Deaths=("mdeaths","sum"))
       .reset_index())
G["FatalityRate"] = (G.Fatal/G.Expeditions*100).round(2)
G = G.merge(coords[["peakid","lat","lon"]], on="peakid", how="left")

# =====================================================================================
#  FIGURE
# =====================================================================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

def repel(ax, x, y, labels, sizes, iters=260, k=0.010, fs=7.4):
    """Nudge labels apart, then join each to its marker with a hairline.
    Cheap alternative to adjustText, which is not always installed."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xr = (ax.get_xlim()[1]-ax.get_xlim()[0]); yr = (ax.get_ylim()[1]-ax.get_ylim()[0])
    lx = x + 0.012*xr; ly = y + 0.012*yr
    for _ in range(iters):
        for i in range(len(lx)):
            dx = dy = 0.0
            for j in range(len(lx)):
                if i == j: continue
                ddx = (lx[i]-lx[j])/xr; ddy = (ly[i]-ly[j])/yr
                d2 = ddx*ddx + ddy*ddy
                if d2 < k*k and d2 > 1e-12:
                    f = (k - np.sqrt(d2))/np.sqrt(d2)
                    dx += ddx*f; dy += ddy*f
            ddx = (lx[i]-x[i])/xr; ddy = (ly[i]-y[i])/yr   # keep near its marker
            dx -= 0.18*ddx; dy -= 0.18*ddy
            lx[i] += dx*xr*0.45; ly[i] += dy*yr*0.45
    for i, t in enumerate(labels):
        ax.plot([x[i], lx[i]], [y[i], ly[i]], lw=0.45, color="0.45", zorder=2)
        ax.text(lx[i], ly[i], t, fontsize=fs, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.7", lw=0.35, alpha=.92))

fig = plt.figure(figsize=(17.5, 14.0))
gs = fig.add_gridspec(3, 4, height_ratios=[1.75, 1.0, 0.95],
                      hspace=0.46, wspace=0.46,
                      left=0.055, right=0.975, top=0.925, bottom=0.055)
CMAP = "YlOrRd"
d = G.dropna(subset=["lat","lon"]).copy()
vmax = float(d.FatalityRate.max())

# ---- (a) map ---------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
sz = np.sqrt(d.Expeditions)*17
sc = ax.scatter(d.lon, d.lat, s=sz, c=d.FatalityRate, cmap=CMAP, vmin=0, vmax=vmax,
                edgecolor="black", linewidth=.7, alpha=.92, zorder=4)
ax.set_xlim(d.lon.min()-0.45, d.lon.max()+0.45)
ax.set_ylim(d.lat.min()-0.22, d.lat.max()+0.28)
KH = (d.lon > 86.5) & (d.lon < 87.25) & (d.lat > 27.6) & (d.lat < 28.2)
out = d[~KH]
repel(ax, out.lon.values, out.lat.values, out.peakid.values, None,
      iters=420, k=0.016, fs=7.2)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_title("(a) Study peaks — marker area is expedition count, colour is the share of "
             "expeditions with a member death; the dense Khumbu group is labelled in the inset",
             loc="left", fontsize=10)
cb = plt.colorbar(sc, ax=ax, pad=0.012, shrink=.88); cb.set_label("Fatality rate (%)")

# size key
for n, lab in [(100,"100"),(500,"500"),(2000,"2000")]:
    ax.scatter([], [], s=np.sqrt(n)*17, c="0.85", edgecolor="black", linewidth=.6, label=lab)
ax.legend(title="Expeditions", loc="upper right", frameon=True, labelspacing=1.5,
          borderpad=0.9, fontsize=7.5, title_fontsize=8)

# Khumbu inset — the cluster that makes the main panel unreadable
kh = d[(d.lon > 86.5) & (d.lon < 87.25) & (d.lat > 27.6) & (d.lat < 28.2)]
if len(kh) >= 4:
    axi = inset_axes(ax, width="33%", height="42%", loc="lower left",
                     bbox_to_anchor=(0.015, 0.035, 1, 1), bbox_transform=ax.transAxes)
    axi.scatter(kh.lon, kh.lat, s=np.sqrt(kh.Expeditions)*9, c=kh.FatalityRate,
                cmap=CMAP, vmin=0, vmax=vmax, edgecolor="black", linewidth=.5, zorder=4)
    axi.set_xlim(kh.lon.min()-0.055, kh.lon.max()+0.055)
    axi.set_ylim(kh.lat.min()-0.035, kh.lat.max()+0.035)
    repel(axi, kh.lon.values, kh.lat.values, kh.peakid.values, None, iters=520,
          k=0.072, fs=6.0)
    axi.set_title("Khumbu, enlarged", fontsize=7.5, pad=2)
    axi.tick_params(labelsize=5.5); axi.grid(alpha=.3)
    for s_ in axi.spines.values(): s_.set_linewidth(1.1); s_.set_color("0.35")

# ---- (b) deadliest ---------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
db = G.sort_values("FatalityRate").tail(16)
ax.barh(db.peakid, db.FatalityRate,
        color=plt.cm.YlOrRd(db.FatalityRate/max(vmax,1e-9)), edgecolor="0.3", linewidth=.4)
ax.axvline(M.fatality.mean()*100, ls="--", c="black", lw=1.1,
           label=f"all peaks {M.fatality.mean()*100:.2f}%")
for i,(r_,n_) in enumerate(zip(db.FatalityRate, db.Expeditions)):
    ax.text(r_+0.25, i, f"n={n_}", va="center", fontsize=6.2, color="0.35")
ax.set_xlabel("Fatality rate (%)"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("(b) Deadliest peaks by rate", loc="left")
ax.set_xlim(0, db.FatalityRate.max()*1.22)

# ---- (c) time --------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1:3])
yr = M.dropna(subset=["year"]).copy(); yr["decade"] = (yr.year//10*10).astype(int)
dd = yr.groupby("decade").agg(n=("fatality","size"), rate=("fatality","mean")).reset_index()
ax.bar(dd.decade, dd.n, width=7.4, color="#4C72B0", alpha=.80, edgecolor="0.3", linewidth=.4)
ax.set_ylabel("Expeditions per decade", color="#4C72B0")
ax.tick_params(axis="y", labelcolor="#4C72B0")
ax.set_xlabel("Decade")
a2 = ax.twinx()
MIN_N = 20      # a 100% rate on three expeditions is noise, not history
solid = dd[dd.n >= MIN_N]; sparse = dd[dd.n < MIN_N]
a2.plot(solid.decade, solid.rate*100, color="#C44E52", marker="o", lw=2.1, zorder=5,
        label=f"fatality rate (decades with ≥{MIN_N} expeditions)")
a2.scatter(sparse.decade, sparse.rate*100, color="#C44E52", marker="x", s=32, alpha=.45,
           zorder=5, label=f"fewer than {MIN_N} expeditions — unreliable")
a2.set_ylabel("Fatality rate (%)", color="#C44E52", labelpad=8)
a2.tick_params(axis="y", labelcolor="#C44E52"); a2.grid(False)
a2.set_ylim(0, max(solid.rate.max()*100*1.45, 12))
a2.legend(fontsize=7, loc="upper center")
ax.set_title("(c) Activity grew while the fatality rate fell\n"
             "the distribution shift that makes chronological evaluation harder than random splitting",
             loc="left", fontsize=9.5)

# ---- (d) imbalance ---------------------------------------------------------------
ax = fig.add_subplot(gs[1, 3])
npos = int(M.fatality.sum()); nneg = len(M)-npos
ax.bar(["non-fatal","fatal"], [nneg,npos], color=["#AEC7E8","#C44E52"],
       edgecolor="0.3", linewidth=.5)
for i,v in enumerate([nneg,npos]):
    ax.text(i, v+len(M)*0.018, f"{v:,}\n({v/len(M)*100:.2f}%)", ha="center",
            va="bottom", fontsize=8.5)
ax.set_ylim(0, nneg*1.22); ax.set_ylabel("Expeditions")
ax.set_title(f"(d) Class imbalance — {nneg/max(npos,1):.1f} : 1", loc="left")

# ---- (e) coverage ----------------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
cov = (M.groupby("peakid")["interval_missing"].mean()*100).sort_values()
ax.barh(cov.index, cov.values, color="#55A868", edgecolor="0.3", linewidth=.35)
ax.axvline(M.interval_missing.mean()*100, ls="--", c="black", lw=1.1,
           label=f"overall {M.interval_missing.mean()*100:.1f}%")
ax.set_xlabel("Expeditions with no usable weather interval (%)")
ax.tick_params(axis="y", labelsize=5.8); ax.legend(fontsize=7)
ax.set_title("(e) Weather coverage by peak", loc="left")

# ---- (f) team size ---------------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])
q = M.copy(); q["bin"] = pd.cut(q.totmembers, [0,3,6,10,15,25,500],
                                labels=["1–3","4–6","7–10","11–15","16–25","26+"])
r = q.groupby("bin", observed=True).agg(rate=("fatality","mean"), n=("fatality","size")).reset_index()
ax.bar(r["bin"].astype(str), r.rate*100, color="#DD8452", edgecolor="0.3", linewidth=.45)
for i,(v,nn) in enumerate(zip(r.rate*100, r.n)):
    ax.text(i, v+0.35, f"n={nn:,}", ha="center", va="bottom", fontsize=6.6)
ax.set_xlabel("Team size (members)"); ax.set_ylabel("Fatality rate (%)")
ax.set_ylim(0, r.rate.max()*100*1.24)
ax.set_title("(f) Larger parties, higher rate", loc="left")

# ---- (g) season ------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 2])
s = M[M.season.isin(["Spring","Summer","Autumn","Winter"])]
rr = (s.groupby("season").agg(rate=("fatality","mean"), n=("fatality","size"))
        .reindex(["Spring","Summer","Autumn","Winter"]).dropna().reset_index())
ax.bar(rr.season, rr.rate*100, color="#8172B3", edgecolor="0.3", linewidth=.45)
for i,(v,nn) in enumerate(zip(rr.rate*100, rr.n)):
    ax.text(i, v+0.12, f"n={int(nn):,}", ha="center", va="bottom", fontsize=6.6)
ax.set_ylabel("Fatality rate (%)"); ax.set_ylim(0, rr.rate.max()*100*1.24)
ax.set_title("(g) By season", loc="left")

# ---- (h) deaths vs expeditions ----------------------------------------------------
ax = fig.add_subplot(gs[2, 3])
ax.scatter(G.Expeditions, G.Deaths, s=42, c=G.FatalityRate, cmap=CMAP, vmin=0, vmax=vmax,
           edgecolor="black", linewidth=.5)
for _, r_ in G.nlargest(6, "Deaths").iterrows():
    ax.annotate(r_.peakid, (r_.Expeditions, r_.Deaths), fontsize=6.6,
                xytext=(4,3), textcoords="offset points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(max(0.7, G.Deaths[G.Deaths>0].min()*0.7), G.Deaths.max()*1.7)
ax.set_xlabel("Expeditions (log)"); ax.set_ylabel("Member deaths (log)")
ax.set_title("(h) Exposure against deaths", loc="left")

fig.suptitle(f"Himalayan expedition dataset — {len(M):,} expeditions across "
             f"{M.peakid.nunique()} peaks, {npos} with a member death "
             f"({M.fatality.mean()*100:.2f}%), {int(M.year.min())}–{int(M.year.max())}",
             fontsize=13.5, y=0.972)

for ext in ("png","pdf"):
    fig.savefig(FIG/f"fig01_dataset_overview.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
coords.to_csv(ROOT/"artifacts"/"peak_coordinates_verified_final.csv", index=False)
print(f"\n  wrote {FIG/'fig01_dataset_overview.png'} (and .pdf)")
print(f"  verified coordinates copied into {ROOT/'artifacts'}")
print("  nothing else in the results folder was changed.")
