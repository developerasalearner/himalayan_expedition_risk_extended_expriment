# The two source datasets

Neither dataset is redistributed here. Both are public, both carry their own
terms, and `master_experiment_v5.py` downloads them automatically through
`kagglehub` on first run.

---

## 1. Expedition records

**The Himalayan Database**, the archive begun by Elizabeth Hawley, obtained
through its public Kaggle release.

| | |
|---|---|
| Kaggle | `siddharth0935/himalayan-expeditions` |
| Version used | **1** |
| Source stated on the dataset page | The Himalayan Database |
| Tables | `exped.csv`, `peaks.csv`, `members.csv`, `refer.csv` |
| Rows used | 11,425 expeditions, of which 9,457 retained |
| Accessed | 29 September 2026 |

The pipeline reads `exped.csv` only. `members.csv` is read by
`scripts/verify_death_causes.py`, which reproduces the death-date and
cause-of-death counts quoted in the Discussion.

## 2. Meteorological record

A per-peak compilation of the **Open-Meteo Historical Weather API**, which
sources its reanalysis from the Copernicus Climate Change Service.

| | |
|---|---|
| Kaggle | `bonesclarke26/historic-weather-data-for-himalayan-peaks` |
| Version used | **6** |
| Upstream | Open-Meteo Historical Weather API |
| Licence | **CC BY 4.0** — attribution is a licence condition |
| Granularity | **hourly**, timestamped UTC, 1 January 1980 to 31 December 2024 |
| Coverage | 27 peaks, one Parquet file each, with a `.txt` coverage report |
| Variables | 15 available; 5 used |
| Accessed | 29 September 2026 |

**Version numbers matter here.** Kaggle datasets are mutable and this one has
been revised at least six times, with peaks and years added. Neither dataset
has a DOI, so the version number is doing the work a DOI normally would.

Three properties of this archive caused real errors in an earlier version of
this study, and are documented in [`../docs/CHANGELOG.md`](../docs/CHANGELOG.md):

- it is **hourly**, not daily, so counters must collapse to days first;
- **snowfall is in centimetres**, not millimetres;
- the Y/N archive flags in the expedition tables are stored as **booleans**,
  so a parser testing for the string `"Y"` silently returns zero for every row.

---

## Getting a Kaggle token

```
kaggle.com -> your profile -> Settings -> API -> Create New Token
```

That downloads `kaggle.json`. Install it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Confirm it works before committing to a long run:

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('siddharth0935/himalayan-expeditions'))"
```

**Never commit the token.** `.gitignore` excludes `kaggle.json`, `.kaggle/` and
anything matching `*access_token*`, but check before making a repository public.

## Manual fallback

If `kagglehub` cannot reach Kaggle, download both datasets from the web
interface, unzip them, and point the script at the folders by editing the two
`kagglehub.dataset_download(...)` calls near the top of the data stage in
`scripts/master_experiment_v5.py` to the local paths. Nothing else changes.

## Attribution

The weather compilation is CC BY 4.0, so any reuse must credit Open-Meteo. The
manuscript does so in its Data section and its data availability statement.
