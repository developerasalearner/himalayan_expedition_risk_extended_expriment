# Updating the repository on GitHub

Written for the **GitHub website**. No git installation needed.

This package **replaces** the contents of an existing repository, which holds
the earlier v4 run. Those results are superseded; see
[`CHANGELOG.md`](CHANGELOG.md) for what changed and why.

---

## 0. First, settle the repository name

The manuscript's data availability statement points at:

```
https://github.com/developerasalearner/himalayan-expedition-risk
```

If your repository is currently named anything else, **that link will 404 for
a reviewer.** Fix it one of two ways before submitting:

- **Rename the repository** to `himalayan-expedition-risk`
  (Settings → General → Repository name → Rename). GitHub redirects the old
  URL, so nothing breaks; or
- **Change the URL in the manuscript** to match the existing name, in the data
  availability statement and anywhere else it appears.

Renaming is tidier, and the hyphenated name reads better in a paper.

---

## 1. Remove the old files

The previous package had a different script name and a 32-sheet workbook, so
stale files will sit alongside the new ones if you only add.

On the repository page, delete these first:

- `scripts/master_experiment.py` — replaced by `master_experiment_v5.py`
- the whole `results/` folder — replaced wholesale

To delete a folder on the website, open any file inside it, use the
**⋯ → Delete file** menu, and repeat; GitHub removes a folder automatically
once it is empty. Alternatively delete the files in small batches and commit
each batch.

If that is tedious, the simplest route is to delete the repository and create
a fresh one with the correct name, then upload this package in one go. You
lose the commit history, which for a package like this does not matter.

---

## 2. Upload

On the repository page, **Add file → Upload files**.

Open the unzipped `himalayan-expedition-risk` folder, select **everything
inside it**, and drag it into the browser. Folder structure is preserved, so
`scripts/`, `docs/`, `data/` and `results/` arrive intact.

There are about 80 files totalling roughly 4 MB, comfortably inside GitHub's
limits for a single upload.

Commit message: *"v5 run: corrected weather aggregation, training-side
imputation, feature-family ablation"*.

### One file will not upload by dragging

`.gitignore` starts with a dot, and macOS Finder hides it. Either press
**Cmd+Shift+.** in Finder to show hidden files before selecting, or create it
on GitHub afterwards with **Add file → Create new file**, name it
`.gitignore`, and paste the contents from this package.

---

## 3. Before you make it public

Search the repository for credentials. The previous package was prepared after
a Kaggle token was accidentally exposed, so this is not hypothetical.

In the GitHub search box, with the repository selected, search for:

```
kaggle
token
KGAT
access_token
```

Expected hits are documentation mentioning the *need* for a token, in
`data/README.md` and `docs/REPRODUCE.md`. **Any actual key is a problem**:
delete it, commit, and expire the key at kaggle.com → Settings → API →
Expire Token. Deleting a file does not remove it from history, so if a key was
ever committed, expire it regardless.

`.gitignore` excludes `kaggle.json`, `.kaggle/` and anything matching
`*access_token*`.

---

## 4. Check it works

Open the repository as a stranger would:

- Does `README.md` render, with the tables intact?
- Do the links to `docs/REPRODUCE.md`, `docs/RESULTS_MAP.md` and
  `data/README.md` resolve?
- Is `results/Himalayan_MasterResults.xlsx` downloadable?
- Do the figures in `results/figures/` preview?
- Does the clone URL in the README match the repository's actual name?

Then, from a clean directory, follow your own instructions:

```bash
git clone https://github.com/developerasalearner/himalayan-expedition-risk.git
cd himalayan-expedition-risk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_results.py --new results --reference results
```

That last command needs no Kaggle token and no long run: it reads the archived
tables and re-derives the paper's conclusions. If it prints thirteen `yes`
lines and exits 0, the package is sound.

---

## 5. Timing

Keep the repository **private** until the paper is accepted, then switch it to
public (Settings → General → Danger Zone → Change visibility).

Reviewers at most journals do not need access during review, and the data
availability statement can say the repository will be made public on
acceptance. If a reviewer does ask, GitHub's private-repository invitations
break anonymity, so send a zip through the editorial system instead.
