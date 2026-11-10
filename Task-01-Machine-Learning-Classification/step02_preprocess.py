# -*- coding: utf-8 -*-
"""
Step 02 - Label mapping, duplicate merge, stratified 3-way split (train/val/test),
          leakage measurement, level representations.
Input : outputs/data_step1.csv
Output: outputs/data/dataset_raw.csv, outputs/data/dataset_dedup.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from common import (SEED, OUT_DIR, DATA_DIR, TAB_DIR, ORIG_TARGET, TARGET,
                    get_logger, section, savefig, get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step02_preprocess")

# ---------------------- A. Load + label mapping ----------------------
section(log, "A. Load + label mapping")
df = pd.read_csv(os.path.join(OUT_DIR, "data_step1.csv"))
log.info("Loaded data_step1.csv: shape=%s", df.shape)
df[TARGET] = (df[ORIG_TARGET] == -1).astype(int)
FEATURES = get_features(df)
log.info("Mapping: class=-1 -> y=1 (phishing) | class=1 -> y=0 (legitimate)")
log.info("Counts:\n%s", df[TARGET].value_counts().rename({1: "phishing(y=1)", 0: "legitimate(y=0)"}).to_string())
for f in ("HTTPS", "AnchorURL", "PrefixSuffix-"):
    log.info("Sanity check - phishing rate (mean y) by level of %s:\n%s",
             f, df.groupby(f)[TARGET].mean().round(3).to_string())

raw = df.drop(columns=[ORIG_TARGET]).reset_index(drop=True)

# ---------------------- B. Merge duplicate records -------------------
section(log, "B. Merge duplicate records")
grp = df.groupby(FEATURES)[TARGET].agg(n_rows="size", n_phish="sum").reset_index()
grp["n_legit"] = grp["n_rows"] - grp["n_phish"]
log.info("Rows before merge                 : %d", len(df))
log.info("Unique feature vectors            : %d", len(grp))
log.info("Duplicate rows removed by merging : %d (%.2f%%)", len(df) - len(grp),
         (len(df) - len(grp)) / len(df) * 100)

conf = grp[(grp.n_phish > 0) & (grp.n_legit > 0)]
ties = grp[grp.n_phish == grp.n_legit]
log.info("Conflicting groups (both classes) : %d  (rows involved: %d)", len(conf), int(conf.n_rows.sum()))
log.info("Tie groups (50/50) -> DROPPED     : %d  (rows: %d)", len(ties), int(ties.n_rows.sum()))
grp = grp[grp.n_phish != grp.n_legit].copy()
grp[TARGET] = (grp.n_phish > grp.n_legit).astype(int)   # majority vote
log.info("Conflicts resolved by majority vote; groups kept: %d", len(grp))

top = grp.sort_values("n_rows", ascending=False).head(10)[["n_rows", "n_phish", "n_legit"]]
log.info("Top-10 largest duplicate groups:\n%s", top.to_string())
log.info("Group-size statistics:\n%s", grp["n_rows"].describe().round(3).to_string())
top.to_csv(os.path.join(TAB_DIR, "table08_top_duplicate_groups.csv"))

dedup = grp[FEATURES + [TARGET, "n_rows"]].reset_index(drop=True)
log.info("Dedup dataset shape: %s", dedup.shape)
log.info("Dedup class counts (y):\n%s", dedup[TARGET].value_counts().to_string())
log.info("Dedup class percent (y):\n%s", (dedup[TARGET].value_counts(normalize=True) * 100).round(3).to_string())

vc = grp["n_rows"].value_counts().sort_index()
plt.figure(figsize=(8, 4.5))
plt.bar(vc.index, vc.values, color="#4c72b0")
plt.yscale("log")
plt.xlabel("Group size (number of identical rows)")
plt.ylabel("Number of groups (log scale)")
plt.title("Distribution of duplicate-group sizes")
savefig(log, "fig07_duplicate_group_sizes")


# ---------------------- C. Stratified split --------------------------
def split_3way(d):
    y = d[TARGET].values
    idx = np.arange(len(d))
    i_tr, i_tmp = train_test_split(idx, test_size=0.40, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    sub = np.empty(len(d), dtype=object)
    sub[i_tr], sub[i_va], sub[i_te] = "train", "val", "test"
    d = d.copy()
    d["subset"] = sub
    return d


def leak_rate(d, subset):
    """Share of rows in `subset` whose feature vector also appears in train."""
    h = pd.util.hash_pandas_object(d[FEATURES], index=False).values
    m = d["subset"].values
    return float(np.isin(h[m == subset], h[m == "train"]).mean())


section(log, "C. Stratified 60/20/20 split + leakage measurement")
raw_s = split_3way(raw)
dedup_s = split_3way(dedup)

rows = []
for name, d in (("raw", raw_s), ("dedup", dedup_s)):
    for s in ("train", "val", "test"):
        part = d[d.subset == s]
        rows.append(dict(dataset=name, subset=s, n=len(part),
                         phishing=int(part[TARGET].sum()),
                         legit=int(len(part) - part[TARGET].sum()),
                         phishing_pct=round(part[TARGET].mean() * 100, 2),
                         leak_from_train_pct=0.0 if s == "train" else round(leak_rate(d, s) * 100, 2)))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(TAB_DIR, "table09_split_summary.csv"), index=False)
log.info("Split summary (leak_from_train_pct = %% of rows whose feature vector also exists in train):\n%s",
         summ.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
labels = (summ.dataset + "\n" + summ.subset).values
ax[0].bar(labels, summ.legit, label="legitimate (y=0)", color="#55a868")
ax[0].bar(labels, summ.phishing, bottom=summ.legit, label="phishing (y=1)", color="#c44e52")
ax[0].set_ylabel("Number of samples")
ax[0].set_title("Class distribution per subset")
ax[0].legend()
leak = summ[summ.subset != "train"]
sns.barplot(data=leak, x="subset", y="leak_from_train_pct", hue="dataset", ax=ax[1])
for c in ax[1].containers:
    ax[1].bar_label(c, fmt="%.1f%%")
ax[1].set_ylabel("Rows also present in train (%)")
ax[1].set_title("Data leakage: raw vs. deduplicated")
savefig(log, "fig08_split_and_leakage")

# ---------------------- D. Level representations ---------------------
section(log, "D. Level representations (computed on dedup-train for preview)")
Xtr = dedup_s.loc[dedup_s.subset == "train", FEATURES]
zero_cells = int((Xtr == 0).sum().sum())
log.info("Cells equal to 0 (suspicious level) in dedup-train: %d of %d (%.2f%%)",
         zero_cells, Xtr.size, zero_cells / Xtr.size * 100)
rep_rows = []
for mode in ("ordinal", "onehot", "merge0_neg", "merge0_pos"):
    R = make_representation(Xtr, mode)
    dropped = sorted(set(FEATURES) - set(R.columns)) if mode.startswith("merge") else []
    rep_rows.append(dict(mode=mode, n_features=R.shape[1], dropped_constant=";".join(dropped)))
rep = pd.DataFrame(rep_rows)
rep.to_csv(os.path.join(TAB_DIR, "table10_representations.csv"), index=False)
log.info("Representations:\n%s", rep.to_string(index=False))

# ---------------------- E. Save --------------------------------------
section(log, "E. Saving datasets")
raw_s.to_csv(os.path.join(DATA_DIR, "dataset_raw.csv"), index=False)
dedup_s.to_csv(os.path.join(DATA_DIR, "dataset_dedup.csv"), index=False)
log.info("Saved dataset_raw.csv   : %s (cols: features + y + subset)", raw_s.shape)
log.info("Saved dataset_dedup.csv : %s (cols: features + y + n_rows + subset)", dedup_s.shape)
log.info("DONE.")