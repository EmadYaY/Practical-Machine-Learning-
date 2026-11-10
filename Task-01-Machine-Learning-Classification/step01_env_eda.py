# -*- coding: utf-8 -*-
"""
Step 01 - Environment check + Data loading + EDA
Dataset : Phishing Websites (phishing.csv)
Outputs : outputs/logs, outputs/figures (PNG+PDF, 300 dpi), outputs/tables
"""
import os
import sys
import time
import platform
import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_selection import mutual_info_classif

# ----------------------------- Config -----------------------------
DATA_PATH = "phishing.csv"
ID_COL = "Index"
TARGET = "class"
POS_LABEL = 1          # only used for the conditional heatmap (verify meaning later)
SEED = 42
SHOW_PLOTS = False     # True => also display figures on screen

OUT_DIR = "outputs"
LOG_DIR = os.path.join(OUT_DIR, "logs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in (LOG_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- Logger -----------------------------
logger = logging.getLogger("step01")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(os.path.join(LOG_DIR, "step01_eda.log"), mode="w", encoding="utf-8")
_sh = logging.StreamHandler(sys.stdout)
for h in (_fh, _sh):
    h.setFormatter(_fmt)
    logger.addHandler(h)


def section(title):
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def savefig(name):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    logger.info("Figure saved: %s (.png/.pdf)", name)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


T_START = time.time()

# ------------------------ A. Environment --------------------------
section("A. Environment")
logger.info("Python       : %s", sys.version.replace("\n", " "))
logger.info("Platform     : %s", platform.platform())
for name, mod in [("numpy", np), ("pandas", pd), ("matplotlib", matplotlib),
                  ("seaborn", sns), ("scikit-learn", sklearn)]:
    logger.info("%-13s: %s", name, mod.__version__)
logger.info("Random seed  : %d", SEED)

# ------------------------ B. Loading ------------------------------
section("B. Loading data")
if not os.path.exists(DATA_PATH):
    logger.error("File not found: %s (put it next to this script)", DATA_PATH)
    sys.exit(1)

t0 = time.time()
df = pd.read_csv(DATA_PATH)
logger.info("Loaded in %.3f s", time.time() - t0)
logger.info("Shape        : %s", df.shape)
logger.info("Columns (%d) : %s", len(df.columns), list(df.columns))
logger.info("Dtype counts :\n%s", df.dtypes.value_counts().to_string())
logger.info("Memory (MB)  : %.3f", df.memory_usage(deep=True).sum() / 1024 ** 2)
logger.info("Head(5):\n%s", df.head().to_string())

# ------------------------ C. Data quality -------------------------
section("C. Data quality")
features = [c for c in df.columns if c not in (ID_COL, TARGET)]
logger.info("Number of features (without %s and %s): %d", ID_COL, TARGET, len(features))
logger.info("Total missing values : %d", int(df.isna().sum().sum()))
logger.info("ID column '%s' unique : %s", ID_COL, df[ID_COL].is_unique)

all_vals = sorted(pd.unique(df[features].values.ravel()).tolist())
logger.info("Distinct values across all features : %s", all_vals)
logger.info("Distinct values of target           : %s", sorted(df[TARGET].unique().tolist()))

n_levels = pd.Series({c: df[c].nunique() for c in features})
logger.info("Features grouped by number of distinct levels:\n%s",
            n_levels.value_counts().sort_index().to_string())
logger.info("Binary (2-level) features : %s", n_levels[n_levels == 2].index.tolist())

dup_feat = int(df.duplicated(subset=features).sum())
dup_full = int(df.duplicated(subset=features + [TARGET]).sum())
conflicts = int((df.groupby(features)[TARGET].nunique() > 1).sum())
logger.info("Duplicate rows (features only)        : %d", dup_feat)
logger.info("Duplicate rows (features + target)    : %d", dup_full)
logger.info("Feature-vectors with CONFLICTING class: %d", conflicts)

desc = df[features].describe().T
desc.to_csv(os.path.join(TAB_DIR, "table01_descriptive_stats.csv"))
logger.info("Descriptive statistics:\n%s", desc.round(4).to_string())

# ------------------------ D. Class distribution -------------------
section("D. Class distribution")
counts = df[TARGET].value_counts().sort_index()
pct = counts / counts.sum() * 100
ratio = counts.max() / counts.min()
logger.info("Class counts :\n%s", counts.to_string())
logger.info("Class percent:\n%s", pct.round(3).to_string())
logger.info("Imbalance ratio (max/min): %.4f", ratio)
pd.DataFrame({"count": counts, "percent": pct.round(3)}).to_csv(
    os.path.join(TAB_DIR, "table02_class_distribution.csv"))

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar([str(i) for i in counts.index], counts.values, color=["#4c72b0", "#dd8452"][:len(counts)])
for b, c, p in zip(bars, counts.values, pct.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}\n({p:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Class label")
ax.set_ylabel("Number of samples")
ax.set_title("Distribution of data - class")
ax.set_ylim(0, counts.max() * 1.18)
savefig("fig01_class_distribution")

# ------------------------ E. Feature levels -----------------------
section("E. Feature level distribution")
level_tab = (pd.DataFrame({c: df[c].value_counts(normalize=True) for c in features}).T
             .reindex(columns=all_vals).fillna(0))
level_tab.to_csv(os.path.join(TAB_DIR, "table03_feature_level_proportions.csv"))
logger.info("Proportion of each level per feature:\n%s", level_tab.round(4).to_string())

palette = {-1: "#c44e52", 0: "#ccb974", 1: "#55a868"}
level_plot = level_tab.sort_values(by=all_vals[0])
level_plot.plot(kind="barh", stacked=True, figsize=(9, 9),
                color=[palette.get(v, "#999999") for v in level_plot.columns], width=0.85)
plt.xlabel("Proportion of samples")
plt.ylabel("Feature")
plt.title("Level distribution of features")
plt.legend(title="Level", bbox_to_anchor=(1.01, 1), loc="upper left")
savefig("fig02_feature_levels")

# ------------------------ F. Correlations -------------------------
section("F. Correlation analysis")
corr_target = df[features + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
corr_target.to_csv(os.path.join(TAB_DIR, "table04_corr_with_target.csv"), header=["pearson_corr"])
logger.info("Pearson correlation of each feature with target:\n%s", corr_target.round(4).to_string())

plt.figure(figsize=(8, 8))
plt.barh(corr_target.index, corr_target.values,
         color=["#c44e52" if v < 0 else "#4c72b0" for v in corr_target.values])
plt.xlabel("Pearson correlation with class")
plt.title("Feature-target correlation")
savefig("fig03_corr_with_target")

corr = df[features].corr()
iu = np.triu_indices_from(corr, k=1)
pairs = pd.DataFrame({"feature_1": corr.index[iu[0]], "feature_2": corr.columns[iu[1]],
                      "corr": corr.values[iu]})
pairs["abs_corr"] = pairs["corr"].abs()
top_pairs = pairs.sort_values("abs_corr", ascending=False).head(10)
top_pairs.to_csv(os.path.join(TAB_DIR, "table05_top_correlated_pairs.csv"), index=False)
logger.info("Top 10 most correlated feature pairs:\n%s", top_pairs.round(4).to_string(index=False))

plt.figure(figsize=(11, 9.5))
sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, square=True,
            xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7)
plt.title("Feature correlation matrix")
savefig("fig04_feature_corr_heatmap")

# ------------------------ G. Mutual information -------------------
section("G. Mutual information with target")
mi = mutual_info_classif(df[features].astype(int), df[TARGET], discrete_features=True, random_state=SEED)
mi_s = pd.Series(mi, index=features).sort_values()
mi_s.to_csv(os.path.join(TAB_DIR, "table06_mutual_information.csv"), header=["mutual_info"])
logger.info("Mutual information (nats):\n%s", mi_s.sort_values(ascending=False).round(5).to_string())

plt.figure(figsize=(8, 8))
plt.barh(mi_s.index, mi_s.values, color="#4c72b0")
plt.xlabel("Mutual information (nats)")
plt.title("Feature importance by mutual information")
savefig("fig05_mutual_information")

# ------------------------ H. Level vs class -----------------------
section("H. P(class = POS_LABEL | feature level)")
cond = pd.DataFrame({c: df.groupby(c)[TARGET].apply(lambda s: (s == POS_LABEL).mean())
                     for c in features}).T.reindex(columns=all_vals)
cond.to_csv(os.path.join(TAB_DIR, "table07_class_given_level.csv"))
logger.info("P(class=%s | level):\n%s", POS_LABEL, cond.round(3).to_string())

plt.figure(figsize=(6.5, 10))
sns.heatmap(cond, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            cbar_kws={"label": f"P(class = {POS_LABEL} | level)"})
plt.xlabel("Feature level")
plt.ylabel("Feature")
plt.title("Class probability per feature level")
savefig("fig06_class_given_level")

# ------------------------ I. Save step output ---------------------
section("I. Saving step output")
df_out = df.drop(columns=[ID_COL])
df_out.to_csv(os.path.join(OUT_DIR, "data_step1.csv"), index=False)
logger.info("Saved %s with shape %s (ID column dropped, NO rows removed)",
            os.path.join(OUT_DIR, "data_step1.csv"), df_out.shape)
logger.info("Total runtime: %.2f s", time.time() - T_START)
logger.info("DONE.")