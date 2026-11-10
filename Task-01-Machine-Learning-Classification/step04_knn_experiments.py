# -*- coding: utf-8 -*-
"""
Step 04 - KNN experiments. Uses TRAIN (CV) and VALIDATION only; TEST stays locked.
Hyper-parameters are selected by 5-fold stratified CV on train, never by validation.
Input : outputs/data/dataset_raw.csv, outputs/data/dataset_dedup.csv
Output: tables 13+, figures fig11-fig20, outputs/data/best_knn_config.json
"""
import os
import json
import time
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, log_loss, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from common import (SEED, DATA_DIR, TAB_DIR, TARGET, get_logger, section, savefig,
                    get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step04_knn_experiments")

# ----------------------------- Config -----------------------------
QUICK = False   # True => much smaller grids (about 2-3 minutes) for a dry run
if QUICK:
    K_ALL, K_LIST, GRID_K = list(range(1, 16)), [1, 5, 11, 21], [1, 5, 11, 21]
else:
    K_ALL = list(range(1, 31))
    K_LIST = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    GRID_K = K_LIST
NF_LIST = [3, 5, 8, 10, 15, 20, 25, 30]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
MI_SCORE = functools.partial(mutual_info_classif, discrete_features=True, random_state=SEED)
CLASS_NAMES = ["Legitimate (0)", "Phishing (1)"]
T0 = time.time()


def tick(msg):
    log.info("[elapsed %.1fs] %s", time.time() - T0, msg)


# ----------------------------- Helpers ----------------------------
def load(fn):
    d = pd.read_csv(os.path.join(DATA_DIR, fn))
    feats = get_features(d)
    tr, va = d[d["subset"] == "train"], d[d["subset"] == "val"]
    return dict(Xtr=tr[feats].reset_index(drop=True), ytr=tr[TARGET].values,
                Xva=va[feats].reset_index(drop=True), yva=va[TARGET].values)


def rep_pair(Xtr, Xva, mode):
    """Level representation; constant columns are decided on TRAIN only."""
    if mode.startswith("merge"):
        v = -1 if mode == "merge0_neg" else 1
        a, b = Xtr.mask(Xtr == 0, v), Xva.mask(Xva == 0, v)
        keep = a.columns[a.nunique() > 1]
        return a[keep], b[keep]
    return make_representation(Xtr, mode), make_representation(Xva, mode)


def build(k, metric="euclidean", weights="uniform", scaler=None, n_feat=None):
    steps = []
    if n_feat is not None:
        steps.append(("fs", SelectKBest(score_func=MI_SCORE, k=n_feat)))
    if scaler is not None:
        steps.append(("sc", scaler()))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)))
    return Pipeline(steps) if len(steps) > 1 else steps[0][1]


def evaluate(R, k, **kw):
    Xtr, ytr, Xva, yva = R
    s = cross_val_score(build(k, **kw), Xtr, ytr, cv=CV, scoring="accuracy")
    m = build(k, **kw).fit(Xtr, ytr)
    return dict(cv_mean=float(s.mean()), cv_std=float(s.std()),
                val_acc=float(accuracy_score(yva, m.predict(Xva))))


def bce(y, p1, k):
    """Binary cross-entropy of the Laplace-smoothed KNN posterior (uniform weights).
    Raw KNN probabilities are 0/1 for small k, which makes BCE infinite; add-one
    smoothing p = (n_pos + 1) / (k + 2) keeps it finite and comparable across K."""
    p = (p1 * k + 1.0) / (k + 2.0)
    return float(log_loss(y, p, labels=[0, 1]))


def lineplot_cv_val(df, hue, title, fname):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for h, g in df.groupby(hue, sort=False):
        ax[0].errorbar(g["K"], g["cv_mean"], yerr=g["cv_std"], marker="o", capsize=2, label=str(h))
        ax[1].plot(g["K"], g["val_acc"], marker="o", label=str(h))
    ax[0].set_title(f"{title} - 5-fold CV accuracy (train)")
    ax[1].set_title(f"{title} - validation accuracy")
    for a in ax:
        a.set_xlabel("K (number of neighbors)")
    ax[0].set_ylabel("Accuracy")
    ax[1].legend(title=hue)
    savefig(log, fname)


def best_row(df):
    """Selection rule: highest CV mean; ties -> lower std, then larger K (smoother model)."""
    return df.sort_values(["cv_mean", "cv_std", "K"], ascending=[False, True, False]).iloc[0]


# ----------------------------- A. Load ----------------------------
section(log, "A0. Load (train/val only)")
data = {"raw": load("dataset_raw.csv"), "dedup": load("dataset_dedup.csv")}
D = data["dedup"]
for n, d in data.items():
    log.info("%-6s | train=%s | val=%s", n, d["Xtr"].shape, d["Xva"].shape)
log.info("CV protocol: %s", CV)
log.info("QUICK mode: %s | K_ALL=%s | K_LIST=%s", QUICK, K_ALL, K_LIST)

# ----------------------------- A. K sweep -------------------------
section(log, "A. K sweep: error, accuracy and BCE vs K (raw vs dedup)")
rows = []
for name, d in data.items():
    for k in K_ALL:
        m = KNeighborsClassifier(n_neighbors=k).fit(d["Xtr"], d["ytr"])
        ptr = m.predict_proba(d["Xtr"])[:, 1]
        pva = m.predict_proba(d["Xva"])[:, 1]
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), d["Xtr"], d["ytr"],
                             cv=CV, scoring="accuracy")
        rows.append(dict(dataset=name, K=k,
                         train_acc=accuracy_score(d["ytr"], m.predict(d["Xtr"])),
                         cv_mean=cv.mean(), cv_std=cv.std(),
                         val_acc=accuracy_score(d["yva"], m.predict(d["Xva"])),
                         train_bce=bce(d["ytr"], ptr, k), val_bce=bce(d["yva"], pva, k)))
    tick(f"K sweep finished for {name}")
sweepA = pd.DataFrame(rows)
sweepA.to_csv(os.path.join(TAB_DIR, "table13_k_sweep.csv"), index=False)
log.info("K sweep table:\n%s", sweepA.round(4).to_string(index=False))

best_k = {}
for name in data:
    b = best_row(sweepA[sweepA.dataset == name])
    best_k[name] = int(b.K)
    log.info("[%s] best K by CV = %d (CV=%.4f +- %.4f | val=%.4f)", name, b.K, b.cv_mean, b.cv_std, b.val_acc)
K_BEST = best_k["dedup"]
log.info("K_BEST (dedup) = %d ; K_BEST (raw) = %d  <- compare: leakage shifts the optimum", K_BEST, best_k["raw"])

g = sweepA[sweepA.dataset == "dedup"]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(g.K, g.train_acc, "o-", label="Training")
ax.plot(g.K, g.cv_mean, "s-", label="5-fold CV (train)")
ax.fill_between(g.K, g.cv_mean - g.cv_std, g.cv_mean + g.cv_std, alpha=0.2)
ax.plot(g.K, g.val_acc, "^-", label="Validation")
ax.axvline(K_BEST, ls="--", color="gray", label=f"best K = {K_BEST}")
ax.set_xlabel("K (number of neighbors)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN accuracy vs K (deduplicated data)")
ax.legend()
savefig(log, "fig11_accuracy_vs_k_dedup")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(g.K, 1 - g.train_acc, "o-", label="Training error")
ax[0].plot(g.K, 1 - g.val_acc, "^-", label="Validation error")
ax[0].set_title("Error rate vs K")
ax[0].set_ylabel("Error rate")
ax[1].plot(g.K, g.train_bce, "o-", label="Training BCE")
ax[1].plot(g.K, g.val_bce, "^-", label="Validation BCE")
ax[1].set_title("BCE loss vs K (Laplace-smoothed posterior)")
ax[1].set_ylabel("Binary cross-entropy")
for a in ax:
    a.set_xlabel("K (number of neighbors)")
    a.legend()
savefig(log, "fig12_error_and_bce_vs_k_dedup")

lineplot_cv_val(sweepA, "dataset", "Raw vs dedup", "fig13_k_sweep_raw_vs_dedup")

# ----------------------------- Representations --------------------
REPS = {m: rep_pair(D["Xtr"], D["Xva"], m) for m in ("ordinal", "onehot", "merge0_neg", "merge0_pos")}
R_ord = (D["Xtr"], D["ytr"], D["Xva"], D["yva"])
RR = {m: (a, D["ytr"], b, D["yva"]) for m, (a, b) in REPS.items()}
RR["ordinal"] = R_ord

# ----------------------------- B. Metrics -------------------------
section(log, "B. Distance metrics (dedup, ordinal representation)")
rows = []
for metric in ("euclidean", "manhattan", "chebyshev", "hamming"):
    for k in K_LIST:
        rows.append(dict(metric=metric, K=k, **evaluate(R_ord, k, metric=metric)))
    tick(f"metric {metric} done")
repB = pd.DataFrame(rows)
repB.to_csv(os.path.join(TAB_DIR, "table14_metrics.csv"), index=False)
log.info("Metric comparison:\n%s", repB.round(4).to_string(index=False))
log.info("Best per metric:\n%s", repB.groupby("metric", sort=False).apply(best_row, include_groups=False).round(4).to_string())
lineplot_cv_val(repB, "metric", "Distance metric", "fig14_distance_metrics")

# ----------------------------- C. Weights -------------------------
section(log, "C. Weighting scheme (dedup, euclidean)")
rows = []
for w in ("uniform", "distance"):
    for k in K_LIST:
        rows.append(dict(weights=w, K=k, **evaluate(R_ord, k, weights=w)))
repC = pd.DataFrame(rows)
repC.to_csv(os.path.join(TAB_DIR, "table15_weights.csv"), index=False)
log.info("Weights comparison:\n%s", repC.round(4).to_string(index=False))
lineplot_cv_val(repC, "weights", "Weighting", "fig15_weights")

# ----------------------------- D. Levels / Merge ------------------
section(log, "D. Level representations: ordinal / one-hot / merge0_neg / merge0_pos")
for m_, (a, _) in REPS.items():
    log.info("Representation %-11s -> %d features", m_, a.shape[1])
rows = []
for mode, R in RR.items():
    for k in K_LIST:
        rows.append(dict(representation=mode, K=k, **evaluate(R, k)))
    tick(f"representation {mode} done")
repD = pd.DataFrame(rows)
repD.to_csv(os.path.join(TAB_DIR, "table16_representations.csv"), index=False)
log.info("Representation comparison:\n%s", repD.round(4).to_string(index=False))
log.info("Best per representation:\n%s",
         repD.groupby("representation", sort=False).apply(best_row, include_groups=False).round(4).to_string())
lineplot_cv_val(repD, "representation", "Level representation", "fig16_representations")

# ----------------------------- E. Scaling -------------------------
section(log, "E. Feature scaling (inside Pipeline; scaler fitted on training folds only)")
stds = D["Xtr"].std().sort_values()
log.info("Feature std on train: min=%s (%.3f) | max=%s (%.3f)", stds.index[0], stds.iloc[0], stds.index[-1], stds.iloc[-1])
rows = []
for sname, sc in {"none": None, "StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler}.items():
    for k in K_LIST:
        rows.append(dict(scaler=sname, K=k, **evaluate(R_ord, k, scaler=sc)))
    tick(f"scaler {sname} done")
repE = pd.DataFrame(rows)
repE.to_csv(os.path.join(TAB_DIR, "table17_scaling.csv"), index=False)
log.info("Scaling comparison:\n%s", repE.round(4).to_string(index=False))
lineplot_cv_val(repE, "scaler", "Scaling", "fig17_scaling")

# ----------------------------- F. Feature selection ---------------
section(log, "F. Feature selection by mutual information (computed on TRAIN only)")
mi = pd.Series(mutual_info_classif(D["Xtr"].values.astype(int), D["ytr"],
                                   discrete_features=True, random_state=SEED),
               index=D["Xtr"].columns).sort_values(ascending=False)
mi.to_csv(os.path.join(TAB_DIR, "table18_mi_train.csv"), header=["mutual_info"])
log.info("MI ranking on train (nats):\n%s", mi.round(5).to_string())
rows = []
for k in sorted({1, 5, K_BEST}):
    for nf in NF_LIST:
        rows.append(dict(K=k, n_features=nf, **evaluate(R_ord, k, n_feat=nf)))
    tick(f"feature selection done for K={k}")
repF = pd.DataFrame(rows)
repF.to_csv(os.path.join(TAB_DIR, "table19_feature_selection.csv"), index=False)
log.info("Feature-selection results:\n%s", repF.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
for k, gg in repF.groupby("K"):
    ln, = ax[0].plot(gg.n_features, gg.cv_mean, "o-", label=f"K={k} (CV)")
    ax[0].plot(gg.n_features, gg.val_acc, "^--", color=ln.get_color(), alpha=0.7, label=f"K={k} (val)")
ax[0].set_xlabel("Number of selected features (top-N by MI)")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("KNN accuracy vs number of features")
ax[0].legend(fontsize=8)
ms = mi.sort_values()
ax[1].barh(ms.index, ms.values, color="#4c72b0")
ax[1].set_xlabel("Mutual information (nats)")
ax[1].set_title("MI ranking (train)")
ax[1].tick_params(axis="y", labelsize=7)
savefig(log, "fig18_feature_selection")

# ----------------------------- G. Joint grid ----------------------
section(log, "G. Joint grid: representation x metric x weights x K (selected by CV only)")
top_reps = repD.groupby("representation")["cv_mean"].max().sort_values(ascending=False).index[:3].tolist()
log.info("Top-3 representations (by best CV in section D): %s", top_reps)
rows = []
for rep_name in top_reps:
    for metric in ("euclidean", "manhattan", "hamming"):
        for w in ("uniform", "distance"):
            for k in GRID_K:
                rows.append(dict(representation=rep_name, metric=metric, weights=w, K=k,
                                 **evaluate(RR[rep_name], k, metric=metric, weights=w)))
    tick(f"grid finished for representation {rep_name}")
grid = pd.DataFrame(rows).sort_values(["cv_mean", "cv_std", "K"], ascending=[False, True, False]).reset_index(drop=True)
grid.to_csv(os.path.join(TAB_DIR, "table20_joint_grid.csv"), index=False)
log.info("Joint grid: %d configurations. Top-15 by CV:\n%s", len(grid), grid.head(15).round(4).to_string())
best = grid.iloc[0]
log.info("BEST CONFIG (by CV): %s", best.to_dict())

top = grid.head(15).iloc[::-1]
labels = [f"{r.representation} | {r.metric} | {r.weights} | K={r.K}" for r in top.itertuples()]
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.barh(labels, top.cv_mean, xerr=top.cv_std, color="#4c72b0", alpha=0.85, label="5-fold CV (train)")
ax.scatter(top.val_acc, labels, color="#c44e52", zorder=3, label="Validation")
ax.set_xlim(max(0.85, top.cv_mean.min() - 0.02), min(1.0, top.val_acc.max() + 0.02))
ax.set_xlabel("Accuracy")
ax.set_title("Top-15 KNN configurations (selected by CV)")
ax.tick_params(axis="y", labelsize=8)
ax.legend(loc="lower right")
savefig(log, "fig19_top_configurations")

# ----------------------------- H. Learning curves -----------------
section(log, "H. Learning curves (baseline k=1 vs best configuration)")
Rb = RR[best.representation]
est_base = build(1)
est_best = build(int(best.K), metric=best.metric, weights=best.weights)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for a, (title, est, X) in zip(ax, (("Baseline (k=1, euclidean, ordinal)", est_base, R_ord[0]),
                                   (f"Best ({best.representation}, {best.metric}, {best.weights}, K={int(best.K)})", est_best, Rb[0]))):
    sizes, trs, tes = learning_curve(est, X, D["ytr"], train_sizes=np.linspace(0.1, 1.0, 10),
                                     cv=CV, scoring="accuracy", shuffle=True, random_state=SEED)
    log.info("[%s]\n%s", title, pd.DataFrame({"train_size": sizes, "train_acc": trs.mean(1),
                                               "cv_acc": tes.mean(1), "cv_std": tes.std(1)}).round(4).to_string(index=False))
    a.plot(sizes, trs.mean(1), "o-", label="Training")
    a.plot(sizes, tes.mean(1), "s-", label="Cross-validation")
    a.fill_between(sizes, tes.mean(1) - tes.std(1), tes.mean(1) + tes.std(1), alpha=0.2)
    a.set_title(title, fontsize=9)
    a.set_xlabel("Number of training samples")
    a.legend()
ax[0].set_ylabel("Accuracy")
savefig(log, "fig20_learning_curves")

# ----------------------------- I. Validation report ---------------
section(log, "I. Validation report: baseline vs best (TEST still locked)")
rows = []
for tag, est, R in (("baseline", est_base, R_ord), ("best", est_best, Rb)):
    m = est.fit(R[0], R[1])
    p = m.predict(R[2])
    rows.append(dict(model=tag, accuracy=accuracy_score(R[3], p), error=1 - accuracy_score(R[3], p),
                     precision_phish=precision_score(R[3], p), recall_phish=recall_score(R[3], p),
                     f1_phish=f1_score(R[3], p), f1_macro=f1_score(R[3], p, average="macro")))
    log.info("[%s] confusion matrix on VAL (rows=actual, cols=pred):\n%s", tag, confusion_matrix(R[3], p, labels=[0, 1]))
    log.info("[%s] classification report on VAL:\n%s", tag,
             classification_report(R[3], p, target_names=CLASS_NAMES, digits=4))
final = pd.DataFrame(rows)
final.to_csv(os.path.join(TAB_DIR, "table21_val_baseline_vs_best.csv"), index=False)
log.info("Baseline vs best on validation:\n%s", final.round(4).to_string(index=False))

cfg = dict(representation=str(best.representation), metric=str(best.metric), weights=str(best.weights),
           K=int(best.K), cv_mean=float(best.cv_mean), cv_std=float(best.cv_std), val_acc=float(best.val_acc))
with open(os.path.join(DATA_DIR, "best_knn_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
log.info("Saved best config: %s", cfg)
tick("ALL DONE (test subset untouched)")