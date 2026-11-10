# -*- coding: utf-8 -*-
"""
Step 03 - Evaluation protocols + baseline KNN (k=1, p=2) on RAW vs DEDUP data.
Input : outputs/data/dataset_raw.csv, outputs/data/dataset_dedup.csv
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

from common import SEED, DATA_DIR, TAB_DIR, TARGET, get_logger, section, savefig, get_features

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step03_knn_baseline")
CLASS_NAMES = ["Legitimate (0)", "Phishing (1)"]


def make_knn(k):
    return KNeighborsClassifier(n_neighbors=k, p=2, metric="minkowski",
                                weights="uniform", algorithm="auto")


# ---------------------------- A. Load --------------------------------
section(log, "A. Load datasets (train/val only - test stays locked)")
data = {}
for name, fn in (("raw", "dataset_raw.csv"), ("dedup", "dataset_dedup.csv")):
    d = pd.read_csv(os.path.join(DATA_DIR, fn))
    feats = get_features(d)
    tr, va = d[d["subset"] == "train"], d[d["subset"] == "val"]
    data[name] = dict(Xtr=tr[feats].values, ytr=tr[TARGET].values,
                      Xva=va[feats].values, yva=va[TARGET].values)
    log.info("%-6s | n_features=%d | train=%d (phishing %.2f%%) | val=%d (phishing %.2f%%)",
             name, len(feats), len(tr), tr[TARGET].mean() * 100, len(va), va[TARGET].mean() * 100)
log.info("KNN parameters: %s", make_knn(1).get_params())


# ---------------------------- B. Baseline ----------------------------
def metric_row(name, subset, y, p):
    return dict(dataset=name, subset=subset,
                accuracy=accuracy_score(y, p), error=1 - accuracy_score(y, p),
                precision_phish=precision_score(y, p, zero_division=0),
                recall_phish=recall_score(y, p, zero_division=0),
                f1_phish=f1_score(y, p, zero_division=0),
                f1_macro=f1_score(y, p, average="macro", zero_division=0),
                f1_micro=f1_score(y, p, average="micro", zero_division=0))


section(log, "B. Baseline KNN (k=1, p=2, Euclidean, uniform weights)")
base_rows, cms = [], {}
for name, D in data.items():
    m = make_knn(1)
    t0 = time.perf_counter(); m.fit(D["Xtr"], D["ytr"]); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); p_tr = m.predict(D["Xtr"]); t_ptr = time.perf_counter() - t0
    t0 = time.perf_counter(); p_va = m.predict(D["Xva"]); t_pva = time.perf_counter() - t0
    log.info("[%s] fit=%.4fs | predict(train)=%.4fs | predict(val)=%.4fs", name, t_fit, t_ptr, t_pva)
    base_rows += [metric_row(name, "train", D["ytr"], p_tr), metric_row(name, "val", D["yva"], p_va)]
    cm = confusion_matrix(D["yva"], p_va, labels=[0, 1])
    cms[name] = cm
    log.info("[%s] Confusion matrix on VAL (rows=actual, cols=predicted, labels=[0,1]):\n%s", name, cm)
    log.info("[%s] Classification report on VAL:\n%s", name,
             classification_report(D["yva"], p_va, target_names=CLASS_NAMES, digits=4, zero_division=0))

base = pd.DataFrame(base_rows)
base.to_csv(os.path.join(TAB_DIR, "table11_baseline_metrics.csv"), index=False)
log.info("Baseline metrics (phish = positive class; micro-F1 equals accuracy for single-label data):\n%s",
         base.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, cm) in zip(axes, cms.items()):
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion matrix - {name} (validation, k=1)")
savefig(log, "fig09_baseline_confusion_matrices")

# ---------------------------- C. Protocols ---------------------------
section(log, "C. Evaluation protocols (accuracy): full training, hold-out, stratified K-fold")
proto_rows = []
for name, D in data.items():
    for k in (1, 5):
        m = make_knn(k).fit(D["Xtr"], D["ytr"])
        acc_full = accuracy_score(D["ytr"], m.predict(D["Xtr"]))
        acc_ho = accuracy_score(D["yva"], m.predict(D["Xva"]))
        proto_rows.append(dict(dataset=name, K=k, protocol="Full training (train->train)",
                               acc_mean=acc_full, acc_std=0.0))
        proto_rows.append(dict(dataset=name, K=k, protocol="Hold-out (train->val)",
                               acc_mean=acc_ho, acc_std=0.0))
        for nf in (5, 10):
            skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=SEED)
            cv = cross_validate(make_knn(k), D["Xtr"], D["ytr"], cv=skf, scoring="accuracy",
                                return_train_score=True, n_jobs=1)
            proto_rows.append(dict(dataset=name, K=k, protocol=f"Stratified {nf}-fold CV (on train)",
                                   acc_mean=cv["test_score"].mean(), acc_std=cv["test_score"].std(),
                                   train_acc_in_cv=cv["train_score"].mean()))
            log.info("[%s | k=%d | %d-fold] fold accuracies: %s", name, k, nf, np.round(cv["test_score"], 4))

proto = pd.DataFrame(proto_rows)
proto["error_mean"] = 1 - proto["acc_mean"]
proto.to_csv(os.path.join(TAB_DIR, "table12_evaluation_protocols.csv"), index=False)
log.info("Protocol comparison:\n%s", proto.round(4).to_string(index=False))

protos = proto["protocol"].unique().tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, k in zip(axes, (1, 5)):
    sub = proto[proto["K"] == k]
    for i, (name, col) in enumerate((("raw", "#c44e52"), ("dedup", "#4c72b0"))):
        s = sub[sub["dataset"] == name].set_index("protocol").loc[protos]
        xs = np.arange(len(protos)) + (i - 0.5) * 0.25
        ax.errorbar(xs, s["acc_mean"], yerr=s["acc_std"], fmt="o", capsize=4, color=col, label=name)
        for x, v in zip(xs, s["acc_mean"]):
            ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=col)
    ax.set_xticks(np.arange(len(protos)))
    ax.set_xticklabels([p.replace(" (", "\n(") for p in protos], fontsize=8)
    ax.set_title(f"KNN (k={k}, p=2)")
axes[0].set_ylabel("Accuracy")
axes[0].legend(title="Dataset")
savefig(log, "fig10_evaluation_protocols")

log.info("TEST subset was NOT touched in this step.")
log.info("DONE.")
