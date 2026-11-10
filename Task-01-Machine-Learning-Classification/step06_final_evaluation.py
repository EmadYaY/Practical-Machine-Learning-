# -*- coding: utf-8 -*-
"""
Step 06 - FINAL evaluation on the TEST set (opened ONCE).
Models are refit on TRAIN+VAL with the hyper-parameters fixed in steps 4-5 and scored once on TEST.
Also: honest training error (KNN leave-one-out), bootstrap CIs, McNemar tests,
      operating points, and supplementary per-epoch BCE/accuracy curves (MLP).
Output: tables 29-34, figures fig29-fig35, outputs/data/test_predictions.csv
"""
import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, log_loss, brier_score_loss, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve, classification_report)

from common import (SEED, DATA_DIR, TAB_DIR, TARGET, get_logger, section, savefig,
                    get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step06_final_evaluation")

EPS = 1e-3          # probability clipping for BCE
N_BOOT = 2000       # bootstrap resamples of the test set
N_EPOCHS = 150      # MLP epochs for the supplementary curves
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
CLASS_NAMES = ["Legitimate (0)", "Phishing (1)"]
T0 = time.time()


def tick(msg):
    log.info("[elapsed %.1fs] %s", time.time() - T0, msg)


def rep_pair(Xa, Xb, mode):
    if mode.startswith("merge"):
        v = -1 if mode == "merge0_neg" else 1
        a, b = Xa.mask(Xa == 0, v), Xb.mask(Xb == 0, v)
        keep = a.columns[a.nunique() > 1]
        return a[keep], b[keep]
    return make_representation(Xa, mode), make_representation(Xb, mode)


def bce(y, p):
    return float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def knn_loo(m, X, y):
    """Exact leave-one-out predictions of a fitted KNN (valid because rows are unique after dedup)."""
    K = m.n_neighbors
    dist, idx = m.kneighbors(X, n_neighbors=K + 1)
    n = len(y)
    self_mask = idx == np.arange(n)[:, None]
    if not (self_mask.sum(axis=1) == 1).all():
        raise RuntimeError("Self-neighbour is not unique; LOO shortcut invalid.")
    nbr_idx = idx[~self_mask].reshape(n, K)
    nbr_dist = dist[~self_mask].reshape(n, K)
    w = 1.0 / np.maximum(nbr_dist, 1e-12) if m.weights == "distance" else np.ones_like(nbr_dist)
    p1 = (w * y[nbr_idx]).sum(axis=1) / w.sum(axis=1)
    return (p1 > 0.5).astype(int), p1


# ----------------------------- A. Load ----------------------------
section(log, "A. Load data and fixed hyper-parameters")
d = pd.read_csv(os.path.join(DATA_DIR, "dataset_dedup.csv"))
FEATS = get_features(d)
P = {s: d[d["subset"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
fit_df = pd.concat([P["train"], P["val"]], ignore_index=True)
Xfit, yfit = fit_df[FEATS], fit_df[TARGET].values
Xte, yte = P["test"][FEATS], P["test"][TARGET].values
log.info("Fit set (train+val): %s | phishing %.2f%%", Xfit.shape, yfit.mean() * 100)
log.info("Test set           : %s | phishing %.2f%%", Xte.shape, yte.mean() * 100)

with open(os.path.join(DATA_DIR, "best_knn_config.json"), encoding="utf-8") as f:
    knn_cfg = json.load(f)
with open(os.path.join(DATA_DIR, "best_dt_config.json"), encoding="utf-8") as f:
    dt_cfg = json.load(f)
log.info("KNN config: %s", knn_cfg)
log.info("DT  config: %s", dt_cfg)

pr = pd.read_csv(os.path.join(TAB_DIR, "table25_dt_pruning.csv"))
top = pr.sort_values(["cv_mean", "leaves"], ascending=[False, True]).iloc[0]
thr = top.cv_mean - top.cv_std / np.sqrt(5)
simple = pr[pr.cv_mean >= thr].sort_values(["leaves", "cv_mean"], ascending=[True, False]).iloc[0]
ALPHA_SIMPLE = float(simple.alpha)
log.info("1-SE simple tree: ccp_alpha=%.6f | leaves=%d | CV=%.4f (selected from table25, NOT from test)",
         ALPHA_SIMPLE, simple.leaves, simple.cv_mean)

SPECS = [
    ("KNN baseline", "ordinal", lambda: KNeighborsClassifier(n_neighbors=1)),
    ("KNN best", knn_cfg["representation"],
     lambda: KNeighborsClassifier(n_neighbors=knn_cfg["K"], metric=knn_cfg["metric"], weights=knn_cfg["weights"])),
    ("DT default", "ordinal", lambda: DecisionTreeClassifier(random_state=SEED)),
    ("DT class-style", "ordinal",
     lambda: DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=SEED)),
    ("DT final (CV)", dt_cfg["representation"],
     lambda: DecisionTreeClassifier(criterion=dt_cfg["criterion"], max_depth=dt_cfg["max_depth"],
                                    min_samples_leaf=dt_cfg["min_samples_leaf"],
                                    ccp_alpha=dt_cfg["ccp_alpha"], random_state=SEED)),
    ("DT simple (1-SE)", "ordinal",
     lambda: DecisionTreeClassifier(criterion="entropy", ccp_alpha=ALPHA_SIMPLE, random_state=SEED)),
]


# ----------------------------- B. MLP curves ----------------------
section(log, "B. Supplementary MLP: per-epoch training/validation BCE and accuracy (train -> val)")


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4,
                         batch_size=64, learning_rate_init=1e-3, random_state=SEED)


log.info("MLP parameters: %s", make_mlp().get_params())
A_tr, A_va = rep_pair(P["train"][FEATS], P["val"][FEATS], "onehot")
A_tr, A_va = A_tr.values.astype(float), A_va.values.astype(float)
ytr_, yva_ = P["train"][TARGET].values, P["val"][TARGET].values
mlp = make_mlp()
hist = []
for ep in range(1, N_EPOCHS + 1):
    mlp.partial_fit(A_tr, ytr_, classes=np.array([0, 1]))      # one epoch per call (shuffled mini-batches)
    ptr, pva = mlp.predict_proba(A_tr)[:, 1], mlp.predict_proba(A_va)[:, 1]
    hist.append(dict(epoch=ep, train_bce=bce(ytr_, ptr), val_bce=bce(yva_, pva),
                     train_acc=accuracy_score(ytr_, ptr >= 0.5), val_acc=accuracy_score(yva_, pva >= 0.5)))
H = pd.DataFrame(hist)
H.to_csv(os.path.join(TAB_DIR, "table34_mlp_history.csv"), index=False)
E_STAR = int(H.loc[H.val_bce.idxmin(), "epoch"])
log.info("MLP history (every 10th epoch):\n%s", H[H.epoch % 10 == 0].round(4).to_string(index=False))
log.info("Best epoch by validation BCE = %d (val BCE=%.4f, val acc=%.4f); best val acc epoch = %d (%.4f)",
         E_STAR, H.val_bce.min(), H.loc[H.val_bce.idxmin(), "val_acc"],
         int(H.loc[H.val_acc.idxmax(), "epoch"]), H.val_acc.max())

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(H.epoch, H.train_bce, label="Training BCE")
ax.plot(H.epoch, H.val_bce, label="Validation BCE")
ax.axvline(E_STAR, ls="--", color="gray", label=f"min val BCE (epoch {E_STAR})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy")
ax.set_title("Training and validation BCE loss (MLP, supplementary)")
ax.legend()
savefig(log, "fig34_mlp_bce_curves")

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(H.epoch, H.train_acc, label="Training accuracy")
ax.plot(H.epoch, H.val_acc, label="Validation accuracy")
ax.axvline(E_STAR, ls="--", color="gray", label=f"epoch {E_STAR}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training and validation accuracy (MLP, supplementary)")
ax.legend()
savefig(log, "fig35_mlp_accuracy_curves")
tick("MLP curves done")

# ----------------------------- C. Test evaluation -----------------
section(log, "C. TEST SET OPENED (first and only time): refit on train+val, score once")
R = {}
for name, rep, make in SPECS:
    A, B = rep_pair(Xfit, Xte, rep)
    m = make().fit(A, yfit)
    score, pred = m.predict_proba(B)[:, 1], m.predict(B)
    resub = 1 - accuracy_score(yfit, m.predict(A))
    if isinstance(m, KNeighborsClassifier):
        honest, kind = 1 - accuracy_score(yfit, knn_loo(m, A, yfit)[0]), "leave-one-out"
    else:
        honest = 1 - cross_val_score(make(), A, yfit, cv=CV5, scoring="accuracy").mean()
        kind = "5-fold CV"
    extra = f" | depth={m.get_depth()} leaves={m.get_n_leaves()}" if hasattr(m, "get_depth") else ""
    log.info("[%s] rep=%s n_features=%d%s | resubstitution error=%.4f | %s error=%.4f",
             name, rep, A.shape[1], extra, resub, kind, honest)
    R[name] = dict(pred=pred, score=score, resub=resub, honest=honest, kind=kind)

A_fit, A_te = rep_pair(Xfit, Xte, "onehot")
A_fit, A_te = A_fit.values.astype(float), A_te.values.astype(float)
mlp_f = make_mlp()
for _ in range(E_STAR):
    mlp_f.partial_fit(A_fit, yfit, classes=np.array([0, 1]))
s_mlp = mlp_f.predict_proba(A_te)[:, 1]
R["MLP (suppl.)"] = dict(pred=(s_mlp >= 0.5).astype(int), score=s_mlp,
                         resub=1 - accuracy_score(yfit, mlp_f.predict(A_fit)),
                         honest=1 - float(H.loc[H.epoch == E_STAR, "val_acc"].iloc[0]), kind="validation")
log.info("[MLP (suppl.)] trained %d epochs on train+val (epoch chosen on validation BCE)", E_STAR)
names = list(R.keys())
COL = dict(zip(names, sns.color_palette("tab10", len(names))))

rows = []
for nm in names:
    y_p, s = R[nm]["pred"], R[nm]["score"]
    rows.append(dict(model=nm, accuracy=accuracy_score(yte, y_p), error=1 - accuracy_score(yte, y_p),
                     precision_phish=precision_score(yte, y_p, zero_division=0),
                     recall_phish=recall_score(yte, y_p, zero_division=0),
                     f1_phish=f1_score(yte, y_p, zero_division=0), f1_macro=f1_score(yte, y_p, average="macro"),
                     mcc=matthews_corrcoef(yte, y_p), roc_auc=roc_auc_score(yte, s),
                     pr_auc=average_precision_score(yte, s), bce=bce(yte, s), brier=brier_score_loss(yte, s)))
    log.info("[%s] confusion matrix on TEST (rows=actual, cols=pred):\n%s", nm, confusion_matrix(yte, y_p, labels=[0, 1]))
    log.info("[%s] classification report on TEST:\n%s", nm,
             classification_report(yte, y_p, target_names=CLASS_NAMES, digits=4, zero_division=0))
M = pd.DataFrame(rows).set_index("model")
M.to_csv(os.path.join(TAB_DIR, "table29_test_metrics.csv"))
log.info("TEST metrics (phishing = positive class; BCE uses clipped probabilities):\n%s", M.round(4).to_string())

pd.DataFrame({"y_true": yte, **{f"{nm}__pred": R[nm]["pred"] for nm in names},
              **{f"{nm}__score": R[nm]["score"] for nm in names}}).to_csv(
    os.path.join(DATA_DIR, "test_predictions.csv"), index=False)
tick("test evaluation done")

# ----------------------------- D. Bootstrap CIs -------------------
section(log, f"D. Bootstrap 95% confidence intervals ({N_BOOT} resamples of the test set)")
rng = np.random.default_rng(SEED)
BOOT = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
yb = yte[BOOT]
ci_rows, CI = [], {k: {} for k in ("accuracy", "f1_phish", "roc_auc")}
for nm in names:
    pb, sb = R[nm]["pred"][BOOT], R[nm]["score"][BOOT]
    acc = (yb == pb).mean(axis=1)
    tp, fp, fn = ((yb == 1) & (pb == 1)).sum(1), ((yb == 0) & (pb == 1)).sum(1), ((yb == 1) & (pb == 0)).sum(1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    auc = np.array([roc_auc_score(yb[i], sb[i]) for i in range(N_BOOT)])
    for key, arr in (("accuracy", acc), ("f1_phish", f1), ("roc_auc", auc)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        CI[key][nm] = (M.loc[nm, key], lo, hi)
        ci_rows.append(dict(model=nm, metric=key, point=M.loc[nm, key], ci_low=lo, ci_high=hi))
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(os.path.join(TAB_DIR, "table30_bootstrap_ci.csv"), index=False)
log.info("Bootstrap CIs:\n%s", ci_df.round(4).to_string(index=False))
tick("bootstrap done")

# ----------------------------- E. McNemar -------------------------
section(log, "E. McNemar exact tests on paired test predictions")
PAIRS = [("KNN best", "DT final (CV)"), ("KNN best", "DT simple (1-SE)"), ("KNN best", "KNN baseline"),
         ("DT final (CV)", "DT default"), ("DT final (CV)", "DT simple (1-SE)"), ("KNN best", "MLP (suppl.)")]
rows = []
for a, b in PAIRS:
    ca, cb = R[a]["pred"] == yte, R[b]["pred"] == yte
    n_a_only, n_b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n_a_only, n_a_only + n_b_only, 0.5).pvalue if (n_a_only + n_b_only) > 0 else 1.0
    rows.append(dict(model_A=a, model_B=b, A_correct_B_wrong=n_a_only, A_wrong_B_correct=n_b_only,
                     p_value=p, significant_at_0_05=bool(p < 0.05)))
mc = pd.DataFrame(rows)
mc.to_csv(os.path.join(TAB_DIR, "table31_mcnemar.csv"), index=False)
log.info("McNemar results:\n%s", mc.round(4).to_string(index=False))

# ----------------------------- F. Generalisation gap --------------
section(log, "F. Training error vs honest error vs test error")
gap = pd.DataFrame([dict(model=nm, resub_error=R[nm]["resub"], honest_error=R[nm]["honest"],
                         honest_kind=R[nm]["kind"], test_error=M.loc[nm, "error"]) for nm in names])
gap["test_minus_resub"] = gap.test_error - gap.resub_error
gap.to_csv(os.path.join(TAB_DIR, "table32_generalisation_gap.csv"), index=False)
log.info("Generalisation gap (KNN honest = leave-one-out, DT honest = 5-fold CV, MLP = validation):\n%s",
         gap.round(4).to_string(index=False))

gp = gap.set_index("model")[["resub_error", "honest_error", "test_error"]]
gp.columns = ["Training (resubstitution) error", "Honest error (LOO / CV / val)", "Test error"]
gp.plot(kind="bar", figsize=(12, 5.2), color=["#4c72b0", "#dd8452", "#c44e52"])
plt.ylabel("Error rate")
plt.title("Training error vs honest error vs test error")
plt.xticks(rotation=25, ha="right")
savefig(log, "fig33_generalisation_gap")

# ----------------------------- G. Operating points ----------------
section(log, "G. Descriptive operating points: best precision at a given phishing recall (TEST)")
rows = []
for nm in names:
    p_, r_, _ = precision_recall_curve(yte, R[nm]["score"])
    row = dict(model=nm)
    for tgt in (0.90, 0.95, 0.98, 0.99):
        ok = r_ >= tgt
        row[f"precision@recall>={tgt}"] = float(p_[ok].max()) if ok.any() else np.nan
    rows.append(row)
op = pd.DataFrame(rows)
op.to_csv(os.path.join(TAB_DIR, "table33_operating_points.csv"), index=False)
log.info("Operating points (descriptive only; choosing a deployment threshold needs validation data):\n%s",
         op.round(4).to_string(index=False))

# ----------------------------- H. Figures -------------------------
section(log, "H. Figures")
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for ax, nm in zip(axes.ravel(), names):
    cm = confusion_matrix(yte, R[nm]["pred"], labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"])
    ax.set_title(f"{nm}\nacc={M.loc[nm, 'accuracy']:.4f}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
for ax in axes.ravel()[len(names):]:
    ax.axis("off")
savefig(log, "fig29_test_confusion_matrices")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for nm in names:
    fpr, tpr, _ = roc_curve(yte, R[nm]["score"])
    for a in ax:
        a.plot(fpr, tpr, color=COL[nm], lw=1.6, label=f"{nm} (AUC={M.loc[nm, 'roc_auc']:.4f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax[0].set_title("Receiver operating characteristic (ROC) curve")
ax[1].set_xlim(0, 0.15)
ax[1].set_ylim(0.85, 1.005)
ax[1].set_title("ROC - zoom (FPR <= 0.15)")
for a in ax:
    a.set_xlabel("False positive rate")
    a.set_ylabel("True positive rate (phishing recall)")
ax[0].legend(loc="lower right", fontsize=8)
savefig(log, "fig30_test_roc")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for nm in names:
    p_, r_, _ = precision_recall_curve(yte, R[nm]["score"])
    for a in ax:
        a.plot(r_, p_, color=COL[nm], lw=1.6, label=f"{nm} (AP={M.loc[nm, 'pr_auc']:.4f})")
ax[0].axhline(yte.mean(), color="k", ls="--", lw=1, label=f"Prevalence ({yte.mean():.2f})")
ax[0].set_title("Precision-recall curve")
ax[1].set_xlim(0.85, 1.005)
ax[1].set_ylim(0.85, 1.005)
ax[1].set_title("Precision-recall - zoom")
for a in ax:
    a.set_xlabel("Recall (phishing)")
    a.set_ylabel("Precision (phishing)")
ax[0].legend(loc="lower left", fontsize=8)
savefig(log, "fig31_test_precision_recall")

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, key, title in zip(axes, ("accuracy", "f1_phish", "roc_auc"), ("Accuracy", "F1 (phishing)", "ROC AUC")):
    pts = np.array([CI[key][nm][0] for nm in names])
    lo = np.array([CI[key][nm][1] for nm in names])
    hi = np.array([CI[key][nm][2] for nm in names])
    yy = np.arange(len(names))
    ax.errorbar(pts, yy, xerr=[pts - lo, hi - pts], fmt="o", capsize=4, color="#4c72b0")
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title(f"{title} with 95% bootstrap CI")
savefig(log, "fig32_test_metrics_with_ci")

log.info("DONE. Test set was evaluated exactly once.")
tick("ALL DONE")