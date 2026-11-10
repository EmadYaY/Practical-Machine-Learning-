# -*- coding: utf-8 -*-
"""
Step 05 - Decision Tree experiments.
Uses TRAIN (5-fold CV) and VALIDATION only; TEST stays locked.
Input : outputs/data/dataset_raw.csv, outputs/data/dataset_dedup.csv
Output: tables 22-27, figures fig21-fig28, outputs/data/best_dt_config.json, *.dot files
"""
import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, log_loss, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from common import (SEED, DATA_DIR, TAB_DIR, TARGET, get_logger, section, savefig,
                    get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step05_decision_tree")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
EPS = 1e-3                                   # probability clipping for BCE
CLASS_NAMES = ["Legitimate (0)", "Phishing (1)"]
CLASS_SHORT = ["Legit", "Phish"]
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
    if mode.startswith("merge"):
        v = -1 if mode == "merge0_neg" else 1
        a, b = Xtr.mask(Xtr == 0, v), Xva.mask(Xva == 0, v)
        keep = a.columns[a.nunique() > 1]
        return a[keep], b[keep]
    return make_representation(Xtr, mode), make_representation(Xva, mode)


def dt(**kw):
    return DecisionTreeClassifier(random_state=SEED, **kw)


def cv_scores(est, X, y):
    return cross_val_score(est, X, y, cv=CV, scoring="accuracy")


def bce(y, p1):
    """Binary cross-entropy with probabilities clipped to [EPS, 1-EPS] (pure leaves give 0/1)."""
    return float(log_loss(y, np.clip(p1, EPS, 1 - EPS), labels=[0, 1]))


def select(df, comp="leaves"):
    """Return (best by CV, simplest within 1 standard error of the best). Ties -> lower complexity."""
    top = df.sort_values(["cv_mean", comp], ascending=[False, True]).iloc[0]
    thr = top.cv_mean - top.cv_std / np.sqrt(5)
    simple = df[df.cv_mean >= thr].sort_values([comp, "cv_mean"], ascending=[True, False]).iloc[0]
    return top, simple


def val_metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), error=1 - accuracy_score(y, p),
                precision_phish=precision_score(y, p, zero_division=0),
                recall_phish=recall_score(y, p, zero_division=0),
                f1_phish=f1_score(y, p, zero_division=0), f1_macro=f1_score(y, p, average="macro"))


# ----------------------------- A. Load ----------------------------
section(log, "A. Load (train/val only)")
data = {"raw": load("dataset_raw.csv"), "dedup": load("dataset_dedup.csv")}
D = data["dedup"]
for n, d in data.items():
    log.info("%-6s | train=%s | val=%s", n, d["Xtr"].shape, d["Xva"].shape)
log.info("CV protocol: %s | BCE clipping eps=%g", CV, EPS)

# ----------------------------- B. Baselines -----------------------
section(log, "B. Baselines: default tree vs class-style tree (entropy, max_depth=4)")
BASELINES = {"default (gini, depth=None)": {},
             "class-style (entropy, depth=4)": dict(criterion="entropy", max_depth=4)}
rows = []
for dname, d in data.items():
    for tag, kw in BASELINES.items():
        m = dt(**kw).fit(d["Xtr"], d["ytr"])
        p_tr, p_va = m.predict(d["Xtr"]), m.predict(d["Xva"])
        cv = cv_scores(dt(**kw), d["Xtr"], d["ytr"])
        rows.append(dict(dataset=dname, model=tag, depth=m.get_depth(), leaves=m.get_n_leaves(),
                         train_acc=accuracy_score(d["ytr"], p_tr), cv_mean=cv.mean(), cv_std=cv.std(),
                         val_acc=accuracy_score(d["yva"], p_va),
                         val_precision_phish=precision_score(d["yva"], p_va, zero_division=0),
                         val_recall_phish=recall_score(d["yva"], p_va, zero_division=0),
                         val_f1_phish=f1_score(d["yva"], p_va, zero_division=0)))
        log.info("[%s | %s] confusion matrix on VAL (rows=actual, cols=pred):\n%s",
                 dname, tag, confusion_matrix(d["yva"], p_va, labels=[0, 1]))
base = pd.DataFrame(rows)
base.to_csv(os.path.join(TAB_DIR, "table22_dt_baselines.csv"), index=False)
log.info("Baseline trees:\n%s", base.round(4).to_string(index=False))

# ----------------------------- C. Depth sweep ---------------------
section(log, "C. Depth sweep (dedup): gini vs entropy, depth 1..25")
rows = []
for crit in ("gini", "entropy"):
    for depth in range(1, 26):
        m = dt(criterion=crit, max_depth=depth).fit(D["Xtr"], D["ytr"])
        cv = cv_scores(dt(criterion=crit, max_depth=depth), D["Xtr"], D["ytr"])
        rows.append(dict(criterion=crit, depth=depth, actual_depth=m.get_depth(), leaves=m.get_n_leaves(),
                         train_acc=accuracy_score(D["ytr"], m.predict(D["Xtr"])),
                         cv_mean=cv.mean(), cv_std=cv.std(),
                         val_acc=accuracy_score(D["yva"], m.predict(D["Xva"])),
                         train_bce=bce(D["ytr"], m.predict_proba(D["Xtr"])[:, 1]),
                         val_bce=bce(D["yva"], m.predict_proba(D["Xva"])[:, 1])))
sweepC = pd.DataFrame(rows)
sweepC.to_csv(os.path.join(TAB_DIR, "table23_dt_depth_sweep.csv"), index=False)
log.info("Depth sweep:\n%s", sweepC.round(4).to_string(index=False))
best_c, simple_c = select(sweepC)
log.info("BEST by CV         : criterion=%s depth=%d leaves=%d CV=%.4f+-%.4f val=%.4f",
         best_c.criterion, best_c.depth, best_c.leaves, best_c.cv_mean, best_c.cv_std, best_c.val_acc)
log.info("SIMPLEST within 1SE: criterion=%s depth=%d leaves=%d CV=%.4f+-%.4f val=%.4f",
         simple_c.criterion, simple_c.depth, simple_c.leaves, simple_c.cv_mean, simple_c.cv_std, simple_c.val_acc)
tick("depth sweep done")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for a, crit in zip(ax, ("gini", "entropy")):
    g = sweepC[sweepC.criterion == crit]
    a.plot(g.depth, g.train_acc, "o-", label="Training")
    a.plot(g.depth, g.cv_mean, "s-", label="5-fold CV (train)")
    a.fill_between(g.depth, g.cv_mean - g.cv_std, g.cv_mean + g.cv_std, alpha=0.2)
    a.plot(g.depth, g.val_acc, "^-", label="Validation")
    a.set_title(f"Decision Tree accuracy vs depth ({crit})")
    a.set_xlabel("max_depth")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
savefig(log, "fig21_dt_accuracy_vs_depth")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for crit, ls in (("gini", "-"), ("entropy", "--")):
    g = sweepC[sweepC.criterion == crit]
    ax[0].plot(g.depth, g.train_bce, ls, marker="o", color="#4c72b0", label=f"Training ({crit})")
    ax[0].plot(g.depth, g.val_bce, ls, marker="^", color="#c44e52", label=f"Validation ({crit})")
    ax[1].plot(g.depth, g.leaves, ls, marker="o", label=crit)
ax[0].set_title("BCE loss vs depth (clipped probabilities)")
ax[0].set_ylabel("Binary cross-entropy")
ax[1].set_title("Number of leaves vs depth")
ax[1].set_ylabel("Leaves")
ax[1].set_yscale("log")
for a in ax:
    a.set_xlabel("max_depth")
    a.legend()
savefig(log, "fig22_dt_bce_and_leaves_vs_depth")

# ----------------------------- D. Grid ----------------------------
section(log, "D. Grid: criterion x max_depth x min_samples_leaf (dedup)")
rows = []
for crit in ("gini", "entropy"):
    for depth in (3, 4, 5, 6, 8, 10, 12, None):
        for msl in (1, 2, 5, 10, 20):
            kw = dict(criterion=crit, max_depth=depth, min_samples_leaf=msl)
            m = dt(**kw).fit(D["Xtr"], D["ytr"])
            cv = cv_scores(dt(**kw), D["Xtr"], D["ytr"])
            rows.append(dict(criterion=crit, depth_label="None" if depth is None else str(depth),
                             min_samples_leaf=msl, leaves=m.get_n_leaves(),
                             cv_mean=cv.mean(), cv_std=cv.std(),
                             val_acc=accuracy_score(D["yva"], m.predict(D["Xva"]))))
gridD = pd.DataFrame(rows)
gridD.to_csv(os.path.join(TAB_DIR, "table24_dt_grid.csv"), index=False)
log.info("Grid (%d configs), top-15 by CV:\n%s", len(gridD),
         gridD.sort_values(["cv_mean", "leaves"], ascending=[False, True]).head(15).round(4).to_string(index=False))
best_d, simple_d = select(gridD)
log.info("BEST by CV         : %s", best_d.to_dict())
log.info("SIMPLEST within 1SE: %s", simple_d.to_dict())
tick("grid done")

depth_order = ["3", "4", "5", "6", "8", "10", "12", "None"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, crit in zip(ax, ("gini", "entropy")):
    pv = (gridD[gridD.criterion == crit]
          .pivot(index="depth_label", columns="min_samples_leaf", values="cv_mean").reindex(depth_order))
    sns.heatmap(pv, annot=True, fmt=".4f", cmap="YlGnBu", ax=a, cbar_kws={"label": "CV accuracy"})
    a.set_title(f"CV accuracy grid ({crit})")
    a.set_xlabel("min_samples_leaf")
    a.set_ylabel("max_depth")
savefig(log, "fig23_dt_grid_heatmap")

# ----------------------------- E. Pruning -------------------------
section(log, "E. Cost-complexity pruning (ccp_alpha) with the best criterion from C")
crit_best = best_c.criterion
path = dt(criterion=crit_best).cost_complexity_pruning_path(D["Xtr"], D["ytr"])
alphas = np.unique(path.ccp_alphas)[:-1]            # last alpha prunes the tree to the root
log.info("criterion=%s | distinct alphas=%d | range [%.3e, %.3e]", crit_best, len(alphas), alphas.min(), alphas.max())
sel = alphas[np.unique(np.linspace(0, len(alphas) - 1, 60).astype(int))]
rows = []
for al in sel:
    m = dt(criterion=crit_best, ccp_alpha=float(al)).fit(D["Xtr"], D["ytr"])
    cv = cv_scores(dt(criterion=crit_best, ccp_alpha=float(al)), D["Xtr"], D["ytr"])
    rows.append(dict(alpha=float(al), leaves=m.get_n_leaves(), depth=m.get_depth(),
                     train_acc=accuracy_score(D["ytr"], m.predict(D["Xtr"])),
                     cv_mean=cv.mean(), cv_std=cv.std(),
                     val_acc=accuracy_score(D["yva"], m.predict(D["Xva"]))))
sweepE = pd.DataFrame(rows)
sweepE.to_csv(os.path.join(TAB_DIR, "table25_dt_pruning.csv"), index=False)
log.info("Pruning sweep:\n%s", sweepE.round(5).to_string(index=False))
best_e, simple_e = select(sweepE)
log.info("BEST by CV         : %s", best_e.to_dict())
log.info("SIMPLEST within 1SE: %s", simple_e.to_dict())
tick("pruning done")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(sweepE.alpha, sweepE.train_acc, "o-", label="Training")
ax[0].plot(sweepE.alpha, sweepE.cv_mean, "s-", label="5-fold CV (train)")
ax[0].fill_between(sweepE.alpha, sweepE.cv_mean - sweepE.cv_std, sweepE.cv_mean + sweepE.cv_std, alpha=0.2)
ax[0].plot(sweepE.alpha, sweepE.val_acc, "^-", label="Validation")
ax[0].axvline(best_e.alpha, ls="--", color="gray", label="best alpha (CV)")
ax[0].set_title(f"Accuracy vs ccp_alpha ({crit_best})")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
ax[1].plot(sweepE.alpha, sweepE.leaves, "o-", color="#55a868")
ax[1].set_title("Number of leaves vs ccp_alpha")
ax[1].set_ylabel("Leaves")
for a in ax:
    a.set_xscale("symlog", linthresh=1e-4)
    a.set_xlabel("ccp_alpha (symlog scale)")
savefig(log, "fig24_dt_pruning")

# ----------------------------- F. Final selection -----------------
section(log, "F. Final configuration (selected by CV only) + level representations")
cands = [
    dict(source="C depth sweep", kw=dict(criterion=best_c.criterion, max_depth=int(best_c.depth)),
         cv_mean=best_c.cv_mean, cv_std=best_c.cv_std, leaves=int(best_c.leaves)),
    dict(source="D grid", kw=dict(criterion=best_d.criterion,
                                  max_depth=None if best_d.depth_label == "None" else int(best_d.depth_label),
                                  min_samples_leaf=int(best_d.min_samples_leaf)),
         cv_mean=best_d.cv_mean, cv_std=best_d.cv_std, leaves=int(best_d.leaves)),
    dict(source="E pruning", kw=dict(criterion=crit_best, ccp_alpha=float(best_e.alpha)),
         cv_mean=best_e.cv_mean, cv_std=best_e.cv_std, leaves=int(best_e.leaves)),
]
log.info("Candidates:\n%s", pd.DataFrame([{**{k: v for k, v in c.items() if k != "kw"}, "kw": str(c["kw"])}
                                           for c in cands]).round(4).to_string(index=False))
log.info("1-SE (simplest) alternatives: C -> criterion=%s depth=%d leaves=%d | D -> %s | E -> alpha=%.5f leaves=%d",
         simple_c.criterion, simple_c.depth, simple_c.leaves,
         f"{simple_d.criterion}, depth={simple_d.depth_label}, msl={simple_d.min_samples_leaf}, leaves={simple_d.leaves}",
         simple_e.alpha, simple_e.leaves)
final = sorted(cands, key=lambda c: (-c["cv_mean"], c["leaves"]))[0]
KW = final["kw"]
log.info("FINAL tree hyper-parameters (best CV): %s  [source: %s]", KW, final["source"])

REPS = {m: rep_pair(D["Xtr"], D["Xva"], m) for m in ("ordinal", "onehot", "merge0_neg", "merge0_pos")}
rows = []
for mode, (a, b) in REPS.items():
    m = dt(**KW).fit(a, D["ytr"])
    cv = cv_scores(dt(**KW), a, D["ytr"])
    rows.append(dict(representation=mode, n_features=a.shape[1], depth=m.get_depth(), leaves=m.get_n_leaves(),
                     train_acc=accuracy_score(D["ytr"], m.predict(a)), cv_mean=cv.mean(), cv_std=cv.std(),
                     val_acc=accuracy_score(D["yva"], m.predict(b))))
repF = pd.DataFrame(rows)
repF.to_csv(os.path.join(TAB_DIR, "table26_dt_representations.csv"), index=False)
log.info("Representation comparison with the final hyper-parameters:\n%s", repF.round(4).to_string(index=False))
REP = repF.sort_values("cv_mean", ascending=False, kind="stable").iloc[0].representation   # ties -> ordinal first
log.info("Selected representation (best CV, ties -> ordinal): %s", REP)
Xtr_f, Xva_f = REPS[REP]
ytr, yva = D["ytr"], D["yva"]

# ----------------------------- G. Final model ---------------------
section(log, "G. Final model")
model = dt(**KW).fit(Xtr_f, ytr)
log.info("Final tree: depth=%d | leaves=%d | nodes=%d | features used=%d",
         model.get_depth(), model.get_n_leaves(), model.tree_.node_count, int((model.feature_importances_ > 0).sum()))
cfg = dict(representation=REP, criterion=KW.get("criterion"), max_depth=KW.get("max_depth"),
           min_samples_leaf=KW.get("min_samples_leaf", 1), ccp_alpha=KW.get("ccp_alpha", 0.0),
           cv_mean=float(final["cv_mean"]), cv_std=float(final["cv_std"]))
with open(os.path.join(DATA_DIR, "best_dt_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
log.info("Saved best config: %s", cfg)

# ----------------------------- H. Visualisations ------------------
section(log, "H. Tree plots, rules, importances, learning curve")
cls_m = dt(criterion="entropy", max_depth=4).fit(D["Xtr"], D["ytr"])
cols_o = list(D["Xtr"].columns)
fig = plt.figure(figsize=(26, 11))
plot_tree(cls_m, feature_names=cols_o, class_names=CLASS_SHORT, filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (entropy, max_depth=4) - class-style tree")
savefig(log, "fig25_tree_class_style")
log.info("Text rules of the class-style tree:\n%s", export_text(cls_m, feature_names=cols_o, show_weights=True))
export_graphviz(cls_m, out_file=os.path.join(DATA_DIR, "tree_class_style.dot"), feature_names=cols_o,
                class_names=CLASS_SHORT, filled=True, rounded=True, special_characters=True)

fig = plt.figure(figsize=(26, 11))
plot_tree(model, max_depth=4, feature_names=list(Xtr_f.columns), class_names=CLASS_SHORT,
          filled=True, rounded=True, fontsize=8)
plt.title(f"Final Decision Tree ({KW}) - displayed to depth 4")
savefig(log, "fig26_tree_final")
export_graphviz(model, out_file=os.path.join(DATA_DIR, "tree_final.dot"), feature_names=list(Xtr_f.columns),
                class_names=CLASS_SHORT, filled=True, rounded=True, special_characters=True)
log.info("Saved .dot files (render with Graphviz if installed): tree_class_style.dot, tree_final.dot")

imp = pd.Series(model.feature_importances_, index=Xtr_f.columns).sort_values(ascending=False)
perm = permutation_importance(model, Xva_f, yva, n_repeats=20, random_state=SEED, scoring="accuracy")
imp_tab = pd.DataFrame({"impurity_importance": imp,
                        "perm_importance_val": pd.Series(perm.importances_mean, index=Xva_f.columns)[imp.index],
                        "perm_std": pd.Series(perm.importances_std, index=Xva_f.columns)[imp.index]})
imp_tab.to_csv(os.path.join(TAB_DIR, "table27_dt_importances.csv"))
log.info("Feature importances (top 15):\n%s", imp_tab.head(15).round(4).to_string())
top = imp_tab.head(15).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].barh(top.index, top.impurity_importance, color="#4c72b0")
ax[0].set_title("Impurity-based importance (train)")
ax[1].barh(top.index, top.perm_importance_val, xerr=top.perm_std, color="#dd8452")
ax[1].set_title("Permutation importance (validation, accuracy drop)")
savefig(log, "fig27_dt_feature_importance")

sizes, trs, tes = learning_curve(dt(**KW), Xtr_f, ytr, train_sizes=np.linspace(0.1, 1.0, 10),
                                 cv=CV, scoring="accuracy", shuffle=True, random_state=SEED)
log.info("Learning curve (final tree):\n%s", pd.DataFrame({"train_size": sizes, "train_acc": trs.mean(1),
                                                            "cv_acc": tes.mean(1), "cv_std": tes.std(1)}).round(4).to_string(index=False))
plt.figure(figsize=(7.5, 5))
plt.plot(sizes, trs.mean(1), "o-", label="Training")
plt.plot(sizes, tes.mean(1), "s-", label="Cross-validation")
plt.fill_between(sizes, tes.mean(1) - tes.std(1), tes.mean(1) + tes.std(1), alpha=0.2)
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.title("Learning curve - final Decision Tree")
plt.legend()
savefig(log, "fig28_dt_learning_curve")

# ----------------------------- I. Validation report ---------------
section(log, "I. Validation report (TEST still locked)")
default_m = dt().fit(D["Xtr"], D["ytr"])
entries = (("default (gini, depth=None)", default_m, D["Xtr"], D["Xva"]),
           ("class-style (entropy, depth=4)", cls_m, D["Xtr"], D["Xva"]),
           (f"final ({REP})", model, Xtr_f, Xva_f))
rows = []
for tag, m, A, B in entries:
    p = m.predict(B)
    rows.append(dict(model=tag, depth=m.get_depth(), leaves=m.get_n_leaves(),
                     train_acc=accuracy_score(D["ytr"], m.predict(A)),
                     val_bce=bce(yva, m.predict_proba(B)[:, 1]), **val_metrics(yva, p)))
    log.info("[%s] confusion matrix on VAL:\n%s", tag, confusion_matrix(yva, p, labels=[0, 1]))
    log.info("[%s] classification report on VAL:\n%s", tag,
             classification_report(yva, p, target_names=CLASS_NAMES, digits=4, zero_division=0))
rep_val = pd.DataFrame(rows)
rep_val.to_csv(os.path.join(TAB_DIR, "table28_dt_val_report.csv"), index=False)
log.info("Decision Tree validation summary:\n%s", rep_val.round(4).to_string(index=False))
tick("ALL DONE (test subset untouched)")