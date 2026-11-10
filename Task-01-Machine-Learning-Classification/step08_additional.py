# -*- coding: utf-8 -*-
"""
Step 08 - Additional experiments (added after reviewing the previous students' reports)
  A. Decision Tree at several depths (3, 4, 10, unlimited): CV, confusion matrix, readable tree
  B. Fold-by-fold cross-validation table for the tree models
  C. SVM kernels (linear / poly / rbf / sigmoid) on the same data (train CV + validation only)
  D. Supplementary regression: DecisionTreeRegressor and KNeighborsRegressor (sklearn diabetes data)
  E. KNN tie-sensitivity check (neighbour-search algorithm changes K=1 accuracy)
  F. Table of sample rows of the raw data
TEST set is NOT used in this step.
Needs: outputs/data/dataset_dedup.csv (step 2), common.py, phishing.csv (for the sample table).
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score, pairwise_distances)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split

from common import (SEED, DATA_DIR, OUT_DIR, TAB_DIR, TARGET, get_logger, section, savefig,
                    get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step08_additional")
TEX_DIR = os.path.join(OUT_DIR, "tex")
os.makedirs(TEX_DIR, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
T0 = time.time()
_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(_MAP.get(c, c) for c in str(s))


def write_tex(name, df, dec=4, bold=None, rename=None):
    """LaTeX tabular body (no caption). bold = {column: 'max'|'min'}; best value in bold."""
    df = df.reset_index(drop=True)
    rename, bold = rename or {}, bold or {}
    cells = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            txt = s.map(lambda v: "--" if pd.isna(v) else f"{v:.{dec}f}")
        elif pd.api.types.is_integer_dtype(s):
            txt = s.map(lambda v: str(int(v)))
        else:
            txt = s.map(esc)
        if c in bold:
            num = pd.to_numeric(s, errors="coerce")
            best = num.max() if bold[c] == "max" else num.min()
            txt = pd.Series([f"\\textbf{{{t}}}" if (pd.notna(v) and np.isclose(v, best)) else t
                             for t, v in zip(txt, num)])
        cells[c] = txt.reset_index(drop=True)
    spec = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(esc(rename.get(c, c)) for c in df.columns) + r" \\", r"\midrule"]
    for i in range(len(df)):
        lines.append(" & ".join(cells[c].iloc[i] for c in df.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TEX_DIR, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("TEX written: %s.tex (%d rows)", name, len(df))


# ----------------------------- load -----------------------------
section(log, "0. Load dedup data (train/val only)")
d = pd.read_csv(os.path.join(DATA_DIR, "dataset_dedup.csv"))
FEATS = get_features(d)
tr, va = d[d["subset"] == "train"], d[d["subset"] == "val"]
Xtr, ytr = tr[FEATS].reset_index(drop=True), tr[TARGET].values
Xva, yva = va[FEATS].reset_index(drop=True), va[TARGET].values
Otr, Ova = make_representation(Xtr, "onehot"), make_representation(Xva, "onehot")
log.info("train=%s val=%s | one-hot train=%s", Xtr.shape, Xva.shape, Otr.shape)

# ----------------------------- A. depth comparison -------------------------
section(log, "A. Decision Tree at several depths (criterion=entropy, ordinal features)")
rows, cms = [], {}
for label, depth in (("3", 3), ("4", 4), ("10", 10), ("Unlimited", None)):
    kw = dict(criterion="entropy", max_depth=depth, random_state=SEED)
    m = DecisionTreeClassifier(**kw).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(**kw), Xtr, ytr, cv=CV, scoring="accuracy")
    p = m.predict(Xva)
    cm = confusion_matrix(yva, p, labels=[0, 1])
    cms[label] = cm
    rows.append(dict(max_depth=label, actual_depth=m.get_depth(), leaves=m.get_n_leaves(),
                     train_acc=accuracy_score(ytr, m.predict(Xtr)), cv_mean=cv.mean(), cv_std=cv.std(),
                     val_acc=accuracy_score(yva, p), val_error=1 - accuracy_score(yva, p),
                     FP=int(cm[0, 1]), FN=int(cm[1, 0])))
    log.info("[depth=%s] confusion matrix on VAL (rows=actual, cols=pred):\n%s", label, cm)
dep = pd.DataFrame(rows)
log.info("Depth comparison:\n%s", dep.round(4).to_string(index=False))
best = dep.sort_values(["cv_mean", "leaves"], ascending=[False, True]).iloc[0]
log.info("Lowest CV error: max_depth=%s (CV=%.4f)", best.max_depth, best.cv_mean)
write_tex("dt_depth_compare", dep, bold={"cv_mean": "max", "val_acc": "max"},
          rename={"actual_depth": "depth", "train_acc": "Train acc.", "cv_mean": "CV acc.", "cv_std": "CV std",
                  "val_acc": "Val acc.", "val_error": "Val error"})

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (lab, cm) in zip(axes, cms.items()):
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"])
    r = dep[dep.max_depth == lab].iloc[0]
    ax.set_title(f"max_depth={lab}\nval acc={r.val_acc:.4f}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
savefig(log, "fig38_dt_depth_confusion")

m3 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED).fit(Xtr, ytr)
fig = plt.figure(figsize=(22, 9.5))
plot_tree(m3, feature_names=list(Xtr.columns), class_names=["Legit", "Phish"], filled=True, rounded=True,
          fontsize=11, impurity=True, proportion=False)
fig.legend(handles=[Patch(facecolor="#e58139", label="orange = majority class Legit (class 0)"),
                    Patch(facecolor="#399de5", label="blue = majority class Phish (class 1)"),
                    Patch(facecolor="white", edgecolor="gray", label="deeper colour = purer node; white = about 50/50")],
           loc="lower center", ncol=3, fontsize=12, frameon=True)
plt.title("Decision Tree, entropy, max_depth=3 (training data)")
savefig(log, "fig39_tree_depth3")

# ----------------------------- C. SVM kernels (before B because B uses it) ---------------
section(log, "C. SVM kernels (C=1, gamma='scale'), train CV + validation only")
rows = []
for rep_name, (A, B) in {"ordinal": (Xtr, Xva), "onehot": (Otr, Ova)}.items():
    for kern in ("linear", "poly", "rbf", "sigmoid"):
        t0 = time.perf_counter()
        m = SVC(kernel=kern, C=1.0, gamma="scale", random_state=SEED).fit(A, ytr)
        fit_s = time.perf_counter() - t0
        cv = cross_val_score(SVC(kernel=kern, C=1.0, gamma="scale", random_state=SEED), A, ytr, cv=CV)
        p = m.predict(B)
        rows.append(dict(representation=rep_name, kernel=kern, cv_mean=cv.mean(), cv_std=cv.std(),
                         val_acc=accuracy_score(yva, p), val_precision=precision_score(yva, p),
                         val_recall=recall_score(yva, p), val_f1=f1_score(yva, p), fit_seconds=fit_s))
svm = pd.DataFrame(rows)
log.info("SVM kernel comparison:\n%s", svm.round(4).to_string(index=False))
bsv = svm.sort_values(["cv_mean", "cv_std"], ascending=[False, True]).iloc[0]
log.info("Best SVM by CV: %s / %s (CV=%.4f)", bsv.representation, bsv.kernel, bsv.cv_mean)
write_tex("svm_kernels", svm, bold={"cv_mean": "max"},
          rename={"cv_mean": "CV acc.", "cv_std": "CV std", "val_acc": "Val acc.", "val_precision": "Val prec.",
                  "val_recall": "Val rec.", "val_f1": "Val F1", "fit_seconds": "Fit (s)"})
fig, ax = plt.subplots(figsize=(9, 4.8))
lab = [f"{r.representation}\n{r.kernel}" for r in svm.itertuples()]
ax.bar(lab, svm.cv_mean, yerr=svm.cv_std, capsize=3, color="#4c72b0", alpha=0.85, label="5-fold CV (train)")
ax.scatter(lab, svm.val_acc, color="#c44e52", zorder=3, label="Validation")
ax.set_ylim(max(0.5, svm.cv_mean.min() - 0.03), 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("SVM kernels: CV accuracy and validation accuracy")
plt.xticks(fontsize=8)
ax.legend()
savefig(log, "fig41_svm_kernels")

# ----------------------------- B. fold-by-fold CV ---------------------------
section(log, "B. Fold-by-fold 5-fold CV accuracy (train only)")
specs = {
    "DT depth 3 (entropy, ordinal)": (DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED), Xtr),
    "DT depth 4 (entropy, ordinal)": (DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=SEED), Xtr),
    "DT default (gini, unlimited)": (DecisionTreeClassifier(random_state=SEED), Xtr),
    "DT final (entropy, pruned, one-hot)": (DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.0006781737064963241,
                                                                    random_state=SEED), Otr),
    "SVM (best kernel by CV)": (SVC(kernel=bsv.kernel, C=1.0, gamma="scale", random_state=SEED),
                                Otr if bsv.representation == "onehot" else Xtr),
}
frows, fold_cols = [], [f"fold{i}" for i in range(1, 6)]
for name, (est, A) in specs.items():
    s = cross_val_score(est, A, ytr, cv=CV, scoring="accuracy")
    frows.append({"model": name, **dict(zip(fold_cols, s)), "mean": s.mean(), "std": s.std()})
fold = pd.DataFrame(frows)
log.info("Fold accuracies:\n%s", fold.round(4).to_string(index=False))
write_tex("cv_folds", fold, bold={"mean": "max"})
fig, ax = plt.subplots(figsize=(9.5, 4.6))
data_ = [fold.loc[i, fold_cols].astype(float).values for i in range(len(fold))]
ax.boxplot(data_, orientation="horizontal", tick_labels=[n.replace(" (", "\n(") for n in fold.model], showmeans=True)
for i, v in enumerate(data_, 1):
    ax.scatter(v, np.full(len(v), i), color="#c44e52", zorder=3, s=14)
ax.set_xlabel("Accuracy per fold (5-fold stratified CV on train)")
ax.set_title("Fold-by-fold cross-validation accuracy")
savefig(log, "fig40_cv_folds")

# ----------------------------- D. regression (supplementary) ----------------
section(log, "D. Supplementary regression on sklearn 'diabetes' (442 samples, 10 features, target = disease progression)")
Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=SEED)
log.info("train=%s test=%s | target mean=%.2f std=%.2f", Xd_tr.shape, Xd_te.shape, yd.mean(), yd.std())
KCV = KFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_rmse(est):
    s = cross_val_score(est, Xd_tr, yd_tr, cv=KCV, scoring="neg_root_mean_squared_error")
    return -s.mean(), s.std()


drows = []
for dpt in list(range(1, 11)) + [None]:
    mean, std = cv_rmse(DecisionTreeRegressor(max_depth=dpt, random_state=SEED))
    drows.append(dict(model="DecisionTree", param=("Unlimited" if dpt is None else str(dpt)), cv_rmse=mean, cv_std=std))
for k in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50):
    mean, std = cv_rmse(KNeighborsRegressor(n_neighbors=k))
    drows.append(dict(model="KNN", param=str(k), cv_rmse=mean, cv_std=std))
reg = pd.DataFrame(drows)
log.info("Regression CV (RMSE, lower is better):\n%s", reg.round(3).to_string(index=False))
bt = reg[reg.model == "DecisionTree"].sort_values("cv_rmse").iloc[0]
bk = reg[reg.model == "KNN"].sort_values("cv_rmse").iloc[0]
log.info("Best DT depth by CV=%s (RMSE=%.2f) | best KNN K=%s (RMSE=%.2f)", bt.param, bt.cv_rmse, bk.param, bk.cv_rmse)

final = {"Mean predictor (baseline)": DummyRegressor(strategy="mean"),
         "DecisionTree (unlimited)": DecisionTreeRegressor(random_state=SEED),
         f"DecisionTree (depth {bt.param})": DecisionTreeRegressor(
             max_depth=None if bt.param == "Unlimited" else int(bt.param), random_state=SEED),
         f"KNN (K={bk.param})": KNeighborsRegressor(n_neighbors=int(bk.param))}
frow, preds = [], {}
for name, m in final.items():
    m.fit(Xd_tr, yd_tr)
    ptr, pte = m.predict(Xd_tr), m.predict(Xd_te)
    preds[name] = pte
    frow.append(dict(model=name, train_rmse=mean_squared_error(yd_tr, ptr) ** 0.5,
                     test_rmse=mean_squared_error(yd_te, pte) ** 0.5, test_mae=mean_absolute_error(yd_te, pte),
                     train_r2=r2_score(yd_tr, ptr), test_r2=r2_score(yd_te, pte)))
fin = pd.DataFrame(frow)
log.info("Regression, test set (used once):\n%s", fin.round(3).to_string(index=False))
write_tex("regression_results", fin, dec=3, bold={"test_rmse": "min", "test_r2": "max"},
          rename={"train_rmse": "Train RMSE", "test_rmse": "Test RMSE", "test_mae": "Test MAE",
                  "train_r2": "Train R2", "test_r2": "Test R2"})
cv_tab = reg.copy()
write_tex("regression_cv", cv_tab, dec=3, rename={"cv_rmse": "CV RMSE", "cv_std": "CV std"})

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
g = reg[reg.model == "DecisionTree"]
ax[0].errorbar(g.param, g.cv_rmse, yerr=g.cv_std, marker="o", capsize=3)
ax[0].set_xlabel("max_depth")
ax[0].set_ylabel("CV RMSE")
ax[0].set_title("DecisionTreeRegressor: CV error vs depth")
g = reg[reg.model == "KNN"]
ax[1].errorbar(g.param.astype(int), g.cv_rmse, yerr=g.cv_std, marker="o", capsize=3, color="#dd8452")
ax[1].set_xlabel("K")
ax[1].set_ylabel("CV RMSE")
ax[1].set_title("KNeighborsRegressor: CV error vs K")
savefig(log, "fig42_regression_cv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for a, name in zip(ax, [n for n in final if n.startswith(("DecisionTree (depth", "KNN"))]):
    a.scatter(yd_te, preds[name], s=18, alpha=0.8)
    lo, hi = yd_te.min(), yd_te.max()
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_title(f"{name}\nR2={fin.loc[fin.model == name, 'test_r2'].iloc[0]:.3f}")
    a.set_xlabel("Actual")
ax[0].set_ylabel("Predicted")
savefig(log, "fig43_regression_pred_vs_actual")

# ----------------------------- E. KNN tie sensitivity ------------------------
section(log, "E. KNN (k=1, Euclidean, ordinal): effect of neighbour-search algorithm (ties between equidistant neighbours)")
Dm = pairwise_distances(Xva.values.astype(float), Xtr.values.astype(float))
mn = Dm.min(axis=1)
tied = (Dm <= mn[:, None] + 1e-9).sum(axis=1) > 1
conflict = sum(len(set(ytr[np.where(Dm[i] <= mn[i] + 1e-9)[0]])) > 1 for i in np.where(tied)[0])
log.info("Validation points with >1 equidistant nearest training neighbour: %d of %d", int(tied.sum()), len(tied))
log.info("... of which the tied neighbours carry conflicting labels: %d", conflict)
erows = []
for alg in ("brute", "kd_tree", "ball_tree"):
    est = lambda a=alg: KNeighborsClassifier(n_neighbors=1, algorithm=a)
    cvs = cross_val_score(est(), Xtr, ytr, cv=CV)
    erows.append(dict(model="baseline K=1 Euclidean ordinal", algorithm=alg, cv_acc=cvs.mean(),
                      val_acc=accuracy_score(yva, est().fit(Xtr, ytr).predict(Xva))))
for alg in ("brute", "ball_tree"):
    est = lambda a=alg: KNeighborsClassifier(n_neighbors=9, metric="hamming", weights="distance", algorithm=a)
    cvs = cross_val_score(est(), Otr, ytr, cv=CV)
    erows.append(dict(model="best K=9 Hamming distance one-hot", algorithm=alg, cv_acc=cvs.mean(),
                      val_acc=accuracy_score(yva, est().fit(Otr, ytr).predict(Ova))))
tie = pd.DataFrame(erows)
log.info("KNN accuracy by neighbour-search algorithm:\n%s", tie.round(4).to_string(index=False))
write_tex("knn_tie_sensitivity", tie, rename={"cv_acc": "CV acc.", "val_acc": "Val acc."})

# ----------------------------- F. sample rows --------------------------------
section(log, "F. Sample rows of the raw data")
if os.path.exists("phishing.csv"):
    raw = pd.read_csv("phishing.csv")
    cols = ["UsingIP", "LongURL", "SubDomains", "HTTPS", "DomainRegLen", "AnchorURL", "WebsiteTraffic", "PageRank", "class"]
    smp = raw[cols].head(5)
    log.info("First 5 rows (selected columns):\n%s", smp.to_string())
    write_tex("data_sample", smp)
log.info("TEST set was not used in this step.")
log.info("[elapsed %.1fs] DONE", time.time() - T0)
