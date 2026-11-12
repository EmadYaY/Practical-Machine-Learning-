# -*- coding: utf-8 -*-
"""
ML Task-01 - Phishing website detection: KNN vs Decision Tree (single-file pipeline).

Usage : python ml_task01.py [phishing.csv]
Output: outputs/figures (png+pdf), outputs/tables (csv), outputs/tex (tabular bodies), outputs/run.log

Protocol: duplicates merged -> stratified 60/20/20 split. All hyper-parameters are chosen by
5-fold CV on TRAIN only. The TEST set is opened exactly once (section 5).
"""
import os
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, roc_curve, log_loss, confusion_matrix)
import logging

# ------------------------------------------------------------------ config
SEED, EPS, N_BOOT, N_EPOCHS = 42, 1e-3, 2000, 150
DATA = sys.argv[1] if len(sys.argv) > 1 else "phishing.csv"
OUT = "outputs"
FIG, TAB, TEX = (os.path.join(OUT, d) for d in ("figures", "tables", "tex"))
for d in (FIG, TAB, TEX):
    os.makedirs(d, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
K_ALL, K_GRID = range(1, 31), [1, 3, 5, 7, 9, 11, 15, 21, 31]
REPS = ("ordinal", "onehot", "merge0_neg", "merge0_pos")
TERN = ["LongURL", "SubDomains", "HTTPS", "AnchorURL", "LinksInScriptTags",
        "ServerFormHandler", "WebsiteTraffic", "LinksPointingToPage"]   # 3-level features
HDR = {"cv_mean": "CV acc.", "cv_std": "CV std", "val_acc": "Val acc.", "train_acc": "Train acc."}
_ESC = str.maketrans({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                      "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"})

sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler(os.path.join(OUT, "run.log"), "w", "utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.info


# ------------------------------------------------------------------ helpers
def save_fig(name):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{FIG}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def tex(name, df, dec=4, bold=None):
    """Save a table as CSV and as a LaTeX tabular body. bold={col: 'max'|'min'} bolds the best value."""
    df = df.reset_index(drop=True)
    df.to_csv(f"{TAB}/{name}.csv", index=False)
    cell = lambda v: ("--" if pd.isna(v) else f"{v:.{dec}f}") if isinstance(v, (float, np.floating)) \
        else str(v).translate(_ESC)
    rows = [[cell(v) for v in r] for r in df.itertuples(index=False)]
    for c, how in (bold or {}).items():
        col, j = pd.to_numeric(df[c]), list(df.columns).index(c)
        best = col.max() if how == "max" else col.min()
        for i in range(len(df)):
            if np.isclose(col[i], best):
                rows[i][j] = r"\textbf{%s}" % rows[i][j]
    spec = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    head = " & ".join(HDR.get(c, c).translate(_ESC) for c in df.columns) + r" \\"
    body = [" & ".join(r) + r" \\" for r in rows]
    with open(f"{TEX}/{name}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join([r"\begin{tabular}{%s}" % spec, r"\toprule", head, r"\midrule", *body,
                           r"\bottomrule", r"\end{tabular}"]) + "\n")


def make_rep(mode, ref):
    """Level representation (stateless). Constant columns created by merging are decided on `ref` (train)."""
    if mode == "ordinal":
        return lambda X: X
    if mode == "onehot":      # 8 ternary features -> 3 indicators each; binary features kept
        def f(X):
            out = X.drop(columns=TERN).copy()
            for c in TERN:
                for v in (-1, 0, 1):
                    out[f"{c}={v}"] = (X[c] == v).astype(int)
            return out
        return f
    v = -1 if mode == "merge0_neg" else 1      # merge level 0 (suspicious) into -1 or +1
    cols = (keep := ref.mask(ref == 0, v).nunique() > 1)[keep].index
    return lambda X: X.mask(X == 0, v)[cols]


def evaluate(est, A, y_tr, B, y_va):
    """5-fold CV accuracy on train (+ validation accuracy for reporting only)."""
    cv = cross_val_score(est, A, y_tr, cv=CV, scoring="accuracy")
    return dict(cv_mean=cv.mean(), cv_std=cv.std(),
                val_acc=accuracy_score(y_va, clone(est).fit(A, y_tr).predict(B)))


def bce(y, p):                       # clipped probabilities (pure leaves give 0/1)
    return float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def cm_heatmap(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"])
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")


# ------------------------------------------------------------------ 1. data
log("== 1. Data: label mapping, duplicate merge, stratified 60/20/20 split ==")
raw0 = pd.read_csv(DATA).drop(columns=["Index"], errors="ignore")
FEATS = [c for c in raw0.columns if c != "class"]
df = raw0.assign(y=(raw0["class"] == -1).astype(int))             # y=1 phishing, y=0 legitimate
log("shape=%s | features=%d | missing=%d | phishing=%.2f%%", raw0.shape, len(FEATS),
    int(raw0.isna().sum().sum()), df.y.mean() * 100)

sample_cols = [c for c in ("UsingIP", "LongURL", "SubDomains", "HTTPS", "DomainRegLen", "AnchorURL",
                           "WebsiteTraffic", "PageRank", "class") if c in raw0.columns]
tex("data_sample", raw0[sample_cols].head(5))

cnt = raw0["class"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
bars = plt.bar(range(len(cnt)), cnt.values, tick_label=[str(i) for i in cnt.index], color=["#4c72b0", "#dd8452"])
for b, c in zip(bars, cnt.values):
    plt.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({c / cnt.sum():.1%})", ha="center", va="bottom")
plt.ylim(0, cnt.max() * 1.18)
plt.xlabel("Class label")
plt.ylabel("Number of samples")
plt.title("Distribution of data - class")
save_fig("fig01_class_distribution")

g = df.groupby(FEATS)["y"].agg(n_rows="size", n_phish="sum").reset_index()
ties = g.n_phish * 2 == g.n_rows
conf = (g.n_phish > 0) & (g.n_phish < g.n_rows)
log("rows=%d | unique vectors=%d | duplicate rows=%d | conflicting groups=%d | tie groups dropped=%d",
    len(df), len(g), len(df) - len(g), conf.sum(), ties.sum())
g = g[~ties].copy()
g["y"] = (g.n_phish * 2 > g.n_rows).astype(int)                     # majority vote
dedup = g[FEATS + ["y"]].reset_index(drop=True)


def split3(d):
    y, idx = d["y"].values, np.arange(len(d))
    i_tr, i_tmp = train_test_split(idx, test_size=0.40, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    sub = np.empty(len(d), dtype=object)
    sub[i_tr], sub[i_va], sub[i_te] = "train", "val", "test"
    return d.assign(subset=sub)


def leak(d, s):      # share of rows in subset s whose feature vector also exists in train
    h, m = pd.util.hash_pandas_object(d[FEATS], index=False).values, d["subset"].values
    return float(np.isin(h[m == s], h[m == "train"]).mean())


sp = {"raw": split3(df[FEATS + ["y"]]), "dedup": split3(dedup)}
tex("split_summary", pd.DataFrame([
    {"Dataset": n, "Subset": s, "N": int((d.subset == s).sum()),
     "Phishing (%)": round(d.y[d.subset == s].mean() * 100, 2),
     "Leak from train (%)": 0.0 if s == "train" else round(leak(d, s) * 100, 2)}
    for n, d in sp.items() for s in ("train", "val", "test")]), dec=2)
log("dedup=%s | leakage val/test (raw): %.1f%% / %.1f%%  (dedup: %.1f%% / %.1f%%)", dedup.shape,
    leak(sp["raw"], "val") * 100, leak(sp["raw"], "test") * 100,
    leak(sp["dedup"], "val") * 100, leak(sp["dedup"], "test") * 100)

P = {s: sp["dedup"][sp["dedup"].subset == s].reset_index(drop=True) for s in ("train", "val", "test")}
Xtr, ytr, Xva, yva = P["train"][FEATS], P["train"].y.values, P["val"][FEATS], P["val"].y.values

# ------------------------------------------------------------------ 2. KNN
log("== 2. KNN (train CV + validation only) ==")
knn = lambda k, metric="euclidean", weights="uniform": KNeighborsClassifier(
    n_neighbors=k, metric=metric, weights=weights)

rows = []
for k in K_ALL:                                      # 2a. K sweep: accuracy, error and BCE
    m = knn(k).fit(Xtr, ytr)
    lap = lambda X, y: float(log_loss(y, (m.predict_proba(X)[:, 1] * k + 1) / (k + 2), labels=[0, 1]))
    rows.append(dict(K=k, train_acc=accuracy_score(ytr, m.predict(Xtr)),
                     **evaluate(knn(k), Xtr, ytr, Xva, yva), train_bce=lap(Xtr, ytr), val_bce=lap(Xva, yva)))
sw = pd.DataFrame(rows)
tex("knn_k_sweep", sw[sw.K.isin([1, 2, 3, 5, 7, 9, 11, 15, 21, 30])], bold={"cv_mean": "max", "val_bce": "min"})
k_star = int(sw.sort_values(["cv_mean", "cv_std", "K"], ascending=[False, True, False]).K.iloc[0])
log("K sweep: best K by CV = %d", k_star)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(sw.K, sw.train_acc, "o-", label="Training")
ax[0].plot(sw.K, sw.cv_mean, "s-", label="5-fold CV (train)")
ax[0].fill_between(sw.K, sw.cv_mean - sw.cv_std, sw.cv_mean + sw.cv_std, alpha=0.2)
ax[0].plot(sw.K, sw.val_acc, "^-", label="Validation")
ax[0].set(title="KNN accuracy vs K", ylabel="Accuracy")
ax[1].plot(sw.K, sw.train_bce, "o-", label="Training BCE")
ax[1].plot(sw.K, sw.val_bce, "^-", label="Validation BCE")
ax[1].set(title="BCE vs K (Laplace-smoothed posterior)", ylabel="Binary cross-entropy")
for a in ax:
    a.axvline(k_star, ls="--", color="gray", label=f"best K (CV) = {k_star}")
    a.set_xlabel("K (number of neighbors)")
    a.legend()
save_fig("fig02_knn_vs_k")

rep_cv = pd.DataFrame([dict(representation=r, K=k, **evaluate(                    # 2b. level representations
    knn(k), (f := make_rep(r, Xtr))(Xtr), ytr, f(Xva), yva)) for r in REPS for k in K_GRID])
tex("knn_representations", rep_cv.pivot(index="K", columns="representation", values="cv_mean")[list(REPS)]
    .reset_index(), bold={r: "max" for r in REPS})
top_reps = rep_cv.groupby("representation").cv_mean.max().sort_values(ascending=False).index[:3].tolist()

rows = []                                                       # 2c. joint grid, selected by CV only
for r in top_reps:
    f = make_rep(r, Xtr)
    A, B = f(Xtr), f(Xva)
    for metric in ("euclidean", "manhattan", "hamming"):
        for w in ("uniform", "distance"):
            for k in K_GRID:
                rows.append(dict(representation=r, metric=metric, weights=w, K=k,
                                 **evaluate(knn(k, metric, w), A, ytr, B, yva)))
grid = pd.DataFrame(rows).sort_values(["cv_mean", "cv_std", "K"], ascending=[False, True, False]
                                      ).reset_index(drop=True)
best = grid.iloc[0]
kcfg = dict(rep=best.representation, metric=best.metric, weights=best.weights, K=int(best.K))
log("KNN best (by CV): %s | CV=%.4f +- %.4f | val=%.4f", kcfg, best.cv_mean, best.cv_std, best.val_acc)
tex("knn_grid_top10", grid.head(10), bold={"cv_mean": "max"})
tex("knn_factors", pd.concat([grid.groupby(c).cv_mean.max().rename("best CV acc.").rename_axis("level").reset_index()
                              .assign(factor=c) for c in ("representation", "metric", "weights")])
    [["factor", "level", "best CV acc."]])

# ------------------------------------------------------------------ 3. Decision Tree
log("== 3. Decision Tree (train CV + validation only) ==")
dt = lambda **kw: DecisionTreeClassifier(random_state=SEED, **kw)

rows = [dict(criterion=c, depth=d, leaves=(m := dt(criterion=c, max_depth=d).fit(Xtr, ytr)).get_n_leaves(),
             train_acc=accuracy_score(ytr, m.predict(Xtr)),
             **evaluate(dt(criterion=c, max_depth=d), Xtr, ytr, Xva, yva))
        for c in ("gini", "entropy") for d in range(1, 26)]       # 3a. depth sweep
ds = pd.DataFrame(rows)
best_c = ds.sort_values(["cv_mean", "leaves"], ascending=[False, True]).iloc[0]
crit = best_c.criterion
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for a, c in zip(ax, ("gini", "entropy")):
    g_ = ds[ds.criterion == c]
    a.plot(g_.depth, g_.train_acc, "o-", label="Training")
    a.plot(g_.depth, g_.cv_mean, "s-", label="5-fold CV (train)")
    a.fill_between(g_.depth, g_.cv_mean - g_.cv_std, g_.cv_mean + g_.cv_std, alpha=0.2)
    a.plot(g_.depth, g_.val_acc, "^-", label="Validation")
    a.set(title=f"Decision Tree accuracy vs depth ({c})", xlabel="max_depth")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
save_fig("fig03_dt_vs_depth")

alphas = np.unique(dt(criterion=crit).cost_complexity_pruning_path(Xtr, ytr).ccp_alphas)[:-1]
sel = alphas[np.unique(np.linspace(0, len(alphas) - 1, 60).astype(int))]    # 3b. cost-complexity pruning
pr = pd.DataFrame([dict(alpha=float(a), leaves=dt(criterion=crit, ccp_alpha=float(a)).fit(Xtr, ytr).get_n_leaves(),
                        **evaluate(dt(criterion=crit, ccp_alpha=float(a)), Xtr, ytr, Xva, yva)) for a in sel])
best_e = pr.sort_values(["cv_mean", "leaves"], ascending=[False, True]).iloc[0]
cands = [(best_c.cv_mean, best_c.leaves, dict(criterion=crit, max_depth=int(best_c.depth))),
         (best_e.cv_mean, best_e.leaves, dict(criterion=crit, ccp_alpha=float(best_e.alpha)))]
KW = sorted(cands, key=lambda c: (-c[0], c[1]))[0][2]                    # best CV, ties -> fewer leaves
log("DT best (by CV): %s", KW)

rows = []                                                       # 3c. representations with the chosen tree
for r in REPS:
    f = make_rep(r, Xtr)
    A, B = f(Xtr), f(Xva)
    m = dt(**KW).fit(A, ytr)
    rows.append(dict(representation=r, n_features=A.shape[1], leaves=m.get_n_leaves(),
                     train_acc=accuracy_score(ytr, m.predict(A)), **evaluate(dt(**KW), A, ytr, B, yva)))
dt_rep = pd.DataFrame(rows)
tex("dt_representations", dt_rep, bold={"cv_mean": "max"})
REP = dt_rep.sort_values("cv_mean", ascending=False, kind="stable").representation.iloc[0]   # ties -> ordinal
log("DT representation (best CV, ties -> ordinal): %s", REP)

rows, cms, trees = [], {}, {}                                   # 3d. class-style depths (3, 4, 10, unlimited)
for lab, depth in (("3", 3), ("4", 4), ("10", 10), ("Unlimited", None)):
    kw = dict(criterion="entropy", max_depth=depth)
    m = trees[lab] = dt(**kw).fit(Xtr, ytr)
    cms[lab] = confusion_matrix(yva, m.predict(Xva), labels=[0, 1])
    rows.append(dict(max_depth=lab, depth=m.get_depth(), leaves=m.get_n_leaves(),
                     train_acc=accuracy_score(ytr, m.predict(Xtr)), **evaluate(dt(**kw), Xtr, ytr, Xva, yva),
                     FP=int(cms[lab][0, 1]), FN=int(cms[lab][1, 0])))
tex("dt_depth_compare", pd.DataFrame(rows), bold={"cv_mean": "max", "val_acc": "max"})
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for a, (lab, cm), r in zip(ax, cms.items(), rows):
    cm_heatmap(a, yva, trees[lab].predict(Xva), f"max_depth={lab}\nval acc={r['val_acc']:.4f}")
save_fig("fig04_dt_depth_confusion")
plt.figure(figsize=(26, 11))
plot_tree(trees["4"], feature_names=FEATS, class_names=["Legit", "Phish"], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (entropy, max_depth=4)")
save_fig("fig05_tree_depth4")

# ------------------------------------------------------------------ 4. MLP curves (BCE/accuracy per epoch)
log("== 4. MLP: per-epoch training/validation BCE and accuracy ==")
mlp = lambda: MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4,
                            batch_size=64, learning_rate_init=1e-3, random_state=SEED)
f_oh = make_rep("onehot", Xtr)
Otr, Ova = f_oh(Xtr).values.astype(float), f_oh(Xva).values.astype(float)
m, hist = mlp(), []
for ep in range(1, N_EPOCHS + 1):                    # one epoch per partial_fit call
    m.partial_fit(Otr, ytr, classes=np.array([0, 1]))
    ptr, pva = m.predict_proba(Otr)[:, 1], m.predict_proba(Ova)[:, 1]
    hist.append(dict(epoch=ep, train_bce=bce(ytr, ptr), val_bce=bce(yva, pva),
                     train_acc=accuracy_score(ytr, ptr >= 0.5), val_acc=accuracy_score(yva, pva >= 0.5)))
H = pd.DataFrame(hist)
E_STAR = int(H.loc[H.val_bce.idxmin(), "epoch"])
log("best epoch by validation BCE = %d (val BCE=%.4f, val acc=%.4f)", E_STAR, H.val_bce.min(),
    H.loc[H.val_bce.idxmin(), "val_acc"])
H.to_csv(f"{TAB}/mlp_history.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, (cols, yl) in zip(ax, ((("train_bce", "val_bce"), "Binary cross-entropy"),
                              (("train_acc", "val_acc"), "Accuracy"))):
    a.plot(H.epoch, H[cols[0]], label="Training")
    a.plot(H.epoch, H[cols[1]], label="Validation")
    a.axvline(E_STAR, ls="--", color="gray", label=f"min val BCE (epoch {E_STAR})")
    a.set(xlabel="Epoch", ylabel=yl)
    a.legend()
save_fig("fig06_mlp_curves")

# ------------------------------------------------------------------ 5. final test evaluation (opened once)
log("== 5. TEST SET OPENED (once): refit on train+val, score once ==")
fit = pd.concat([P["train"], P["val"]], ignore_index=True)
Xfit, yfit, Xte, yte = fit[FEATS], fit.y.values, P["test"][FEATS], P["test"].y.values
SPECS = {"KNN baseline": ("ordinal", KNeighborsClassifier(n_neighbors=1)),
         "KNN best": (kcfg["rep"], knn(kcfg["K"], kcfg["metric"], kcfg["weights"])),
         "DT class-style": ("ordinal", dt(criterion="entropy", max_depth=4)),
         "DT final": (REP, dt(**KW))}
R = {}
for name, (rep_name, est) in SPECS.items():
    f = make_rep(rep_name, Xfit)
    m = est.fit(f(Xfit), yfit)
    R[name] = (m.predict(f(Xte)), m.predict_proba(f(Xte))[:, 1])
f = make_rep("onehot", Xfit)
m = mlp()
for _ in range(E_STAR):
    m.partial_fit(f(Xfit).values.astype(float), yfit, classes=np.array([0, 1]))
s = m.predict_proba(f(Xte).values.astype(float))[:, 1]
R["MLP"] = ((s >= 0.5).astype(int), s)
names = list(R)

tm = pd.DataFrame([dict(model=n, accuracy=accuracy_score(yte, p), precision=precision_score(yte, p),
                        recall=recall_score(yte, p), f1=f1_score(yte, p), roc_auc=roc_auc_score(yte, s_),
                        bce=bce(yte, s_)) for n, (p, s_) in R.items()])
tex("test_metrics", tm, bold={"accuracy": "max", "precision": "max", "recall": "max", "f1": "max",
                              "roc_auc": "max", "bce": "min"})
log("TEST metrics:\n%s", tm.round(4).to_string(index=False))
pd.DataFrame({"y_true": yte, **{f"{n}__pred": R[n][0] for n in names},
              **{f"{n}__score": R[n][1] for n in names}}).to_csv(f"{TAB}/test_predictions.csv", index=False)

rng = np.random.default_rng(SEED)                                # bootstrap 95% CI (accuracy, F1)
BOOT = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
yb, ci = yte[BOOT], []
for n in names:
    pb = R[n][0][BOOT]
    tp, fp, fn = ((yb == 1) & (pb == 1)).sum(1), ((yb == 0) & (pb == 1)).sum(1), ((yb == 1) & (pb == 0)).sum(1)
    fmt = lambda pt, arr: f"{pt:.4f} [{np.percentile(arr, 2.5):.4f}, {np.percentile(arr, 97.5):.4f}]"
    ci.append({"model": n, "Accuracy [95% CI]": fmt(tm.set_index("model").accuracy[n], (yb == pb).mean(1)),
               "F1 [95% CI]": fmt(tm.set_index("model").f1[n], 2 * tp / np.maximum(2 * tp + fp + fn, 1))})
tex("test_bootstrap_ci", pd.DataFrame(ci))

rows = []                                                       # McNemar exact test on paired predictions
for a, b in (("KNN best", "DT final"), ("KNN best", "KNN baseline"), ("KNN best", "MLP")):
    ca, cb = R[a][0] == yte, R[b][0] == yte
    na, nb = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(na, na + nb, 0.5).pvalue if na + nb else 1.0
    rows.append({"Model A": a, "Model B": b, "A right, B wrong": na, "A wrong, B right": nb,
                 "p-value": "<0.0001" if p < 1e-4 else f"{p:.4f}", "Significant (0.05)": "Yes" if p < 0.05 else "No"})
tex("mcnemar", pd.DataFrame(rows))

fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
for ax, n in zip(axes, names):
    cm_heatmap(ax, yte, R[n][0], f"{n}\nacc={tm.set_index('model').accuracy[n]:.4f}")
save_fig("fig07_test_confusion")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for n in names:
    fpr, tpr, _ = roc_curve(yte, R[n][1])
    for a in ax:
        a.plot(fpr, tpr, lw=1.6, label=f"{n} (AUC={tm.set_index('model').roc_auc[n]:.4f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax[0].set_title("ROC curve")
ax[1].set(xlim=(0, 0.15), ylim=(0.85, 1.005), title="ROC - zoom (FPR <= 0.15)")
for a in ax:
    a.set(xlabel="False positive rate", ylabel="True positive rate (phishing recall)")
ax[0].legend(loc="lower right", fontsize=8)
save_fig("fig08_test_roc")

pkgs = {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn")}
tex("hyperparameters", pd.DataFrame([
    ("KNN baseline", "n_neighbors / metric / weights / representation", "1 / euclidean / uniform / ordinal"),
    ("KNN best", "n_neighbors / metric / weights / representation",
     f"{kcfg['K']} / {kcfg['metric']} / {kcfg['weights']} / {kcfg['rep']}"),
    ("DT class-style", "criterion / max_depth", "entropy / 4"),
    ("DT final", "parameters / representation", f"{KW} / {REP}"),
    ("MLP (suppl.)", "hidden / activation / solver / alpha / batch / lr / epochs",
     f"(64,) / relu / adam / 1e-4 / 64 / 1e-3 / {E_STAR}"),
    ("All", "random_state / CV / split", f"{SEED} / stratified 5-fold / 60-20-20"),
    ("Environment", "Python / libraries", f"{sys.version.split()[0]} / " + ", ".join(f"{k} {v}" for k, v in pkgs.items())),
], columns=["Model", "Parameter(s)", "Value(s)"]))
log("DONE. Test set evaluated exactly once.")
