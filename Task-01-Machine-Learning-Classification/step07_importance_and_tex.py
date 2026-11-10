# -*- coding: utf-8 -*-
"""
Step 07 - (A) importance aggregated per ORIGINAL feature (MI, DT impurity, DT/KNN grouped permutation)
          (B) export every report table as a LaTeX tabular body (no caption) into outputs/tex/
The TEST set is NOT used here (only train/val and the tables written by steps 1-6).
"""
import os
import sys
import json
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import (SEED, OUT_DIR, DATA_DIR, TAB_DIR, TARGET, get_logger, section, savefig,
                    get_features, make_representation)

sns.set_theme(style="whitegrid", context="notebook")
log = get_logger("step07_importance_and_tex")
TEX_DIR = os.path.join(OUT_DIR, "tex")
os.makedirs(TEX_DIR, exist_ok=True)
N_REP_DT, N_REP_KNN = 30, 10
T0 = time.time()


def tick(msg):
    log.info("[elapsed %.1fs] %s", time.time() - T0, msg)


# =============================== A. Importance ===============================
section(log, "A. Load data, configs, refit final models on TRAIN (same as steps 4-5)")
d = pd.read_csv(os.path.join(DATA_DIR, "dataset_dedup.csv"))
FEATS = get_features(d)
tr, va = d[d["subset"] == "train"], d[d["subset"] == "val"]
Xtr, ytr = tr[FEATS].reset_index(drop=True), tr[TARGET].values
Xva, yva = va[FEATS].reset_index(drop=True), va[TARGET].values
with open(os.path.join(DATA_DIR, "best_knn_config.json"), encoding="utf-8") as f:
    knn_cfg = json.load(f)
with open(os.path.join(DATA_DIR, "best_dt_config.json"), encoding="utf-8") as f:
    dt_cfg = json.load(f)


def rep_fn(Xref, mode):
    """Stateless level representation; merged-constant columns are decided on train only."""
    if mode.startswith("merge"):
        v = -1 if mode == "merge0_neg" else 1
        keep = Xref.mask(Xref == 0, v).nunique() > 1
        cols = keep[keep].index
        return lambda X: X.mask(X == 0, v)[cols]
    return lambda X: make_representation(X, mode)


f_dt, f_knn = rep_fn(Xtr, dt_cfg["representation"]), rep_fn(Xtr, knn_cfg["representation"])
dt = DecisionTreeClassifier(criterion=dt_cfg["criterion"], max_depth=dt_cfg["max_depth"],
                            min_samples_leaf=dt_cfg["min_samples_leaf"], ccp_alpha=dt_cfg["ccp_alpha"],
                            random_state=SEED).fit(f_dt(Xtr), ytr)
knn = KNeighborsClassifier(n_neighbors=knn_cfg["K"], metric=knn_cfg["metric"],
                           weights=knn_cfg["weights"]).fit(f_knn(Xtr), ytr)
acc_dt = accuracy_score(yva, dt.predict(f_dt(Xva)))
acc_knn = accuracy_score(yva, knn.predict(f_knn(Xva)))
log.info("Sanity check - val accuracy: DT=%.4f (step 5: 0.9411) | KNN=%.4f (step 4: 0.9549)", acc_dt, acc_knn)

# --- impurity importance aggregated per original feature
imp_cols = pd.Series(dt.feature_importances_, index=f_dt(Xtr).columns)
orig = imp_cols.index.str.split("=").str[0]
imp_agg = imp_cols.groupby(orig).sum().reindex(FEATS).fillna(0.0)
log.info("DT impurity importance: sums to %.4f over %d original features", imp_agg.sum(), len(FEATS))


# --- grouped permutation importance (permute the ORIGINAL feature, then rebuild representation)
def group_perm(predict, X, y, n_rep, seed):
    rng = np.random.default_rng(seed)
    base = accuracy_score(y, predict(X))
    out = {}
    for ft in X.columns:
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[ft] = rng.permutation(X[ft].values)
            drops.append(base - accuracy_score(y, predict(Xp)))
        out[ft] = (float(np.mean(drops)), float(np.std(drops)))
    return pd.DataFrame(out, index=["mean", "std"]).T


p_dt = group_perm(lambda X: dt.predict(f_dt(X)), Xva, yva, N_REP_DT, SEED)
tick("DT grouped permutation done")
p_knn = group_perm(lambda X: knn.predict(f_knn(X)), Xva, yva, N_REP_KNN, SEED)
tick("KNN grouped permutation done")

mi = pd.read_csv(os.path.join(TAB_DIR, "table18_mi_train.csv"), index_col=0)["mutual_info"]
imp = pd.DataFrame({"MI_train": mi.reindex(FEATS), "DT_impurity": imp_agg,
                    "DT_perm": p_dt["mean"], "DT_perm_std": p_dt["std"],
                    "KNN_perm": p_knn["mean"], "KNN_perm_std": p_knn["std"]})
for c in ("MI_train", "DT_impurity", "DT_perm", "KNN_perm"):
    imp[f"rank_{c}"] = imp[c].rank(ascending=False, method="min").astype(int)
imp = imp.sort_values("DT_perm", ascending=False)
imp.index.name = "feature"
imp.to_csv(os.path.join(TAB_DIR, "table35_importance_original_features.csv"))
log.info("Importance per original feature (permutation = accuracy drop on VALIDATION):\n%s",
         imp.round(4).to_string())
log.info("Features whose permutation never hurts (mean drop <= 0): DT=%d | KNN=%d",
         int((imp.DT_perm <= 0).sum()), int((imp.KNN_perm <= 0).sum()))

scores = imp[["MI_train", "DT_impurity", "DT_perm", "KNN_perm"]]
rho = scores.corr(method="spearman")
rho.to_csv(os.path.join(TAB_DIR, "table36_importance_rank_agreement.csv"))
log.info("Spearman rank correlation between importance measures:\n%s", rho.round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 6.2))
for a, (col, std, title) in zip(ax, (("DT_impurity", None, "DT impurity importance (aggregated)"),
                                     ("DT_perm", "DT_perm_std", "DT permutation importance (val)"),
                                     ("KNN_perm", "KNN_perm_std", "KNN permutation importance (val)"))):
    t = imp.sort_values(col, ascending=False).head(15).iloc[::-1]
    a.barh(t.index, t[col], xerr=None if std is None else t[std], color="#4c72b0", capsize=2)
    a.set_title(title)
    a.set_xlabel("Importance" if std is None else "Accuracy drop")
    a.tick_params(axis="y", labelsize=8)
savefig(log, "fig36_importance_original_features")

plt.figure(figsize=(6.2, 5))
sns.heatmap(rho, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, cbar_kws={"label": "Spearman rho"})
plt.title("Agreement between importance rankings")
savefig(log, "fig37_importance_rank_agreement")

# =============================== B. LaTeX export =============================
section(log, "B. Export report tables as LaTeX tabular bodies (no caption) -> outputs/tex/")
_MAP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(_MAP.get(c, c) for c in str(s))


HDR = {"K": "K", "train_acc": "Train acc.", "cv_mean": "CV acc.", "cv_std": "CV std", "val_acc": "Val acc.",
       "train_bce": "Train BCE", "val_bce": "Val BCE", "accuracy": "Acc.", "error": "Error",
       "precision_phish": "Prec.", "recall_phish": "Rec.", "f1_phish": "F1", "f1_macro": "F1-macro",
       "f1_micro": "F1-micro", "roc_auc": "ROC AUC", "pr_auc": "PR AUC", "bce": "BCE", "brier": "Brier",
       "mcc": "MCC", "n_features": "#Features", "leaves": "Leaves", "depth": "Depth", "dataset": "Dataset",
       "subset": "Subset", "model": "Model", "representation": "Representation", "metric": "Metric",
       "weights": "Weights", "criterion": "Criterion", "resub_error": "Training (resub.) error",
       "honest_error": "Honest error", "honest_kind": "Honest error type", "test_error": "Test error",
       "test_minus_resub": "Test - resub."}


def fmt_col(s, dec):
    if pd.api.types.is_bool_dtype(s):
        return s.map(lambda v: "Yes" if v else "No")
    if pd.api.types.is_integer_dtype(s):
        return s.map(lambda v: str(int(v)))
    if pd.api.types.is_float_dtype(s):
        return s.map(lambda v: "--" if pd.isna(v) else f"{v:.{dec}f}")
    return s.map(lambda v: "--" if (not isinstance(v, str) and pd.isna(v)) else esc(v))


WRITTEN = []


def ex(name, df, dec=4, bold=None, rename=None):
    df = df.reset_index(drop=True)
    rn, bold, cols = {**HDR, **(rename or {})}, bold or {}, list(df.columns)
    cells = {}
    for c in cols:
        s = fmt_col(df[c], dec)
        if c in bold:
            num = pd.to_numeric(df[c], errors="coerce")
            best = num.max() if bold[c] == "max" else num.min()
            s = pd.Series([f"\\textbf{{{x}}}" if (pd.notna(v) and np.isclose(v, best)) else x
                           for x, v in zip(s, num)])
        cells[c] = s.reset_index(drop=True)
    spec = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else "l"
                   for c in cols)
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(esc(rn.get(c, c)) for c in cols) + r" \\", r"\midrule"]
    for i in range(len(df)):
        lines.append(" & ".join(cells[c].iloc[i] for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(TEX_DIR, f"{name}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    WRITTEN.append(name)
    log.info("TEX written: %-32s rows=%d cols=%d", name + ".tex", len(df), len(cols))


def rd(fn, **kw):
    return pd.read_csv(os.path.join(TAB_DIR, fn), **kw)


# ---- data description
t = rd("table02_class_distribution.csv")
t["class"] = t["class"].map({-1: "-1 (phishing)", 1: "1 (legitimate)"})
ex("class_distribution", t, dec=3, rename={"class": "Original label", "count": "Count", "percent": "Percent (%)"})
ex("split_summary", rd("table09_split_summary.csv"), dec=2,
   rename={"n": "N", "phishing": "Phishing", "legit": "Legitimate", "phishing_pct": "Phishing (%)",
           "leak_from_train_pct": "Leak from train (%)"})
ex("representation_sizes", rd("table10_representations.csv"),
   rename={"mode": "Representation", "dropped_constant": "Dropped (constant)"})

# ---- step 3
ex("baseline_knn", rd("table11_baseline_metrics.csv")[["dataset", "subset", "accuracy", "error",
                                                       "precision_phish", "recall_phish", "f1_phish"]])
ex("evaluation_protocols", rd("table12_evaluation_protocols.csv")[["dataset", "K", "protocol", "acc_mean",
                                                                   "acc_std", "error_mean"]],
   rename={"protocol": "Protocol", "acc_mean": "Acc. mean", "acc_std": "Acc. std", "error_mean": "Error mean"})

# ---- step 4
KS = [1, 2, 3, 5, 7, 9, 11, 15, 21, 30]
sw = rd("table13_k_sweep.csv")
sw = sw[sw.K.isin(KS)]
for ds in ("dedup", "raw"):
    ex(f"k_sweep_{ds}", sw[sw.dataset == ds].drop(columns="dataset"), bold={"cv_mean": "max", "val_bce": "min"})


def piv(fn, col, order):
    return rd(fn).pivot(index="K", columns=col, values="cv_mean")[order].reset_index()


for name, fn, col, order in (("metrics_cv", "table14_metrics.csv", "metric", ["euclidean", "manhattan", "chebyshev", "hamming"]),
                             ("weights_cv", "table15_weights.csv", "weights", ["uniform", "distance"]),
                             ("representations_knn_cv", "table16_representations.csv", "representation",
                              ["ordinal", "onehot", "merge0_neg", "merge0_pos"]),
                             ("scaling_cv", "table17_scaling.csv", "scaler",
                              ["none", "StandardScaler", "MinMaxScaler"])):
    p = piv(fn, col, order)
    ex(name, p, bold={c: "max" for c in order})

fs = rd("table19_feature_selection.csv").pivot(index="n_features", columns="K", values="cv_mean").reset_index()
fs.columns = ["#Features"] + [f"K={k}" for k in fs.columns[1:]]
ex("feature_selection_cv", fs, bold={c: "max" for c in fs.columns[1:]})
ex("joint_grid_top10", rd("table20_joint_grid.csv").head(10), bold={"cv_mean": "max"})
ex("knn_val_baseline_vs_best", rd("table21_val_baseline_vs_best.csv"))

# ---- step 5
ex("dt_baselines", rd("table22_dt_baselines.csv")[["dataset", "model", "depth", "leaves", "train_acc", "cv_mean",
                                                    "cv_std", "val_acc", "val_f1_phish"]],
   rename={"val_f1_phish": "Val F1"})
dsw = rd("table23_dt_depth_sweep.csv")
DL = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
for crit in ("entropy", "gini"):
    t = dsw[(dsw.criterion == crit) & (dsw.depth.isin(DL))]
    ex(f"dt_depth_{crit}", t[["depth", "leaves", "train_acc", "cv_mean", "cv_std", "val_acc", "train_bce", "val_bce"]],
       bold={"cv_mean": "max", "val_bce": "min"})
g = rd("table24_dt_grid.csv", keep_default_na=False).sort_values(["cv_mean", "leaves"], ascending=[False, True]).head(10)
ex("dt_grid_top10", g, rename={"depth_label": "max_depth"}, bold={"cv_mean": "max"})
pr = rd("table25_dt_pruning.csv")
idx = np.unique(np.append(np.linspace(0, len(pr) - 1, 12).astype(int), pr.cv_mean.idxmax()))
t = pr.iloc[idx].copy()
t["alpha"] = t["alpha"].map(lambda v: f"{v:.5f}")
ex("dt_pruning", t, rename={"alpha": "ccp_alpha"}, bold={"cv_mean": "max"})
ex("dt_representations", rd("table26_dt_representations.csv"), bold={"cv_mean": "max"})
ex("dt_val_report", rd("table28_dt_val_report.csv"))

# ---- step 6 + 7
ti = imp.head(15)
ex("importance_top15", pd.DataFrame({
    "Feature": ti.index, "MI (train)": ti.MI_train.values, "DT impurity": ti.DT_impurity.values,
    "DT perm. (drop)": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(ti.DT_perm, ti.DT_perm_std)],
    "KNN perm. (drop)": [f"{m:.4f} +/- {s:.4f}" for m, s in zip(ti.KNN_perm, ti.KNN_perm_std)],
    "Rank MI": ti.rank_MI_train.values, "Rank DT imp.": ti.rank_DT_impurity.values,
    "Rank DT perm.": ti.rank_DT_perm.values, "Rank KNN perm.": ti.rank_KNN_perm.values}))
ex("importance_rank_agreement", rho.rename(index={"MI_train": "MI (train)", "DT_impurity": "DT impurity",
                                                  "DT_perm": "DT perm.", "KNN_perm": "KNN perm."},
                                           columns={"MI_train": "MI (train)", "DT_impurity": "DT impurity",
                                                    "DT_perm": "DT perm.", "KNN_perm": "KNN perm."}
                                           ).rename_axis("Measure").reset_index(), dec=3)

tm = rd("table29_test_metrics.csv")
ex("test_metrics", tm, bold={"accuracy": "max", "error": "min", "precision_phish": "max", "recall_phish": "max",
                              "f1_phish": "max", "f1_macro": "max", "mcc": "max", "roc_auc": "max", "pr_auc": "max",
                              "bce": "min", "brier": "min"})
ci = rd("table30_bootstrap_ci.csv")
ci["txt"] = ci.apply(lambda r: f"{r.point:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]", axis=1)
pc = ci.pivot(index="model", columns="metric", values="txt").reindex(tm["model"].tolist())
pc = pc[["accuracy", "f1_phish", "roc_auc"]].reset_index()
ex("test_bootstrap_ci", pc, rename={"accuracy": "Accuracy [95% CI]", "f1_phish": "F1 phishing [95% CI]",
                                    "roc_auc": "ROC AUC [95% CI]"})
mc = rd("table31_mcnemar.csv")
mc["p_value"] = mc["p_value"].map(lambda p: "<0.0001" if p < 1e-4 else f"{p:.4f}")
ex("mcnemar", mc, rename={"model_A": "Model A", "model_B": "Model B", "A_correct_B_wrong": "A right, B wrong",
                          "A_wrong_B_correct": "A wrong, B right", "p_value": "p-value",
                          "significant_at_0_05": "Significant (0.05)"})
ex("generalisation_gap", rd("table32_generalisation_gap.csv"))
ex("operating_points", rd("table33_operating_points.csv"),
   rename={"precision@recall>=0.9": "P@R>=0.90", "precision@recall>=0.95": "P@R>=0.95",
           "precision@recall>=0.98": "P@R>=0.98", "precision@recall>=0.99": "P@R>=0.99"})

# ---- libraries
LIBS = [("Python", ".".join(map(str, sys.version_info[:3])), "Language runtime"),
        ("numpy", version("numpy"), "Arrays, bootstrap resampling"),
        ("pandas", version("pandas"), "Data loading, grouping, tables"),
        ("scipy", version("scipy"), "Exact binomial test (McNemar)"),
        ("scikit-learn", version("scikit-learn"), "KNN, Decision Tree, MLP, CV, metrics, mutual information"),
        ("matplotlib", version("matplotlib"), "Figures"),
        ("seaborn", version("seaborn"), "Heatmaps and plot styling")]
ex("libraries", pd.DataFrame(LIBS, columns=["Library", "Version", "Role in this work"]))

# ---- selected hyper-parameters
pr_all = rd("table25_dt_pruning.csv")
top = pr_all.sort_values(["cv_mean", "leaves"], ascending=[False, True]).iloc[0]
alpha_simple = float(pr_all[pr_all.cv_mean >= top.cv_mean - top.cv_std / np.sqrt(5)]
                     .sort_values(["leaves", "cv_mean"], ascending=[True, False]).iloc[0].alpha)
H = rd("table34_mlp_history.csv")
e_star = int(H.loc[H.val_bce.idxmin(), "epoch"])
HP = [("KNN baseline", "n_neighbors / metric / weights", "1 / minkowski (p=2) / uniform"),
      ("KNN baseline", "representation", "ordinal"),
      ("KNN best", "n_neighbors / metric / weights", f"{knn_cfg['K']} / {knn_cfg['metric']} / {knn_cfg['weights']}"),
      ("KNN best", "representation", knn_cfg["representation"]),
      ("DT default", "criterion / max_depth / min_samples_leaf / ccp_alpha", "gini / None / 1 / 0"),
      ("DT class-style", "criterion / max_depth", "entropy / 4"),
      ("DT final (CV)", "criterion / max_depth / min_samples_leaf",
       f"{dt_cfg['criterion']} / {dt_cfg['max_depth']} / {dt_cfg['min_samples_leaf']}"),
      ("DT final (CV)", "ccp_alpha / representation", f"{dt_cfg['ccp_alpha']:.6f} / {dt_cfg['representation']}"),
      ("DT simple (1-SE)", "criterion / ccp_alpha / representation", f"entropy / {alpha_simple:.6f} / ordinal"),
      ("MLP (suppl.)", "hidden_layer_sizes / activation / solver", "(64,) / relu / adam"),
      ("MLP (suppl.)", "learning_rate_init / alpha (L2) / batch_size", "0.001 / 0.0001 / 64"),
      ("MLP (suppl.)", "epochs (min validation BCE)", str(e_star)),
      ("All models", "random_state", str(SEED))]
ex("hyperparameters", pd.DataFrame(HP, columns=["Model", "Parameter(s)", "Value(s)"]))

section(log, "C. Summary")
log.info("LaTeX files written (%d): %s", len(WRITTEN), ", ".join(sorted(WRITTEN)))
tick("ALL DONE (test subset untouched)")