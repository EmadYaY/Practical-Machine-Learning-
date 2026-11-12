[Persian - پارسی](README_task01_fa.md)

## Reference Report

The following report is provided as a reference for this assignment:

[View the Reference Report (Google Drive)](https://drive.google.com/file/d/1_TCKn9jrwUNjK58ZOlgNBC9UiMYBoPho/view)

-----------------
# Task-01: Phishing Website Detection (KNN vs Decision Tree)

Binary classification on the UCI *Phishing Websites* data (`phishing.csv`: 11,054 rows, 30 features with values in {-1, 0, 1}; label mapped so that `y = 1` is phishing). KNN and Decision Tree are tuned with 5-fold CV on the training part only and compared once on a held-out test set. An MLP is added only to draw per-epoch loss/accuracy curves.

Dataset: R. Mohammad and L. McCluskey, UCI Machine Learning Repository, DOI 10.24432/C51W2X (CC BY 4.0).

## Two implementations

The task is implemented twice with the same protocol (merge duplicates, stratified 60/20/20 split, CV-only model selection, test set opened once).

| | Full pipeline (`step01` ... `step07`, `common.py`) | Compact (`ml_task01.py`) |
|---|---|---|
| Run | one script per stage, in order | one command, a few minutes |
| Hand-off between stages | CSV/JSON checkpoints in `outputs/data` | none (one process) |
| Output folder | `outputs/` | `outputs_compact/` |
| EDA (class balance, correlations, mutual information, level heatmaps) | yes | class balance only |
| Duplicates and leakage | raw vs dedup compared in baselines, protocols and K sweep | leakage rate + raw vs dedup KNN table |
| KNN experiments | K sweep (raw and dedup), 4 metrics incl. Chebyshev, weights, scaling, feature selection, 4 representations, 162-config grid, learning curves | K sweep (dedup), 4 representations, 162-config grid (Euclidean / Manhattan / Hamming, uniform / distance) |
| Decision Tree experiments | default and class-style baselines, depth sweep, criterion x depth x leaf-size grid, pruning, representations, importances, learning curve, `.dot` export | depth sweep, pruning, 4 representations, depths 3/4/10/unlimited with confusion matrices, depth-4 tree plot |
| Test-set models | 7 (KNN baseline, KNN best, DT default, DT class-style, DT final, DT 1-SE, MLP) | 5 (KNN baseline, KNN best, DT class-style, DT final, MLP) |
| Test metrics | Acc, P, R, F1, macro-F1, MCC, ROC-AUC, PR-AUC, BCE, Brier; operating points | Acc, P, R, F1, ROC-AUC, BCE; train and test error |
| Uncertainty / tests | bootstrap CI (Acc, F1, AUC), McNemar (6 pairs), leave-one-out vs CV vs test error | bootstrap CI (Acc, F1), McNemar (3 pairs) |
| Feature importance | impurity, grouped permutation (DT, KNN), rank agreement | none |
| Logs | one log per stage, with full classification reports and confusion matrices | one `run.log` |
| LaTeX tables | all report tables exported (`outputs/tex`) | tables used by the report (`outputs_compact/tex`) |

**Where the full pipeline is better:** each stage can be re-run alone (e.g. only the tree stage) because earlier results are checkpointed; every experiment keeps its own log, table and figure; and it covers experiments the compact version leaves out (scaling, feature selection, Chebyshev distance, learning curves, feature importance, PR curves, operating points, leave-one-out error).

**Where the compact version is better:** a single command reproduces every number used in the report, nothing depends on files left by an earlier stage, and shared helpers are defined once.

For the models they share, both versions give identical test results (see below).

## Compact version: output structure

```
ml_task01.py
phishing.csv
outputs_compact/
  run.log                 # selected configs and test metrics
  figures/                # fig01 ... fig08 (.png and .pdf, 300 dpi)
  tables/                 # every table as .csv, plus mlp_history.csv and test_predictions.csv
  tex/                    # the same tables as LaTeX tabular bodies (no caption)
```

| Figure | Content |
|---|---|
| `fig01_class_distribution` | class counts of the original data |
| `fig02_knn_vs_k` | KNN accuracy (train / CV / validation) and BCE vs K |
| `fig03_dt_vs_depth` | tree accuracy vs depth, gini and entropy |
| `fig04_dt_depth_confusion` | validation confusion matrices for depth 3, 4, 10, unlimited |
| `fig05_tree_depth4` | entropy tree, max depth 4 |
| `fig06_mlp_curves` | MLP BCE and accuracy per epoch |
| `fig07_test_confusion` | test confusion matrices of all models |
| `fig08_test_roc` | test ROC curves (full and zoomed) |

Tables in `tex/` and `tables/`: `data_sample`, `split_summary`, `leakage_effect`, `knn_k_sweep`, `knn_representations`, `knn_grid_top10`, `knn_factors`, `dt_representations`, `dt_depth_compare`, `test_metrics`, `test_bootstrap_ci`, `mcnemar`, `hyperparameters`.

## Compact version: code walkthrough

| Block | What it does |
|---|---|
| Config and helpers | seeds, CV object, grids; `tex()` writes a table as CSV and LaTeX; `make_rep()` builds the four level representations (`ordinal`, `onehot`, `merge0_neg`, `merge0_pos`); `evaluate()` returns CV mean/std and validation accuracy |
| 1. Data | maps `class` to `y`, merges duplicate feature vectors (majority vote, ties dropped), splits 60/20/20 stratified (raw and dedup), measures train-to-val/test leakage |
| 2. KNN | K sweep with accuracy and Laplace-smoothed BCE; leakage effect on K; representation comparison; joint grid over the top-3 representations x metric x weights x K, chosen by CV only |
| 3. Decision Tree | depth sweep (gini, entropy); cost-complexity pruning; best of both by CV (ties: fewer leaves); representation choice; depths 3/4/10/unlimited with confusion matrices and the depth-4 tree plot |
| 4. MLP | `partial_fit` per epoch on one-hot features, train/validation BCE and accuracy; epoch with minimum validation BCE is kept |
| 5. Test evaluation | refits the five models on train+validation and scores the test set once; metrics, bootstrap CI (2000 resamples), McNemar exact test, confusion matrices, ROC; writes the hyper-parameter table |

## Full pipeline: stages

| Script | Role |
|---|---|
| `common.py` | paths, logger, figure saving, level representations |
| `step01_env_eda.py` | environment check, data quality, EDA figures and tables |
| `step02_preprocess.py` | label mapping, duplicate merge, split, leakage measurement |
| `step03_knn_baseline.py` | baseline KNN (k=1) on raw vs dedup, evaluation protocols |
| `step04_knn_experiments.py` | K, metric, weights, representation, scaling, feature selection, joint grid, learning curves |
| `step05_decision_tree.py` | depth, criterion, grid, pruning, representations, tree plots, importances |
| `step06_final_evaluation.py` | test set (opened once), bootstrap CI, McNemar, error gaps, operating points, MLP curves |
| `step07_importance_and_tex.py` | importance per original feature, LaTeX export of all tables |

## Test results (identical in both versions)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN baseline (k=1, Euclidean, ordinal) | 0.9186 | 0.9372 | 0.9025 | 0.9195 | 0.9191 |
| KNN best (one-hot, Hamming, distance, K=9) | 0.9567 | 0.9706 | 0.9445 | 0.9574 | 0.9933 |
| DT class-style (entropy, depth 4) | 0.9255 | 0.9321 | 0.9227 | 0.9274 | 0.9756 |
| DT final (entropy, pruned, one-hot) | 0.9472 | 0.9525 | 0.9445 | 0.9485 | 0.9677 |
| MLP (45 epochs, supplementary) | 0.9619 | 0.9615 | 0.9647 | 0.9631 | 0.9938 |

## How to run

```bash
# compact
python ml_task01.py phishing.csv

# full pipeline (run in this order)
python step01_env_eda.py
python step02_preprocess.py
python step03_knn_baseline.py
python step04_knn_experiments.py
python step05_decision_tree.py
python step06_final_evaluation.py
python step07_importance_and_tex.py
```

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn.
