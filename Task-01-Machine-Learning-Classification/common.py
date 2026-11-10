# -*- coding: utf-8 -*-
"""Shared helpers for all steps (logger, figure saving, level representations)."""
import os
import sys
import logging
import matplotlib.pyplot as plt

SEED = 42
OUT_DIR = "outputs"
LOG_DIR = os.path.join(OUT_DIR, "logs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
DATA_DIR = os.path.join(OUT_DIR, "data")
SHOW_PLOTS = False
for _d in (LOG_DIR, FIG_DIR, TAB_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

ORIG_TARGET = "class"   # original: -1 = phishing, 1 = legitimate
TARGET = "y"            # ours:      1 = phishing, 0 = legitimate
NON_FEATURE_COLS = {"Index", "class", "y", "n_rows", "subset"}

TERNARY_COLS = ["LongURL", "SubDomains", "HTTPS", "AnchorURL",
                "LinksInScriptTags", "ServerFormHandler",
                "WebsiteTraffic", "LinksPointingToPage"]


def get_features(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"), mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def section(logger, title):
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def savefig(logger, name):
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    logger.info("Figure saved: %s (.png/.pdf)", name)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def make_representation(X, mode):
    """
    Level representations of the feature matrix (stateless, no leakage):
      ordinal    : original values in {-1, 0, 1}
      onehot     : 8 ternary features -> 3 indicator columns each; binary features kept
      merge0_neg : merge level 0 (suspicious) into -1 (phishing-like)
      merge0_pos : merge level 0 (suspicious) into +1 (legitimate-like)
    Constant columns created by merging are dropped.
    """
    if mode == "ordinal":
        return X.copy()
    if mode == "onehot":
        out = X.drop(columns=TERNARY_COLS).copy()
        for c in TERNARY_COLS:
            for v in (-1, 0, 1):
                out[f"{c}={v}"] = (X[c] == v).astype(int)
        return out
    if mode in ("merge0_neg", "merge0_pos"):
        val = -1 if mode == "merge0_neg" else 1
        out = X.mask(X == 0, val)
        return out.loc[:, out.nunique() > 1]
    raise ValueError(f"unknown mode: {mode}")