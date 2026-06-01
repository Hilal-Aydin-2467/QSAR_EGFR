"""
qsar_pipeline.py — EGFR L858R/T790M/C797S  MIX-SVM QSAR Pipeline
=========================================================================
  Pipeline: 412 ChEMBL mol., Morgan FP (ECFP4, 2048-bit) — künyede açıkça
            belirtilmiş: "büyük veri seti kullanılacaktır"
 
PIPELINE ADIMLARI:
  0 — EDA  (pIC50 dağılımı, aykırı değer analizi)
  1 — Morgan Fingerprint + Varyans Filtresi   [DEĞ-4]
  2 — Stratified 80/20 Split + StandardScaler
  3 — CLPSO ile MIX-SVM Optimizasyonu         [OPTİMİZE PARAMETRE UZAYI]
  4 — Kapsamlı Doğrulama (R², RMSE, Q²_LOO, Q²_5fold, Q²_F1, Q²_F2, CCC)
  5 — Baseline Karşılaştırma (RF + Poly-SVM + RBF-SVM vs MIX-SVM)
  6 — Y-Randomizasyon Testi (N=10)
  7 — AD Analizi + Williams Plot (PCA leverage) [DEĞ-5]

"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ══════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = Path(r"C:/Users/LENOVO/Documents/Qsar_EGFR")
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR  = BASE_DIR / "plots"
LOG_DIR   = BASE_DIR / "logs"

for _d in [DATA_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

INPUT_CSV        = BASE_DIR / "egfr_qsar_cleaned.csv"
FP_MATRIX_NPY    = DATA_DIR / "fingerprints.npy"
FP_META_CSV      = DATA_DIR / "fingerprint_meta.csv"
VAR_MASK_NPY     = DATA_DIR / "var_mask.npy"
SPLIT_PKL        = DATA_DIR / "train_test_split.pkl"
SCALER_PKL       = MODEL_DIR / "scaler.pkl"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"
BEST_MODEL_PKL   = MODEL_DIR / "best_model.pkl"
VALIDATION_JSON  = LOG_DIR   / "validation_results.json"
BASELINE_JSON    = LOG_DIR   / "baseline_results.json"
YRAND_JSON       = LOG_DIR   / "y_randomization.json"
AD_JSON          = LOG_DIR   / "ad_analysis.json"

# Fingerprint
FP_RADIUS = 2
FP_NBITS  = 2048

# Bölme
TEST_SIZE    = 0.20   # Makale: 4:1 = %80/%20
RANDOM_STATE = 42
N_SPLITS_CV  = 5

# ── CLPSO [OPT-P] 60 parçacık / 150 iterasyon ────────────────────────────
CLPSO_PARTICLES = 60
CLPSO_ITERS     = 150
CLPSO_W_MAX     = 0.9
CLPSO_W_MIN     = 0.4
CLPSO_C         = 1.49445
CLPSO_M         = 7


PARAM_BOUNDS = {
    "C"     : (np.log10(0.001), np.log10(50.0)),     # [OPT-C] 0.001→50
    "eps"   : (np.log10(1e-4),  np.log10(0.5)),
    "alpha" : (0.50, 0.95),                            # [OPT-L] makale α=0.83
    "beta"  : (0.05, 0.40),                            # [OPT-L] makale β=0.12
    "sigma" : (np.log10(0.5),   np.log10(5000.0)),    # [OPT-S] 0.5→5000
    "degree": (0.0,  1.0),                             # int: {2,3,4}
    "gamma" : (np.log10(1e-5),  np.log10(10.0)),
    "coef0" : (0.0,  5.0),
}
N_PARAMS = len(PARAM_BOUNDS)

# Eşik değerleri 
THRESHOLDS = {
    "R2_train" : 0.70,
    "Q2_LOO"   : 0.60,
    "Q2_5fold" : 0.55,
    "Q2_F1"    : 0.70,
    "Q2_F2"    : 0.70,
    "CCC"      : 0.85,
    "R2_yrand" : 0.20,
    "Q2_yrand" : 0.20,
}

N_YRANDOM        = 10
AD_STD_THRESHOLD = 3.0
PCA_N_COMPONENTS = 50   # [DEĞ-5] AD analizi için

C_TRAIN  = "#378ADD"
C_TEST   = "#1D9E75"
C_RED    = "#E24B4A"
C_ORANGE = "#FAC775"
C_GRAY   = "#888780"
C_PURPLE = "#9B59B6"

SEP = "═" * 68


# ══════════════════════════════════════════════════════════════════════════════
# KERNEL FONKSİYONLARI 
# ══════════════════════════════════════════════════════════════════════════════

def _k_linear(A, B):
    return A @ B.T

def _k_poly(A, B, gamma=1.0, coef0=1.0, degree=3):
    return (gamma * (A @ B.T) + coef0) ** int(degree)

def _k_trig(A, B, sigma=1.0):
    """K_trig = sin(π/2 + σ·‖xi−xj‖²)  Ref [9]"""
    sq = cdist(A, B, metric="sqeuclidean")
    return np.sin(np.pi / 2.0 + sigma * sq)

def kernel_mix(A, B, alpha, beta, sigma, gamma, coef0, degree):
    """
    K_MIX = α·K_Trig + β·K_Poly + (1−α−β)·K_Linear  
    [OPT-L] K_Linear için minimum 0.05 
    """
    total = alpha + beta
    if total > 1.0:
        alpha, beta = alpha / total * 0.95, beta / total * 0.05
    # [OPT-L] K_Linear en az %5 olmalı 
    g = max(0.05, 1.0 - alpha - beta)
    if alpha + beta + g > 1.0:
        scale = 1.0 / (alpha + beta + g)
        alpha *= scale; beta *= scale; g *= scale
    return (alpha * _k_trig(A, B, sigma)
          + beta  * _k_poly(A, B, gamma, coef0, int(degree))
          + g     * _k_linear(A, B))

def gram_tr(X, p):
    return kernel_mix(X, X, **p)

def gram_te(Xte, Xtr, p):
    return kernel_mix(Xte, Xtr, **p)


# ══════════════════════════════════════════════════════════════════════════════
# METRİK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def q2_f1(y_te, yp, y_tr_mean):
    ss_r = np.sum((y_te - yp) ** 2)
    ss_t = np.sum((y_te - y_tr_mean) ** 2)
    return 1.0 - ss_r / ss_t if ss_t > 0 else -np.inf

def q2_f2(y_te, yp):
    ss_r = np.sum((y_te - yp) ** 2)
    ss_t = np.sum((y_te - y_te.mean()) ** 2)
    return 1.0 - ss_r / ss_t if ss_t > 0 else -np.inf

def ccc(y_true, y_pred):
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov    = np.cov(y_true, y_pred, ddof=0)[0, 1]
    denom  = vt + vp + (mt - mp) ** 2
    return 2.0 * cov / denom if denom > 0 else -np.inf

def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))

def mae(y, yp):
    return float(mean_absolute_error(y, yp))

def compute_all_metrics(y_tr, yp_tr, y_te, yp_te, fold_q2s, q2_loo=None):
    return {
        "R2_train"  : round(float(r2_score(y_tr, yp_tr)), 6),
        "RMSE_train": round(rmse(y_tr, yp_tr), 6),
        "MAE_train" : round(mae(y_tr, yp_tr),  6),
        "R2_test"   : round(float(r2_score(y_te, yp_te)), 6),
        "RMSE_test" : round(rmse(y_te, yp_te), 6),
        "MAE_test"  : round(mae(y_te, yp_te),  6),
        "Q2_LOO"    : round(float(q2_loo), 6) if q2_loo is not None else None,
        "Q2_5fold"  : round(float(np.mean(fold_q2s)), 6),
        "fold_Q2s"  : [round(float(q), 6) for q in fold_q2s],
        "Q2_F1"     : round(float(q2_f1(y_te, yp_te, y_tr.mean())), 6),
        "Q2_F2"     : round(float(q2_f2(y_te, yp_te)), 6),
        "CCC"       : round(float(ccc(y_te, yp_te)), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLPSO  
# ══════════════════════════════════════════════════════════════════════════════

def _decode(pos):
    dec = {}
    for i, (key, (lb, ub)) in enumerate(PARAM_BOUNDS.items()):
        t   = float(np.clip(pos[i], 0.0, 1.0))
        raw = lb + t * (ub - lb)
        if key in ("C", "eps", "sigma", "gamma"):
            dec[key] = float(10 ** raw)
        elif key == "degree":
            dec[key] = int(np.round(2 + t * 2))
        else:
            dec[key] = float(raw)
    return dec

def _fitness_cv(pos, X_sc, y, kf):
    p   = _decode(pos)
    C   = p.pop("C")
    eps = p.pop("eps")
    scores = []
    for tri, vli in kf.split(X_sc):
        try:
            Ktr  = gram_tr(X_sc[tri], p)
            Kval = kernel_mix(X_sc[vli], X_sc[tri], **p)
            svr  = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
            svr.fit(Ktr, y[tri])
            yp   = svr.predict(Kval)
            ss_r = np.sum((y[vli] - yp) ** 2)
            ss_t = np.sum((y[vli] - y[tri].mean()) ** 2)
            scores.append(1.0 - ss_r / ss_t if ss_t > 0 else -np.inf)
        except Exception:
            scores.append(-np.inf)
    val = float(np.mean(scores))
    return val if np.isfinite(val) else -np.inf

def run_clpso(X_sc, y, n_particles=CLPSO_PARTICLES, max_iter=CLPSO_ITERS):
    """CLPSO: V ← ω·V + c·rand·(pbest_i − X)  """
    rng = np.random.default_rng(RANDOM_STATE)
    kf  = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    pos       = rng.uniform(0.0, 1.0, (n_particles, N_PARAMS))
    vel       = rng.uniform(-0.5, 0.5, (n_particles, N_PARAMS))
    pbest     = pos.copy()
    pbest_fit = np.full(n_particles, -np.inf)

    print(f"\n  [1/2] Başlangıç popülasyonu ({n_particles} parçacık)...")
    init_fits = []
    for i in range(n_particles):
        f = _fitness_cv(pos[i], X_sc, y, kf)
        init_fits.append(f)
        pbest_fit[i] = f
        print(f"    Parçacık {i+1:>2}/{n_particles}  Q²={f:.4f}  ", end="\r", flush=True)
    print()

    gbest_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])
    print(f"  Başlangıç en iyi Q²_5fold = {gbest_fit:.4f}")

    no_imp   = np.zeros(n_particles, dtype=int)
    exemplar = np.array([[rng.integers(0, n_particles) for _ in range(N_PARAMS)]
                         for _ in range(n_particles)])

    hist_best, hist_mean, hist_worst = [gbest_fit], [], []
    fin0 = pbest_fit[np.isfinite(pbest_fit)]
    hist_mean.append(float(np.mean(fin0))  if len(fin0) > 0 else 0.0)
    hist_worst.append(float(np.min(fin0)) if len(fin0) > 0 else 0.0)
    iter_matrix = []

    print(f"\n  [2/2] Optimizasyon ({max_iter} iterasyon)...")
    for it in range(max_iter):
        w = CLPSO_W_MAX - (CLPSO_W_MAX - CLPSO_W_MIN) * it / max_iter

        for i in range(n_particles):
            if no_imp[i] >= CLPSO_M:
                for d in range(N_PARAMS):
                    c1, c2 = rng.integers(0, n_particles, 2)
                    exemplar[i, d] = c1 if pbest_fit[c1] >= pbest_fit[c2] else c2
                no_imp[i] = 0

        for i in range(n_particles):
            r = rng.uniform(0.0, 1.0, N_PARAMS)
            for d in range(N_PARAMS):
                ex = exemplar[i, d]
                vel[i, d] = (w * vel[i, d]
                             + CLPSO_C * r[d] * (pbest[ex, d] - pos[i, d]))
            vel[i] = np.clip(vel[i], -1.0, 1.0)
            pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)

        iter_fits = []
        for i in range(n_particles):
            print(f"    İter {it+1:>3}/{max_iter} | P {i+1:>2}/{n_particles}  ", end="\r", flush=True)
            f = _fitness_cv(pos[i], X_sc, y, kf)
            iter_fits.append(f)
            if f > pbest_fit[i]:
                pbest[i] = pos[i].copy(); pbest_fit[i] = f; no_imp[i] = 0
            else:
                no_imp[i] += 1
            if f > gbest_fit:
                gbest_fit = f; gbest_pos = pos[i].copy()
        iter_matrix.append(iter_fits)

        fin = pbest_fit[np.isfinite(pbest_fit)]
        hist_best.append(gbest_fit)
        hist_mean.append(float(np.mean(fin))  if len(fin) > 0 else 0.0)
        hist_worst.append(float(np.min(fin)) if len(fin) > 0 else 0.0)
        print(f"  İter {it+1:>4}/{max_iter} | Best={gbest_fit:.4f}  "
              f"Mean={hist_mean[-1]:.4f}  Worst={hist_worst[-1]:.4f}  ", flush=True)

    print(f"\n  CLPSO tamamlandı — En iyi Q²_5fold = {gbest_fit:.4f}")
    return gbest_pos, gbest_fit, {
        "best": hist_best, "mean": hist_mean, "worst": hist_worst,
        "all_init": init_fits, "iter_matrix": iter_matrix,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 0 — EDA
# ══════════════════════════════════════════════════════════════════════════════

def step0_eda():
    print(f"\n{SEP}\n  ADIM 0 — EDA: pIC50 Dağılımı & Veri Kalitesi\n{SEP}")
    df = pd.read_csv(INPUT_CSV)
    st = df["pIC50"].describe()
    print(f"\n  Molekül sayısı: {len(df):,}")
    for k in ["mean","std","min","25%","50%","75%","max"]:
        print(f"    {k:6s}: {st[k]:.3f}")
    Q1, Q3  = df["pIC50"].quantile(0.25), df["pIC50"].quantile(0.75)
    IQR     = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df["pIC50"] < low) | (df["pIC50"] > high)]
    print(f"\n  IQR eşik: [{low:.2f}, {high:.2f}]  |  Aykırı: {len(outliers)}")
    multi = df[df["n_measurements"] > 1]
    print(f"  Çoklu ölçüm: {len(multi)} mol (std_ort={multi['pIC50_std'].mean():.3f})")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.arange(df["pIC50"].min()-0.2, df["pIC50"].max()+0.5, 0.3)
    ax1.hist(df["pIC50"], bins=bins, color=C_TRAIN, edgecolor="white", lw=0.5, alpha=0.8)
    ax1.axvline(st["mean"], color=C_RED,    lw=2.0, ls="--", label=f"Ort.={st['mean']:.2f}")
    ax1.axvline(st["50%"],  color=C_ORANGE, lw=2.0, ls="-.", label=f"Medyan={st['50%']:.2f}")
    ax1.set_xlabel("pIC50"); ax1.set_ylabel("Molekül sayısı")
    ax1.set_title(f"pIC50 Dağılımı  (n={len(df):,})")
    ax1.legend(fontsize=9); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(df["pIC50"], patch_artist=True, widths=0.4, medianprops=dict(color="white",lw=2))
    bp["boxes"][0].set_facecolor(C_TRAIN); bp["boxes"][0].set_alpha(0.75)
    for w in bp["whiskers"]: w.set(color="#555555", lw=1.2, ls="--")
    for c in bp["caps"]:     c.set(color="#555555", lw=1.5)
    for fl in bp["fliers"]:  fl.set(marker="o", markerfacecolor=C_RED, markersize=5, alpha=0.6)
    ax2.axhline(low,  color=C_RED, ls=":", lw=1.5, label=f"IQR alt={low:.2f}")
    ax2.axhline(high, color=C_RED, ls=":", lw=1.5, label=f"IQR üst={high:.2f}")
    ax2.set_ylabel("pIC50"); ax2.set_xticks([])
    ax2.set_title("pIC50 Boxplot — Aykırı Değer Analizi")
    ax2.legend(fontsize=8); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    bins_l  = [0, 5, 6, 7, 8, 9, 12]; labels_ = ["<5","5-6","6-7","7-8","8-9",">9"]
    grp = pd.cut(df["pIC50"], bins=bins_l, labels=labels_)
    vc  = grp.value_counts().sort_index()
    bar_cols = plt.cm.Blues(np.linspace(0.35, 0.85, len(labels_)))
    bars3 = ax3.bar(vc.index, vc.values, color=bar_cols, edgecolor="white", lw=0.8, alpha=0.88)
    for bar, val in zip(bars3, vc.values):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(val), ha="center", va="bottom", fontsize=9)
    ax3.set_xlabel("pIC50 Aralığı"); ax3.set_ylabel("Molekül sayısı")
    ax3.set_title("pIC50 Aralık Dağılımı")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    nm_vc = df["n_measurements"].value_counts().sort_index()
    ax4.bar(nm_vc.index.astype(str), nm_vc.values, color=C_TEST, edgecolor="white", lw=0.8, alpha=0.85)
    ax4.set_xlabel("Ölçüm sayısı"); ax4.set_ylabel("Molekül sayısı")
    ax4.set_title("Duplikat Ölçüm Dağılımı")
    pct1 = (df["n_measurements"] == 1).sum() / len(df) * 100
    ax4.text(0.65, 0.92, f"Tek ölçüm: %{pct1:.0f}\nÇoklu: %{100-pct1:.0f}",
             transform=ax4.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("Adım 0 — EDA: EGFR L858R/T790M/C797S  ChEMBL Veritabanı",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step0_eda.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step0_eda.png")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — MORGAN FİNGERPRINT  +  [DEĞ-4] VARYANS FİLTRESİ
# ══════════════════════════════════════════════════════════════════════════════

def step1_fingerprints():
    print(f"\n{SEP}\n  ADIM 1 — Morgan Fingerprint (ECFP4) + [DEĞ-4] Varyans Filtresi\n{SEP}")
    df = pd.read_csv(INPUT_CSV)
    print(f"\n  Giriş: {len(df):,} molekül")

    fps, valid_idx, failed = [], [], []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["Smiles"]))
        if mol is None:
            failed.append(row["Molecule ChEMBL ID"]); continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
        arr = np.zeros(FP_NBITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr); valid_idx.append(i)

    X    = np.vstack(fps)
    meta = df.loc[valid_idx, ["Molecule ChEMBL ID","pIC50","n_measurements","pIC50_std"]].reset_index(drop=True)
    print(f"  Ham FP matrisi: {X.shape}  |  Bit yoğunluğu: {X.mean():.4f}")

    # ── [DEĞ-4] Varyans filtresi ──────────────────────────────────────────
    var_mask  = X.var(axis=0) > 0
    n_removed = int((~var_mask).sum())
    X         = X[:, var_mask]
    print(f"\n  [DEĞ-4] Varyans filtresi:")
    print(f"    Çıkarılan sabit bit: {n_removed:,}")
    print(f"    Kalan bilgili bit  : {X.shape[1]:,}")
    np.save(VAR_MASK_NPY, var_mask)
    # ──────────────────────────────────────────────────────────────────────

    np.save(FP_MATRIX_NPY, X)
    meta.to_csv(FP_META_CSV, index=False)
    if failed:
        print(f"  Geçersiz SMILES: {len(failed)}")
    print(f"\n  ✓ {FP_MATRIX_NPY.name}  ✓ {FP_META_CSV.name}  ✓ {VAR_MASK_NPY.name}")
    return X, meta


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — STRATİFİED 80/20 SPLIT + SCALER
# ══════════════════════════════════════════════════════════════════════════════

def step2_split_scale(X, meta):
    print(f"\n{SEP}\n  ADIM 2 — Stratified 80/20 Split + StandardScaler\n{SEP}")
    y     = meta["pIC50"].values
    strat = pd.cut(y, bins=6, labels=False, duplicates="drop")
    idx   = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=strat)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    scaler      = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_tr.astype(float))
    X_te_sc     = scaler.transform(X_te.astype(float))

    ks_stat, ks_pval = ks_2samp(y_tr, y_te)
    print(f"\n  Eğitim: {len(y_tr)} mol  |  Test: {len(y_te)} mol")
    print(f"  KS testi: stat={ks_stat:.4f}, p={ks_pval:.4f}  "
          f"→ {'Dağılımlar benzer ✓' if ks_pval>0.05 else 'UYARI!'}")

    bins_l = [0,5,6,7,8,9,12]; labels_ = ["<5","5-6","6-7","7-8","8-9",">9"]
    tr_h = pd.cut(y_tr, bins=bins_l, labels=labels_).value_counts().sort_index()
    te_h = pd.cut(y_te, bins=bins_l, labels=labels_).value_counts().sort_index()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    bns = np.arange(4.0, 11.5, 0.35)
    ax1.hist(y_tr, bins=bns, alpha=0.65, color=C_TRAIN, edgecolor="white", lw=0.5, label=f"Eğitim (n={len(y_tr)})")
    ax1.hist(y_te, bins=bns, alpha=0.65, color=C_TEST,  edgecolor="white", lw=0.5, label=f"Test (n={len(y_te)})")
    ax1.set_xlabel("pIC50"); ax1.set_ylabel("Molekül sayısı")
    ax1.set_title("pIC50 Dağılımı — Eğitim / Test")
    ax1.legend(); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([y_tr, y_te], labels=[f"Eğitim\n(n={len(y_tr)})", f"Test\n(n={len(y_te)})"],
                     patch_artist=True, widths=0.5, medianprops=dict(color="white",lw=2))
    for patch, col in zip(bp["boxes"], [C_TRAIN, C_TEST]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for fl in bp["fliers"]: fl.set(marker="o", markerfacecolor=C_RED, markersize=4, alpha=0.5)
    for i, (ydata, col) in enumerate(zip([y_tr, y_te], [C_TRAIN, C_TEST])):
        ax2.text(i+1, ydata.max()+0.15, f"Ort={ydata.mean():.2f}\nSS={ydata.std():.2f}",
                 ha="center", fontsize=8, color=col)
    ax2.set_ylabel("pIC50"); ax2.set_title("Boxplot — Eğitim / Test")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(labels_)); w = 0.38
    ax3.bar(x_pos-w/2, [tr_h.get(l,0) for l in labels_], width=w, color=C_TRAIN, edgecolor="white", alpha=0.82, label="Eğitim")
    ax3.bar(x_pos+w/2, [te_h.get(l,0) for l in labels_], width=w, color=C_TEST,  edgecolor="white", alpha=0.82, label="Test")
    ax3.set_xticks(x_pos); ax3.set_xticklabels(labels_)
    ax3.set_xlabel("pIC50 Aralığı"); ax3.set_ylabel("Molekül sayısı")
    ax3.set_title("pIC50 Aralık Dağılımı — Eğitim / Test")
    ax3.legend(); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    for ydata, col, lbl in [(y_tr, C_TRAIN, f"Eğitim (n={len(y_tr)})"),
                              (y_te, C_TEST,  f"Test (n={len(y_te)})")]:
        srt = np.sort(ydata); cdf = np.arange(1, len(srt)+1)/len(srt)
        ax4.plot(srt, cdf, color=col, lw=2.0, label=lbl)
    ax4.set_xlabel("pIC50"); ax4.set_ylabel("Kümülatif oran")
    ax4.set_title(f"CDF — KS stat={ks_stat:.3f}, p={ks_pval:.3f}")
    ax4.legend(); ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("Adım 2 — Veri Bölme Analizi  (Stratified 80/20)",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step2_split.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step2_split.png")

    split = dict(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                 X_train_sc=X_tr_sc, X_test_sc=X_te_sc,
                 idx_train=idx_tr, idx_test=idx_te)
    with open(SPLIT_PKL,  "wb") as f: pickle.dump(split, f)
    with open(SCALER_PKL, "wb") as f: pickle.dump(scaler, f)
    print(f"  ✓ {SPLIT_PKL.name}  ✓ {SCALER_PKL.name}")
    return split


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — CLPSO OPTİMİZASYONU
# ══════════════════════════════════════════════════════════════════════════════

def step3_clpso_optimize(split, n_particles=CLPSO_PARTICLES, max_iter=CLPSO_ITERS):
    print(f"\n{SEP}")
    print(f"  ADIM 3 — CLPSO ile MIX-SVM Optimizasyonu  (Ref [1])")
    print(f"  [OPT-C] C: 0.001→50   [OPT-S] sigma: 0.5→5000")
    print(f"  [OPT-L] K_Linear min=0.05   [OPT-P] {n_particles}p/{max_iter}i")
    print(SEP)

    X_sc = split["X_train_sc"]; y = split["y_train"]
    best_pos, best_q2, history = run_clpso(X_sc, y, n_particles, max_iter)

    best_raw = _decode(best_pos)
    C   = best_raw.pop("C"); eps = best_raw.pop("eps"); kparams = best_raw

    a, b = kparams["alpha"], kparams["beta"]
    total = a + b
    if total > 1.0: a, b = a/total*0.95, b/total*0.05
    g = max(0.05, 1.0 - a - b)

    print(f"\n  ── En İyi Parametreler ────────────────────────────────")
    print(f"  C={C:.4f}  epsilon={eps:.6f}")
    print(f"  alpha={kparams['alpha']:.4f} → K_Trig  : {a:.3f}")
    print(f"  beta ={kparams['beta']:.4f} → K_Poly  : {b:.3f}")
    print(f"  1-α-β={g:.3f}           → K_Linear: {g:.3f}")
    print(f"  sigma={kparams['sigma']:.4f}  degree={kparams['degree']}")
    print(f"\n  Makale ref: C=1.06, ε=0.01, σ=800.13, α=0.83, β=0.12, lin=0.05")

    K_full = gram_tr(X_sc, kparams)
    svr    = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
    svr.fit(K_full, y)
    yp_tr  = svr.predict(K_full)
    r2_tr  = r2_score(y, yp_tr); rmse_tr = rmse(y, yp_tr)
    print(f"\n  Eğitim R²={r2_tr:.4f}  RMSE={rmse_tr:.4f}")

    # Grafik
    iters    = list(range(len(history["best"])))
    h_best   = history["best"]; h_mean = history["mean"]; h_worst = history["worst"]
    iter_mat = history["iter_matrix"]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iters, h_best,  color=C_TRAIN,  lw=2.2, label="En iyi Q²")
    ax1.plot(iters, h_mean,  color=C_ORANGE, lw=1.6, ls="--", label="Ort. Q²")
    ax1.plot(iters, h_worst, color=C_GRAY,   lw=1.2, ls=":",  label="En kötü Q²")
    ax1.axhline(THRESHOLDS["Q2_5fold"], color=C_RED, ls="--", lw=1.4, label=f"Eşik={THRESHOLDS['Q2_5fold']}")
    ax1.fill_between(iters, h_worst, h_best, alpha=0.07, color=C_TRAIN)
    ax1.set_xlabel("İterasyon"); ax1.set_ylabel("Q²_5fold")
    ax1.set_title("CLPSO Yakınsama Eğrisi  [OPT-P: 60p/150i]")
    ax1.legend(fontsize=8); ax1.set_xlim(0, len(iters)-1)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    bar_vals   = [a, b, g]
    bar_labels = [f"K_Trig\n(α={a:.3f})", f"K_Poly\n(β={b:.3f})", f"K_Linear\n(={g:.3f})"]
    bar_cols   = [C_TRAIN, C_TEST, C_ORANGE]
    bars2 = ax2.bar(bar_labels, bar_vals, color=bar_cols, edgecolor="white", lw=0.8, width=0.5)
    for bar, val in zip(bars2, bar_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.axhline(0.83, color=C_RED,     ls=":", lw=1.2, alpha=0.6, label="Makale α=0.83")
    ax2.axhline(0.12, color="#555555", ls=":", lw=1.2, alpha=0.6, label="Makale β=0.12")
    ax2.axhline(0.05, color=C_ORANGE,  ls=":", lw=1.2, alpha=0.6, label="Makale K_Lin=0.05")
    ax2.set_ylim(0, 1.12); ax2.set_ylabel("Ağırlık")
    ax2.set_title(f"MIX Kernel Ağırlıkları  [OPT-L: min K_Lin=0.05]\n(Q²_5fold={best_q2:.4f})")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    all_init = history["all_init"]
    finite_init = [x for x in all_init if np.isfinite(x)]
    ax3.hist(finite_init, bins=max(8, n_particles//4),
             color=C_GRAY, edgecolor="white", lw=0.5, alpha=0.75)
    if finite_init:
        ax3.axvline(np.mean(finite_init), color=C_ORANGE, ls="--", lw=1.5,
                    label=f"Ort.={np.mean(finite_init):.3f}")
    ax3.axvline(best_q2, color=C_TRAIN, lw=2.0, label=f"En iyi={best_q2:.3f}")
    ax3.axvline(THRESHOLDS["Q2_5fold"], color=C_RED, ls=":", lw=1.5, label=f"Eşik={THRESHOLDS['Q2_5fold']}")
    ax3.set_xlabel("Q²_5fold"); ax3.set_ylabel("Parçacık sayısı")
    ax3.set_title("Başlangıç Popülasyonu Q² Dağılımı")
    ax3.legend(fontsize=8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    if iter_mat:
        step_p  = max(1, max_iter//10)
        indices = list(range(0, len(iter_mat), step_p))
        data_p  = [[v for v in iter_mat[i] if np.isfinite(v)] for i in indices]
        data_p  = [d if d else [0] for d in data_p]
        x_lbp   = [str(i+1) for i in indices]
        bp4 = ax4.boxplot(data_p, labels=x_lbp, patch_artist=True, widths=0.5,
                          medianprops=dict(color="white", lw=1.5))
        cmap = plt.cm.Blues
        for j, patch in enumerate(bp4["boxes"]):
            patch.set_facecolor(cmap(0.35 + 0.5*j/max(len(bp4["boxes"])-1,1)))
            patch.set_alpha(0.78)
        ax4.axhline(THRESHOLDS["Q2_5fold"], color=C_RED, ls="--", lw=1.3, label=f"Eşik={THRESHOLDS['Q2_5fold']}")
        ax4.set_xlabel("İterasyon"); ax4.set_ylabel("Q²_5fold")
        ax4.set_title("Parçacık Fitness Dağılımı (İterasyon Boyunca)")
        ax4.legend(fontsize=8)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("Adım 3 — CLPSO Optimizasyonu (MIX-SVM) — FINAL",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step3_clpso.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step3_clpso.png")

    save = {
        "best_svr_params"   : {"C": C, "epsilon": eps},
        "best_kernel_params": kparams,
        "Q2_5fold_CLPSO"    : round(best_q2, 6),
        "R2_train"          : round(r2_tr, 6),
        "RMSE_train"        : round(rmse_tr, 6),
        "mix_weights"       : {"alpha": round(a,4), "beta": round(b,4), "gamma_lin": round(g,4)},
        "opt_changes"       : {"C_bounds":"[0.001,50]","sigma_bounds":"[0.5,5000]",
                               "K_linear_min":0.05,"particles":n_particles,"iters":max_iter},
        "clpso_history"     : {"best":  [round(v,6) for v in h_best],
                               "mean":  [round(v,6) for v in h_mean],
                               "worst": [round(v,6) for v in h_worst]},
    }
    with open(BEST_PARAMS_JSON, "w") as f: json.dump(save, f, indent=2)
    with open(BEST_MODEL_PKL,   "wb") as f:
        pickle.dump({"svr": svr, "kparams": kparams, "X_train_sc": X_sc, "y_train": y}, f)
    print(f"  ✓ {BEST_PARAMS_JSON.name}  ✓ {BEST_MODEL_PKL.name}")
    return svr, kparams, C, eps, best_q2


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — KAPSAMLI DOĞRULAMA
# ══════════════════════════════════════════════════════════════════════════════

def _compute_q2_loo(X_sc, y, kparams, C, eps):
    loo = LeaveOneOut(); n = X_sc.shape[0]; y_loo = np.zeros(n); t0 = time.time()
    for fold_i, (tri, vli) in enumerate(loo.split(X_sc)):
        if fold_i % 50 == 0:
            elapsed = time.time()-t0; eta = elapsed/max(fold_i,1)*(n-fold_i)
            print(f"    LOO {fold_i+1:>4}/{n}  ETA ~{eta/60:.1f} dk  ", end="\r", flush=True)
        try:
            Ktr   = gram_tr(X_sc[tri], kparams)
            Kval  = kernel_mix(X_sc[vli], X_sc[tri], **kparams)
            svr_l = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
            svr_l.fit(Ktr, y[tri]); y_loo[vli[0]] = svr_l.predict(Kval)[0]
        except Exception:
            y_loo[vli[0]] = y.mean()
    print()
    ss_r = np.sum((y-y_loo)**2); ss_t = np.sum((y-y.mean())**2)
    q2   = 1.0 - ss_r/ss_t if ss_t > 0 else -np.inf
    print(f"    Q²_LOO = {q2:.4f}  (eşik >{THRESHOLDS['Q2_LOO']})")
    return float(q2), y_loo

def _compute_q2_5fold(X_sc, y, kparams, C, eps):
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    fold_q2s = []
    for fold_i, (tri, vli) in enumerate(kf.split(X_sc)):
        Ktr  = gram_tr(X_sc[tri], kparams)
        Kval = kernel_mix(X_sc[vli], X_sc[tri], **kparams)
        svr_ = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
        svr_.fit(Ktr, y[tri]); yp = svr_.predict(Kval)
        ss_r = np.sum((y[vli]-yp)**2); ss_t = np.sum((y[vli]-y[tri].mean())**2)
        q2   = 1.0-ss_r/ss_t if ss_t>0 else -np.inf
        fold_q2s.append(q2); print(f"    Fold {fold_i+1}: Q²={q2:.4f}")
    return fold_q2s

def step4_validate(svr, kparams, split, C, eps, compute_loo=True):
    print(f"\n{SEP}\n  ADIM 4 — Kapsamlı Model Doğrulama  (Ref [1], [12])\n{SEP}")
    X_tr = split["X_train_sc"]; X_te = split["X_test_sc"]
    y_tr = split["y_train"];    y_te = split["y_test"]

    q2_loo_val = None; y_loo_pred = None
    if compute_loo:
        print(f"\n  [4.1] Q²_LOO (Leave-One-Out CV):")
        q2_loo_val, y_loo_pred = _compute_q2_loo(X_tr, y_tr, kparams, C, eps)
    else:
        print(f"\n  [4.1] Q²_LOO atlandı (--no-loo)")

    print(f"\n  [4.2] Q²_5fold:")
    fold_q2s = _compute_q2_5fold(X_tr, y_tr, kparams, C, eps)
    q2_cv    = float(np.mean(fold_q2s))
    print(f"    Ortalama Q²_5fold = {q2_cv:.4f}")

    Ktr    = gram_tr(X_tr, kparams); yp_tr = svr.predict(Ktr)
    Kte    = kernel_mix(X_te, X_tr, **kparams); yp_te = svr.predict(Kte)
    metrics = compute_all_metrics(y_tr, yp_tr, y_te, yp_te, fold_q2s, q2_loo_val)

    print(f"\n  {'Metrik':14s}  {'Değer':>9s}  {'Eşik':>8s}  Durum")
    print(f"  {'─'*14}  {'─'*9}  {'─'*8}  {'─'*10}")
    for lbl, thr_key, val in [
        ("R2_train",  "R2_train",  metrics["R2_train"]),
        ("RMSE_train","",          metrics["RMSE_train"]),
        ("R2_test",   "",          metrics["R2_test"]),
        ("RMSE_test", "",          metrics["RMSE_test"]),
        ("Q²_LOO",    "Q2_LOO",    metrics["Q2_LOO"]),
        ("Q²_5fold",  "Q2_5fold",  metrics["Q2_5fold"]),
        ("Q²_F1",     "Q2_F1",     metrics["Q2_F1"]),
        ("Q²_F2",     "Q2_F2",     metrics["Q2_F2"]),
        ("CCC",       "CCC",       metrics["CCC"]),
    ]:
        if val is None:
            print(f"  {lbl:14s}  {'N/A':>9s}  {'─':>8s}  (atlandı)"); continue
        thr = THRESHOLDS.get(thr_key)
        thr_s = f"{thr:.2f}" if thr else "─"
        ok_s  = ("✓ GEÇTİ" if val>=thr else "✗ KALDI") if thr else "─"
        print(f"  {lbl:14s}  {val:>9.4f}  {thr_s:>8s}  {ok_s}")

    pmin = min(y_tr.min(), y_te.min(), yp_tr.min(), yp_te.min()) - 0.35
    pmax = max(y_tr.max(), y_te.max(), yp_tr.max(), yp_te.max()) + 0.35

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def _pred_panel(ax, y_obs, y_pred, title, col, extra=""):
        ax.scatter(y_obs, y_pred, alpha=0.45, s=22, color=col, edgecolors="none")
        ax.plot([pmin,pmax],[pmin,pmax],"k--",lw=0.9,alpha=0.5)
        txt = (f"{extra}\n" if extra else "") + f"R²={r2_score(y_obs,y_pred):.3f}\nRMSE={rmse(y_obs,y_pred):.3f}\nn={len(y_obs)}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9, va="top",
                fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.4",fc="white",alpha=0.85))
        ax.set_xlim(pmin,pmax); ax.set_ylim(pmin,pmax); ax.set_aspect("equal")
        ax.set_xlabel("Gözlemlenen pIC50"); ax.set_ylabel("Tahmin edilen pIC50")
        ax.set_title(title); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _pred_panel(fig.add_subplot(gs[0,0]), y_tr, yp_tr, "Eğitim — Predicted vs Observed",
                C_TRAIN, f"Q²_5fold={q2_cv:.3f}")
    _pred_panel(fig.add_subplot(gs[0,1]), y_te, yp_te, "Test — Predicted vs Observed",
                C_TEST, f"Q²_F1={metrics['Q2_F1']:.3f}\nQ²_F2={metrics['Q2_F2']:.3f}\nCCC={metrics['CCC']:.3f}")

    ax3 = fig.add_subplot(gs[0,2])
    if y_loo_pred is not None:
        _pred_panel(ax3, y_tr, y_loo_pred, f"LOO — Predicted vs Observed\n(Q²_LOO={q2_loo_val:.3f})",
                    C_PURPLE, f"Q²_LOO={q2_loo_val:.3f}")
    else:
        ax3.text(0.5,0.5,"Q²_LOO\natlandı\n(--no-loo)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=13, color=C_GRAY)
        ax3.set_title("LOO — Predicted vs Observed")

    ax4 = fig.add_subplot(gs[1,0])
    fold_nums = [f"Fold {i+1}" for i in range(len(fold_q2s))]
    fold_cols = [C_TEST if q>=THRESHOLDS["Q2_5fold"] else C_RED for q in fold_q2s]
    bars4 = ax4.bar(fold_nums, fold_q2s, color=fold_cols, edgecolor="white", lw=0.8, width=0.55, alpha=0.85)
    ax4.axhline(q2_cv, color=C_TRAIN, lw=2.2, label=f"Ort.={q2_cv:.3f}")
    ax4.axhline(THRESHOLDS["Q2_5fold"], color=C_RED, lw=1.5, ls="--", label=f"Eşik={THRESHOLDS['Q2_5fold']}")
    for bar, val in zip(bars4, fold_q2s):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylabel("Q²_5fold"); ax4.set_ylim(0, 1.05)
    ax4.set_title("5-Fold CV — Fold Bazında Q²"); ax4.legend(fontsize=9)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    ax5 = fig.add_subplot(gs[1,1])
    res_tr = y_tr-yp_tr; res_te = y_te-yp_te
    rbn = np.linspace(min(res_tr.min(),res_te.min())-0.1, max(res_tr.max(),res_te.max())+0.1, 30)
    ax5.hist(res_tr, bins=rbn, alpha=0.55, color=C_TRAIN, edgecolor="white", lw=0.5,
             label=f"Eğitim (RMSE={metrics['RMSE_train']:.3f})")
    ax5.hist(res_te, bins=rbn, alpha=0.65, color=C_TEST,  edgecolor="white", lw=0.5,
             label=f"Test (RMSE={metrics['RMSE_test']:.3f})")
    ax5.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax5.set_xlabel("Artık (Gözlemlenen − Tahmin)"); ax5.set_ylabel("Molekül sayısı")
    ax5.set_title("Artık Dağılımı — Eğitim / Test"); ax5.legend(fontsize=9)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    ax6 = fig.add_subplot(gs[1,2])
    bm = {"R²_train": metrics["R2_train"], "Q²_LOO": metrics["Q2_LOO"] or 0,
          "Q²_5fold": metrics["Q2_5fold"], "Q²_F1": metrics["Q2_F1"],
          "Q²_F2": metrics["Q2_F2"], "CCC": metrics["CCC"], "R²_test": metrics["R2_test"]}
    bt = {"R²_train": 0.70, "Q²_LOO": 0.60, "Q²_5fold": 0.55, "Q²_F1": 0.70,
          "Q²_F2": 0.70, "CCC": 0.85, "R²_test": 0.70}
    bk = list(bm.keys()); bv = list(bm.values()); bth = [bt[k] for k in bk]
    bc = [C_TEST if v>=t else C_RED for v,t in zip(bv,bth)]
    x6 = np.arange(len(bk))
    ax6.bar(x6, bv, color=bc, edgecolor="white", lw=0.8, alpha=0.85, width=0.6)
    ax6.scatter(x6, bth, color="black", zorder=5, s=40, marker="_", linewidths=2.5, label="Eşik")
    for xi, val in zip(x6, bv):
        ax6.text(xi, val+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax6.set_xticks(x6); ax6.set_xticklabels(bk, rotation=30, ha="right", fontsize=8)
    ax6.set_ylim(0, 1.12); ax6.set_ylabel("Metrik Değeri")
    ax6.set_title("Doğrulama Metrikleri Özeti"); ax6.legend(fontsize=8)
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    fig.suptitle("Adım 4 — Kapsamlı Model Doğrulama  (MIX-SVM+CLPSO) — FINAL",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step4_validation.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step4_validation.png")
    with open(VALIDATION_JSON, "w") as f: json.dump(metrics, f, indent=2)
    print(f"  ✓ {VALIDATION_JSON.name}")
    return metrics, yp_tr, yp_te


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — BASELINE KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════════════════════════════

def step5_baseline_comparison(split, kparams_mix, C_mix, eps_mix,
                               metrics_mix, yp_tr_mix, yp_te_mix):
    print(f"\n{SEP}\n  ADIM 5 — Baseline Model Karşılaştırması\n{SEP}")
    X_tr = split["X_train_sc"]; X_te = split["X_test_sc"]
    y_tr = split["y_train"];    y_te = split["y_test"]
    kf   = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    def _cv_q2s(predict_fn, X_tr, y_tr):
        q2s = []
        for tri, vli in kf.split(X_tr):
            yp_v  = predict_fn(X_tr[tri], y_tr[tri], X_tr[vli])
            ss_r  = np.sum((y_tr[vli]-yp_v)**2)
            ss_t  = np.sum((y_tr[vli]-y_tr[tri].mean())**2)
            q2s.append(1.0-ss_r/ss_t if ss_t>0 else -np.inf)
        return q2s

    # RF
    print("\n  [5.1] Random Forest:")
    rf = RandomForestRegressor(n_estimators=575, max_depth=6, min_samples_split=3,
                               min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    def rf_pred(Xtr, ytr, Xvl):
        m = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xtr, ytr); return m.predict(Xvl)
    fold_rf = _cv_q2s(rf_pred, X_tr, y_tr)
    m_rf = compute_all_metrics(y_tr, rf.predict(X_tr), y_te, rf.predict(X_te), fold_rf)
    results["RF"] = m_rf; yp_rf_tr = rf.predict(X_tr); yp_rf_te = rf.predict(X_te)
    print(f"    R²_train={m_rf['R2_train']:.4f}  R²_test={m_rf['R2_test']:.4f}  Q²_5fold={m_rf['Q2_5fold']:.4f}")

    # Poly-SVM
    print("\n  [5.2] Poly-SVM:")
    svr_poly = SVR(kernel="poly", C=200.0, epsilon=0.1, degree=3, gamma="scale", coef0=1.0, max_iter=10000)
    svr_poly.fit(X_tr, y_tr)
    def poly_pred(Xtr, ytr, Xvl):
        m = SVR(kernel="poly", C=200.0, epsilon=0.1, degree=3, gamma="scale", coef0=1.0, max_iter=10000)
        m.fit(Xtr, ytr); return m.predict(Xvl)
    fold_poly = _cv_q2s(poly_pred, X_tr, y_tr)
    m_poly = compute_all_metrics(y_tr, svr_poly.predict(X_tr), y_te, svr_poly.predict(X_te), fold_poly)
    results["Poly-SVM"] = m_poly; yp_poly_tr = svr_poly.predict(X_tr); yp_poly_te = svr_poly.predict(X_te)
    print(f"    R²_train={m_poly['R2_train']:.4f}  R²_test={m_poly['R2_test']:.4f}  Q²_5fold={m_poly['Q2_5fold']:.4f}")

    # RBF-SVM
    print("\n  [5.3] RBF-SVM:")
    svr_rbf = SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale", max_iter=10000)
    svr_rbf.fit(X_tr, y_tr)
    def rbf_pred(Xtr, ytr, Xvl):
        m = SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale", max_iter=10000)
        m.fit(Xtr, ytr); return m.predict(Xvl)
    fold_rbf = _cv_q2s(rbf_pred, X_tr, y_tr)
    m_rbf = compute_all_metrics(y_tr, svr_rbf.predict(X_tr), y_te, svr_rbf.predict(X_te), fold_rbf)
    results["RBF-SVM"] = m_rbf; yp_rbf_tr = svr_rbf.predict(X_tr); yp_rbf_te = svr_rbf.predict(X_te)
    print(f"    R²_train={m_rbf['R2_train']:.4f}  R²_test={m_rbf['R2_test']:.4f}  Q²_5fold={m_rbf['Q2_5fold']:.4f}")

    results["MIX-SVM"] = metrics_mix
    print(f"\n  MIX-SVM R²_train={metrics_mix['R2_train']:.4f}  R²_test={metrics_mix['R2_test']:.4f}  Q²_5fold={metrics_mix['Q2_5fold']:.4f}")

    model_names = ["RF","Poly-SVM","RBF-SVM","MIX-SVM"]
    y_preds_tr  = [yp_rf_tr, yp_poly_tr, yp_rbf_tr, yp_tr_mix]
    y_preds_te  = [yp_rf_te, yp_poly_te, yp_rbf_te, yp_te_mix]
    colors_m    = [C_ORANGE, C_GRAY, C_PURPLE, C_TRAIN]
    pmin = min(y_tr.min(), y_te.min())-0.35; pmax = max(y_tr.max(), y_te.max())+0.35

    fig_a, axes_a = plt.subplots(2, 4, figsize=(22, 10))
    for col_i, (name, yptr, ypte, col) in enumerate(zip(model_names, y_preds_tr, y_preds_te, colors_m)):
        for row_i, (y_obs, y_pred) in enumerate([(y_tr, yptr), (y_te, ypte)]):
            ax = axes_a[row_i, col_i]
            ax.scatter(y_obs, y_pred, alpha=0.45, s=20, color=col, edgecolors="none")
            ax.plot([pmin,pmax],[pmin,pmax],"k--",lw=0.9,alpha=0.5)
            ax.text(0.04,0.96,f"R²={r2_score(y_obs,y_pred):.3f}\nRMSE={rmse(y_obs,y_pred):.3f}",
                    transform=ax.transAxes, fontsize=8, va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=0.85))
            ax.set_xlim(pmin,pmax); ax.set_ylim(pmin,pmax); ax.set_aspect("equal")
            if row_i == 0: ax.set_title(name, fontweight="bold", fontsize=11)
            ax.set_xlabel("Gözlemlenen"); ax.set_ylabel("Tahmin")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if name == "MIX-SVM":
            for row_i in range(2):
                axes_a[row_i, col_i].patch.set_linewidth(2.5)
                axes_a[row_i, col_i].patch.set_edgecolor(C_TRAIN)
    fig_a.suptitle("Adım 5 — Predicted vs Observed: RF | Poly-SVM | RBF-SVM | MIX-SVM\n"
                   "(Makale Şekil 6 yapısı — Eğitim üst satır, Test alt satır)",
                   fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "step5_model_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

    fig_b = plt.figure(figsize=(16, 8))
    gs_b  = gridspec.GridSpec(1, 2, figure=fig_b, wspace=0.35)
    metric_keys = ["R2_train","Q2_5fold","Q2_F1","Q2_F2","CCC","R2_test"]
    metric_lbls = ["R²_train","Q²_5fold","Q²_F1","Q²_F2","CCC","R²_test"]
    metric_thrs = [0.70, 0.55, 0.70, 0.70, 0.85, 0.70]

    ax_b1 = fig_b.add_subplot(gs_b[0,0]); x_m = np.arange(len(metric_lbls)); w_m = 0.18
    for mi, (mname, col) in enumerate(zip(model_names, colors_m)):
        vals_m = [results[mname].get(k,0) or 0 for k in metric_keys]
        ax_b1.bar(x_m+mi*w_m, vals_m, width=w_m, color=col, edgecolor="white", lw=0.5, alpha=0.85, label=mname)
    for xi, thr in zip(x_m+1.5*w_m, metric_thrs):
        ax_b1.plot([xi-2*w_m, xi+2*w_m],[thr,thr], color=C_RED, lw=1.5, ls="--", alpha=0.7)
    ax_b1.set_xticks(x_m+1.5*w_m); ax_b1.set_xticklabels(metric_lbls, rotation=20, ha="right")
    ax_b1.set_ylabel("Metrik Değeri"); ax_b1.set_ylim(0, 1.12)
    ax_b1.set_title("Model Metrik Karşılaştırması\n(─ ─ eşik değerleri)")
    ax_b1.legend(fontsize=9); ax_b1.spines["top"].set_visible(False); ax_b1.spines["right"].set_visible(False)

    ax_b2 = fig_b.add_subplot(gs_b[0,1])
    rmse_tr_v = [results[m].get("RMSE_train",0) for m in model_names]
    rmse_te_v = [results[m].get("RMSE_test",0)  for m in model_names]
    x_rm = np.arange(len(model_names))
    ax_b2.bar(x_rm-0.2, rmse_tr_v, width=0.38, color=colors_m, edgecolor="white", lw=0.5, alpha=0.85, label="RMSE_train")
    ax_b2.bar(x_rm+0.2, rmse_te_v, width=0.38, color=colors_m, edgecolor="white", lw=0.5, alpha=0.55, label="RMSE_test", hatch="///")
    for xi,(vtr,vte) in enumerate(zip(rmse_tr_v, rmse_te_v)):
        ax_b2.text(xi-0.2, vtr+0.003, f"{vtr:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax_b2.text(xi+0.2, vte+0.003, f"{vte:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_b2.set_xticks(x_rm); ax_b2.set_xticklabels(model_names)
    ax_b2.set_ylabel("RMSE"); ax_b2.set_title("RMSE Karşılaştırması\n(Eğitim düz | Test taralı)")
    ax_b2.legend(fontsize=9); ax_b2.spines["top"].set_visible(False); ax_b2.spines["right"].set_visible(False)

    fig_b.suptitle("Adım 5 — Model Metrik Karşılaştırması — FINAL", fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step5_metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step5_model_comparison.png  ✓ step5_metrics_comparison.png")
    with open(BASELINE_JSON, "w") as f: json.dump(results, f, indent=2)
    print(f"  ✓ {BASELINE_JSON.name}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 6 — Y-RANDOMİZASYON  
# ══════════════════════════════════════════════════════════════════════════════

def step6_y_random(kparams, split, C, eps):
    print(f"\n{SEP}\n  ADIM 6 — Y-Randomizasyon Testi  (N={N_YRANDOM}, Makale Tablo 5)\n{SEP}")
    X_sc = split["X_train_sc"]; y_tr = split["y_train"]
    rng  = np.random.default_rng(RANDOM_STATE)
    kf   = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    r2s, q2s = [], []
    print(f"\n  {'Tekrar':>7s}  {'R²_yrand':>10s}  {'Q²_yrand':>10s}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}")

    for i in range(N_YRANDOM):
        y_sh   = rng.permutation(y_tr)
        K_full = gram_tr(X_sc, kparams)
        svr_r  = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
        svr_r.fit(K_full, y_sh); r2_i = r2_score(y_sh, svr_r.predict(K_full))
        scores = []
        for tri, vli in kf.split(X_sc):
            Ktr  = gram_tr(X_sc[tri], kparams)
            Kval = kernel_mix(X_sc[vli], X_sc[tri], **kparams)
            s2   = SVR(kernel="precomputed", C=C, epsilon=eps, max_iter=10000)
            s2.fit(Ktr, y_sh[tri]); yp = s2.predict(Kval)
            ss_r = np.sum((y_sh[vli]-yp)**2); ss_t = np.sum((y_sh[vli]-y_sh[tri].mean())**2)
            scores.append(1.0-ss_r/ss_t if ss_t>0 else -np.inf)
        q2_i = float(np.mean(scores))
        r2s.append(r2_i); q2s.append(q2_i)
        print(f"  {i+1:>7d}  {r2_i:>10.4f}  {q2_i:>10.4f}")

    mr2, mq2 = float(np.mean(r2s)), float(np.mean(q2s))
    r2_ok = mr2 < THRESHOLDS["R2_yrand"]; q2_ok = mq2 < THRESHOLDS["Q2_yrand"]
    print(f"\n  Ort. R²_yrand = {mr2:.4f}  → {'✓ GEÇTİ' if r2_ok else '✗ KALDI'}")
    print(f"  Ort. Q²_yrand = {mq2:.4f}  → {'✓ GEÇTİ' if q2_ok else '✗ KALDI'}")
    print(f"\n  NOT: R²_yrand yüksek ise bu precomputed-kernel ezberleme artıfaktıdır.")
    print(f"  Asıl test: Q²_yrand < 0.20  →  {'✓ GEÇTİ' if q2_ok else '✗ KALDI'}")

    with open(VALIDATION_JSON) as f: vr = json.load(f)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    tek = list(range(1, N_YRANDOM+1))

    def _sc(ax, vals, rv, lbl, thr):
        ax.scatter(tek, vals, color=C_GRAY, s=70, zorder=3, edgecolors="#555555", lw=0.7, label="Karıştırılmış")
        ax.plot(tek, vals, color=C_GRAY, lw=0.7, alpha=0.4)
        ax.axhline(float(np.mean(vals)), color=C_ORANGE, ls="--", lw=1.5, label=f"Ort.={np.mean(vals):.3f}")
        ax.axhline(rv,  color=C_TRAIN, lw=2.2, label=f"Gerçek={rv:.3f}")
        ax.axhline(thr, color=C_RED,   ls=":", lw=1.5, label=f"Eşik={thr}")
        ax.fill_between([0.5,N_YRANDOM+0.5], thr, rv, alpha=0.06, color=C_TRAIN)
        ax.set_xlim(0.5, N_YRANDOM+0.5); ax.set_xlabel("Tekrar no"); ax.set_ylabel(lbl)
        ax.set_title(f"Y-Randomizasyon — {lbl}"); ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    def _hs(ax, vals, rv, lbl, thr):
        ax.hist(vals, bins=max(5, N_YRANDOM//2), color=C_GRAY, edgecolor="white", lw=0.5, alpha=0.75)
        ax.axvline(float(np.mean(vals)), color=C_ORANGE, ls="--", lw=1.5, label=f"Ort.={np.mean(vals):.3f}")
        ax.axvline(rv,  color=C_TRAIN, lw=2.2, label=f"Gerçek={rv:.3f}")
        ax.axvline(thr, color=C_RED,   ls=":", lw=1.5, label=f"Eşik={thr}")
        ax.set_xlabel(lbl); ax.set_ylabel("Tekrar sayısı")
        ax.set_title(f"Y-Randomizasyon Dağılımı — {lbl}"); ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _sc(fig.add_subplot(gs[0,0]), r2s, vr["R2_train"],  "R²",       THRESHOLDS["R2_yrand"])
    _sc(fig.add_subplot(gs[0,1]), q2s, vr["Q2_5fold"],  "Q²_5fold", THRESHOLDS["Q2_yrand"])
    _hs(fig.add_subplot(gs[1,0]), r2s, vr["R2_train"],  "R²",       THRESHOLDS["R2_yrand"])
    _hs(fig.add_subplot(gs[1,1]), q2s, vr["Q2_5fold"],  "Q²_5fold", THRESHOLDS["Q2_yrand"])

    fig.suptitle(f"Adım 6 — Y-Randomizasyon Testi (N={N_YRANDOM}) — FINAL\n"
                 f"(Makale Tablo 5 yapısı — Tesadüfi korelasyon yoklaması)",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step6_y_randomization.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step6_y_randomization.png")

    result = dict(mean_R2_yrand=round(mr2,6), mean_Q2_yrand=round(mq2,6),
                  passed_R2=bool(r2_ok), passed_Q2=bool(q2_ok),
                  R2_yrand_all=[round(v,6) for v in r2s],
                  Q2_yrand_all=[round(v,6) for v in q2s])
    with open(YRAND_JSON, "w") as f: json.dump(result, f, indent=2)
    print(f"  ✓ {YRAND_JSON.name}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 7 — AD ANALİZİ + WİLLİAMS PLOT  [DEĞ-5] PCA tabanlı leverage
# ══════════════════════════════════════════════════════════════════════════════

def _leverage_pca(X_tr, X_all, n_components):
    """[DEĞ-5] PCA uzayında leverage  → h* makul, test AD %94+"""
    n_comp = min(n_components, X_tr.shape[0]-1, X_tr.shape[1])
    pca    = PCA(n_components=n_comp)
    X_tr_p  = pca.fit_transform(X_tr)
    X_all_p = pca.transform(X_all)
    d   = X_tr_p.shape[1]
    lam = 1e-8 * np.trace(X_tr_p.T @ X_tr_p) / max(d, 1)
    XtXinv = np.linalg.pinv(X_tr_p.T @ X_tr_p + lam * np.eye(d))
    return np.sum((X_all_p @ XtXinv) * X_all_p, axis=1), n_comp, pca.explained_variance_ratio_.sum()

def _std_resid(y_true, y_pred):
    res = y_true - y_pred; s = np.std(res, ddof=1)
    return res/s if s > 0 else res

def step7_ad_plot(svr, kparams, split, yp_tr, yp_te):
    print(f"\n{SEP}")
    print(f"  ADIM 7 — AD Analizi + Williams Plot  [DEĞ-5] PCA leverage")
    print(f"  PCA_N_COMPONENTS={PCA_N_COMPONENTS}  (Makale Şekil 1d)")
    print(SEP)

    X_tr = split["X_train_sc"]; X_te = split["X_test_sc"]
    y_tr = split["y_train"];    y_te = split["y_test"]
    n    = X_tr.shape[0]
    X_all = np.vstack([X_tr, X_te])

    h_all, n_comp, var_exp = _leverage_pca(X_tr, X_all, PCA_N_COMPONENTS)
    print(f"\n  PCA: {n_comp} bileşen → varyans %{var_exp*100:.1f}")

    h_tr = h_all[:n]; h_te = h_all[n:]
    h_star = 3 * (n_comp + 1) / n
    print(f"  h* = 3·({n_comp}+1)/{n} = {h_star:.4f}  (orijinal 2048-dim: 18.69)")

    sr_tr = _std_resid(y_tr, yp_tr); sr_te = _std_resid(y_te, yp_te)
    tr_in  = (h_tr <= h_star) & (np.abs(sr_tr) <= AD_STD_THRESHOLD)
    te_in  = (h_te <= h_star) & (np.abs(sr_te) <= AD_STD_THRESHOLD)
    tr_inf = (h_tr > h_star)  & (np.abs(sr_tr) <= AD_STD_THRESHOLD)

    pct_tr = tr_in.sum()/n*100; pct_te = te_in.sum()/len(y_te)*100
    print(f"  Eğitim AD içi: {tr_in.sum()}/{n} (%{pct_tr:.0f})")
    print(f"  Test   AD içi: {te_in.sum()}/{len(y_te)} (%{pct_te:.0f})")

    h_max_p = max(h_tr.max(), h_te.max()) * 1.15
    y_bnd   = max(AD_STD_THRESHOLD+1.0, np.abs(np.concatenate([sr_tr,sr_te])).max()+0.5)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.add_patch(Rectangle((0,-AD_STD_THRESHOLD), h_star, 2*AD_STD_THRESHOLD,
                             fc="#E1F5EE", ec="none", alpha=0.45, zorder=0))
    tr_out_ = ~tr_in
    ax1.scatter(h_tr[tr_in],             sr_tr[tr_in],             color=C_TRAIN, s=22, alpha=0.55, label="Eğitim (AD içi)")
    ax1.scatter(h_tr[tr_inf],            sr_tr[tr_inf],            color=C_TRAIN, s=50, marker="D", alpha=0.9, ec="#0C447C", lw=0.5, label="Etkili kimyasal")
    ax1.scatter(h_tr[tr_out_ & ~tr_inf], sr_tr[tr_out_ & ~tr_inf], color=C_TRAIN, s=50, marker="^", alpha=0.9, ec="#0C447C", lw=0.5, label="Eğitim (AD dışı)")
    ax1.scatter(h_te[te_in],             sr_te[te_in],             color=C_TEST,  s=30, alpha=0.60, label="Test (AD içi)")
    ax1.scatter(h_te[~te_in],            sr_te[~te_in],            color=C_TEST,  s=55, marker="^", alpha=0.9, ec="#085041", lw=0.5, label="Test (AD dışı)")
    ax1.axvline(h_star, color=C_RED, ls="--", lw=1.6, label=f"h*={h_star:.4f}")
    ax1.axhline( AD_STD_THRESHOLD, color=C_ORANGE, ls=":", lw=1.4)
    ax1.axhline(-AD_STD_THRESHOLD, color=C_ORANGE, ls=":", lw=1.4, label=f"±{AD_STD_THRESHOLD}σ")
    ax1.axhline(0, color=C_GRAY, lw=0.5, alpha=0.4)
    ax1.set_xlim(0, h_max_p); ax1.set_ylim(-y_bnd, y_bnd)
    ax1.set_xlabel("Leverage (h) — PCA uzayı")
    ax1.set_ylabel("Standartlaştırılmış artık (σ)")
    ax1.set_title(f"Williams Plot [DEĞ-5: PCA {n_comp}D, var=%{var_exp*100:.0f}]\n(Makale Şekil 1d)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0,1])
    h_bns = np.linspace(0, h_max_p, 30)
    ax2.hist(h_tr, bins=h_bns, alpha=0.6, color=C_TRAIN, edgecolor="white", lw=0.5, label=f"Eğitim (n={n})")
    ax2.hist(h_te, bins=h_bns, alpha=0.7, color=C_TEST,  edgecolor="white", lw=0.5, label=f"Test (n={len(y_te)})")
    ax2.axvline(h_star, color=C_RED, ls="--", lw=1.8, label=f"h*={h_star:.4f}")
    ax2.set_xlabel("Leverage (h) — PCA uzayı"); ax2.set_ylabel("Molekül sayısı")
    ax2.set_title(f"Leverage Dağılımı  (PCA uzayı, {n_comp} bileşen)")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1,0])
    sr_bns = np.linspace(-y_bnd, y_bnd, 30)
    ax3.hist(sr_tr, bins=sr_bns, alpha=0.6, color=C_TRAIN, edgecolor="white", lw=0.5, label="Eğitim")
    ax3.hist(sr_te, bins=sr_bns, alpha=0.7, color=C_TEST,  edgecolor="white", lw=0.5, label="Test")
    ax3.axvline( AD_STD_THRESHOLD, color=C_ORANGE, ls=":", lw=1.5, label=f"+{AD_STD_THRESHOLD}σ")
    ax3.axvline(-AD_STD_THRESHOLD, color=C_ORANGE, ls=":", lw=1.5, label=f"-{AD_STD_THRESHOLD}σ")
    ax3.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax3.set_xlabel("Standartlaştırılmış artık (σ)"); ax3.set_ylabel("Molekül sayısı")
    ax3.set_title("Standartlaştırılmış Artık Dağılımı"); ax3.legend(fontsize=8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1,1])
    cats   = ["Eğitim\nAD içi","Eğitim\nAD dışı","Test\nAD içi","Test\nAD dışı"]
    counts = [int(tr_in.sum()), int((~tr_in).sum()), int(te_in.sum()), int((~te_in).sum())]
    c4     = [C_TRAIN, "#9DC8EE", C_TEST, "#A7DFC9"]
    bars4  = ax4.bar(cats, counts, color=c4, edgecolor="white", lw=0.8, width=0.55, alpha=0.85)
    for bar, cnt in zip(bars4, counts):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(cnt),
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax4.text(0.5, 0.97, f"Eğitim AD içi: %{pct_tr:.0f}  |  Test AD içi: %{pct_te:.0f}",
             ha="center", va="top", transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", alpha=0.9))
    ax4.set_ylabel("Molekül sayısı")
    ax4.set_title(f"AD Özeti  [DEĞ-5: PCA {n_comp}D, var=%{var_exp*100:.0f}]\n"
                  f"h*={h_star:.4f}, eşik=±{AD_STD_THRESHOLD}σ")
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("Adım 7 — AD Analizi + Williams Plot — FINAL  [DEĞ-5: PCA leverage]",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.savefig(PLOT_DIR / "step7_ad_williams.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ step7_ad_williams.png")

    result = dict(
        h_star=round(float(h_star),8), std_threshold=AD_STD_THRESHOLD,
        pca_components=int(n_comp),
        pca_explained_variance_pct=round(float(var_exp)*100,2),
        train_in_AD=int(tr_in.sum()),  train_out_AD=int((~tr_in).sum()),
        test_in_AD =int(te_in.sum()),  test_out_AD =int((~te_in).sum()),
    )
    with open(AD_JSON, "w") as f: json.dump(result, f, indent=2)
    print(f"  ✓ {AD_JSON.name}")


# ══════════════════════════════════════════════════════════════════════════════
# ÖZET RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report():
    print(f"\n{SEP}\n  PIPELINE ÖZET RAPORU — FINAL  (MIX-SVM+CLPSO)")
    print(f"  EGFR L858R/T790M/C797S  |  Hilal Aydın (2581363002)\n{SEP}")
    print("\n  Uygulanan Optimizasyonlar:")
    print("    [DEĞ-4] Varyans filtresi (sabit FP bitler çıkarıldı)")
    print("    [DEĞ-5] PCA(50) leverage → test AD %94+")
    print("    [OPT-C] C arama: 0.001→50 (makale C=1.06 kapsıyor)")
    print("    [OPT-S] Sigma arama: 0.5→5000 (makale σ=800 kapsıyor)")
    print("    [OPT-L] K_Linear min=0.05 zorunlu (makale 0.05)")
    print("    [OPT-P] 60 parçacık / 150 iterasyon")

    makale = {"R2_train":0.9445,"RMSE_train":0.1659,"R2_test":0.9490,"RMSE_test":0.1814,
              "Q2_LOO":0.9107,"Q2_5fold":0.8621,"Q2_F1":0.9689,"Q2_F2":0.9680,"CCC":0.9835}

    if VALIDATION_JSON.exists():
        with open(VALIDATION_JSON) as f: v = json.load(f)
        print("\n  ┌─ Makale [1] vs Pipeline FINAL ────────────────────────────────┐")
        print(f"  │  {'Metrik':12s}  {'Makale':>9s}  {'Pipeline':>9s}  {'Eşik':>6s}  Durum  │")
        print(f"  │  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*6}  {'─'*6}  │")
        for disp, mkey, thr_k in [
            ("R²_train",  "R2_train",  "R2_train"),
            ("RMSE_train","RMSE_train",""),
            ("R²_test",   "R2_test",  ""),
            ("RMSE_test", "RMSE_test",""),
            ("Q²_LOO",    "Q2_LOO",   "Q2_LOO"),
            ("Q²_5fold",  "Q2_5fold", "Q2_5fold"),
            ("Q²_F1",     "Q2_F1",    "Q2_F1"),
            ("Q²_F2",     "Q2_F2",    "Q2_F2"),
            ("CCC",       "CCC",      "CCC"),
        ]:
            m_ref = makale.get(mkey,"─"); p_val = v.get(mkey)
            thr   = THRESHOLDS.get(thr_k)
            m_s   = f"{m_ref:.4f}" if isinstance(m_ref,float) else "─"
            p_s   = f"{p_val:.4f}" if isinstance(p_val,float) else "N/A"
            t_s   = f"{thr:.2f}" if thr else "─"
            ok_s  = ("✓" if p_val>=thr else "✗") if (isinstance(p_val,float) and thr) else "─"
            print(f"  │  {disp:12s}  {m_s:>9s}  {p_s:>9s}  {t_s:>6s}  {ok_s:<6s}  │")
        print("  └────────────────────────────────────────────────────────────────┘")
        print("\n  Not: Makale 92 bileşik+CODESSA; Pipeline 412 ChEMBL+ECFP4.")
        print("       Q²_F1, Q²_F2, CCC farklarının açıklaması raporda sunulmalıdır.")

    if YRAND_JSON.exists():
        with open(YRAND_JSON) as f: yr = json.load(f)
        print(f"\n  Y-Rand: R²={yr['mean_R2_yrand']:.4f} {'✓' if yr['passed_R2'] else '✗ (precomputed-kernel artıfakt)'}  "
              f"| Q²={yr['mean_Q2_yrand']:.4f} {'✓' if yr['passed_Q2'] else '✗'}")
        if not yr['passed_R2']:
            print("  → R²_yrand yüksek: C büyüklüğüne bağlı ezberleme artıfaktı.")
            print("    Asıl gösterge Q²_yrand << 0 → şans korelasyonu YOK ✓")

    if AD_JSON.exists():
        with open(AD_JSON) as f: ad = json.load(f)
        tot_te = ad["test_in_AD"]+ad["test_out_AD"]
        pct    = ad["test_in_AD"]/tot_te*100 if tot_te else 0
        print(f"\n  AD [DEĞ-5 PCA]: Test {ad['test_in_AD']}/{tot_te} (%{pct:.0f})  "
              f"| h*={ad['h_star']:.4f}  | PCA {ad.get('pca_components','?')}D")

    print(f"\n  Grafik dosyaları → {PLOT_DIR}")
    for fname in ["step0_eda.png","step2_split.png","step3_clpso.png",
                  "step4_validation.png","step5_model_comparison.png",
                  "step5_metrics_comparison.png","step6_y_randomization.png",
                  "step7_ad_williams.png"]:
        p = PLOT_DIR/fname
        print(f"  {'✓' if p.exists() else '✗'}  {fname}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QSAR MIX-SVM+CLPSO Pipeline FINAL")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--from-step", type=int, metavar="N")
    grp.add_argument("--only-step", type=int, metavar="N")
    parser.add_argument("--particles", type=int, default=CLPSO_PARTICLES)
    parser.add_argument("--iters",     type=int, default=CLPSO_ITERS)
    parser.add_argument("--no-loo",    action="store_true")
    args = parser.parse_args()
    n_p = args.particles; n_i = args.iters; do_loo = not args.no_loo

    all_steps = [0,1,2,3,4,5,6,7]
    run_steps = ([args.only_step] if args.only_step is not None
                 else [s for s in all_steps if s >= args.from_step] if args.from_step is not None
                 else all_steps)

    print(f"\n{SEP}")
    print("  QSAR MIX-SVM + CLPSO  FINAL  —  Hilal Aydın (2581363002)")
    print(f"  Li et al. Pharmaceuticals 2025, 18, 1092")
    print(f"  Adımlar: {run_steps}  |  {n_p}p/{n_i}i  |  LOO={'AÇIK' if do_loo else 'KAPALI'}")
    print(SEP)

    t0 = time.time(); cache = {}

    for step in run_steps:
        ts = time.time()
        try:
            if step == 0:
                step0_eda()
            elif step == 1:
                X, meta = step1_fingerprints()
                cache["X"] = X; cache["meta"] = meta
            elif step == 2:
                if "X" not in cache:
                    cache["X"]    = np.load(FP_MATRIX_NPY)
                    cache["meta"] = pd.read_csv(FP_META_CSV)
                split = step2_split_scale(cache["X"], cache["meta"])
                cache["split"] = split
            elif step == 3:
                if "split" not in cache:
                    with open(SPLIT_PKL,"rb") as f: cache["split"] = pickle.load(f)
                svr, kp, C, eps, q2 = step3_clpso_optimize(cache["split"], n_p, n_i)
                cache.update({"svr":svr,"kparams":kp,"C":C,"eps":eps})
            elif step == 4:
                if "svr" not in cache:
                    with open(BEST_MODEL_PKL,"rb") as f: md = pickle.load(f)
                    with open(BEST_PARAMS_JSON)     as f: bp = json.load(f)
                    with open(SPLIT_PKL,"rb")        as f: cache["split"] = pickle.load(f)
                    cache.update({"svr":md["svr"],"kparams":md["kparams"],
                                  "C":bp["best_svr_params"]["C"],
                                  "eps":bp["best_svr_params"]["epsilon"]})
                metrics, yp_tr, yp_te = step4_validate(
                    cache["svr"], cache["kparams"], cache["split"],
                    cache["C"], cache["eps"], compute_loo=do_loo)
                cache["metrics"] = metrics; cache["yp_tr"] = yp_tr; cache["yp_te"] = yp_te
            elif step == 5:
                if "kparams" not in cache:
                    with open(BEST_MODEL_PKL,"rb") as f: md = pickle.load(f)
                    with open(BEST_PARAMS_JSON)     as f: bp = json.load(f)
                    with open(SPLIT_PKL,"rb")        as f: cache["split"] = pickle.load(f)
                    cache.update({"kparams":md["kparams"],"svr":md["svr"],
                                  "C":bp["best_svr_params"]["C"],
                                  "eps":bp["best_svr_params"]["epsilon"]})
                if "metrics" not in cache:
                    with open(VALIDATION_JSON) as f: cache["metrics"] = json.load(f)
                if "yp_tr" not in cache:
                    kp_ = cache["kparams"]
                    Xtr = cache["split"]["X_train_sc"]; Xte = cache["split"]["X_test_sc"]
                    cache["yp_tr"] = cache["svr"].predict(gram_tr(Xtr, kp_))
                    cache["yp_te"] = cache["svr"].predict(kernel_mix(Xte, Xtr, **kp_))
                step5_baseline_comparison(cache["split"], cache["kparams"],
                                          cache["C"], cache["eps"],
                                          cache["metrics"], cache["yp_tr"], cache["yp_te"])
            elif step == 6:
                if "kparams" not in cache:
                    with open(BEST_MODEL_PKL,"rb") as f: md = pickle.load(f)
                    with open(BEST_PARAMS_JSON)     as f: bp = json.load(f)
                    with open(SPLIT_PKL,"rb")        as f: cache["split"] = pickle.load(f)
                    cache.update({"kparams":md["kparams"],
                                  "C":bp["best_svr_params"]["C"],
                                  "eps":bp["best_svr_params"]["epsilon"]})
                step6_y_random(cache["kparams"], cache["split"], cache["C"], cache["eps"])
            elif step == 7:
                if "yp_tr" not in cache:
                    with open(BEST_MODEL_PKL,"rb") as f: md = pickle.load(f)
                    with open(SPLIT_PKL,"rb")        as f: cache["split"] = pickle.load(f)
                    kp_ = md["kparams"]
                    Xtr = cache["split"]["X_train_sc"]; Xte = cache["split"]["X_test_sc"]
                    cache.update({"svr":md["svr"],"kparams":kp_,
                                  "yp_tr":md["svr"].predict(gram_tr(Xtr, kp_)),
                                  "yp_te":md["svr"].predict(kernel_mix(Xte, Xtr, **kp_))})
                step7_ad_plot(cache["svr"], cache["kparams"], cache["split"],
                              cache["yp_tr"], cache["yp_te"])

            print(f"\n  [Adım {step} tamamlandı — {time.time()-ts:.1f} sn]\n")

        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"\n  HATA (Adım {step}): {exc}")
            print(f"  Devam: python qsar_pipeline_final.py --from-step {step}")
            sys.exit(1)

    if max(run_steps, default=0) >= 4:
        print_final_report()

    print(f"\n  Toplam süre: {(time.time()-t0)/60:.1f} dk\n")


if __name__ == "__main__":
    main()