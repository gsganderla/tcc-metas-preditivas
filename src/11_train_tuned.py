#!/usr/bin/env python3
"""
11_train_tuned.py
=================
Treinamento com ajuste de hiperparâmetros via validação cruzada temporal.

Modelos
-------
0. Naïve — Persistência: ŷ_{t} = y_{t-1} (lag-1 do target); referência mínima
1. OLS  — Regressão Linear Múltipla (MQO), baseline sem regularização
2. RF   — Random Forest: Grid Search exaustivo por EQM
           Parâmetros: n_estimators × max_depth × max_features
3. XGB  — XGBoost: Random Search com regularização L1 (reg_alpha) e L2 (reg_lambda)
           Parâmetros: n_estimators, max_depth, learning_rate, subsample,
                       colsample_bytree, reg_alpha, reg_lambda
4. LGBM — LightGBM: Random Search (mesmos eixos + num_leaves)

Critério de otimização: EQM negativo (neg_mean_squared_error)
Validação cruzada: janela expansível temporal por período (5 folds) em todo o tuning.

Saídas
------
  data/processed/tuning/       — CSV com todos os resultados de Grid/Random Search
  data/processed/models/tuned/ — modelos ajustados (.joblib)
  data/processed/cv_results_tuned.csv — comparação final OLS × RF × XGB × LGBM
  logs/train_tuned.log
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor as _sm_vif
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PANEL     = ROOT / "data" / "processed" / "panel_features_clean.parquet"
OUT_DIR   = ROOT / "data" / "processed"
TUNING_DIR = OUT_DIR / "tuning"
MODEL_DIR = OUT_DIR / "models" / "tuned"
LOG_DIR   = ROOT / "logs"

# ── Configuração ───────────────────────────────────────────────────────────────
TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]

FINS = ["vol_credito_rs_mil", "captacao_rs_mil", "ativo_total_rs_mil",
        "patrimonio_liq_rs_mil", "carteira_credito_rs_mil"]
MACRO_COLS = ["selic_aa", "ipca_acum_trim", "cambio_brl_usd",
              "ibc_br", "inpc_acum_trim", "ipa_agro", "concessoes_cred"]
PIB_COLS   = ["pib_corrente_rs_mil", "pib_per_capita_rs",
              "vab_agro_rs_mil", "vab_industria_rs_mil", "vab_servicos_rs_mil",
              "share_agro_mun"]
COMOD_COLS = ["ipa_agro_idx", "milho_rs_60kg", "boi_gordo_rs_arroba", "leite_rs_litro"]
CR_COLS    = ["cr_total_rs_mi", "cr_custeio_rs_mi",
              "cr_investimento_rs_mi", "cr_comercializacao_rs_mi"]
FEAT_CAT   = ["segmento_num", "porte_num"]
FEAT_SAFRA = ["dummy_plantio_verao", "dummy_colheita_verao",
              "dummy_plantio_inv", "dummy_colheita_inv",
              "dummy_sul", "dummy_plantio_verao_sul", "dummy_colheita_verao_sul"]

ALL_FEATURES = ([f"L{l}_{v}" for l in [1, 2, 4] for v in FINS]
                + [f"L1_{c}" for c in MACRO_COLS]
                + PIB_COLS
                + [f"L1_{c}" for c in COMOD_COLS]
                + [f"L1_{c}" for c in CR_COLS]
                + FEAT_CAT + FEAT_SAFRA)

# ── VIF ────────────────────────────────────────────────────────────────────────
VIF_THRESH  = 10.0        # limiar: VIF > 10 → alta multicolinearidade
VIF_FIG_DIR = ROOT / "reports" / "figures" / "vif"
TARGET_LABELS = {
    "vol_credito_rs_mil": "Volume de Crédito (R$ mil)",
    "captacao_rs_mil":    "Captação (R$ mil)",
}

# ── CV temporal ────────────────────────────────────────────────────────────────
N_SPLITS         = 5
MIN_TRAIN_PERIODS = 8
N_ITER_RANDOM    = 30
RANDOM_STATE     = 42

# ── Espaços de hiperparâmetros ─────────────────────────────────────────────────
# RF — Grid Search exaustivo (3³ = 27 combinações)
GRID_RF = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth":    [None, 10, 20],
    "model__max_features": ["sqrt", "log2", 0.5],
}

# XGBoost — Random Search (L1 = reg_alpha, L2 = reg_lambda)
DIST_XGB = {
    "model__n_estimators":     [100, 200, 300, 500],
    "model__max_depth":        [3, 4, 5, 6, 7],
    "model__learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "model__reg_alpha":        [0, 0.001, 0.01, 0.1, 1.0],   # L1
    "model__reg_lambda":       [0.1, 0.5, 1.0, 2.0, 5.0],   # L2
}

# LightGBM — Random Search
DIST_LGBM = {
    "model__n_estimators":     [100, 200, 300, 500],
    "model__num_leaves":       [15, 31, 63, 127],
    "model__max_depth":        [-1, 5, 8, 10],
    "model__learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "model__reg_alpha":        [0, 0.001, 0.01, 0.1, 1.0],   # L1
    "model__reg_lambda":       [0.1, 0.5, 1.0, 2.0, 5.0],   # L2
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "train_tuned.log", encoding="utf-8"),
        ],
    )


# ── Validação cruzada temporal ─────────────────────────────────────────────────
def temporal_cv_sklearn(periodo_col: pd.Series) -> list[tuple]:
    """
    Converte os períodos em pares (train_positions, test_positions) compatíveis
    com GridSearchCV / RandomizedSearchCV do scikit-learn.

    Janela expansível: cada fold acrescenta test_size períodos ao treino anterior.
    As posições são inteiras (iloc) referentes à ordem atual do DataFrame.
    """
    periodos  = sorted(periodo_col.unique())
    n         = len(periodos)
    test_size = max(1, (n - MIN_TRAIN_PERIODS) // N_SPLITS)
    arr       = periodo_col.values

    splits = []
    for i in range(N_SPLITS):
        split = MIN_TRAIN_PERIODS + i * test_size
        if split >= n:
            break
        train_p   = set(periodos[:split])
        test_p    = set(periodos[split: split + test_size])
        if not test_p:
            break
        train_pos = np.where(np.isin(arr, list(train_p)))[0]
        test_pos  = np.where(np.isin(arr, list(test_p)))[0]
        splits.append((train_pos, test_pos))

    return splits


# ── Pipelines base ─────────────────────────────────────────────────────────────
def pipeline_ols() -> Pipeline:
    """OLS puro: apenas imputer (MQO não requer escalonamento)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LinearRegression()),
    ])


def pipeline_rf() -> Pipeline:
    """Random Forest com imputer."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def pipeline_xgb() -> Pipeline:
    """XGBoost com imputer."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                 verbosity=0, eval_metric="rmse")),
    ])


def pipeline_lgbm() -> Pipeline:
    """LightGBM com imputer."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ])


# ── Métricas ───────────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred) -> dict:
    return {
        "r2":   round(r2_score(y_true, y_pred), 6),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "mae":  round(mean_absolute_error(y_true, y_pred), 2),
    }


def avaliar_cv(pipeline, X: pd.DataFrame, y: pd.Series,
               cv_splits: list[tuple]) -> pd.DataFrame:
    """
    Avalia o pipeline nos folds temporais (out-of-sample).
    Treina em cada fold do zero para medir desempenho real de generalização.
    """
    records = []
    for fold, (tr_pos, te_pos) in enumerate(cv_splits, 1):
        X_tr, y_tr = X.iloc[tr_pos], y.iloc[tr_pos]
        X_te, y_te = X.iloc[te_pos], y.iloc[te_pos]
        mask_tr    = y_tr.notna()
        mask_te    = y_te.notna()
        pipeline.fit(X_tr[mask_tr], y_tr[mask_tr])
        y_pred = pipeline.predict(X_te[mask_te])
        m = _metrics(y_te[mask_te].values, y_pred)
        records.append({"fold": fold, "n_train": mask_tr.sum(),
                        "n_test": mask_te.sum(), **m})
    return pd.DataFrame(records)


# ── VIF — diagnóstico de multicolinearidade (apenas OLS) ──────────────────────
def calcular_vif(X_arr: np.ndarray,
                 feature_names: list[str]) -> pd.DataFrame:
    """
    Calcula o VIF de cada coluna de X (sem constante).
    Retorna DataFrame ordenado por VIF decrescente.
    VIF = 1/(1 - R²_j) onde R²_j é de regressão de j sobre demais features.
    """
    vif_vals = []
    for i in range(X_arr.shape[1]):
        col = X_arr[:, i]
        if col.std() < 1e-10:   # constante → VIF matematicamente indefinido
            vif_vals.append(np.nan)
            continue
        try:
            v = float(_sm_vif(X_arr, i))
        except Exception:
            v = np.inf          # colinearidade perfeita → posto deficiente
        vif_vals.append(v)
    df = pd.DataFrame({"feature": feature_names, "VIF": vif_vals})
    # NaN (constantes) ficam no final; infinitos lideram o ranking
    return df.sort_values("VIF", ascending=False, na_position="last").reset_index(drop=True)


def vif_iterativo(feats: list[str], X_df: pd.DataFrame,
                  thresh: float = VIF_THRESH
                  ) -> tuple[list, pd.DataFrame, pd.DataFrame, list]:
    """
    Remove iterativamente a feature com maior VIF acima de `thresh`.
    Cada iteração re-calcula o VIF sobre as features restantes.

    Retorna
    -------
    feats_ok  : lista de features sem alta multicolinearidade
    df_antes  : tabela VIF original (todas as features)
    df_depois : tabela VIF após tratamento
    removidas : [(feature, vif_no_momento_da_remocao), ...]
    """
    imp = SimpleImputer(strategy="median")
    feats_cur = list(feats)

    # VIF antes
    X0 = imp.fit_transform(X_df[feats_cur].astype(float))
    df_antes = calcular_vif(X0, feats_cur)

    removidas: list[tuple[str, float]] = []
    while len(feats_cur) >= 2:
        X_imp = imp.fit_transform(X_df[feats_cur].astype(float))
        df_vif = calcular_vif(X_imp, feats_cur)
        max_vif = df_vif.loc[0, "VIF"]
        if not (np.isinf(max_vif) or max_vif > thresh):
            break
        feat_rem = df_vif.loc[0, "feature"]
        removidas.append((feat_rem, float(max_vif)))
        feats_cur.remove(feat_rem)

    # VIF depois
    X_fim = imp.fit_transform(X_df[feats_cur].astype(float))
    df_depois = calcular_vif(X_fim, feats_cur)
    return feats_cur, df_antes, df_depois, removidas


def fig_vif(df_antes: pd.DataFrame, df_depois: pd.DataFrame,
            removidas: list[tuple], target: str) -> None:
    """
    Gráfico horizontal de barras: VIF antes (esq.) e depois (dir.) do tratamento.
    Linhas de referência em VIF=5 (moderada) e VIF=10 (alta).
    """
    VIF_FIG_DIR.mkdir(parents=True, exist_ok=True)

    def _clip(v: float) -> float:
        if pd.isna(v):      return 0.0   # constante → sem barra
        if np.isinf(v):     return 500.0
        return min(v, 500.0)

    def _lbl(v: float) -> str:
        if pd.isna(v):  return "const."
        if np.isinf(v): return "∞"
        return f"{v:.1f}"

    def _color(v: float) -> str:
        if pd.isna(v):              return "#AAAAAA"  # cinza: constante
        if np.isinf(v) or v > 10:  return "#CC0000"
        if v > 5:                   return "#FFA500"
        return "#55A868"

    def _panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
        # Ascending sort → highest VIF at top; NaN (constantes) at bottom
        dfs = df.sort_values("VIF", ascending=True, na_position="first").reset_index(drop=True)
        feats_p = dfs["feature"].tolist()
        vifs_p  = dfs["VIF"].tolist()
        vdisp   = [_clip(v) for v in vifs_p]
        colors  = [_color(v) for v in vifs_p]

        y = np.arange(len(feats_p))
        ax.barh(y, vdisp, color=colors, edgecolor="white", alpha=0.85, height=0.72)

        finite_vifs = [_clip(v) for v in vifs_p if not pd.isna(v)]
        ref_lim = max(max(finite_vifs, default=0) * 1.12, 15)
        ax.axvline(5,  color="#FFA500", linewidth=1.2, linestyle="--",
                   alpha=0.7, label="VIF = 5 (moderada)")
        ax.axvline(10, color="#CC0000", linewidth=1.5, linestyle="-",
                   alpha=0.8, label="VIF = 10 (alta)")

        for i, (v_raw, v_d) in enumerate(zip(vifs_p, vdisp)):
            if v_d > 0:
                ax.text(v_d + ref_lim * 0.01, i,
                        _lbl(v_raw), va="center", fontsize=6.5)
            else:
                ax.text(ref_lim * 0.01, i,
                        _lbl(v_raw), va="center", fontsize=6.5, color="#777777")

        ax.set_yticks(y)
        ax.set_yticklabels(feats_p, fontsize=6.5)
        ax.set_xlim(0, ref_lim)
        ax.set_xlabel("VIF", fontsize=9)
        ax.set_title(title, fontsize=10, pad=5)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    n_antes  = len(df_antes)
    n_depois = len(df_depois)
    h = max(5, max(n_antes, n_depois) * 0.22)
    fig, axes = plt.subplots(1, 2, figsize=(16, h))
    fig.suptitle(
        f"Diagnóstico VIF — OLS — {TARGET_LABELS[target]}\n"
        f"(limiar = {VIF_THRESH:.0f}; removidas iterativamente: {len(removidas)})",
        fontsize=11, fontweight="bold",
    )

    _panel(axes[0], df_antes,  f"Antes  ({n_antes} features)")
    _panel(axes[1], df_depois, f"Depois ({n_depois} features)")

    if removidas:
        note_feats = ", ".join(f[0] for f in removidas[:6])
        if len(removidas) > 6:
            note_feats += f"… (+{len(removidas) - 6})"
        fig.text(0.5, 0.00,
                 f"Removidas: {note_feats}",
                 ha="center", fontsize=7.5, color="#555555", style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fname = VIF_FIG_DIR / f"vif_{target}.png"
    plt.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close()
    logging.info(f"    Figura VIF salva: {fname.name}")


# ── Baseline naïve ────────────────────────────────────────────────────────────
def avaliar_naive(X_df: pd.DataFrame, y_s: pd.Series,
                  cv_splits: list[tuple], target: str) -> pd.DataFrame:
    """Persistência: ŷ = L1_{target}. Sem treinamento."""
    lag_col = f"L1_{target}"
    records = []
    for fold, (_, te_pos) in enumerate(cv_splits, 1):
        y_te   = y_s.iloc[te_pos]
        y_pred = X_df.iloc[te_pos][lag_col]
        mask   = y_te.notna() & y_pred.notna()
        m = _metrics(y_te[mask].values, y_pred[mask].values.astype(float))
        records.append({"fold": fold, "n_train": 0, "n_test": int(mask.sum()), **m})
    return pd.DataFrame(records)


# ── Tuning ─────────────────────────────────────────────────────────────────────
def tunar_rf(X: np.ndarray, y: np.ndarray,
             cv_splits: list[tuple], target: str) -> tuple[Pipeline, dict, pd.DataFrame]:
    """
    Grid Search exaustivo para Random Forest.
    Critério: neg_mean_squared_error (EQM).
    """
    gs = GridSearchCV(
        estimator=pipeline_rf(),
        param_grid=GRID_RF,
        scoring="neg_mean_squared_error",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    gs.fit(X, y)

    df_res = pd.DataFrame(gs.cv_results_)
    df_res["rmse_cv"] = np.sqrt(-df_res["mean_test_score"])
    df_res = df_res.sort_values("rmse_cv")
    df_res["target"] = target
    df_res["modelo"] = "RandomForest"

    best_params = {k.replace("model__", ""): v
                   for k, v in gs.best_params_.items()}
    return gs.best_estimator_, best_params, df_res


def tunar_xgb(X: np.ndarray, y: np.ndarray,
              cv_splits: list[tuple], target: str) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Random Search para XGBoost (L1/L2 incluídos no espaço de busca)."""
    rs = RandomizedSearchCV(
        estimator=pipeline_xgb(),
        param_distributions=DIST_XGB,
        n_iter=N_ITER_RANDOM,
        scoring="neg_mean_squared_error",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE,
        return_train_score=False,
    )
    rs.fit(X, y)

    df_res = pd.DataFrame(rs.cv_results_)
    df_res["rmse_cv"] = np.sqrt(-df_res["mean_test_score"])
    df_res = df_res.sort_values("rmse_cv")
    df_res["target"] = target
    df_res["modelo"] = "XGBoost"

    best_params = {k.replace("model__", ""): v
                   for k, v in rs.best_params_.items()}
    return rs.best_estimator_, best_params, df_res


def tunar_lgbm(X: np.ndarray, y: np.ndarray,
               cv_splits: list[tuple], target: str) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Random Search para LightGBM (L1/L2 incluídos no espaço de busca)."""
    rs = RandomizedSearchCV(
        estimator=pipeline_lgbm(),
        param_distributions=DIST_LGBM,
        n_iter=N_ITER_RANDOM,
        scoring="neg_mean_squared_error",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE,
        return_train_score=False,
    )
    rs.fit(X, y)

    df_res = pd.DataFrame(rs.cv_results_)
    df_res["rmse_cv"] = np.sqrt(-df_res["mean_test_score"])
    df_res = df_res.sort_values("rmse_cv")
    df_res["target"] = target
    df_res["modelo"] = "LightGBM"

    best_params = {k.replace("model__", ""): v
                   for k, v in rs.best_params_.items()}
    return rs.best_estimator_, best_params, df_res


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Treinamento com Tuning – Cresol TCC ===")

    df = pd.read_parquet(PANEL)
    logging.info(f"Painel: {df.shape[0]} linhas × {df.shape[1]} colunas")

    all_cv_records = []
    all_tuning_dfs = []
    best_params_log = {}

    for target in TARGETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"TARGET: {target}")
        logging.info(f"{'='*60}")

        outros = [t for t in TARGETS if t != target]
        feats  = [f for f in ALL_FEATURES if f in df.columns and f not in [target] + outros]
        mask   = df[target].notna()
        X_df   = df.loc[mask, feats]
        y_s    = df.loc[mask, target]
        periodo = df.loc[mask, "periodo_str"].reset_index(drop=True)
        X_df    = X_df.reset_index(drop=True)
        y_s     = y_s.reset_index(drop=True)

        logging.info(f"  Amostras: {len(y_s)}, features: {len(feats)}")

        # ── Splits temporais para sklearn ──────────────────────────────────────
        cv_splits = temporal_cv_sklearn(periodo)
        logging.info(f"  Folds CV: {len(cv_splits)}")
        for i, (tr, te) in enumerate(cv_splits, 1):
            p_tr = sorted(set(periodo.iloc[tr]))
            p_te = sorted(set(periodo.iloc[te]))
            logging.info(f"    Fold {i}: treino [{p_tr[0]}–{p_tr[-1]}] "
                         f"({len(tr)} obs) | teste [{p_te[0]}–{p_te[-1]}] ({len(te)} obs)")

        X_np = X_df.values  # array numpy para Grid/Random Search
        y_np = y_s.values

        best_params_log[target] = {}

        # ── 1. OLS (MQO) — sem tuning ─────────────────────────────────────────
        logging.info("\n  [1] OLS (MQO) — baseline…")
        ols = pipeline_ols()
        df_cv_ols = avaliar_cv(ols, X_df, y_s, cv_splits)
        df_cv_ols["modelo"] = "OLS"
        df_cv_ols["target"] = target
        all_cv_records.append(df_cv_ols)
        logging.info(f"      R² médio={df_cv_ols['r2'].mean():.4f}  "
                     f"RMSE médio={df_cv_ols['rmse'].mean():,.0f}")

        # Refit OLS em todos os dados
        ols.fit(X_np, y_np)
        joblib.dump({"pipeline": ols, "features": feats},
                    MODEL_DIR / f"ols_{target}.joblib")

        # ── 1b. Diagnóstico VIF (Regressão Linear — OLS apenas) ───────────────
        logging.info("\n  [1b] Diagnóstico de Multicolinearidade — VIF…")
        feats_vif, df_vif_antes, df_vif_depois, removidas_vif = vif_iterativo(
            feats, X_df
        )

        n_alto = int((df_vif_antes["VIF"].replace(np.inf, 1e9) > VIF_THRESH).sum())
        logging.info(f"      Features originais : {len(feats)} | com VIF > {VIF_THRESH:.0f}: {n_alto}")

        # ── Tabela VIF ANTES ──────────────────────────────────────────────────
        logging.info(f"\n      {'Feature':<42} {'VIF':>8}  {'Status'}")
        logging.info(f"      {'-'*62}")
        for _, row in df_vif_antes.iterrows():
            v = row["VIF"]
            if pd.isna(v):
                v_str, status = "   const.", "← variância zero (coluna constante)"
            elif np.isinf(v):
                v_str, status = "       ∞", "← ALTO"
            else:
                v_str = f"{v:8.2f}"
                status = "← ALTO" if v > VIF_THRESH else ("← mod." if v > 5 else "")
            logging.info(f"      {row['feature']:<42} {v_str}  {status}")

        if not removidas_vif:
            logging.info(f"\n      Nenhuma feature com VIF > {VIF_THRESH:.0f}. Sem tratamento necessário.")
        else:
            # ── Remoção iterativa ─────────────────────────────────────────────
            logging.info(f"\n      Removidas iterativamente ({len(removidas_vif)}):")
            for ft, v in removidas_vif:
                v_str = "∞" if np.isinf(v) else f"{v:.2f}"
                logging.info(f"        VIF = {v_str:>8} → {ft}")

            logging.info(f"\n      Features restantes: {len(feats_vif)}")
            logging.info(f"\n      {'Feature':<42} {'VIF':>8}  {'Status'}")
            logging.info(f"      {'-'*62}")
            for _, row in df_vif_depois.iterrows():
                v = row["VIF"]
                if pd.isna(v):
                    v_str, status = "   const.", "← variância zero"
                else:
                    v_str = f"{v:8.2f}"
                    status = "← mod." if v > 5 else ""
                logging.info(f"      {row['feature']:<42} {v_str}  {status}")

        # ── Salvar tabelas VIF ────────────────────────────────────────────────
        def _vif_csv(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()
            d["VIF"] = d["VIF"].apply(
                lambda v: None if pd.isna(v) else (999.0 if np.isinf(v) else round(v, 4)))
            d["status"] = d["VIF"].apply(
                lambda v: "constante" if v is None else (
                    "alto" if v > VIF_THRESH else ("moderado" if v > 5 else "ok")))
            return d

        _vif_csv(df_vif_antes).to_csv(
            TUNING_DIR / f"vif_antes_{target}.csv", index=False, encoding="utf-8-sig")
        _vif_csv(df_vif_depois).to_csv(
            TUNING_DIR / f"vif_depois_{target}.csv", index=False, encoding="utf-8-sig")
        logging.info(f"\n      CSVs VIF salvos em: {TUNING_DIR}")

        # ── Figura VIF ────────────────────────────────────────────────────────
        fig_vif(df_vif_antes, df_vif_depois, removidas_vif, target)

        # ── OLS com features VIF-tratadas (comparação) ────────────────────────
        if removidas_vif:
            X_df_vif = X_df[feats_vif].copy()
            for c in FEAT_CAT:
                if c in X_df_vif.columns:
                    X_df_vif[c] = X_df_vif[c].astype(float)
            ols_vif = pipeline_ols()
            df_cv_ols_vif = avaliar_cv(ols_vif, X_df_vif, y_s, cv_splits)

            r2_orig  = df_cv_ols["r2"].mean()
            r2_vif   = df_cv_ols_vif["r2"].mean()
            rmse_orig = df_cv_ols["rmse"].mean()
            rmse_vif  = df_cv_ols_vif["rmse"].mean()
            delta_r2   = r2_vif   - r2_orig
            delta_rmse = rmse_vif - rmse_orig
            logging.info(f"\n      Comparação OLS original vs OLS VIF-tratado:")
            logging.info(
                f"        OLS_orig ({len(feats):2d} features): "
                f"R²={r2_orig:.4f}  RMSE={rmse_orig:,.0f}")
            logging.info(
                f"        OLS_VIF  ({len(feats_vif):2d} features): "
                f"R²={r2_vif:.4f}  RMSE={rmse_vif:,.0f}"
                f"  (ΔR²={delta_r2:+.4f}  ΔRMSE={delta_rmse:+,.0f})")

            # Salvar OLS VIF-tratado
            ols_vif.fit(X_df_vif.values, y_s.values)
            joblib.dump({"pipeline": ols_vif, "features": feats_vif},
                        MODEL_DIR / f"ols_vif_{target}.joblib")
            logging.info(f"        Modelo salvo: ols_vif_{target}.joblib")

        # ── 2. Random Forest — Grid Search ────────────────────────────────────
        logging.info(f"\n  [2] Random Forest — Grid Search "
                     f"({len(GRID_RF['model__n_estimators']) * len(GRID_RF['model__max_depth']) * len(GRID_RF['model__max_features'])} "
                     f"combinações × {len(cv_splits)} folds)…")
        rf_best, rf_params, df_gs = tunar_rf(X_np, y_np, cv_splits, target)
        all_tuning_dfs.append(
            df_gs[["target", "modelo", "rmse_cv",
                   "param_model__n_estimators", "param_model__max_depth",
                   "param_model__max_features"]].head(10)
        )
        logging.info(f"      Melhores params RF: {rf_params}")
        logging.info(f"      RMSE_cv (melhor): {df_gs['rmse_cv'].iloc[0]:,.0f}")
        best_params_log[target]["RF"] = rf_params

        df_cv_rf = avaliar_cv(rf_best, X_df, y_s, cv_splits)
        df_cv_rf["modelo"] = "RandomForest_tuned"
        df_cv_rf["target"] = target
        all_cv_records.append(df_cv_rf)
        logging.info(f"      R² médio={df_cv_rf['r2'].mean():.4f}  "
                     f"RMSE médio={df_cv_rf['rmse'].mean():,.0f}")

        # Salvar RF
        rf_best.fit(X_np, y_np)
        joblib.dump({"pipeline": rf_best, "features": feats, "params": rf_params},
                    MODEL_DIR / f"rf_tuned_{target}.joblib")
        df_gs.to_csv(TUNING_DIR / f"gridsearch_rf_{target}.csv",
                     index=False, encoding="utf-8-sig")

        # ── 3. XGBoost — Random Search ────────────────────────────────────────
        logging.info(f"\n  [3] XGBoost — Random Search "
                     f"({N_ITER_RANDOM} iterações × {len(cv_splits)} folds)…")
        xgb_best, xgb_params, df_rs_xgb = tunar_xgb(X_np, y_np, cv_splits, target)
        logging.info(f"      Melhores params XGB: {xgb_params}")
        logging.info(f"      RMSE_cv (melhor): {df_rs_xgb['rmse_cv'].iloc[0]:,.0f}")
        best_params_log[target]["XGBoost"] = xgb_params

        df_cv_xgb = avaliar_cv(xgb_best, X_df, y_s, cv_splits)
        df_cv_xgb["modelo"] = "XGBoost_tuned"
        df_cv_xgb["target"] = target
        all_cv_records.append(df_cv_xgb)
        logging.info(f"      R² médio={df_cv_xgb['r2'].mean():.4f}  "
                     f"RMSE médio={df_cv_xgb['rmse'].mean():,.0f}")

        xgb_best.fit(X_np, y_np)
        joblib.dump({"pipeline": xgb_best, "features": feats, "params": xgb_params},
                    MODEL_DIR / f"xgb_tuned_{target}.joblib")
        df_rs_xgb.to_csv(TUNING_DIR / f"randomsearch_xgb_{target}.csv",
                         index=False, encoding="utf-8-sig")

        # ── 4. LightGBM — Random Search ───────────────────────────────────────
        logging.info(f"\n  [4] LightGBM — Random Search "
                     f"({N_ITER_RANDOM} iterações × {len(cv_splits)} folds)…")
        lgbm_best, lgbm_params, df_rs_lgbm = tunar_lgbm(X_np, y_np, cv_splits, target)
        logging.info(f"      Melhores params LGBM: {lgbm_params}")
        logging.info(f"      RMSE_cv (melhor): {df_rs_lgbm['rmse_cv'].iloc[0]:,.0f}")
        best_params_log[target]["LightGBM"] = lgbm_params

        df_cv_lgbm = avaliar_cv(lgbm_best, X_df, y_s, cv_splits)
        df_cv_lgbm["modelo"] = "LightGBM_tuned"
        df_cv_lgbm["target"] = target
        all_cv_records.append(df_cv_lgbm)
        logging.info(f"      R² médio={df_cv_lgbm['r2'].mean():.4f}  "
                     f"RMSE médio={df_cv_lgbm['rmse'].mean():,.0f}")

        lgbm_best.fit(X_np, y_np)
        joblib.dump({"pipeline": lgbm_best, "features": feats, "params": lgbm_params},
                    MODEL_DIR / f"lgbm_tuned_{target}.joblib")
        df_rs_lgbm.to_csv(TUNING_DIR / f"randomsearch_lgbm_{target}.csv",
                          index=False, encoding="utf-8-sig")

        # ── 5. Naïve — Persistência ───────────────────────────────────────────
        logging.info("\n  [5] Naïve — Persistência (ŷ = L1 do target)…")
        df_cv_naive = avaliar_naive(X_df, y_s, cv_splits, target)
        df_cv_naive["modelo"] = "Naive"
        df_cv_naive["target"] = target
        all_cv_records.append(df_cv_naive)
        logging.info(f"      R² médio={df_cv_naive['r2'].mean():.4f}  "
                     f"RMSE médio={df_cv_naive['rmse'].mean():,.0f}")

        # ── Resumo do target ───────────────────────────────────────────────────
        logging.info(f"\n  Resumo — {target}:")
        for df_cv, nome in [(df_cv_ols,   "OLS       "),
                             (df_cv_rf,   "RF_tuned  "),
                             (df_cv_xgb,  "XGB_tuned "),
                             (df_cv_lgbm, "LGBM_tuned"),
                             (df_cv_naive,"Naive     ")]:
            logging.info(f"    {nome}  R²={df_cv['r2'].mean():.4f} ± {df_cv['r2'].std():.4f}"
                         f"  RMSE={df_cv['rmse'].mean():>9,.0f}"
                         f"  MAE={df_cv['mae'].mean():>8,.0f}")

    # ── Consolidar resultados ──────────────────────────────────────────────────
    df_all = pd.concat(all_cv_records, ignore_index=True)
    df_all.to_csv(OUT_DIR / "cv_results_tuned.csv", index=False, encoding="utf-8-sig")

    # Tabela pivot final
    pivot = (df_all.groupby(["target", "modelo"])[["r2", "rmse", "mae"]]
             .mean().round(4))

    logging.info(f"\n{'='*60}")
    logging.info("TABELA COMPARATIVA FINAL (R² médio):")
    logging.info("\n" + pivot["r2"].unstack("modelo").to_string())

    logging.info("\nTABELA COMPARATIVA FINAL (RMSE médio):")
    logging.info("\n" + pivot["rmse"].unstack("modelo").to_string())

    # Salvar melhores parâmetros em JSON
    params_path = OUT_DIR / "best_params_tuned.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_log, f, indent=2, default=str)
    logging.info(f"\nMelhores parâmetros salvos: {params_path.name}")
    logging.info(f"Resultados CV: cv_results_tuned.csv")
    logging.info(f"Modelos: {MODEL_DIR}")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
