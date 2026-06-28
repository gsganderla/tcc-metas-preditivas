#!/usr/bin/env python3
"""
12_model_evaluation.py
=======================
Avaliação completa dos modelos com diagnóstico de instabilidade preditiva.

Métricas por modelo e target (OOS — out-of-sample por fold temporal):
  R²        — variância explicada pelo modelo
  RMSE      — raiz do EQM; sensível a erros extremos (eleva ao quadrado)
  MAE       — erro médio absoluto; robusto a outliers
  RMSE/MAE  — razão de instabilidade:
               < 1.35 → distribuição gaussiana de erros (referência teórica)
               1.35–1.5 → cauda leve
               1.5–2.0 → erros pontuais moderadamente grandes
               > 2.0  → instabilidade preditiva — erros extremos dominam RMSE

Quando RMSE/MAE > 1.5, identifica e reporta as N observações com maior
resíduo absoluto: período, cooperativa, município, UF, real, previsto, desvio.

Saídas
------
  reports/figures/evaluation/metricas_comparativas.png
  reports/figures/evaluation/rmse_mae_ratio.png
  reports/figures/evaluation/ganho_naive.png
  reports/figures/evaluation/residuos_distribuicao.png
  reports/figures/evaluation/actual_vs_predicted.png
  reports/figures/evaluation/erros_pontuais.png
  reports/figures/evaluation/metricas_por_porte.png
  reports/evaluation_metrics.csv
  reports/metricas_por_porte.csv
  reports/large_errors_report.csv
  logs/model_evaluation.log
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
PANEL       = ROOT / "data" / "processed" / "panel_features_clean.parquet"
PARAMS_FILE = ROOT / "data" / "processed" / "best_params_tuned.json"
FIG_DIR     = ROOT / "reports" / "figures" / "evaluation"
REP_DIR     = ROOT / "reports"
LOG_DIR     = ROOT / "logs"

# ── Configuração ───────────────────────────────────────────────────────────────
TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]
MODELS  = ["Naive", "OLS", "RF", "XGBoost", "LightGBM"]

MODEL_LABELS = {
    "Naive":    "Naïve (L1)",
    "OLS":      "OLS (MQO)",
    "RF":       "Random Forest",
    "XGBoost":  "XGBoost",
    "LightGBM": "LightGBM",
}
TARGET_LABELS = {
    "vol_credito_rs_mil": "Volume de Crédito (R$ mil)",
    "captacao_rs_mil":    "Captação (R$ mil)",
}
COLORS = {
    "Naive":    "#9E9E9E",
    "OLS":      "#4C72B0",
    "RF":       "#55A868",
    "XGBoost":  "#C44E52",
    "LightGBM": "#DD8452",
}

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
                + [f"L1_{c}" for c in MACRO_COLS] + PIB_COLS
                + [f"L1_{c}" for c in COMOD_COLS]
                + [f"L1_{c}" for c in CR_COLS]
                + FEAT_CAT + FEAT_SAFRA)

N_SPLITS          = 5
MIN_TRAIN_PERIODS = 8
RANDOM_STATE      = 42

RATIO_LEVE       = 1.35
RATIO_MODERADA   = 1.50
RATIO_GRAVE      = 2.00
N_LARGE_ERRORS   = 20   # top-N erros reportados por modelo/target


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "model_evaluation.log", encoding="utf-8"),
        ],
    )


# ── Splits temporais ───────────────────────────────────────────────────────────
def temporal_cv_sklearn(periodo_col: pd.Series) -> list[tuple]:
    periodos  = sorted(periodo_col.unique())
    n         = len(periodos)
    test_size = max(1, (n - MIN_TRAIN_PERIODS) // N_SPLITS)
    arr       = periodo_col.values
    splits    = []
    for i in range(N_SPLITS):
        split = MIN_TRAIN_PERIODS + i * test_size
        if split >= n:
            break
        train_p   = set(periodos[:split])
        test_p    = set(periodos[split: split + test_size])
        if not test_p:
            break
        splits.append((
            np.where(np.isin(arr, list(train_p)))[0],
            np.where(np.isin(arr, list(test_p)))[0],
        ))
    return splits


# ── Pipelines com melhores parâmetros ─────────────────────────────────────────
def build_pipeline(model_name: str, params: dict) -> Pipeline:
    if model_name == "OLS":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model",   LinearRegression())])
    if model_name == "RF":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model",   RandomForestRegressor(
                             **params, random_state=RANDOM_STATE, n_jobs=-1))])
    if model_name == "XGBoost":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model",   XGBRegressor(
                             **params, random_state=RANDOM_STATE, n_jobs=-1,
                             verbosity=0, eval_metric="rmse"))])
    if model_name == "LightGBM":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model",   LGBMRegressor(
                             **params, random_state=RANDOM_STATE, n_jobs=-1,
                             verbose=-1))])
    raise ValueError(f"Modelo desconhecido: {model_name}")


# ── MAPE com proteção contra zero ─────────────────────────────────────────────
def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, min_denom: float = 1.0) -> float:
    """MAPE descartando observações onde |y_true| < min_denom (evita divisão por zero)."""
    mask = np.abs(y_true) >= min_denom
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Avaliação OOS ──────────────────────────────────────────────────────────────
def oos_predictions(pipeline: Pipeline, X_df: pd.DataFrame, y_s: pd.Series,
                    df_meta: pd.DataFrame,
                    cv_splits: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Treina em cada fold e coleta previsões out-of-sample.
    Retorna (df_predicoes, df_metricas_fold).
    """
    all_preds    = []
    fold_metrics = []

    for fold, (tr_pos, te_pos) in enumerate(cv_splits, 1):
        X_tr = X_df.iloc[tr_pos].values
        y_tr = y_s.iloc[tr_pos].values
        X_te = X_df.iloc[te_pos].values
        y_te = y_s.iloc[te_pos].values
        meta = df_meta.iloc[te_pos].reset_index(drop=True)

        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)

        pipeline.fit(X_tr[mask_tr], y_tr[mask_tr])
        y_pred = pipeline.predict(X_te[mask_te])
        y_true = y_te[mask_te]
        meta_f = meta[mask_te].reset_index(drop=True)

        resid = y_true - y_pred

        chunk = meta_f[["cnpj8", "periodo_str"]].copy()
        for col in ["municipio", "uf", "porte_municipio"]:
            if col in meta_f.columns:
                chunk[col] = meta_f[col].values
        chunk["fold"]         = fold
        chunk["y_true"]       = y_true
        chunk["y_pred"]       = y_pred
        chunk["residual"]     = resid
        chunk["abs_residual"] = np.abs(resid)
        chunk["pct_erro"]     = np.abs(resid) / (np.abs(y_true) + 1e-9) * 100
        all_preds.append(chunk)

        r2_f   = r2_score(y_true, y_pred)
        rmse_f = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_f  = mean_absolute_error(y_true, y_pred)
        fold_metrics.append({
            "fold": fold, "n_test": mask_te.sum(),
            "r2": r2_f, "rmse": rmse_f, "mae": mae_f,
            "rmse_mae": rmse_f / (mae_f + 1e-9),
            "mape": mape_safe(y_true, y_pred),
        })

    df_preds = pd.concat(all_preds, ignore_index=True)
    df_folds = pd.DataFrame(fold_metrics)
    return df_preds, df_folds


# ── Baseline naïve OOS ────────────────────────────────────────────────────────
def oos_naive(X_df: pd.DataFrame, y_s: pd.Series,
              df_meta: pd.DataFrame, cv_splits: list[tuple],
              target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baseline de persistência: ŷ_{i,t} = y_{i,t-1} = L1_{target}.
    Sem treinamento — usa a feature de lag-1 do target diretamente.
    """
    lag_col    = f"L1_{target}"
    all_preds  = []
    fold_mets  = []

    for fold, (_, te_pos) in enumerate(cv_splits, 1):
        y_te   = y_s.iloc[te_pos].values
        y_pred = X_df.iloc[te_pos][lag_col].values.astype(float)
        meta   = df_meta.iloc[te_pos].reset_index(drop=True)

        mask   = ~(np.isnan(y_te) | np.isnan(y_pred))
        y_te_m, y_p_m = y_te[mask], y_pred[mask]
        meta_f = meta[mask].reset_index(drop=True)

        resid  = y_te_m - y_p_m
        chunk  = meta_f[["cnpj8", "periodo_str"]].copy()
        for c in ["municipio", "uf", "porte_municipio"]:
            if c in meta_f.columns:
                chunk[c] = meta_f[c].values
        chunk["fold"]         = fold
        chunk["y_true"]       = y_te_m
        chunk["y_pred"]       = y_p_m
        chunk["residual"]     = resid
        chunk["abs_residual"] = np.abs(resid)
        chunk["pct_erro"]     = np.abs(resid) / (np.abs(y_te_m) + 1e-9) * 100
        all_preds.append(chunk)

        rmse_f = np.sqrt(mean_squared_error(y_te_m, y_p_m))
        mae_f  = mean_absolute_error(y_te_m, y_p_m)
        fold_mets.append({"fold": fold, "n_test": int(mask.sum()),
                          "r2": r2_score(y_te_m, y_p_m),
                          "rmse": rmse_f, "mae": mae_f,
                          "rmse_mae": rmse_f / (mae_f + 1e-9),
                          "mape": mape_safe(y_te_m, y_p_m)})

    return pd.concat(all_preds, ignore_index=True), pd.DataFrame(fold_mets)


# ── Métricas consolidadas ──────────────────────────────────────────────────────
def consolidar_metricas(df_preds: pd.DataFrame,
                         df_folds: pd.DataFrame) -> dict:
    """Métricas globais e por fold; identifica erros pontuais grandes."""
    y_true = df_preds["y_true"].values
    y_pred = df_preds["y_pred"].values

    r2_g   = r2_score(y_true, y_pred)
    rmse_g = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_g  = mean_absolute_error(y_true, y_pred)
    mape_g = mape_safe(y_true, y_pred)
    ratio  = rmse_g / (mae_g + 1e-9)

    abs_r = df_preds["abs_residual"].values
    thr   = np.percentile(abs_r, 95)

    if ratio < RATIO_LEVE:
        classif = "Gaussiana (normal)"
    elif ratio < RATIO_MODERADA:
        classif = "Cauda leve"
    elif ratio < RATIO_GRAVE:
        classif = "Moderada — erros pontuais presentes"
    else:
        classif = "SEVERA — erros extremos dominam RMSE"

    mape_col = df_folds["mape"] if "mape" in df_folds.columns else pd.Series(dtype=float)
    return {
        "r2_global":       round(r2_g,   4),
        "rmse_global":     round(rmse_g, 2),
        "mae_global":      round(mae_g,  2),
        "mape_global":     round(mape_g, 2) if not np.isnan(mape_g) else np.nan,
        "rmse_mae_ratio":  round(ratio,  3),
        "classif_ratio":   classif,
        "r2_mean":         round(df_folds["r2"].mean(),   4),
        "r2_std":          round(df_folds["r2"].std(),    4),
        "rmse_mean":       round(df_folds["rmse"].mean(), 2),
        "rmse_std":        round(df_folds["rmse"].std(),  2),
        "mae_mean":        round(df_folds["mae"].mean(),  2),
        "mae_std":         round(df_folds["mae"].std(),   2),
        "mape_mean":       round(float(np.nanmean(mape_col)), 2) if len(mape_col) else np.nan,
        "mape_std":        round(float(np.nanstd(mape_col)),  2) if len(mape_col) else np.nan,
        "p95_abs_residual": round(thr,  2),
        "p99_abs_residual": round(np.percentile(abs_r, 99), 2),
        "n_large_errors":  int((abs_r > thr).sum()),
        "pct_large":       round((abs_r > thr).mean() * 100, 2),
    }


# ── Métricas estratificadas por porte ─────────────────────────────────────────
PORTE_ORDER  = ["pequeno", "intermediario", "grande"]
PORTE_LABELS_MAP = {
    "pequeno":      "Pequeno\n(<40k hab.)",
    "intermediario":"Interm.\n(40–100k)",
    "grande":       "Grande\n(≥100k)",
}
PORTE_COLORS = {"pequeno": "#8BC34A", "intermediario": "#2196F3", "grande": "#FF5722"}


def metricas_por_porte(oos_dict: dict) -> pd.DataFrame:
    """
    R², RMSE, MAE e MAPE estratificados por porte_municipio.
    Retorna DataFrame: target, modelo, porte, n_obs, r2, rmse, mae, mape.
    """
    records = []
    for (target, mname), data in oos_dict.items():
        df_p = data["preds"].copy()
        if "porte_municipio" not in df_p.columns:
            continue
        for porte, grp in df_p.groupby("porte_municipio"):
            y_t = grp["y_true"].values
            y_p = grp["y_pred"].values
            if len(y_t) < 2:
                continue
            rmse_v = np.sqrt(mean_squared_error(y_t, y_p))
            records.append({
                "target": target,
                "modelo": mname,
                "porte":  porte,
                "n_obs":  len(y_t),
                "r2":     round(r2_score(y_t, y_p), 4),
                "rmse":   round(rmse_v, 2),
                "mae":    round(mean_absolute_error(y_t, y_p), 2),
                "mape":   round(mape_safe(y_t, y_p), 2),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["porte"] = pd.Categorical(df["porte"], categories=PORTE_ORDER, ordered=True)
        df = df.sort_values(["target", "modelo", "porte"]).reset_index(drop=True)
    return df


def fig_metricas_por_porte(df_porte: pd.DataFrame) -> None:
    """
    Barras agrupadas: R², RMSE, MAE e MAPE por modelo e porte.
    4 linhas (métricas) × 2 colunas (targets).
    Barras agrupadas por porte (3 por modelo) para comparar heterogeneidade.
    """
    metrics_info = [
        ("r2",   "R² (OOS)",      "r2_mean"),
        ("rmse", "RMSE (R$ mil)", "rmse_mean"),
        ("mae",  "MAE (R$ mil)",  "mae_mean"),
        ("mape", "MAPE (%)",      "mape_mean"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle(
        "Desempenho Preditivo Estratificado por Porte do Município",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, (metric, ylabel, _) in enumerate(metrics_info):
        for col, target in enumerate(TARGETS):
            ax  = axes[row, col]
            sub = df_porte[df_porte["target"] == target]

            x         = np.arange(len(MODELS))
            n_p       = len(PORTE_ORDER)
            bar_w     = 0.72 / n_p

            for pi, porte in enumerate(PORTE_ORDER):
                grp   = sub[sub["porte"] == porte].set_index("modelo")
                vals  = [grp.loc[m, metric] if m in grp.index else np.nan
                         for m in MODELS]
                offset = (pi - n_p / 2 + 0.5) * bar_w
                ax.bar(x + offset, vals, bar_w * 0.9,
                       label=PORTE_LABELS_MAP[porte].replace("\n", " "),
                       color=PORTE_COLORS[porte], alpha=0.82,
                       edgecolor="white", linewidth=0.6)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                               rotation=15, ha="right", fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(TARGET_LABELS[target], fontsize=10, pad=5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if metric == "mape":
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
            elif metric == "r2":
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
            else:
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, framealpha=0.85, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "metricas_por_porte.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: metricas_por_porte.png")


# ── Figuras ────────────────────────────────────────────────────────────────────
def fig_metricas_comparativas(resumo: pd.DataFrame) -> None:
    """
    Barras agrupadas: R², RMSE, MAE por modelo e target.
    3 linhas (uma por métrica) × 2 colunas (um por target).
    """
    metrics = [("r2_mean",   "r2_std",   "R² (médio)",             True,  (0.70, 1.02)),
               ("rmse_mean", "rmse_std", "RMSE (R$ mil)",           False, None),
               ("mae_mean",  "mae_std",  "MAE (R$ mil)",            False, None),
               ("mape_mean", "mape_std", "MAPE (%)",                 False, None)]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    fig.suptitle("Comparação de Métricas — Modelos Tuned vs. Naïve (CV Temporal OOS)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, (col_m, col_s, ylabel, invert_better, ylim) in enumerate(metrics):
        for c, target in enumerate(TARGETS):
            ax   = axes[row, c]
            data = resumo[resumo["target"] == target].copy()
            x    = np.arange(len(MODELS))
            vals = data.set_index("modelo")[col_m].reindex(MODELS)
            errs = data.set_index("modelo")[col_s].reindex(MODELS)
            bars = ax.bar(x, vals, width=0.55, yerr=errs, capsize=4,
                          color=[COLORS[m] for m in MODELS],
                          edgecolor="white", linewidth=0.8, alpha=0.88)

            # Rótulo de valor em cada barra
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                if col_m == "r2_mean":
                    fmt = f"{v:.4f}"
                elif col_m == "mape_mean":
                    fmt = f"{v:.1f}%"
                else:
                    fmt = f"{v:,.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (errs.max() * 0.05 if not np.isnan(errs.max()) else 0),
                        fmt, ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                               rotation=15, ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(TARGET_LABELS[target], fontsize=10, pad=6)
            if col_m == "r2_mean":
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
            elif col_m == "mape_mean":
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
            else:
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "metricas_comparativas.png",
                dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: metricas_comparativas.png")


def fig_rmse_mae_ratio(resumo: pd.DataFrame) -> None:
    """
    Razão RMSE/MAE por modelo e target.
    Linhas de referência: Gaussiana (1.25), leve (1.35), moderada (1.5), grave (2.0).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    fig.suptitle("Razão RMSE/MAE — Indicador de Instabilidade Preditiva",
                 fontsize=12, fontweight="bold")

    REFS = [(1.25, "#888888", "--", "Gaussiana pura (1.25)"),
            (RATIO_LEVE,    "#3A86FF", "-.", f"Cauda leve ({RATIO_LEVE})"),
            (RATIO_MODERADA,"#FFA500", "-",  f"Moderada ({RATIO_MODERADA})"),
            (RATIO_GRAVE,   "#CC0000", "-",  f"Grave ({RATIO_GRAVE:.1f})")]

    for c, target in enumerate(TARGETS):
        ax   = axes[c]
        data = resumo[resumo["target"] == target].copy()
        vals = data.set_index("modelo")["rmse_mae_ratio"].reindex(MODELS)

        colors = []
        for v in vals:
            if v < RATIO_LEVE:
                colors.append("#55A868")
            elif v < RATIO_MODERADA:
                colors.append("#8ECAE6")
            elif v < RATIO_GRAVE:
                colors.append("#FFA500")
            else:
                colors.append("#CC0000")

        bars = ax.barh([MODEL_LABELS[m] for m in MODELS], vals,
                       color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        for ref_val, ref_col, ref_ls, ref_lbl in REFS:
            ax.axvline(ref_val, color=ref_col, linestyle=ref_ls,
                       linewidth=1.3, alpha=0.8, label=ref_lbl)

        ax.set_xlabel("RMSE / MAE", fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10, pad=6)
        ax.set_xlim(0, max(vals.max() * 1.15, RATIO_GRAVE * 1.1))
        ax.spines[["top", "right"]].set_visible(False)
        if c == 1:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "rmse_mae_ratio.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: rmse_mae_ratio.png")


def fig_residuos_distribuicao(oos_dict: dict) -> None:
    """
    Violin plots dos resíduos OOS por modelo e target.
    Permite visualizar caudas pesadas e assimetria.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Distribuição dos Resíduos OOS (real − previsto)",
                 fontsize=12, fontweight="bold")

    for c, target in enumerate(TARGETS):
        ax   = axes[c]
        dfs  = []
        for m in MODELS:
            df_p = oos_dict[(target, m)]["preds"].copy()
            df_p["modelo"] = MODEL_LABELS[m]
            dfs.append(df_p)
        df_all = pd.concat(dfs)

        order = [MODEL_LABELS[m] for m in MODELS]
        palette = {MODEL_LABELS[m]: COLORS[m] for m in MODELS}
        sns.violinplot(data=df_all, x="modelo", y="residual",
                       order=order, palette=palette,
                       inner="quartile", cut=0.5, ax=ax, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel("Resíduo (R$ mil)", fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10, pad=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "residuos_distribuicao.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: residuos_distribuicao.png")


def fig_actual_vs_predicted(oos_dict: dict, resumo: pd.DataFrame) -> None:
    """
    Scatter Real × Previsto para os dois melhores modelos de cada target.
    2 linhas × 2 colunas.
    """
    # Seleciona os dois melhores por R² global por target
    def top2(target):
        sub = resumo[resumo["target"] == target].copy()
        return sub.sort_values("r2_global", ascending=False)["modelo"].head(2).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Real × Previsto (Melhores 2 Modelos por Target)",
                 fontsize=12, fontweight="bold")

    for row, target in enumerate(TARGETS):
        models_top = top2(target)
        for col, mname in enumerate(models_top):
            ax   = axes[row, col]
            df_p = oos_dict[(target, mname)]["preds"]
            y_t  = df_p["y_true"].values
            y_p  = df_p["y_pred"].values
            folds = df_p["fold"].values

            sc = ax.scatter(y_t, y_p, c=folds, cmap="viridis",
                            s=14, alpha=0.55, edgecolors="none")
            lims = [min(y_t.min(), y_p.min()) * 0.95,
                    max(y_t.max(), y_p.max()) * 1.05]
            ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.7, label="y = x")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Real (R$ mil)", fontsize=9)
            ax.set_ylabel("Previsto (R$ mil)", fontsize=9)
            r2v = oos_dict[(target, mname)]["metricas"]["r2_global"]
            ax.set_title(f"{TARGET_LABELS[target]}\n{MODEL_LABELS[mname]}  "
                         f"(R² = {r2v:.4f})", fontsize=9.5, pad=5)
            plt.colorbar(sc, ax=ax, label="Fold", pad=0.02)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: actual_vs_predicted.png")


def fig_erros_pontuais(oos_dict: dict, resumo: pd.DataFrame) -> None:
    """
    |Resíduo| ao longo dos períodos para o melhor modelo de cada target.
    Destaca os pontos acima do P95 (erros pontuais de grande magnitude).
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    fig.suptitle("Magnitude dos Erros Pontuais ao Longo do Tempo",
                 fontsize=12, fontweight="bold")

    for row, target in enumerate(TARGETS):
        ax = axes[row]
        sub = resumo[resumo["target"] == target].sort_values("r2_global", ascending=False)
        best_model = sub.iloc[0]["modelo"]
        df_p = oos_dict[(target, best_model)]["preds"].copy()

        thr = df_p["abs_residual"].quantile(0.95)
        df_p["flag"] = df_p["abs_residual"] > thr

        # Todos os pontos (pequenos, transparentes)
        ax.scatter(df_p["periodo_str"], df_p["abs_residual"],
                   color=COLORS[best_model], s=12, alpha=0.30, label="Resíduo")

        # Erros grandes
        grande = df_p[df_p["flag"]]
        ax.scatter(grande["periodo_str"], grande["abs_residual"],
                   color="#CC0000", s=45, zorder=5, alpha=0.8,
                   marker="^", label=f"|e| > P95 ({thr:,.0f})")

        ax.axhline(thr, color="#CC0000", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Limiar P95")
        ax.set_ylabel("|Resíduo| (R$ mil)", fontsize=9)
        ax.set_title(f"{TARGET_LABELS[target]} — {MODEL_LABELS[best_model]}",
                     fontsize=10, pad=5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        tick_labels = sorted(df_p["periodo_str"].unique())
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "erros_pontuais.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: erros_pontuais.png")


# ── Ganho sobre o Naïve ───────────────────────────────────────────────────────
def fig_ganho_naive(resumo: pd.DataFrame) -> None:
    """
    Redução percentual de RMSE e MAE de cada modelo vs. o baseline naïve.
    Barras positivas = melhora; negativas = modelo pior que o naïve.
    """
    OUTROS = [m for m in MODELS if m != "Naive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Ganho sobre o Baseline Naïve (redução % em RMSE e MAE)",
                 fontsize=12, fontweight="bold")

    for c, target in enumerate(TARGETS):
        ax    = axes[c]
        sub   = resumo[resumo["target"] == target].set_index("modelo")
        rmse_n = sub.loc["Naive", "rmse_mean"]
        mae_n  = sub.loc["Naive", "mae_mean"]

        labels, rmse_ganho, mae_ganho = [], [], []
        for m in OUTROS:
            labels.append(MODEL_LABELS[m])
            rmse_ganho.append((rmse_n - sub.loc[m, "rmse_mean"]) / rmse_n * 100)
            mae_ganho.append( (mae_n  - sub.loc[m, "mae_mean"])  / mae_n  * 100)

        x  = np.arange(len(OUTROS))
        w  = 0.38
        b1 = ax.bar(x - w / 2, rmse_ganho, w, label="RMSE", alpha=0.85,
                    color=[COLORS[m] for m in OUTROS], edgecolor="white")
        b2 = ax.bar(x + w / 2, mae_ganho,  w, label="MAE",  alpha=0.55,
                    color=[COLORS[m] for m in OUTROS], edgecolor="white",
                    linestyle="--", linewidth=0.8)

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        for bar, v in zip(list(b1) + list(b2), rmse_ganho + mae_ganho):
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if v >= 0 else -1.0),
                    f"{sign}{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel("Redução % vs. Naïve", fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10, pad=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if c == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ganho_naive.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: ganho_naive.png")


# ── Relatório de erros grandes ─────────────────────────────────────────────────
def large_errors_report(oos_dict: dict) -> pd.DataFrame:
    """
    Consolida os N maiores erros OOS por modelo e target.
    Colunas: target, modelo, fold, periodo_str, cnpj8, municipio, uf,
             y_true, y_pred, residual, abs_residual, pct_erro
    """
    records = []
    for (target, mname), data in oos_dict.items():
        df_p = data["preds"].copy()
        df_p["target"] = target
        df_p["modelo"] = mname
        top = df_p.nlargest(N_LARGE_ERRORS, "abs_residual")
        records.append(top)

    df = pd.concat(records, ignore_index=True)
    cols_base = ["target", "modelo", "fold", "periodo_str", "cnpj8"]
    for c in ["municipio", "uf"]:
        if c in df.columns:
            cols_base.append(c)
    cols_base += ["y_true", "y_pred", "residual", "abs_residual", "pct_erro"]
    df = df[[c for c in cols_base if c in df.columns]]
    df = df.sort_values(["target", "modelo", "abs_residual"], ascending=[True, True, False])

    for col in ["y_true", "y_pred", "residual", "abs_residual"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    df["pct_erro"] = df["pct_erro"].round(2)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Avaliação de Modelos — Cresol TCC ===")

    df = pd.read_parquet(PANEL)
    logging.info(f"Painel: {df.shape[0]} linhas × {df.shape[1]} colunas")

    with open(PARAMS_FILE, encoding="utf-8") as f:
        best_params = json.load(f)

    # Colunas de metadados disponíveis para rastreio de erros e estratificação
    meta_cols = ["cnpj8", "periodo_str"] + [
        c for c in ["municipio", "uf", "porte_municipio"] if c in df.columns
    ]

    oos_dict    = {}   # (target, model) → {"preds": df, "folds": df, "metricas": dict}
    resumo_rows = []

    for target in TARGETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"TARGET: {target}")

        outros = [t for t in TARGETS if t != target]
        feats  = [f for f in ALL_FEATURES if f in df.columns and f not in [target] + outros]
        mask   = df[target].notna()
        X_df   = df.loc[mask, feats].reset_index(drop=True)
        y_s    = df.loc[mask, target].reset_index(drop=True)
        df_meta = df.loc[mask, [c for c in meta_cols if c in df.columns]].reset_index(drop=True)
        periodo = df.loc[mask, "periodo_str"].reset_index(drop=True)

        cv_splits = temporal_cv_sklearn(periodo)

        for mname in MODELS:
            logging.info(f"  {MODEL_LABELS[mname]}…")
            if mname == "Naive":
                df_preds, df_folds = oos_naive(X_df, y_s, df_meta, cv_splits, target)
            else:
                params = {} if mname == "OLS" else best_params.get(target, {}).get(
                    mname if mname != "RF" else "RF", {})
                pipe = build_pipeline(mname, params)
                df_preds, df_folds = oos_predictions(pipe, X_df, y_s, df_meta, cv_splits)
            met = consolidar_metricas(df_preds, df_folds)

            oos_dict[(target, mname)] = {
                "preds":   df_preds,
                "folds":   df_folds,
                "metricas": met,
            }

            ratio = met["rmse_mae_ratio"]
            flag  = ("⚠ INSTABILIDADE" if ratio >= RATIO_MODERADA else
                     "→ cauda leve"    if ratio >= RATIO_LEVE     else "")
            logging.info(
                f"    R²={met['r2_global']:.4f}  "
                f"RMSE={met['rmse_global']:>10,.0f}  "
                f"MAE={met['mae_global']:>9,.0f}  "
                f"RMSE/MAE={ratio:.3f}  {flag}"
            )
            logging.info(f"    Distribuição: {met['classif_ratio']} "
                         f"| Erros P95+: {met['n_large_errors']} ({met['pct_large']}%)")

            resumo_rows.append({
                "target":          target,
                "modelo":          mname,
                **met,
            })

    resumo = pd.DataFrame(resumo_rows)

    # ── Log tabela final ───────────────────────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info("TABELA RESUMO — R², RMSE, MAE, MAPE, RMSE/MAE:")
    for target in TARGETS:
        logging.info(f"\n  {TARGET_LABELS[target]}:")
        logging.info(
            f"  {'Modelo':<18} {'R²':>8} {'RMSE':>12} {'MAE':>10} {'MAPE':>8} {'Ratio':>8}  Classificação"
        )
        sub = resumo[resumo["target"] == target].set_index("modelo")
        for m in MODELS:
            row = sub.loc[m]
            mape_str = (f"{row['mape_global']:>7.1f}%"
                        if not pd.isna(row.get("mape_global", np.nan)) else "    N/A ")
            logging.info(
                f"  {MODEL_LABELS[m]:<18} {row['r2_global']:>8.4f} "
                f"{row['rmse_global']:>12,.0f} {row['mae_global']:>10,.0f} "
                f"{mape_str} "
                f"{row['rmse_mae_ratio']:>8.3f}  {row['classif_ratio']}"
            )

    # ── Métricas por porte ────────────────────────────────────────────────────
    df_porte = metricas_por_porte(oos_dict)
    if not df_porte.empty:
        logging.info(f"\n{'='*60}")
        logging.info("MÉTRICAS POR PORTE DO MUNICÍPIO:")
        for target in TARGETS:
            logging.info(f"\n  {TARGET_LABELS[target]}:")
            logging.info(
                f"  {'Modelo':<18} {'Porte':<16} {'n_obs':>6} "
                f"{'R²':>8} {'RMSE':>12} {'MAE':>10} {'MAPE':>8}"
            )
            sub_p = df_porte[df_porte["target"] == target]
            for m in MODELS:
                grp = sub_p[sub_p["modelo"] == m]
                for _, pr in grp.iterrows():
                    mape_str = (f"{pr['mape']:>7.1f}%"
                                if not pd.isna(pr["mape"]) else "    N/A ")
                    logging.info(
                        f"  {MODEL_LABELS[m]:<18} {pr['porte']:<16} {pr['n_obs']:>6} "
                        f"{pr['r2']:>8.4f} {pr['rmse']:>12,.0f} {pr['mae']:>10,.0f} {mape_str}"
                    )

    # ── Figuras ────────────────────────────────────────────────────────────────
    logging.info("\nGerando figuras…")
    fig_metricas_comparativas(resumo)
    fig_rmse_mae_ratio(resumo)
    fig_ganho_naive(resumo)
    fig_residuos_distribuicao(oos_dict)
    fig_actual_vs_predicted(oos_dict, resumo)
    fig_erros_pontuais(oos_dict, resumo)
    if not df_porte.empty:
        fig_metricas_por_porte(df_porte)

    # ── Salvar CSVs ────────────────────────────────────────────────────────────
    resumo_out = resumo.copy()
    resumo_out["target_label"] = resumo_out["target"].map(TARGET_LABELS)
    resumo_out["modelo_label"] = resumo_out["modelo"].map(MODEL_LABELS)
    resumo_out.to_csv(REP_DIR / "evaluation_metrics.csv",
                      index=False, encoding="utf-8-sig")
    logging.info("\nSalvo: evaluation_metrics.csv")

    if not df_porte.empty:
        df_porte.to_csv(REP_DIR / "metricas_por_porte.csv",
                        index=False, encoding="utf-8-sig")
        logging.info("Salvo: metricas_por_porte.csv")

    df_large = large_errors_report(oos_dict)
    df_large.to_csv(REP_DIR / "large_errors_report.csv",
                    index=False, encoding="utf-8-sig")
    logging.info("Salvo: large_errors_report.csv")

    # ── Diagnóstico de instabilidade ───────────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info("DIAGNÓSTICO DE INSTABILIDADE PREDITIVA:")
    for _, row in resumo.iterrows():
        r = row["rmse_mae_ratio"]
        if r >= RATIO_MODERADA:
            logging.info(
                f"  ⚠ {TARGET_LABELS[row['target']]} — {MODEL_LABELS[row['modelo']]}: "
                f"RMSE/MAE={r:.3f} — {row['classif_ratio']}. "
                f"{row['n_large_errors']} obs acima do P95 "
                f"(> R$ {row['p95_abs_residual']:,.0f} de erro)."
            )
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
