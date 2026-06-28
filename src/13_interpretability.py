#!/usr/bin/env python3
"""
13_interpretability.py
======================
Interpretabilidade global e local — modelos preditivos Cresol.

Módulos
-------
A  Resultados comparativos por agência e região (OOS, CV temporal)
B  Feature Importance global — todos os modelos normalizados
C  Permutation Importance OOS — melhor modelo por target
D  SHAP global — beeswarm + summary bar
E  SHAP por período — evolução temporal das contribuições
F  SHAP local por agência — heatmap top cooperativas × top features
G  SHAP Dependence — relação feature ↔ SHAP para top 6 features
H  Análise de sensibilidade — tornado + curvas de resposta + cenários
I  SHAP por estrato de porte — beeswarm + barras comparativas por porte
   (pequeno / intermediário / grande), revelando se os determinantes
   das previsões diferem entre municípios de portes distintos

Premissas
---------
- Melhores modelos tuned (script 11): XGBoost (vol_crédito) e OLS (captação)
- Predições OOS via Time Series Split (5 folds, janela expansível)
- SHAP computado no dataset completo após refit — interpretação do modelo treinado
- Permutation Importance no último fold (OOS honesto)
- shap 0.52: usa API legada (summary_plot, shap_values array) para compatibilidade

Saídas
------
  reports/figures/interpretability/  (8 figuras globais + 4 por porte por target)
  reports/agency_results.csv
  reports/region_results.csv
  reports/feature_importance_all.csv
  reports/permutation_importance.csv
  reports/shap_values.csv (por target)
  reports/shap_por_porte_{target}.csv (mean |SHAP| por feature e porte)
  reports/sensitivity.csv
  logs/interpretability.log
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sk_perm_imp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PANEL      = ROOT / "data" / "processed" / "panel_features_clean.parquet"
PARAMS_F   = ROOT / "data" / "processed" / "best_params_tuned.json"
MODEL_DIR  = ROOT / "data" / "processed" / "models" / "tuned"
FIG_DIR    = ROOT / "reports" / "figures" / "interpretability"
REP_DIR    = ROOT / "reports"
LOG_DIR    = ROOT / "logs"

# ── Configuração ──────────────────────────────────────────────────────────────
TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]
MODELS  = ["OLS", "RF", "XGBoost", "LightGBM"]

# Melhor modelo por target (baseado em R² — scripts 11-12)
BEST = {"vol_credito_rs_mil": "XGBoost", "captacao_rs_mil": "OLS"}

MODEL_LABELS = {
    "OLS": "OLS (MQO)", "RF": "Random Forest",
    "XGBoost": "XGBoost", "LightGBM": "LightGBM",
}
TARGET_LABELS = {
    "vol_credito_rs_mil": "Volume de Crédito (R$ mil)",
    "captacao_rs_mil":    "Captação (R$ mil)",
}
COLORS = {
    "OLS": "#4C72B0", "RF": "#55A868",
    "XGBoost": "#C44E52", "LightGBM": "#DD8452",
}
SUL = {"PR", "SC", "RS"}

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
TOP_AGENCIES      = 15   # agências exibidas nos gráficos
TOP_FEATS_SHAP    = 10   # features no heatmap SHAP por agência
TOP_FEATS_PORTE   = 12   # features exibidas no SHAP por porte

PORTE_ORDER = ["pequeno", "intermediario", "grande"]
PORTE_LABELS_FIG = {
    "pequeno":       "Pequeno (<40k hab.)",
    "intermediario": "Intermediário (40–100k)",
    "grande":        "Grande (>=100k)",
}
PORTE_COLORS = {
    "pequeno":       "#8BC34A",
    "intermediario": "#2196F3",
    "grande":        "#FF5722",
}

# Rótulos amigáveis para as features
FEAT_LABELS = {
    "L1_vol_credito_rs_mil":      "Crédito (t-1)",
    "L2_vol_credito_rs_mil":      "Crédito (t-2)",
    "L4_vol_credito_rs_mil":      "Crédito (t-4)",
    "L1_captacao_rs_mil":         "Captação (t-1)",
    "L2_captacao_rs_mil":         "Captação (t-2)",
    "L4_captacao_rs_mil":         "Captação (t-4)",
    "L1_ativo_total_rs_mil":      "Ativo Total (t-1)",
    "L2_ativo_total_rs_mil":      "Ativo Total (t-2)",
    "L4_ativo_total_rs_mil":      "Ativo Total (t-4)",
    "L1_patrimonio_liq_rs_mil":   "Patrim. Líq. (t-1)",
    "L2_patrimonio_liq_rs_mil":   "Patrim. Líq. (t-2)",
    "L1_carteira_credito_rs_mil": "Carteira Créd. (t-1)",
    "L2_carteira_credito_rs_mil": "Carteira Créd. (t-2)",
    "L4_carteira_credito_rs_mil": "Carteira Créd. (t-4)",
    "L1_selic_aa":                "Selic (t-1)",
    "L1_ipca_acum_trim":          "IPCA (t-1)",
    "L1_cambio_brl_usd":          "Câmbio BRL/USD (t-1)",
    "L1_ibc_br":                  "IBC-Br (t-1)",
    "L1_concessoes_cred":         "Concessões Créd. (t-1)",
    "L1_ipa_agro":                "IPA-Agro (t-1)",
    "L1_inpc_acum_trim":          "INPC (t-1)",
    "pib_corrente_rs_mil":        "PIB Municipal",
    "pib_per_capita_rs":          "PIB per Capita",
    "vab_agro_rs_mil":            "VAB Agro",
    "vab_industria_rs_mil":       "VAB Indústria",
    "vab_servicos_rs_mil":        "VAB Serviços",
    "share_agro_mun":             "Share Agro Mun.",
    "L1_milho_rs_60kg":           "Milho (t-1)",
    "L1_boi_gordo_rs_arroba":     "Boi Gordo (t-1)",
    "L1_leite_rs_litro":          "Leite (t-1)",
    "L1_ipa_agro_idx":            "IPA-Agro Idx (t-1)",
    "L1_cr_total_rs_mi":          "Créd. Rural Total (t-1)",
    "L1_cr_custeio_rs_mi":        "Créd. Rural Custeio (t-1)",
    "L1_cr_investimento_rs_mi":   "Créd. Rural Invest. (t-1)",
    "L1_cr_comercializacao_rs_mi":"Créd. Rural Comerc. (t-1)",
    "segmento_num":               "Segmento",
    "porte_num":                  "Porte Município",
    "dummy_plantio_verao":        "D: Plantio Verão",
    "dummy_colheita_verao":       "D: Colheita Verão",
    "dummy_plantio_inv":          "D: Plantio Inverno",
    "dummy_colheita_inv":         "D: Colheita Inverno",
    "dummy_sul":                  "D: Região Sul",
    "dummy_plantio_verao_sul":    "D: Plantio×Sul",
    "dummy_colheita_verao_sul":   "D: Colheita×Sul",
}

def fl(name: str) -> str:
    return FEAT_LABELS.get(name, name)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "interpretability.log", encoding="utf-8"),
        ],
    )


# ── CV temporal (idêntico scripts 11-12) ──────────────────────────────────────
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
        train_p = set(periodos[:split])
        test_p  = set(periodos[split: split + test_size])
        if not test_p:
            break
        splits.append((
            np.where(np.isin(arr, list(train_p)))[0],
            np.where(np.isin(arr, list(test_p)))[0],
        ))
    return splits


# ── Carregar modelos tuned ────────────────────────────────────────────────────
def load_model(target: str, model_name: str) -> dict:
    fname = {
        "OLS":      f"ols_{target}.joblib",
        "RF":       f"rf_tuned_{target}.joblib",
        "XGBoost":  f"xgb_tuned_{target}.joblib",
        "LightGBM": f"lgbm_tuned_{target}.joblib",
    }[model_name]
    return joblib.load(MODEL_DIR / fname)  # {"pipeline": ..., "features": [...]}


# ── Predições OOS com metadados ───────────────────────────────────────────────
def oos_predictions(pipeline: Pipeline, X_df: pd.DataFrame, y_s: pd.Series,
                    df_meta: pd.DataFrame, cv_splits: list) -> pd.DataFrame:
    chunks = []
    for fold, (tr_pos, te_pos) in enumerate(cv_splits, 1):
        X_tr = X_df.iloc[tr_pos].values;  y_tr = y_s.iloc[tr_pos].values
        X_te = X_df.iloc[te_pos].values;  y_te = y_s.iloc[te_pos].values
        meta = df_meta.iloc[te_pos].reset_index(drop=True)
        mask_tr = ~np.isnan(y_tr);  mask_te = ~np.isnan(y_te)
        pipeline.fit(X_tr[mask_tr], y_tr[mask_tr])
        y_pred = pipeline.predict(X_te[mask_te])
        y_true = y_te[mask_te]
        chunk  = meta[mask_te].reset_index(drop=True).copy()
        chunk["fold"]         = fold
        chunk["y_true"]       = y_true
        chunk["y_pred"]       = y_pred
        chunk["residual"]     = y_true - y_pred
        chunk["abs_residual"] = np.abs(y_true - y_pred)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# ── Métricas auxiliares ───────────────────────────────────────────────────────
def _mape(y_true, y_pred) -> float:
    m = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / np.abs(y_true[m])) * 100)

def _met(y_true, y_pred) -> dict:
    return {
        "r2":   r2_score(y_true, y_pred) if y_true.std() > 0 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": _mape(y_true, y_pred),
    }

def metricas_agencia(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cnpj8, g in df.groupby("cnpj8"):
        if len(g) < 3:
            continue
        yt, yp = g["y_true"].values, g["y_pred"].values
        row = {"cnpj8": cnpj8, "n_obs": len(g),
               "avg_real": yt.mean(), "avg_pred": yp.mean()}
        row.update(_met(yt, yp))
        for col in ["municipio", "uf"]:
            if col in g.columns:
                row[col] = g[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_real", ascending=False)

def metricas_regiao(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uf, g in df.groupby("uf"):
        yt, yp = g["y_true"].values, g["y_pred"].values
        row = {"uf": uf, "n_obs": len(g), "n_coop": g["cnpj8"].nunique(),
               "regiao": "Sul" if uf in SUL else "Não-Sul"}
        row.update(_met(yt, yp))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


# ── Feature Importance (todos os modelos) ─────────────────────────────────────
def extrair_fi(target: str, features: list) -> pd.DataFrame:
    rows = []
    for mname in MODELS:
        data = load_model(target, mname)
        pipe = data["pipeline"]
        mdl  = pipe.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
        elif hasattr(mdl, "coef_"):
            imp = np.abs(mdl.coef_)
        else:
            continue
        imp_norm = imp / (imp.sum() + 1e-12)
        for f, v in zip(features, imp_norm):
            rows.append({"modelo": mname, "feature": f, "importance": v})
    return pd.DataFrame(rows)


# ── Permutation Importance (último fold OOS) ──────────────────────────────────
def perm_importance_last_fold(pipeline: Pipeline, X_df: pd.DataFrame,
                               y_s: pd.Series, periodo: pd.Series,
                               n_repeats: int = 20) -> pd.DataFrame:
    cv = temporal_cv_sklearn(periodo)
    tr_pos, te_pos = cv[-1]
    X_tr = X_df.iloc[tr_pos].values;  y_tr = y_s.iloc[tr_pos].values
    X_te = X_df.iloc[te_pos].values;  y_te = y_s.iloc[te_pos].values
    mask_tr = ~np.isnan(y_tr);  mask_te = ~np.isnan(y_te)
    pipeline.fit(X_tr[mask_tr], y_tr[mask_tr])
    result = sk_perm_imp(pipeline, X_te[mask_te], y_te[mask_te],
                         n_repeats=n_repeats, random_state=RANDOM_STATE,
                         scoring="r2", n_jobs=-1)
    return pd.DataFrame({
        "feature":         X_df.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False)


# ── SHAP ──────────────────────────────────────────────────────────────────────
def compute_shap(pipeline: Pipeline, X_df: pd.DataFrame,
                 model_name: str) -> tuple[np.ndarray, float]:
    """
    Retorna (shap_values: array n×p, expected_value: float).
    Usa API legada (shap_values) compatível com shap 0.52.
    """
    X_imp = pipeline.named_steps["imputer"].transform(X_df.values)
    mdl   = pipeline.named_steps["model"]

    if model_name in ("RF", "XGBoost", "LightGBM"):
        exp = shap.TreeExplainer(mdl)
        sv  = exp.shap_values(X_imp)
        ev  = float(exp.expected_value)
    else:  # OLS — LinearExplainer
        # Pre-subsample background deterministically (maskers.Independent 0.52 não suporta random_state)
        _n_bg = min(300, len(X_imp))
        if _n_bg < len(X_imp):
            _rng = np.random.default_rng(RANDOM_STATE)
            _idx = sorted(_rng.choice(len(X_imp), size=_n_bg, replace=False))
            _X_bg = X_imp[_idx]
        else:
            _X_bg = X_imp
        bg  = shap.maskers.Independent(_X_bg, max_samples=_n_bg)
        exp = shap.LinearExplainer(mdl, bg)
        sv  = exp.shap_values(X_imp)
        ev  = float(exp.expected_value)

    # Garante array 2-D (n_samples × n_features)
    if isinstance(sv, list):
        sv = sv[0]
    return sv, ev, X_imp


def shap_por_periodo(sv: np.ndarray, df_meta: pd.DataFrame,
                      features: list) -> pd.DataFrame:
    """Média de |SHAP| por feature e período."""
    abs_sv = np.abs(sv)
    df_sv  = pd.DataFrame(abs_sv, columns=features)
    df_sv["periodo_str"] = df_meta["periodo_str"].values
    return (df_sv.groupby("periodo_str")[features].mean()
            .reset_index().melt(id_vars="periodo_str",
                                var_name="feature", value_name="mean_abs_shap"))


def shap_por_agencia(sv: np.ndarray, df_meta: pd.DataFrame,
                      features: list, top_n: int = TOP_AGENCIES) -> pd.DataFrame:
    """Média de |SHAP| por feature e cooperativa (top N por avg_real)."""
    abs_sv = np.abs(sv)
    df_sv  = pd.DataFrame(abs_sv, columns=features)
    df_sv["cnpj8"]    = df_meta["cnpj8"].values
    df_sv["avg_real"] = df_meta["y_true"].values if "y_true" in df_meta.columns else 0
    top_ids = (df_sv.groupby("cnpj8")["avg_real"].mean()
               .nlargest(top_n).index.tolist())
    return (df_sv[df_sv["cnpj8"].isin(top_ids)]
            .groupby("cnpj8")[features].mean())


def shap_por_porte(sv: np.ndarray, df_meta: pd.DataFrame,
                    features: list) -> tuple[pd.DataFrame, dict]:
    """
    Calcula mean |SHAP| por feature e por estrato de porte.

    Retorna
    -------
    df_porte : DataFrame (features × portes) com mean |SHAP|
    n_obs    : dict {porte: n_observações}
    """
    abs_sv = np.abs(sv)
    df_sv  = pd.DataFrame(abs_sv, columns=features)
    if "porte_municipio" not in df_meta.columns:
        return pd.DataFrame(), {}

    df_sv["porte"] = df_meta["porte_municipio"].values
    cols_feat = features

    result = {}
    n_obs  = {}
    for porte, grp in df_sv.groupby("porte"):
        result[porte] = grp[cols_feat].mean()
        n_obs[porte]  = len(grp)

    df = pd.DataFrame(result)
    portes_avail = [p for p in PORTE_ORDER if p in df.columns]
    return df[portes_avail], {p: n_obs.get(p, 0) for p in portes_avail}


def _log_shap_por_porte(df_p: pd.DataFrame, n_obs: dict, target: str) -> None:
    """Loga top features por porte e destaca divergências de ranking."""
    if df_p.empty:
        return
    portes = [p for p in PORTE_ORDER if p in df_p.columns]

    logging.info(f"\n  SHAP por porte — {TARGET_LABELS[target]}:")
    header = f"  {'Feature':<30}" + "".join(f"  {PORTE_LABELS_FIG[p][:14]:>16}" for p in portes)
    n_line  = f"  {'(n_obs)':<30}" + "".join(f"  {n_obs.get(p, 0):>16,d}" for p in portes)
    logging.info(header)
    logging.info(n_line)

    global_mean = df_p.mean(axis=1)
    for feat in global_mean.nlargest(12).index:
        vals = "".join(f"  {df_p.loc[feat, p]:>16,.1f}" for p in portes)
        logging.info(f"  {fl(feat):<30}{vals}")

    # Divergências de ranking entre portes
    if len(portes) >= 2:
        norm = df_p.div(df_p.max(axis=1) + 1e-9, axis=0)
        div  = norm.std(axis=1).nlargest(5)
        logging.info(f"\n  Features com maior divergência de ranking entre portes:")
        for feat, std_val in div.items():
            vals = "".join(f"  {df_p.loc[feat, p]:>16,.1f}" for p in portes)
            logging.info(f"  {fl(feat):<30}{vals}  std_norm={std_val:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    logging.info(f"  Figura: {path.name}")


def fig_agencias_regiao(df_preds: pd.DataFrame, df_ag: pd.DataFrame,
                         df_reg: pd.DataFrame, target: str) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Resultados por Agência e Região — {TARGET_LABELS[target]}",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    # ── Painel A: scatter avg_real × avg_pred por cooperativa ────────────────
    ax_a = fig.add_subplot(gs[0, :])
    top15 = df_ag.head(TOP_AGENCIES).copy()
    lims  = [min(top15["avg_real"].min(), top15["avg_pred"].min()) * 0.90,
             max(top15["avg_real"].max(), top15["avg_pred"].max()) * 1.05]
    sc = ax_a.scatter(df_ag["avg_real"], df_ag["avg_pred"],
                      c=df_ag["mape_pct"], cmap="RdYlGn_r",
                      s=df_ag["n_obs"] * 4, alpha=0.75, edgecolors="gray",
                      linewidths=0.4, vmin=0, vmax=30)
    ax_a.plot(lims, lims, "k--", linewidth=1.0, alpha=0.5, label="y = x")
    for _, row in top15.iterrows():
        ax_a.annotate(row.get("municipio", row["cnpj8"])[:12],
                      (row["avg_real"], row["avg_pred"]),
                      fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax_a, label="MAPE (%)", pad=0.01)
    ax_a.set_xlabel("Média Real (R$ mil)");  ax_a.set_ylabel("Média Prevista (R$ mil)")
    ax_a.set_title("Real × Previsto por Cooperativa (tamanho ∝ nº observações; cor = MAPE)")
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=8)

    # ── Painel B: top 15 agências — real vs previsto (barras) ────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    x    = np.arange(len(top15))
    ax_b.barh(x + 0.2, top15["avg_real"], height=0.38,
              color="#4C72B0", alpha=0.85, label="Real")
    ax_b.barh(x - 0.2, top15["avg_pred"], height=0.38,
              color="#C44E52", alpha=0.85, label="Previsto")
    labels = [row.get("municipio", str(row["cnpj8"]))[:14]
              for _, row in top15.iterrows()]
    ax_b.set_yticks(x);  ax_b.set_yticklabels(labels, fontsize=7.5)
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_b.set_xlabel("R$ mil");  ax_b.set_title(f"Top {TOP_AGENCIES} Agências — Médias")
    ax_b.legend(fontsize=8);  ax_b.spines[["top", "right"]].set_visible(False)

    # ── Painel C: MAPE por UF ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    df_reg_s = df_reg.sort_values("mape_pct")
    colors_r  = ["#55A868" if uf in SUL else "#DD8452"
                 for uf in df_reg_s["uf"]]
    bars = ax_c.barh(df_reg_s["uf"], df_reg_s["mape_pct"],
                     color=colors_r, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, df_reg_s["mape_pct"]):
        ax_c.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                  f"{v:.1f}%", va="center", fontsize=8)
    ax_c.set_xlabel("MAPE (%)");  ax_c.set_title("MAPE por UF (verde=Sul)")
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.axvline(df_reg_s["mape_pct"].mean(), color="gray",
                 linestyle="--", linewidth=1, alpha=0.7, label="Média")
    ax_c.legend(fontsize=8)

    _save(FIG_DIR / f"A_agencias_regiao_{target}.png")


def fig_feature_importance(df_fi: pd.DataFrame, target: str) -> None:
    top20 = (df_fi.groupby("feature")["importance"].mean()
             .nlargest(20).index.tolist())
    df_top = df_fi[df_fi["feature"].isin(top20)].copy()
    df_top["feat_lbl"] = df_top["feature"].map(fl)

    # Pivot: features × modelos
    piv = (df_top.pivot_table(index="feature", columns="modelo",
                               values="importance", aggfunc="mean")
           .fillna(0).reindex(columns=MODELS, fill_value=0))
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean", ascending=True).drop(columns="mean")

    fig, ax = plt.subplots(figsize=(11, 8))
    x = np.arange(len(piv))
    w = 0.22
    for i, mname in enumerate(MODELS):
        vals = piv[mname].values if mname in piv.columns else np.zeros(len(piv))
        ax.barh(x + (i - 1.5) * w, vals, height=w,
                label=MODEL_LABELS[mname], color=COLORS[mname], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels([fl(f) for f in piv.index], fontsize=8)
    ax.set_xlabel("Importância normalizada (soma = 1 por modelo)")
    ax.set_title(f"Feature Importance Global — {TARGET_LABELS[target]}\n"
                 f"(RF=MDI, XGB/LGBM=Gain, OLS=|coef| normalizado)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    _save(FIG_DIR / f"B_feat_importance_{target}.png")


def fig_permutation(df_perm: pd.DataFrame, target: str, model_name: str) -> None:
    df_p = df_perm[df_perm["importance_mean"] > -0.005].head(20).sort_values(
        "importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    x  = np.arange(len(df_p))
    ax.barh(x, df_p["importance_mean"], xerr=df_p["importance_std"],
            color=COLORS[model_name], alpha=0.85, capsize=3, height=0.65)
    ax.set_yticks(x)
    ax.set_yticklabels([fl(f) for f in df_p["feature"]], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Queda média em R² ao permutar a feature (OOS — último fold)")
    ax.set_title(f"Permutation Importance — {TARGET_LABELS[target]}\n"
                 f"Modelo: {MODEL_LABELS[model_name]}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    _save(FIG_DIR / f"C_permutation_{target}.png")


def fig_shap_beeswarm(sv: np.ndarray, X_imp: np.ndarray,
                       features: list, target: str, model_name: str) -> None:
    feat_lbls = [fl(f) for f in features]
    plt.figure(figsize=(10, 9))
    shap.summary_plot(sv, X_imp, feature_names=feat_lbls,
                      max_display=20, show=False, plot_size=None)
    plt.title(f"SHAP — Beeswarm Global\n"
              f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}", pad=10)
    _save(FIG_DIR / f"D_shap_beeswarm_{target}.png")


def fig_shap_periodo(df_sp: pd.DataFrame, features: list,
                      target: str, model_name: str) -> None:
    top_feats = (df_sp.groupby("feature")["mean_abs_shap"].mean()
                 .nlargest(TOP_FEATS_SHAP).index.tolist())
    piv = (df_sp[df_sp["feature"].isin(top_feats)]
           .pivot_table(index="feature", columns="periodo_str",
                        values="mean_abs_shap", aggfunc="mean"))
    piv.index = [fl(f) for f in piv.index]
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(piv, cmap="YlOrRd", ax=ax, annot=False,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Média |SHAP| (R$ mil)"})
    ax.set_xlabel("Período");  ax.set_ylabel("")
    ax.set_title(f"SHAP por Período — {TARGET_LABELS[target]}\n"
                 f"{MODEL_LABELS[model_name]} — Evolução da contribuição das features")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save(FIG_DIR / f"E_shap_periodo_{target}.png")


def fig_shap_agencia(df_ag_shap: pd.DataFrame, df_ag: pd.DataFrame,
                      features: list, target: str, model_name: str) -> None:
    top_feats = (df_ag_shap[features].mean()
                 .nlargest(TOP_FEATS_SHAP).index.tolist())
    sub = df_ag_shap[top_feats].copy()
    sub.columns = [fl(f) for f in top_feats]

    # Enriquecer índice com nome da cooperativa
    lookup = df_ag.set_index("cnpj8")[["municipio"]].to_dict()["municipio"]
    sub.index = [lookup.get(c, str(c))[:12] for c in sub.index]
    sub = sub.reindex(sub.mean(axis=1).sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(sub, cmap="Blues", ax=ax, annot=True, fmt=".0f",
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Média |SHAP| (R$ mil)"})
    ax.set_xlabel("Feature");  ax.set_ylabel("Cooperativa")
    ax.set_title(f"SHAP Local por Agência — {TARGET_LABELS[target]}\n"
                 f"{MODEL_LABELS[model_name]} — Top {TOP_AGENCIES} agências "
                 f"por volume médio")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    _save(FIG_DIR / f"F_shap_agencia_{target}.png")


def fig_shap_dependence(sv: np.ndarray, X_imp: np.ndarray,
                         features: list, target: str, model_name: str) -> None:
    top6_idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:6]
    top6_names = [features[i] for i in top6_idx]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"SHAP Dependence — {TARGET_LABELS[target]}\n"
                 f"{MODEL_LABELS[model_name]}", fontsize=12, fontweight="bold")

    for ax, feat_name, feat_idx in zip(axes.flat, top6_names, top6_idx):
        x_vals    = X_imp[:, feat_idx]
        shap_vals = sv[:, feat_idx]
        sc = ax.scatter(x_vals, shap_vals, c=shap_vals, cmap="coolwarm",
                        s=12, alpha=0.55, edgecolors="none",
                        vmin=-np.abs(shap_vals).max(),
                        vmax=np.abs(shap_vals).max())
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlabel(fl(feat_name), fontsize=8.5)
        ax.set_ylabel("SHAP value", fontsize=8)
        ax.set_title(fl(feat_name), fontsize=9, pad=4)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"
                                  if abs(x) > 1000 else f"{x:.2f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        plt.colorbar(sc, ax=ax, pad=0.02)

    _save(FIG_DIR / f"G_shap_dependence_{target}.png")


def fig_shap_beeswarm_por_porte(sv: np.ndarray, X_imp: np.ndarray,
                                 df_meta: pd.DataFrame, features: list,
                                 target: str, model_name: str) -> None:
    """
    I-a: Beeswarm SHAP separado para cada estrato de porte.

    Três figuras independentes (uma por porte), usando o mesmo layout do beeswarm
    global (módulo D), filtradas às observações de cada estrato.
    Permitem comparar direção e magnitude das contribuições por porte.
    """
    if "porte_municipio" not in df_meta.columns:
        logging.info("    porte_municipio não disponível — beeswarms por porte ignorados.")
        return

    feat_lbls = [fl(f) for f in features]
    for porte in PORTE_ORDER:
        mask  = df_meta["porte_municipio"].values == porte
        n_obs = int(mask.sum())
        if n_obs < 5:
            logging.info(f"    {porte}: n={n_obs} — insuficiente para beeswarm, ignorado.")
            continue

        sv_p = sv[mask]
        X_p  = X_imp[mask]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_p, X_p, feature_names=feat_lbls,
                          max_display=15, show=False, plot_size=None)
        plt.title(
            f"SHAP Beeswarm — {PORTE_LABELS_FIG[porte]} (n={n_obs} obs)\n"
            f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}",
            pad=10,
        )
        _save(FIG_DIR / f"I_shap_beeswarm_{porte}_{target}.png")


def fig_shap_barras_porte(df_porte: pd.DataFrame, n_obs: dict,
                            target: str, model_name: str) -> None:
    """
    I-b: Barras horizontais comparativas — mean |SHAP| por porte e feature.

    3 subplots lado a lado (um por porte), cada um ordenado pelo próprio
    ranking de importância. Permite identificar:
      - Quais features dominam a predição em cada estrato
      - Mudanças de ranking entre pequeno / intermediário / grande
      - Features que aparecem no top para um porte mas não para outro
    """
    if df_porte.empty:
        return

    portes = [p for p in PORTE_ORDER if p in df_porte.columns]
    n_p    = len(portes)
    if n_p == 0:
        return

    fig, axes = plt.subplots(1, n_p, figsize=(5.5 * n_p, 7), sharey=False)
    if n_p == 1:
        axes = [axes]

    fig.suptitle(
        f"SHAP — Determinantes por Estrato de Porte\n"
        f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}",
        fontsize=12, fontweight="bold",
    )

    # Calcula ranking global para referência de posição
    global_order = df_porte.mean(axis=1).nlargest(TOP_FEATS_PORTE).index.tolist()

    for ax, porte in zip(axes, portes):
        col    = df_porte[porte]
        top_n  = col.nlargest(TOP_FEATS_PORTE)
        feats_sorted = top_n.index.tolist()  # ordered by THIS porte
        vals   = top_n.values
        x      = np.arange(len(feats_sorted))

        # Cor da barra: porte color; destaque se ranking muito diferente do global
        bar_colors = []
        for feat in feats_sorted:
            rank_global = global_order.index(feat) + 1 if feat in global_order else 99
            rank_local  = feats_sorted.index(feat) + 1
            # diferença de ranking > 4 → destaque mais escuro
            bar_colors.append(PORTE_COLORS[porte] if abs(rank_global - rank_local) <= 4
                               else "#CC0000")

        bars = ax.barh(x, vals, color=bar_colors, alpha=0.82,
                       edgecolor="white", linewidth=0.6)

        # Anotação: rank global entre parênteses
        for i, feat in enumerate(feats_sorted):
            rank_g = global_order.index(feat) + 1 if feat in global_order else ">12"
            ax.text(vals[i] + vals.max() * 0.02, i,
                    f"#{rank_g}", va="center", fontsize=7.5, color="#555555")

        ax.set_yticks(x)
        ax.set_yticklabels([fl(f) for f in feats_sorted], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP| (R$ mil)", fontsize=9)
        ax.set_title(
            f"{PORTE_LABELS_FIG[porte]}\n(n={n_obs.get(porte, 0):,d} obs)",
            fontsize=10, pad=5,
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Nota de leitura: # entre parênteses = ranking global
    fig.text(0.99, 0.01,
             "#N = posição no ranking global (vermelho = divergência > 4 posições)",
             ha="right", va="bottom", fontsize=7.5, color="#555555",
             style="italic")

    _save(FIG_DIR / f"I_shap_barras_porte_{target}.png")


def fig_shap_heatmap_porte(df_porte: pd.DataFrame, n_obs: dict,
                             target: str, model_name: str) -> None:
    """
    I-c: Heatmap features × portes — ranking relativo dentro de cada estrato.

    Valores normalizados por coluna (0=min, 1=max dentro do porte).
    Revela quais features têm importância desproporcional em um estrato
    específico vs. os demais.
    """
    if df_porte.empty:
        return

    portes = [p for p in PORTE_ORDER if p in df_porte.columns]
    if not portes:
        return

    # Top features por importância global
    top_n = min(TOP_FEATS_PORTE + 3, len(df_porte))
    top_feats = df_porte.mean(axis=1).nlargest(top_n).index.tolist()
    sub = df_porte.loc[top_feats, portes].copy()

    # Painel 1: valores absolutos (mean |SHAP|)
    # Painel 2: valores normalizados por coluna (ranking relativo por porte)
    sub_norm = sub.div(sub.max(axis=0) + 1e-9, axis=1)

    col_lbls = [f"{PORTE_LABELS_FIG[p]}\n(n={n_obs.get(p, 0):,d})" for p in portes]
    row_lbls = [fl(f) for f in top_feats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(
        f"SHAP — Importância por Estrato de Porte\n"
        f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}",
        fontsize=12, fontweight="bold",
    )

    # Heatmap absoluto
    sns.heatmap(sub, cmap="YlOrRd", ax=axes[0], annot=True, fmt=".0f",
                linewidths=0.4, linecolor="white",
                xticklabels=col_lbls, yticklabels=row_lbls,
                cbar_kws={"label": "Mean |SHAP| (R$ mil)"})
    axes[0].set_title("Valor absoluto", fontsize=10, pad=5)
    axes[0].set_xlabel("")

    # Heatmap normalizado (0–1 dentro de cada porte)
    sns.heatmap(sub_norm, cmap="Blues", ax=axes[1], annot=True, fmt=".2f",
                linewidths=0.4, linecolor="white",
                xticklabels=col_lbls, yticklabels=row_lbls,
                vmin=0, vmax=1,
                cbar_kws={"label": "Importância relativa (0–1)"})
    axes[1].set_title("Normalizado por porte (ranking relativo)", fontsize=10, pad=5)
    axes[1].set_xlabel("")

    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    _save(FIG_DIR / f"I_shap_heatmap_porte_{target}.png")


def fig_sensibilidade(pipeline: Pipeline, X_ref: np.ndarray, y_base: float,
                       features: list, df_fi: pd.DataFrame,
                       target: str, model_name: str) -> None:
    """
    H1 — Tornado: range de previsão quando cada feature varia P10→P90.
    H2 — Curvas de resposta: previsão vs valor da feature (top 6 contínuas).
    H3 — Tabela de cenários: Pessimista / Base / Otimista.
    """
    # Seleciona top 12 features por importância média
    top12 = (df_fi.groupby("feature")["importance"].mean()
             .nlargest(12).index.tolist())

    # ── H1: Tornado ───────────────────────────────────────────────────────────
    tornado_rows = []
    for feat in top12:
        if feat not in features:
            continue
        fi   = features.index(feat)
        p10  = float(np.nanpercentile(X_ref[:, fi], 10))
        p90  = float(np.nanpercentile(X_ref[:, fi], 90))
        if abs(p90 - p10) < 1e-9:
            continue
        X_lo = X_ref[[0]].copy();  X_lo[0, fi] = p10
        X_hi = X_ref[[0]].copy();  X_hi[0, fi] = p90
        pred_lo = float(pipeline.predict(X_lo)[0])
        pred_hi = float(pipeline.predict(X_hi)[0])
        tornado_rows.append({
            "feature": feat, "p10": p10, "p90": p90,
            "pred_lo": pred_lo, "pred_hi": pred_hi,
            "range": abs(pred_hi - pred_lo),
        })

    df_torn = (pd.DataFrame(tornado_rows)
               .sort_values("range", ascending=True))

    fig1, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(df_torn))
    for i, (_, row) in enumerate(df_torn.iterrows()):
        lo, hi = min(row["pred_lo"], row["pred_hi"]), max(row["pred_lo"], row["pred_hi"])
        color  = "#55A868" if row["pred_hi"] > row["pred_lo"] else "#C44E52"
        ax.barh(i, hi - lo, left=lo - y_base, height=0.55,
                color=color, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels([fl(r["feature"]) for _, r in df_torn.iterrows()], fontsize=8)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Variação na previsão relativa ao cenário-base (R$ mil)")
    ax.set_title(f"Tornado — Sensibilidade das Previsões (P10 → P90)\n"
                 f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    _save(FIG_DIR / f"H1_tornado_{target}.png")

    # ── H2: Curvas de resposta (top 6 features contínuas) ────────────────────
    top6_cont = [r["feature"] for _, r in df_torn.iloc[-6:].iterrows()
                 if r["feature"] not in FEAT_SAFRA + FEAT_CAT][:6]
    if not top6_cont:
        top6_cont = [r["feature"] for _, r in df_torn.iloc[-6:].iterrows()][:6]

    fig2, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle(f"Curvas de Resposta — Simulação de Metas\n"
                  f"{TARGET_LABELS[target]} — {MODEL_LABELS[model_name]}",
                  fontsize=12, fontweight="bold")

    for ax2, feat in zip(axes.flat, top6_cont):
        if feat not in features:
            ax2.axis("off"); continue
        fi   = features.index(feat)
        vals = np.linspace(
            np.nanpercentile(X_ref[:, fi], 5),
            np.nanpercentile(X_ref[:, fi], 95), 60
        )
        preds = []
        for v in vals:
            Xmod = X_ref[[0]].copy();  Xmod[0, fi] = v
            preds.append(float(pipeline.predict(Xmod)[0]))

        ax2.plot(vals, preds, color=COLORS[model_name], linewidth=2)
        ax2.axvline(X_ref[0, fi], color="gray", linestyle="--", linewidth=1,
                    alpha=0.8, label="Valor base")
        ax2.axhline(y_base, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax2.set_xlabel(fl(feat), fontsize=8.5)
        ax2.set_ylabel("Previsão (R$ mil)", fontsize=8)
        ax2.set_title(fl(feat), fontsize=9)
        ax2.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"
                                  if abs(x) > 1000 else f"{x:.2f}"))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(fontsize=7)

    _save(FIG_DIR / f"H2_curvas_resposta_{target}.png")

    # ── H3: Tabela de cenários ────────────────────────────────────────────────
    cenario_rows = []
    for feat in top12:
        if feat not in features:
            continue
        fi   = features.index(feat)
        vals = {"Pessimista": np.nanpercentile(X_ref[:, fi], 10),
                "Base (P50)": np.nanpercentile(X_ref[:, fi], 50),
                "Otimista":   np.nanpercentile(X_ref[:, fi], 90)}
        for cen, val in vals.items():
            Xmod = X_ref[[0]].copy(); Xmod[0, fi] = val
            pred = float(pipeline.predict(Xmod)[0])
            cenario_rows.append({
                "feature": feat, "feature_label": fl(feat),
                "cenario": cen, "valor_feature": round(val, 4),
                "previsao": round(pred, 2),
                "delta_pct": round((pred - y_base) / (abs(y_base) + 1e-9) * 100, 2),
            })

    df_cen = pd.DataFrame(cenario_rows)
    df_cen.to_csv(REP_DIR / f"sensitivity_{target}.csv",
                  index=False, encoding="utf-8-sig")
    logging.info(f"  Salvo: sensitivity_{target}.csv")

    return df_torn


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Interpretabilidade dos Modelos — Cresol TCC ===")

    df = pd.read_parquet(PANEL)
    meta_cols = ["cnpj8", "periodo_str"] + [
        c for c in ["municipio", "uf", "porte_municipio"] if c in df.columns]

    all_agency = [];  all_region = []
    all_fi     = [];  all_perm   = []

    for target in TARGETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"TARGET: {target}")

        outros = [t for t in TARGETS if t != target]
        feats  = [f for f in ALL_FEATURES if f in df.columns and f not in [target] + outros]
        mask   = df[target].notna()
        X_df   = df.loc[mask, feats].reset_index(drop=True)
        X_df   = X_df.astype({c: "float64" for c in FEAT_CAT if c in X_df.columns})
        y_s    = df.loc[mask, target].reset_index(drop=True)
        df_meta = df.loc[mask, [c for c in meta_cols if c in df.columns]].reset_index(drop=True)
        periodo = df.loc[mask, "periodo_str"].reset_index(drop=True)

        cv_splits  = temporal_cv_sklearn(periodo)
        best_model = BEST[target]
        data_best  = load_model(target, best_model)
        pipe_best  = data_best["pipeline"]

        # ── A: Resultados por agência e região ────────────────────────────────
        logging.info("  [A] Predições OOS para análise por agência/região…")
        df_preds = oos_predictions(pipe_best, X_df, y_s, df_meta, cv_splits)
        df_preds_meta = df_preds.copy()
        df_ag    = metricas_agencia(df_preds)
        df_reg   = metricas_regiao(df_preds)

        df_ag["target"] = target;  df_reg["target"] = target
        all_agency.append(df_ag);  all_region.append(df_reg)

        logging.info(f"    Cooperativas avaliadas: {len(df_ag)}")
        logging.info(f"    MAPE médio: {df_ag['mape_pct'].mean():.2f}% ± "
                     f"{df_ag['mape_pct'].std():.2f}%")
        logging.info(f"    Cooperativas com MAPE < 10%: "
                     f"{(df_ag['mape_pct'] < 10).sum()} / {len(df_ag)}")

        fig_agencias_regiao(df_preds, df_ag, df_reg, target)

        # ── B: Feature Importance (todos os modelos) ──────────────────────────
        logging.info("  [B] Feature Importance global…")
        df_fi = extrair_fi(target, feats)
        df_fi["target"] = target
        all_fi.append(df_fi)

        top5 = (df_fi.groupby("feature")["importance"].mean()
                .nlargest(5).to_dict())
        logging.info(f"    Top 5 features (média): "
                     + ", ".join(f"{fl(f)}={v:.4f}" for f, v in top5.items()))

        fig_feature_importance(df_fi, target)

        # ── C: Permutation Importance (último fold OOS) ───────────────────────
        logging.info("  [C] Permutation Importance OOS (último fold)…")
        pipe_perm = load_model(target, best_model)["pipeline"]
        df_perm   = perm_importance_last_fold(pipe_perm, X_df, y_s, periodo)
        df_perm["target"] = target;  df_perm["modelo"] = best_model
        all_perm.append(df_perm)

        top3_perm = df_perm.head(3)[["feature", "importance_mean"]].values
        logging.info(f"    Top 3 perm: "
                     + ", ".join(f"{fl(r[0])}={r[1]:.4f}" for r in top3_perm))
        fig_permutation(df_perm, target, best_model)

        # ── D-G: SHAP ─────────────────────────────────────────────────────────
        logging.info(f"  [D-G] SHAP — {MODEL_LABELS[best_model]}…")

        # Refit no dataset completo para SHAP estável
        pipe_shap = load_model(target, best_model)["pipeline"]
        X_full = X_df.values
        y_full = y_s.values
        mask_full = ~np.isnan(y_full)
        pipe_shap.fit(X_full[mask_full], y_full[mask_full])

        sv, ev, X_imp = compute_shap(pipe_shap, X_df, best_model)
        logging.info(f"    SHAP computado: {sv.shape}")

        # Salvar SHAP values
        df_sv = pd.DataFrame(sv, columns=feats)
        df_sv["cnpj8"]       = df_meta["cnpj8"].values
        df_sv["periodo_str"] = df_meta["periodo_str"].values
        df_sv["y_true"]      = y_s.values
        df_sv.to_csv(REP_DIR / f"shap_values_{target}.csv",
                     index=False, encoding="utf-8-sig")
        logging.info(f"    Salvo: shap_values_{target}.csv")

        # D: Beeswarm
        fig_shap_beeswarm(sv, X_imp, feats, target, best_model)

        # E: Por período
        df_sp = shap_por_periodo(sv, df_meta, feats)
        fig_shap_periodo(df_sp, feats, target, best_model)

        # F: Por agência (precisamos de y_true no df_meta)
        df_meta_y = df_meta.copy()
        df_meta_y["y_true"] = y_s.values
        df_ag_shap = shap_por_agencia(sv, df_meta_y, feats)
        fig_shap_agencia(df_ag_shap, df_ag, feats, target, best_model)

        # G: Dependence
        fig_shap_dependence(sv, X_imp, feats, target, best_model)

        # ── I: SHAP por estrato de porte ──────────────────────────────────────
        logging.info(f"  [I] SHAP por estrato de porte…")
        df_porte, n_obs_porte = shap_por_porte(sv, df_meta, feats)

        if not df_porte.empty:
            _log_shap_por_porte(df_porte, n_obs_porte, target)

            # I-a: Beeswarm por porte (3 figuras separadas)
            fig_shap_beeswarm_por_porte(sv, X_imp, df_meta, feats, target, best_model)

            # I-b: Barras comparativas (1 figura, 3 colunas)
            fig_shap_barras_porte(df_porte, n_obs_porte, target, best_model)

            # I-c: Heatmap features × portes
            fig_shap_heatmap_porte(df_porte, n_obs_porte, target, best_model)

            # Salvar CSV
            df_porte_out = df_porte.copy()
            df_porte_out.index.name = "feature"
            df_porte_out["feature_label"] = [fl(f) for f in df_porte_out.index]
            df_porte_out.reset_index(inplace=True)
            df_porte_out["target"] = target
            df_porte_out.to_csv(REP_DIR / f"shap_por_porte_{target}.csv",
                                index=False, encoding="utf-8-sig")
            logging.info(f"    Salvo: shap_por_porte_{target}.csv")
        else:
            logging.info("    porte_municipio não encontrado em df_meta — módulo I ignorado.")

        # ── H: Análise de Sensibilidade — Simulação de Metas ─────────────────
        logging.info("  [H] Análise de sensibilidade…")
        # Ponto de referência: mediana do último período disponível
        ultimo_periodo = df.loc[mask, "periodo_str"].max()
        mask_ult = (df.loc[mask, "periodo_str"] == ultimo_periodo).values
        X_ult = X_df.values[mask_ult]
        if len(X_ult) == 0:
            X_ult = X_df.values[-10:]
        X_median = np.nanmedian(X_ult, axis=0, keepdims=True)

        # Imputa valores ausentes na referência
        X_median_imp = pipe_shap.named_steps["imputer"].transform(X_median)
        y_base = float(pipe_shap.predict(X_median)[0])

        # Usa X_imp (dataset completo, já transformado) como referência de distribuição
        fig_sensibilidade(pipe_shap, X_imp, y_base, feats, df_fi, target, best_model)

        logging.info(f"    Ponto base ({ultimo_periodo} — mediana): "
                     f"{TARGET_LABELS[target]} = {y_base:,.0f}")

    # ── Consolidar CSVs ───────────────────────────────────────────────────────
    pd.concat(all_agency).to_csv(REP_DIR / "agency_results.csv",
                                 index=False, encoding="utf-8-sig")
    pd.concat(all_region).to_csv(REP_DIR / "region_results.csv",
                                 index=False, encoding="utf-8-sig")
    pd.concat(all_fi).to_csv(REP_DIR / "feature_importance_all.csv",
                              index=False, encoding="utf-8-sig")
    pd.concat(all_perm).to_csv(REP_DIR / "permutation_importance.csv",
                                index=False, encoding="utf-8-sig")

    logging.info(f"\n{'='*60}")
    logging.info("Figuras em: reports/figures/interpretability/")
    logging.info("CSVs em:   reports/")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
