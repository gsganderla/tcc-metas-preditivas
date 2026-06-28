#!/usr/bin/env python3
"""
09_shap_analysis.py
===================
Análise de interpretabilidade dos melhores modelos preditivos.

Gera:
  - SHAP values para o melhor modelo de cada target
  - Gráfico Summary Plot (beeswarm) — importância global + direção do efeito
  - Gráfico Bar Plot — importância média absoluta por feature
  - SHAP Dependence Plots das 4 features mais importantes
  - Permutation Importance como validação cruzada da importância
  - Tabela consolidada: feature_importance_final.csv

Usa os modelos salvos pelo script 08.
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem GUI para salvar PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PANEL     = ROOT / "data" / "processed" / "panel_features_clean.parquet"
MODEL_DIR = ROOT / "data" / "processed" / "models"
FIG_DIR   = ROOT / "reports" / "figures"
OUT_DIR   = ROOT / "data" / "processed"
LOG_DIR   = ROOT / "logs"

TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "shap_analysis.log", encoding="utf-8"),
        ],
    )


# ── Utilitários ────────────────────────────────────────────────────────────────
def _limpar_nome(col: str) -> str:
    """Nomes curtos e legíveis para os gráficos."""
    col = col.replace("_rs_mil", "").replace("_rs_mi", "")
    col = col.replace("_", " ").strip()
    return col


def _carregar_modelo(target: str) -> tuple:
    """Localiza e carrega o modelo .joblib salvo pelo script 08."""
    candidatos = list(MODEL_DIR.glob(f"best_{target}_*.joblib"))
    if not candidatos:
        raise FileNotFoundError(f"Nenhum modelo encontrado para {target} em {MODEL_DIR}")
    caminho = candidatos[0]
    pacote = joblib.load(caminho)
    pipeline = pacote["pipeline"]
    features = pacote["features"]
    nome_modelo = caminho.stem.replace(f"best_{target}_", "")
    logging.info(f"Modelo carregado: {caminho.name}")
    return pipeline, features, nome_modelo


# ── SHAP ───────────────────────────────────────────────────────────────────────
def calcular_shap(pipeline, X: pd.DataFrame, nome_modelo: str) -> shap.Explanation:
    """
    Calcula SHAP values para o pipeline treinado.
    Extrai o modelo interno (pós-imputer/scaler) e aplica o Explainer adequado.
    """
    # Transformar X até o penúltimo passo (imputer + scaler se Ridge)
    X_transf = X.copy()
    steps = list(pipeline.named_steps.items())
    for step_name, step_obj in steps[:-1]:   # tudo menos o "model"
        X_transf = step_obj.transform(X_transf)

    model = pipeline.named_steps["model"]

    if nome_modelo in ("xgboost", "lightgbm"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_transf)
        # Converter para Explanation com nomes de features originais
        explanation = shap.Explanation(
            values=shap_vals.values,
            base_values=shap_vals.base_values,
            data=X.values,
            feature_names=[_limpar_nome(f) for f in X.columns],
        )
    elif nome_modelo == "randomforest":
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_transf)
        explanation = shap.Explanation(
            values=shap_vals.values,
            base_values=shap_vals.base_values,
            data=X.values,
            feature_names=[_limpar_nome(f) for f in X.columns],
        )
    else:  # Ridge / linear
        explainer = shap.LinearExplainer(model, X_transf)
        shap_vals = explainer(X_transf)
        explanation = shap.Explanation(
            values=shap_vals.values,
            base_values=shap_vals.base_values,
            data=X.values,
            feature_names=[_limpar_nome(f) for f in X.columns],
        )
    return explanation


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_summary_beeswarm(explanation: shap.Explanation, target: str,
                          nome_modelo: str, max_display: int = 20) -> None:
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {target}\nModelo: {nome_modelo}", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_beeswarm_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Gráfico salvo: shap_beeswarm_{target}.png")


def plot_bar_importance(explanation: shap.Explanation, target: str,
                        nome_modelo: str, max_display: int = 20) -> None:
    plt.figure(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {target}\nModelo: {nome_modelo}", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_bar_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Gráfico salvo: shap_bar_{target}.png")


def plot_dependence(explanation: shap.Explanation, X: pd.DataFrame,
                    target: str, nome_modelo: str, top_n: int = 4) -> None:
    """Gráficos de dependência SHAP para as top-N features."""
    feat_names = explanation.feature_names
    mean_abs   = np.abs(explanation.values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for plot_i, feat_i in enumerate(top_idx):
        ax = axes[plot_i]
        vals = explanation.values[:, feat_i]
        data = explanation.data[:, feat_i]
        ax.scatter(data, vals, alpha=0.4, s=10, color="#1f77b4")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_xlabel(feat_names[feat_i], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feat_names[feat_i], fontsize=10)

    fig.suptitle(f"SHAP Dependence Plots — {target} ({nome_modelo})", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"shap_dependence_{target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Gráfico salvo: shap_dependence_{target}.png")


def plot_cv_results(target: str) -> None:
    """Boxplot de R² por modelo para o target."""
    csv_path = OUT_DIR / f"cv_results_{target}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    modelos = df["modelo"].unique()
    data = [df[df["modelo"] == m]["r2"].values for m in modelos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=modelos, patch_artist=True)
    ax.set_title(f"R² por Fold — {target}", fontsize=12)
    ax.set_ylabel("R²")
    ax.set_xlabel("Modelo")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"cv_boxplot_{target}.png", dpi=150)
    plt.close(fig)
    logging.info(f"Gráfico salvo: cv_boxplot_{target}.png")


# ── Permutation Importance ─────────────────────────────────────────────────────
def calc_permutation_importance(pipeline, X: pd.DataFrame,
                                y: pd.Series, nome_modelo: str,
                                n_repeats: int = 10) -> pd.DataFrame:
    """
    Permutation importance no conjunto de TREINO completo.
    Menos enviesado que o impurity-based para modelos de árvore.
    """
    # Transformar X até o passo modelo
    X_transf = X.copy()
    steps = list(pipeline.named_steps.items())
    for step_name, step_obj in steps[:-1]:
        X_transf = step_obj.transform(X_transf)

    model = pipeline.named_steps["model"]

    perm = permutation_importance(
        model, X_transf, y, n_repeats=n_repeats,
        random_state=42, n_jobs=-1,
    )
    df_perm = pd.DataFrame({
        "feature":     [_limpar_nome(f) for f in X.columns],
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std":  perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    return df_perm


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Análise SHAP + Importância de Features ===")

    df = pd.read_parquet(PANEL)
    logging.info(f"Painel: {df.shape[0]} linhas × {df.shape[1]} colunas")

    all_imp = []

    for target in TARGETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"Target: {target}")

        try:
            pipeline, features, nome_modelo = _carregar_modelo(target)
        except FileNotFoundError as e:
            logging.error(str(e))
            continue

        feats_avail = [f for f in features if f in df.columns]
        mask = df[target].notna()
        X = df.loc[mask, feats_avail]
        y = df.loc[mask, target]
        logging.info(f"  Amostras: {len(X)}, features: {len(feats_avail)}")

        # 1. SHAP values
        logging.info("  Calculando SHAP values…")
        try:
            explanation = calcular_shap(pipeline, X, nome_modelo)

            # 2. Gráficos SHAP
            plot_summary_beeswarm(explanation, target, nome_modelo)
            plot_bar_importance(explanation, target, nome_modelo)
            plot_dependence(explanation, X, target, nome_modelo)

            # Importância SHAP → tabela
            mean_abs_shap = np.abs(explanation.values).mean(axis=0)
            df_shap = pd.DataFrame({
                "feature": feats_avail,
                "feature_label": [_limpar_nome(f) for f in feats_avail],
                "shap_importance": mean_abs_shap,
            }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
            df_shap["target"] = target
            df_shap["modelo"] = nome_modelo
            all_imp.append(df_shap)

        except Exception as exc:
            logging.error(f"  SHAP falhou: {exc}")

        # 3. Permutation Importance
        logging.info("  Calculando Permutation Importance…")
        try:
            df_perm = calc_permutation_importance(pipeline, X, y, nome_modelo)
            df_perm["target"] = target
            df_perm["modelo"] = nome_modelo
            logging.info(f"\n  Top 10 Permutation Importance ({target}):")
            logging.info("\n" + df_perm.head(10)[["feature", "perm_importance_mean"]].to_string(index=False))
            perm_path = OUT_DIR / f"permutation_importance_{target}.csv"
            df_perm.to_csv(perm_path, index=False, encoding="utf-8-sig")

        except Exception as exc:
            logging.error(f"  Permutation Importance falhou: {exc}")

        # 4. Boxplot CV
        plot_cv_results(target)

    # 5. Tabela consolidada de importância SHAP
    if all_imp:
        df_final = pd.concat(all_imp, ignore_index=True)
        df_final.to_csv(OUT_DIR / "shap_importance_final.csv", index=False, encoding="utf-8-sig")
        logging.info(f"\nImportância SHAP consolidada salva.")

    logging.info(f"\nFiguras salvas em: {FIG_DIR}")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
