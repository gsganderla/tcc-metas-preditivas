#!/usr/bin/env python3
"""
14_metas_comerciais.py
======================
Converte previsões do modelo preditivo em metas comerciais diferenciadas por
porte de município.

Pipeline explícito em DOIS ESTÁGIOS separados:
  [1] Componente técnico-preditivo  — saída do modelo treinado; representa o
      potencial estimado de cada cooperativa no período; não contém julgamento
      de gestão.
  [2] Componente de ajuste estratégico — fatores configuráveis por porte que
      traduzem intenção comercial sobre a referência técnica; calibrar junto
      à governança no ciclo de planejamento.

Diferenciação por porte (zona de configuração editável — ver FATORES_*):
  pequeno        — fator_base conservador + ajuste sazonal trimestral
                   (sensível ao calendário agrícola regional)
  intermediario  — fator_base + fator_expansao de mercado (market share)
  grande         — fator_base + fator_expansao mais agressivo
                   + limites de variação configuráveis (piso/teto)

Todos os fatores têm valor padrão neutro (1.0 / 0.0 / None).
A meta comercial com fatores neutros é idêntica à previsão técnica.

Entradas
--------
  data/processed/panel_features_clean.parquet
  data/processed/models/tuned/{modelo}_{target}.joblib

Saídas
------
  reports/metas_comerciais.csv
  reports/figures/metas/metas_vs_historico.png
  reports/figures/metas/distribuicao_metas.png
  logs/metas_comerciais.log
"""

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PANEL     = ROOT / "data" / "processed" / "panel_features_clean.parquet"
TUNED_DIR = ROOT / "data" / "processed" / "models" / "tuned"
FIG_DIR   = ROOT / "reports" / "figures" / "metas"
REP_DIR   = ROOT / "reports"
LOG_DIR   = ROOT / "logs"

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]
TARGET_LABELS = {
    "vol_credito_rs_mil": "Volume de Crédito (R$ mil)",
    "captacao_rs_mil":    "Captação (R$ mil)",
}

# Modelo usado por target (arquivo em TUNED_DIR: {prefixo}_{target}.joblib)
# Opções: "ols", "rf_tuned", "xgb_tuned", "lgbm_tuned"
MODELO_META = {
    "vol_credito_rs_mil": "xgb_tuned",  # XGBoost — melhor OOS (R²=0.9752)
    "captacao_rs_mil":    "ols",        # OLS     — melhor OOS (R²=0.9767)
}

# Stratos na ordem canônica
PORTE_ORDER = ["pequeno", "intermediario", "grande"]
PORTE_LABELS_FIG = {
    "pequeno":       "Pequeno\n(<40k hab.)",
    "intermediario": "Intermediário\n(40–100k)",
    "grande":        "Grande\n(≥100k)",
}
PORTE_COLORS = {
    "pequeno":       "#8BC34A",
    "intermediario": "#2196F3",
    "grande":        "#FF5722",
}


# ════════════════════════════════════════════════════════════════════════════════
#  ZONA DE CONFIGURAÇÃO
#  ─────────────────────────────────────────────────────────────────────────────
#  Editar os fatores abaixo junto à governança antes de cada ciclo de
#  planejamento.  Valores padrão = neutros (fatores 1.0 / 0.0 / None).
#
#  Estrutura por porte (FatoresPorte):
#    fator_base         — multiplicador global sobre a previsão técnica
#    fator_sazonalidade — ajuste por trimestre Q1..Q4 (aplicado sobre fator_base)
#    fator_expansao     — componente de expansão de mercado (adicional)
#                         meta = prev_tecnica × fator_base × fs × (1 + fator_expansao)
#    piso_variacao      — meta não pode cair abaixo de L1 × piso  (None = sem piso)
#    teto_variacao      — meta não pode crescer acima de L1 × teto (None = sem teto)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FatoresPorte:
    """Parâmetros de ajuste estratégico para um estrato de porte de município."""

    fator_base: float = 1.0
    """Escalonamento global da previsão técnica. 1.0 = neutro."""

    fator_sazonalidade: dict = field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    )
    """Ajuste sazonal por trimestre (Q1–Q4).
    Municípios pequenos são mais sensíveis ao calendário agrícola — ex.:
    Q1=1.05 (colheita verão ativa), Q4=0.95 (início de plantio, maior custo)."""

    fator_expansao: float = 0.0
    """Componente de expansão de mercado adicionado ao potencial técnico.
    0.05 = meta 5 % acima do potencial estimado para capturar market share."""

    piso_variacao: Optional[float] = None
    """Variação mínima em relação ao período anterior (L1).
    Exemplo: 0.90 → meta não pode cair mais de 10 % vs. período anterior.
    None = sem piso."""

    teto_variacao: Optional[float] = None
    """Variação máxima em relação ao período anterior (L1).
    Exemplo: 1.20 → meta não pode crescer mais de 20 % vs. período anterior.
    None = sem teto."""


# ─── Pequeno — conservador, sensível ao calendário agrícola ───────────────────
# Cooperativas em municípios < 40k hab. dependem fortemente do calendário
# agrícola regional. Ajustar fator_sazonalidade por trimestre após análise
# da safra esperada. Manter fator_base conservador para reduzir risco de
# meta inatingível em anos de seca ou geada.
FATORES_PEQUENO = FatoresPorte(
    fator_base=1.0,
    fator_sazonalidade={
        1: 1.0,   # Q1 (jan–mar): colheita verão — ex.: 1.05 se safra favorável
        2: 1.0,   # Q2 (abr–jun): plantio inverno
        3: 1.0,   # Q3 (jul–set): colheita inverno
        4: 1.0,   # Q4 (out–dez): plantio verão — ex.: 0.95 se crédito restritivo
    },
    fator_expansao=0.0,
    piso_variacao=None,   # ex.: 0.92 para proteger cooperativas de menor porte
    teto_variacao=None,   # ex.: 1.15
)

# ─── Intermediário — com componente de expansão de mercado ────────────────────
# Municípios de 40–100k hab. têm maior potencial de captação de novos
# associados e ampliação de carteira. fator_expansao representa a ambição
# incremental sobre o potencial técnico estimado.
FATORES_INTERMEDIARIO = FatoresPorte(
    fator_base=1.0,
    fator_sazonalidade={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    fator_expansao=0.0,   # ex.: 0.05 para +5 % de expansão sobre o potencial
    piso_variacao=None,
    teto_variacao=None,   # ex.: 1.20
)

# ─── Grande — ajustes mais agressivos com limites de variação ─────────────────
# Municípios >= 100k hab. têm mercados mais maduros e competitivos. Permite
# metas mais agressivas, controladas pelos limites piso/teto para evitar
# descontinuidade em relação ao histórico.
FATORES_GRANDE = FatoresPorte(
    fator_base=1.0,
    fator_sazonalidade={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    fator_expansao=0.0,   # ex.: 0.08 para metas mais agressivas
    piso_variacao=None,   # ex.: 0.90
    teto_variacao=None,   # ex.: 1.25
)

FATORES: dict[str, FatoresPorte] = {
    "pequeno":       FATORES_PEQUENO,
    "intermediario": FATORES_INTERMEDIARIO,
    "grande":        FATORES_GRANDE,
}

# ════════════════════════════════════════════════════════════════════════════════
#  Fim da zona de configuração
# ════════════════════════════════════════════════════════════════════════════════


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "metas_comerciais.log", encoding="utf-8"),
        ],
    )


def _log_fatores() -> None:
    logging.info("\nCONFIGURACAO DE FATORES DE AJUSTE (calibrar com governanca):")
    for porte, f in FATORES.items():
        logging.info(
            f"  {porte:<16}: base={f.fator_base}  expansao={f.fator_expansao}"
            f"  piso={f.piso_variacao}  teto={f.teto_variacao}"
        )
        saz = "  ".join(f"Q{q}={v}" for q, v in f.fator_sazonalidade.items())
        logging.info(f"                 sazonalidade: {saz}")


# ════════════════════════════════════════════════════════════════════════════════
#  [1] COMPONENTE TÉCNICO-PREDITIVO
# ════════════════════════════════════════════════════════════════════════════════

def gerar_previsao_tecnica(df: pd.DataFrame, target: str,
                            modelo_key: str) -> pd.DataFrame:
    """
    Carrega o modelo treinado e gera previsões para todo o painel.

    Esta é a referência técnica: representa o potencial estimado de cada
    cooperativa em cada período, sem qualquer ajuste estratégico.
    Não contém julgamento de gestão — é a saída pura do modelo.

    Parâmetros
    ----------
    df         : painel completo (panel_features_clean.parquet)
    target     : nome da variável-alvo
    modelo_key : prefixo do arquivo joblib em TUNED_DIR

    Retorna
    -------
    DataFrame com metadados da cooperativa + previsao_tecnica.
    """
    model_path = TUNED_DIR / f"{modelo_key}_{target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {model_path}\n"
            f"Execute 11_train_tuned.py antes deste script."
        )

    saved = joblib.load(model_path)
    pipe  = saved["pipeline"]
    feats = saved["features"]   # lista exata usada no treino
    logging.info(f"  Modelo: {model_path.name}  ({len(feats)} features)")

    # Converter colunas categóricas para float64 (compatibilidade Int8 nullable)
    X = df[feats].copy()
    for c in feats:
        if c in ("segmento_num", "porte_num") and c in X.columns:
            X[c] = X[c].astype("float64")

    # Metadados para rastreio
    meta_cols = ["cnpj8", "periodo_str"] + [
        c for c in ["municipio", "uf", "porte_municipio"] if c in df.columns
    ]
    df_out = df[meta_cols].copy()
    df_out["target"]    = target
    df_out["trimestre"] = df_out["periodo_str"].str[-1].astype(int)

    # Valor real (para comparação posterior)
    df_out["y_real"] = df[target].values

    # Período anterior (para limites de variação)
    l1_col = f"L1_{target}"
    df_out["L1_target"] = df[l1_col].values if l1_col in df.columns else np.nan

    # Previsão técnica: aplicar modelo a todas as linhas
    df_out["previsao_tecnica"] = pipe.predict(X.values).astype(float)

    return df_out


# ════════════════════════════════════════════════════════════════════════════════
#  [2] COMPONENTE DE AJUSTE ESTRATÉGICO
# ════════════════════════════════════════════════════════════════════════════════

def aplicar_ajuste_estrategico(df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica fatores de ajuste estratégico por porte sobre a previsão técnica,
    produzindo a meta comercial final.

    Fórmula:
        meta = previsao_tecnica × fator_base × fator_sazonalidade(trim)
               × (1 + fator_expansao)
        [com clipping por piso_variacao e teto_variacao se configurados]

    Colunas adicionadas
    -------------------
    fator_base              — escalonamento global do porte
    fator_sazonalidade      — ajuste trimestral do porte
    fator_expansao          — componente de expansão de mercado do porte
    multiplicador_total     — meta / previsao_tecnica (produto de todos os fatores)
    meta_comercial          — valor final da meta (R$ mil)
    diferenca_meta_tecnica  — meta_comercial − previsao_tecnica (R$ mil)
    diferenca_meta_tecnica_pct — idem em %
    """
    df = df_prev.copy()

    if "porte_municipio" not in df.columns:
        df["porte_municipio"] = "pequeno"

    # Inicializar colunas de ajuste com valores neutros
    df["fator_base"]         = 1.0
    df["fator_sazonalidade"] = 1.0
    df["fator_expansao"]     = 0.0
    df["meta_comercial"]     = df["previsao_tecnica"].copy()

    for porte, f in FATORES.items():
        pmask = df["porte_municipio"] == porte
        if not pmask.any():
            continue

        df.loc[pmask, "fator_base"]     = f.fator_base
        df.loc[pmask, "fator_expansao"] = f.fator_expansao

        # Sazonalidade por trimestre dentro do porte
        for q, fs in f.fator_sazonalidade.items():
            qmask = pmask & (df["trimestre"] == q)
            df.loc[qmask, "fator_sazonalidade"] = fs

        # Meta = referência técnica × todos os fatores estratégicos
        df.loc[pmask, "meta_comercial"] = (
            df.loc[pmask, "previsao_tecnica"]
            * df.loc[pmask, "fator_base"]
            * df.loc[pmask, "fator_sazonalidade"]
            * (1.0 + df.loc[pmask, "fator_expansao"])
        )

        # Limites de variação em relação ao período anterior (L1)
        l1_valid = pmask & df["L1_target"].notna() & (df["L1_target"] > 0)
        if f.piso_variacao is not None and l1_valid.any():
            piso = (df.loc[l1_valid, "L1_target"] * f.piso_variacao).values
            df.loc[l1_valid, "meta_comercial"] = np.maximum(
                df.loc[l1_valid, "meta_comercial"].values, piso
            )
        if f.teto_variacao is not None and l1_valid.any():
            teto = (df.loc[l1_valid, "L1_target"] * f.teto_variacao).values
            df.loc[l1_valid, "meta_comercial"] = np.minimum(
                df.loc[l1_valid, "meta_comercial"].values, teto
            )

    # Derivados
    ref = df["previsao_tecnica"].values
    df["multiplicador_total"]        = df["meta_comercial"].values / (np.abs(ref) + 1e-9)
    df["diferenca_meta_tecnica"]     = df["meta_comercial"] - df["previsao_tecnica"]
    df["diferenca_meta_tecnica_pct"] = (
        df["diferenca_meta_tecnica"] / (np.abs(df["previsao_tecnica"]) + 1e-9) * 100
    )

    # Arredondar
    for col in ["previsao_tecnica", "meta_comercial", "diferenca_meta_tecnica"]:
        df[col] = df[col].round(2)
    for col in ["fator_base", "fator_sazonalidade", "fator_expansao",
                "multiplicador_total", "diferenca_meta_tecnica_pct"]:
        df[col] = df[col].round(4)

    return df


# ── Diagnóstico ───────────────────────────────────────────────────────────────
def log_resumo_metas(df_metas: pd.DataFrame) -> None:
    """
    Tabela por período × porte: n cooperativas, total técnico, total meta, delta %.
    Exibe apenas os 4 períodos mais recentes.
    """
    logging.info("\nRESUMO DE METAS COMERCIAIS — ultimos 4 periodos por porte:")
    for target in TARGETS:
        sub = df_metas[df_metas["target"] == target]
        periodos = sorted(sub["periodo_str"].unique())[-4:]
        logging.info(f"\n  {TARGET_LABELS[target]}:")
        logging.info(
            f"  {'Periodo':<10} {'Porte':<16} {'n_coops':>7} "
            f"{'Prev. Tecnica (tot)':>20} {'Meta Comercial (tot)':>21} {'Delta':>8}"
        )
        for per in periodos:
            for porte in PORTE_ORDER:
                grp = sub[(sub["periodo_str"] == per) & (sub["porte_municipio"] == porte)]
                if grp.empty:
                    continue
                prev_tot = grp["previsao_tecnica"].sum()
                meta_tot = grp["meta_comercial"].sum()
                delta    = (meta_tot - prev_tot) / (abs(prev_tot) + 1e-9) * 100
                sign     = "+" if delta >= 0 else ""
                logging.info(
                    f"  {per:<10} {porte:<16} {len(grp):>7} "
                    f"{prev_tot:>20,.0f} {meta_tot:>21,.0f} {sign}{delta:>6.2f}%"
                )


# ── Figuras ───────────────────────────────────────────────────────────────────
def fig_metas_vs_historico(df_metas: pd.DataFrame) -> None:
    """
    Série temporal agregada: real (barras), previsão técnica (linha azul) e
    meta comercial (linha laranja). Soma de todas as cooperativas por período.

    Com fatores neutros, as duas linhas se sobrepõem — evidenciando que a meta
    é idêntica ao potencial técnico até que os fatores sejam calibrados.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "Potencial Técnico vs. Meta Comercial — Evolução Temporal Agregada\n"
        "Soma de todas as cooperativas do Sistema Cresol",
        fontsize=12, fontweight="bold",
    )

    for row, target in enumerate(TARGETS):
        ax  = axes[row]
        sub = df_metas[df_metas["target"] == target].copy()

        agg = (
            sub.groupby("periodo_str", sort=True)
            .agg(
                real     =("y_real",           "sum"),
                prev_tec =("previsao_tecnica",  "sum"),
                meta     =("meta_comercial",    "sum"),
            )
            .reset_index()
        )
        x = np.arange(len(agg))

        # [1] Real (histórico) — barras em cinza claro
        ax.bar(x, agg["real"], color="#CFD8DC", width=0.6, alpha=0.75,
               label="Real (histórico)", edgecolor="white", zorder=1)

        # [1] Previsão técnica — linha azul (componente técnico)
        ax.plot(x, agg["prev_tec"], "o--", color="#1565C0", linewidth=1.8,
                markersize=5, label="[1] Previsão técnica (modelo)", zorder=3)

        # [2] Meta comercial — linha laranja (componente estratégico aplicado)
        ax.plot(x, agg["meta"], "s-", color="#E65100", linewidth=2.2,
                markersize=6, label="[2] Meta comercial (ajuste estratégico)", zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(agg["periodo_str"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(TARGET_LABELS[target], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
        ax.text(0.99, 0.04,
                "Linhas sobrepostas = fatores neutros (1.0)\n"
                "Calibrar FATORES_* para separar as curvas",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#888888",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "metas_vs_historico.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: metas_vs_historico.png")


def fig_distribuicao_metas(df_metas: pd.DataFrame) -> None:
    """
    Distribuição da meta comercial por estrato de porte — período mais recente.

    Violin plot: mostra spread, mediana e distribuição dentro de cada estrato.
    Evidencia a heterogeneidade entre cooperativas do mesmo porte e permite
    identificar outliers que podem requerer ajuste fino individual.
    """
    ultimo_per = sorted(df_metas["periodo_str"].unique())[-1]
    sub_all    = df_metas[df_metas["periodo_str"] == ultimo_per].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Distribuição das Metas Comerciais por Porte — {ultimo_per}\n"
        "(meta comercial por cooperativa; com fatores neutros: meta = previsão técnica)",
        fontsize=11, fontweight="bold",
    )

    for col_ax, target in enumerate(TARGETS):
        ax    = axes[col_ax]
        sub   = sub_all[sub_all["target"] == target]

        groups    = []
        positions = []
        tick_pos  = []
        tick_lbl  = []
        colors    = []

        i = 0
        for porte in PORTE_ORDER:
            grp = sub[sub["porte_municipio"] == porte]["meta_comercial"].dropna()
            if len(grp) < 2:
                continue
            groups.append(grp.values)
            positions.append(i)
            tick_pos.append(i)
            tick_lbl.append(
                f"{PORTE_LABELS_FIG[porte]}\n(n={len(grp)})"
            )
            colors.append(PORTE_COLORS[porte])
            i += 1.8

        if not groups:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(groups, positions=positions, widths=1.1,
                              showmedians=True, showextrema=True)

        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
            pc.set_edgecolor("white")

        for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
            parts[part].set_color("#333333")
            parts[part].set_linewidth(1.2)

        # Mediana anotada
        for pos, grp in zip(positions, groups):
            med = np.median(grp)
            ax.text(pos, med, f" {med:,.0f}", va="center", fontsize=7.5,
                    color="#333333", zorder=5)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=9)
        ax.set_ylabel(TARGET_LABELS[target], fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10, pad=5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "distribuicao_metas.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: distribuicao_metas.png")


def fig_decomposicao_ajuste(df_metas: pd.DataFrame) -> None:
    """
    Decomposição do ajuste estratégico por porte e período.

    Exibe o multiplicador_total (meta / previsão_técnica) para cada porte.
    Com fatores neutros, todos os multiplicadores são 1.0.
    Após calibração, a figura mostra quanto cada porte foi ajustado para cima
    ou para baixo em relação ao potencial técnico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Decomposição do Ajuste Estratégico — Multiplicador (meta / previsão técnica)\n"
        "1.0 = neutro | > 1.0 = meta acima do potencial técnico | < 1.0 = conservador",
        fontsize=11, fontweight="bold",
    )

    for col_ax, target in enumerate(TARGETS):
        ax  = axes[col_ax]
        sub = df_metas[df_metas["target"] == target].copy()

        periodos = sorted(sub["periodo_str"].unique())

        for porte in PORTE_ORDER:
            grp = sub[sub["porte_municipio"] == porte]
            if grp.empty:
                continue
            agg = (
                grp.groupby("periodo_str")["multiplicador_total"]
                .mean()
                .reindex(periodos, fill_value=np.nan)
            )
            x = np.arange(len(periodos))
            ax.plot(x, agg.values, marker="o", markersize=4, linewidth=1.6,
                    color=PORTE_COLORS[porte],
                    label=PORTE_LABELS_FIG[porte].replace("\n", " "))

        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
                   alpha=0.6, label="Neutro (1.0)")
        ax.set_xticks(np.arange(len(periodos)))
        ax.set_xticklabels(periodos, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Multiplicador (meta / prev. técnica)", fontsize=9)
        ax.set_title(TARGET_LABELS[target], fontsize=10, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if col_ax == 0:
            ax.legend(fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "decomposicao_ajuste.png", dpi=160, bbox_inches="tight")
    plt.close()
    logging.info("  Figura: decomposicao_ajuste.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Metas Comerciais — Cresol TCC ===")

    _log_fatores()

    df = pd.read_parquet(PANEL)
    logging.info(
        f"\nPainel: {df.shape[0]} linhas x {df.shape[1]} colunas | "
        f"cooperativas: {df['cnpj8'].nunique()} | "
        f"periodos: {df['periodo_str'].min()} a {df['periodo_str'].max()}"
    )

    all_metas = []

    for target in TARGETS:
        logging.info(f"\n{'='*55}")
        logging.info(f"TARGET: {target}")
        modelo_key = MODELO_META[target]

        # ── [1] Componente técnico-preditivo ──────────────────────────────────
        logging.info("  [1] Gerando previsao tecnica (saida pura do modelo)...")
        df_prev = gerar_previsao_tecnica(df, target, modelo_key)
        n_ok    = df_prev["previsao_tecnica"].notna().sum()
        logging.info(
            f"      {n_ok}/{len(df_prev)} previsoes geradas "
            f"({len(df_prev) - n_ok} linhas sem features suficientes)"
        )

        # ── [2] Componente de ajuste estratégico ──────────────────────────────
        logging.info("  [2] Aplicando ajuste estrategico por porte...")
        df_meta = aplicar_ajuste_estrategico(df_prev)

        n_nao_neutros = (df_meta["multiplicador_total"].round(6) != 1.0).sum()
        logging.info(
            f"      {n_nao_neutros} observacoes com multiplicador != 1.0 "
            f"(todos sao neutros com fatores padrao)"
        )

        # Diagnóstico por porte no período mais recente
        ultimo_per = sorted(df_meta["periodo_str"].unique())[-1]
        sub_ult    = df_meta[df_meta["periodo_str"] == ultimo_per]
        logging.info(f"\n  Periodo mais recente: {ultimo_per}")
        logging.info(f"  {'Porte':<16} {'n':>4}  {'Prev. Tecnica':>16}  {'Meta Comercial':>16}  {'Mult.':>6}")
        for porte in PORTE_ORDER:
            grp = sub_ult[sub_ult["porte_municipio"] == porte]
            if grp.empty:
                continue
            logging.info(
                f"  {porte:<16} {len(grp):>4}  "
                f"{grp['previsao_tecnica'].sum():>16,.0f}  "
                f"{grp['meta_comercial'].sum():>16,.0f}  "
                f"{grp['multiplicador_total'].mean():>6.4f}"
            )

        all_metas.append(df_meta)

    df_metas = pd.concat(all_metas, ignore_index=True)

    log_resumo_metas(df_metas)

    # ── Figuras ───────────────────────────────────────────────────────────────
    logging.info("\nGerando figuras...")
    fig_metas_vs_historico(df_metas)
    fig_distribuicao_metas(df_metas)
    fig_decomposicao_ajuste(df_metas)

    # ── CSV de saída ──────────────────────────────────────────────────────────
    cols_out = [
        "target", "cnpj8", "periodo_str", "trimestre",
        "municipio", "uf", "porte_municipio",
        # Componente [1] — técnico
        "y_real", "L1_target", "previsao_tecnica",
        # Componente [2] — estratégico
        "fator_base", "fator_sazonalidade", "fator_expansao",
        "multiplicador_total",
        # Resultado
        "meta_comercial",
        "diferenca_meta_tecnica", "diferenca_meta_tecnica_pct",
    ]
    cols_out = [c for c in cols_out if c in df_metas.columns]
    out_path = REP_DIR / "metas_comerciais.csv"
    df_metas[cols_out].to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(f"\nSalvo: {out_path.name}  ({len(df_metas)} linhas)")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
