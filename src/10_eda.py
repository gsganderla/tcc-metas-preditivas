#!/usr/bin/env python3
"""
10_eda.py
=========
Análise Exploratória de Dados (EDA) pré-modelagem — Painel Cresol.

Seções:
  1. Estatísticas descritivas (média, mediana, std, CV, skew, kurtosis)
  2. Distribuições — histogramas + boxplots dos targets (antes e após QC)
  3. Anomalias — boxplot de targets por período; outliers visualizados
  4. Correlações — heatmap + top correlações com cada target
  5. Sazonalidade agrícola — padrão trimestral; safra Sul (verão/inverno)
  6. Séries temporais — evolução dos targets por cooperativa e por UF/região

Lê  : data/processed/panel_features.parquet        (pré-QC)
       data/processed/panel_features_clean.parquet  (pós-QC)
Gera: reports/figures/eda/*.png  (8 figuras)
      data/processed/eda_statistics.csv
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PANEL   = ROOT / "data" / "processed" / "panel_features.parquet"
PANELC  = ROOT / "data" / "processed" / "panel_features_clean.parquet"
FIG_DIR = ROOT / "reports" / "figures" / "eda"
OUT_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"

TARGETS  = ["vol_credito_rs_mil", "captacao_rs_mil"]
LABEL_T  = {"vol_credito_rs_mil": "Volume de Crédito (R$ mil)",
             "captacao_rs_mil":   "Captações (R$ mil)"}

# Regiões: Sul (safra verão/inverno) vs outras
SUL = {"PR", "RS", "SC"}

# Paleta consistente
PAL_UF  = sns.color_palette("tab10", 10)
PAL_TGT = ["#1f77b4", "#ff7f0e"]

SAFRA_LABELS = {
    1: "Q1\n(jan–mar)\nColheita verão",
    2: "Q2\n(abr–jun)\nPlantio inverno",
    3: "Q3\n(jul–set)\nColheita inverno",
    4: "Q4\n(out–dez)\nPlantio verão",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "eda.log", encoding="utf-8"),
        ],
    )


def _salvar(fig: plt.Figure, nome: str) -> None:
    p = FIG_DIR / nome
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Salvo: {nome}")


def _fmt_mil(x, _):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


# ── 1. Estatísticas descritivas ───────────────────────────────────────────────
def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula e retorna tabela de estatísticas descritivas para variáveis numéricas."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "variavel":   col,
            "n":          len(s),
            "n_nulos":    df[col].isna().sum(),
            "pct_nulos":  round(df[col].isna().mean() * 100, 2),
            "media":      round(s.mean(), 2),
            "mediana":    round(s.median(), 2),
            "std":        round(s.std(), 2),
            "cv_pct":     round(s.std() / s.mean() * 100, 1) if s.mean() != 0 else np.nan,
            "min":        round(s.min(), 2),
            "p25":        round(s.quantile(0.25), 2),
            "p75":        round(s.quantile(0.75), 2),
            "max":        round(s.max(), 2),
            "skewness":   round(float(stats.skew(s)), 4),
            "kurtosis":   round(float(stats.kurtosis(s)), 4),
        })
    return pd.DataFrame(rows)


# ── 2. Distribuições dos targets ──────────────────────────────────────────────
def plot_distribuicoes(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Histogramas + boxplots dos targets antes e após QC."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Distribuição dos Targets — antes e após Quality Control",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, target in enumerate(TARGETS):
        label = LABEL_T[target]
        color = PAL_TGT[row]

        for col_idx, (data, titulo) in enumerate([
            (df_raw[target],   "Pré-QC"),
            (df_clean[target], "Pós-QC (winsorizado)"),
        ]):
            # Histograma
            ax_h = axes[row, col_idx * 2]
            ax_h.hist(data.dropna(), bins=40, color=color, alpha=0.7, edgecolor="white")
            ax_h.axvline(data.median(), color="red", lw=1.5, ls="--",
                         label=f"Mediana: {data.median():,.0f}")
            ax_h.axvline(data.mean(), color="orange", lw=1.5, ls=":",
                         label=f"Média: {data.mean():,.0f}")
            ax_h.set_title(f"{label}\n{titulo} — Histograma", fontsize=9)
            ax_h.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
            ax_h.legend(fontsize=7)
            skw = data.skew()
            ax_h.text(0.97, 0.95, f"skew={skw:.2f}", transform=ax_h.transAxes,
                      ha="right", va="top", fontsize=8, color="gray")

            # Boxplot
            ax_b = axes[row, col_idx * 2 + 1]
            ax_b.boxplot(data.dropna(), vert=True, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.5),
                         medianprops=dict(color="red", lw=2),
                         flierprops=dict(marker=".", markersize=3, alpha=0.4))
            ax_b.set_title(f"{label}\n{titulo} — Boxplot", fontsize=9)
            ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
            n_out = ((data < data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))) |
                     (data > data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))).sum()
            ax_b.text(0.97, 0.97, f"outliers IQR: {n_out}",
                      transform=ax_b.transAxes, ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    _salvar(fig, "eda_01_distribuicoes.png")


# ── 3. Anomalias — boxplot por período ────────────────────────────────────────
def plot_anomalias_temporal(df: pd.DataFrame) -> None:
    """Boxplot dos targets por período — revela anomalias e tendência temporal."""
    periodos = sorted(df["periodo_str"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle("Anomalias Temporais — Distribuição dos Targets por Trimestre",
                 fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        data_by_period = [df.loc[df["periodo_str"] == p, target].dropna().values
                          for p in periodos]
        bp = ax.boxplot(data_by_period, labels=periodos, patch_artist=True,
                        flierprops=dict(marker=".", markersize=3, color="red", alpha=0.5),
                        medianprops=dict(color="black", lw=2))
        # Colorir por ano
        for i, patch in enumerate(bp["boxes"]):
            ano = periodos[i][:4]
            cores = {"2020": "#aec6cf", "2021": "#77b5fe", "2022": "#5b84c4",
                     "2023": "#3b5998", "2024": "#1b3a6b"}
            patch.set_facecolor(cores.get(ano, "#aaa"))
            patch.set_alpha(0.7)

        ax.set_title(LABEL_T[target], fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylabel("R$ mil")

        # Linha de tendência na mediana
        medianas = [np.median(d) if len(d) else np.nan for d in data_by_period]
        ax.plot(range(1, len(periodos) + 1), medianas, "r--", lw=1.5,
                alpha=0.8, label="Mediana")
        ax.legend(fontsize=8)

    # Legenda de anos
    from matplotlib.patches import Patch
    leg = [Patch(color=c, label=a) for a, c in
           [("2020", "#aec6cf"), ("2021", "#77b5fe"), ("2022", "#5b84c4"),
            ("2023", "#3b5998"), ("2024", "#1b3a6b")]]
    axes[0].legend(handles=leg, title="Ano", fontsize=8, loc="upper left")

    plt.tight_layout()
    _salvar(fig, "eda_02_anomalias_temporais.png")


# ── 4. Correlações ────────────────────────────────────────────────────────────
def plot_correlacao_heatmap(df: pd.DataFrame) -> None:
    """Mapa de calor: targets × features selecionadas."""
    key_feats = (
        TARGETS
        + ["L1_vol_credito_rs_mil", "L1_captacao_rs_mil",
           "L1_ativo_total_rs_mil", "L1_patrimonio_liq_rs_mil",
           "L1_carteira_credito_rs_mil"]
        + ["L1_selic_aa", "L1_ipca_acum_trim", "L1_cambio_brl_usd",
           "L1_ibc_br", "L1_ipa_agro"]
        + ["pib_corrente_rs_mil", "pib_per_capita_rs", "vab_agro_rs_mil"]
        + ["L1_milho_rs_60kg", "L1_boi_gordo_rs_arroba", "L1_leite_rs_litro"]
        + ["share_agro_mun"]
    )
    cols_avail = [c for c in key_feats if c in df.columns]
    corr = df[cols_avail].corr(method="pearson")

    label_map = {
        "vol_credito_rs_mil":       "Vol.Crédito",
        "captacao_rs_mil":          "Captação",
        "L1_vol_credito_rs_mil":    "L1 Vol.Crédito",
        "L1_captacao_rs_mil":       "L1 Captação",
        "L1_ativo_total_rs_mil":    "L1 Ativo Total",
        "L1_patrimonio_liq_rs_mil": "L1 PL",
        "L1_carteira_credito_rs_mil": "L1 Carteira",
        "L1_selic_aa":              "L1 Selic",
        "L1_ipca_acum_trim":        "L1 IPCA",
        "L1_cambio_brl_usd":        "L1 Câmbio",
        "L1_ibc_br":                "L1 IBC-Br",
        "L1_ipa_agro":              "L1 IPA-Agro",
        "pib_corrente_rs_mil":      "PIB Municipal",
        "pib_per_capita_rs":        "PIB per Capita",
        "vab_agro_rs_mil":          "VAB Agro",
        "L1_milho_rs_60kg":         "L1 Milho",
        "L1_boi_gordo_rs_arroba":   "L1 Boi Gordo",
        "L1_leite_rs_litro":        "L1 Leite",
        "share_agro_mun":           "Share Agro Mun.",
    }
    corr.index   = [label_map.get(c, c) for c in corr.index]
    corr.columns = [label_map.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"})
    ax.set_title("Matriz de Correlação — Targets e Features Principais",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _salvar(fig, "eda_03_correlacao_heatmap.png")


def plot_top_correlacoes(df: pd.DataFrame) -> None:
    """Top 15 features mais correlacionadas com cada target."""
    feat_cols = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in TARGETS and not c.startswith("ano")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Top 15 Features — Correlação com os Targets (Pearson)",
                 fontsize=13, fontweight="bold")

    for ax, target, color in zip(axes, TARGETS, PAL_TGT):
        corrs = df[feat_cols + [target]].corr()[target].drop(target)
        top15 = corrs.abs().nlargest(15)
        valores = corrs[top15.index]

        cores = ["#d73027" if v > 0 else "#4575b4" for v in valores.values]
        labels = [c.replace("_rs_mil", "").replace("_rs_mi", "")
                    .replace("_", " ").strip() for c in valores.index]

        ax.barh(range(len(valores)), valores.values, color=cores, alpha=0.8)
        ax.set_yticks(range(len(valores)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coeficiente de Correlação (Pearson)")
        ax.set_title(LABEL_T[target], fontsize=10)
        ax.set_xlim(-1, 1)

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#d73027", label="Positiva"),
                           Patch(color="#4575b4", label="Negativa")],
                  fontsize=8, loc="lower right")

    plt.tight_layout()
    _salvar(fig, "eda_04_top_correlacoes.png")


# ── 5. Sazonalidade agrícola ──────────────────────────────────────────────────
def plot_sazonalidade(df: pd.DataFrame) -> None:
    """
    Padrão trimestral dos targets:
      Q1 (jan–mar) = colheita safra verão (soja/milho)
      Q2 (abr–jun) = plantio safra inverno
      Q3 (jul–set) = colheita safra inverno (trigo/aveia)
      Q4 (out–dez) = plantio safra verão (maior demanda de crédito)
    """
    df = df.copy()
    df["regiao"] = df["uf"].apply(lambda u: "Sul (PR/SC/RS)" if u in SUL else "Outras UFs")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Sazonalidade Agrícola — Variação Trimestral dos Targets\n"
                 "(Média ± IC95%, separado por região Cresol)",
                 fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        for regiao, cor in [("Sul (PR/SC/RS)", "#1f77b4"), ("Outras UFs", "#ff7f0e")]:
            sub = df[df["regiao"] == regiao]
            stats_q = sub.groupby("trimestre")[target].agg(
                media="mean",
                std="std",
                n="count",
            ).reset_index()
            stats_q["se"] = stats_q["std"] / np.sqrt(stats_q["n"])
            stats_q["ic95"] = 1.96 * stats_q["se"]

            xs = stats_q["trimestre"].values
            ax.plot(xs, stats_q["media"], marker="o", color=cor,
                    lw=2, ms=8, label=regiao)
            ax.fill_between(xs,
                            stats_q["media"] - stats_q["ic95"],
                            stats_q["media"] + stats_q["ic95"],
                            color=cor, alpha=0.15)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([SAFRA_LABELS[q] for q in [1, 2, 3, 4]], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
        ax.set_ylabel("R$ mil (média)")
        ax.set_title(LABEL_T[target], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Anotações de safra
        ax.axvspan(0.5, 1.5, color="#ffe599", alpha=0.25, label="Colheita verão")
        ax.axvspan(3.5, 4.5, color="#c6efce", alpha=0.25, label="Plantio verão")

    plt.tight_layout()
    _salvar(fig, "eda_05_sazonalidade.png")


def plot_sazonalidade_por_ano(df: pd.DataFrame) -> None:
    """Variação trimestral por ano — detecta se padrão sazonal é estável."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Sazonalidade por Ano — Padrão Trimestral dos Targets",
                 fontsize=13, fontweight="bold")
    palette = sns.color_palette("Blues_d", n_colors=df["ano"].nunique())
    anos = sorted(df["ano"].unique())

    for ax, target in zip(axes, TARGETS):
        for i, ano in enumerate(anos):
            sub = df[df["ano"] == ano].groupby("trimestre")[target].median()
            if len(sub) < 2:
                continue
            ax.plot(sub.index, sub.values, marker="o", color=palette[i],
                    lw=1.5, ms=6, label=str(ano), alpha=0.9)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([SAFRA_LABELS[q] for q in [1, 2, 3, 4]], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
        ax.set_ylabel("R$ mil (mediana)")
        ax.set_title(LABEL_T[target], fontsize=10)
        ax.legend(title="Ano", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _salvar(fig, "eda_06_sazonalidade_por_ano.png")


# ── 6. Séries temporais ───────────────────────────────────────────────────────
def plot_serie_temporal_uf(df: pd.DataFrame) -> None:
    """Evolução agregada dos targets por UF ao longo do tempo."""
    df = df.copy()
    df["regiao"] = df["uf"].apply(lambda u: "Sul" if u in SUL else u)

    ufs = sorted(df["uf"].unique())
    pal = dict(zip(ufs, sns.color_palette("tab10", len(ufs))))
    periodos = sorted(df["periodo_str"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Evolução Temporal dos Targets por UF — Soma Trimestral",
                 fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        for uf in ufs:
            sub = df[df["uf"] == uf].groupby("periodo_str")[target].sum()
            sub = sub.reindex(periodos)
            n_coops = df[df["uf"] == uf]["cnpj8"].nunique()
            lw = 2.5 if uf in SUL else 1.2
            ax.plot(range(len(periodos)), sub.values, marker="o", ms=4,
                    lw=lw, color=pal[uf], label=f"{uf} ({n_coops} coops)", alpha=0.85)

        ax.set_xticks(range(len(periodos)))
        ax.set_xticklabels(periodos, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
        ax.set_ylabel("R$ mil (soma UF)")
        ax.set_title(LABEL_T[target], fontsize=10)
        ax.legend(title="UF", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _salvar(fig, "eda_07_serie_temporal_uf.png")


def plot_serie_temporal_cooperativas(df: pd.DataFrame) -> None:
    """Séries temporais individuais (fundo) + mediana agregada (destaque)."""
    periodos = sorted(df["periodo_str"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Séries Temporais Individuais por Cooperativa\n"
                 "(linhas finas = cooperativa, linha grossa = mediana geral)",
                 fontsize=13, fontweight="bold")

    for ax, target, color in zip(axes, TARGETS, PAL_TGT):
        for cnpj in df["cnpj8"].unique():
            sub = df[df["cnpj8"] == cnpj].set_index("periodo_str")[target]
            sub = sub.reindex(periodos)
            ax.plot(range(len(periodos)), sub.values, lw=0.5,
                    color=color, alpha=0.2)

        # Mediana geral por período
        med = df.groupby("periodo_str")[target].median().reindex(periodos)
        ax.plot(range(len(periodos)), med.values, lw=3, color="black",
                zorder=5, label="Mediana geral")
        ax.fill_between(
            range(len(periodos)),
            df.groupby("periodo_str")[target].quantile(0.25).reindex(periodos),
            df.groupby("periodo_str")[target].quantile(0.75).reindex(periodos),
            color="gray", alpha=0.2, label="IQR (P25–P75)",
        )

        ax.set_xticks(range(len(periodos)))
        ax.set_xticklabels(periodos, rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_mil))
        ax.set_ylabel("R$ mil")
        ax.set_title(LABEL_T[target], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _salvar(fig, "eda_08_serie_cooperativas.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== EDA – Painel Cresol ===")

    df_raw   = pd.read_parquet(PANEL)
    df_clean = pd.read_parquet(PANELC)
    logging.info(f"Painel bruto:  {df_raw.shape[0]} linhas × {df_raw.shape[1]} cols")
    logging.info(f"Painel limpo: {df_clean.shape[0]} linhas × {df_clean.shape[1]} cols")
    logging.info(f"UFs: {sorted(df_raw['uf'].unique())}")
    logging.info(f"Cooperativas: {df_raw['cnpj8'].nunique()}")

    # 1. Estatísticas descritivas
    logging.info("\n[1] Estatísticas descritivas…")
    df_stats = estatisticas_descritivas(df_clean)
    df_stats.to_csv(OUT_DIR / "eda_statistics.csv", index=False, encoding="utf-8-sig")
    logging.info(f"  Salvo: eda_statistics.csv ({len(df_stats)} variáveis)")

    tgt_stats = df_stats[df_stats["variavel"].isin(TARGETS)][
        ["variavel", "n", "media", "mediana", "std", "cv_pct", "skewness", "kurtosis"]]
    logging.info("\n  Targets:")
    logging.info("\n" + tgt_stats.to_string(index=False))

    # 2. Distribuições
    logging.info("\n[2] Distribuições (pré e pós QC)…")
    plot_distribuicoes(df_raw, df_clean)

    # 3. Anomalias temporais
    logging.info("\n[3] Anomalias temporais…")
    plot_anomalias_temporal(df_raw)

    # 4. Correlações
    logging.info("\n[4] Correlações…")
    plot_correlacao_heatmap(df_clean)
    plot_top_correlacoes(df_clean)

    # 5. Sazonalidade
    logging.info("\n[5] Sazonalidade agrícola…")
    plot_sazonalidade(df_clean)
    plot_sazonalidade_por_ano(df_clean)

    # 6. Séries temporais
    logging.info("\n[6] Séries temporais…")
    plot_serie_temporal_uf(df_clean)
    plot_serie_temporal_cooperativas(df_clean)

    # Resumo final
    logging.info(f"\n{'='*60}")
    logging.info(f"Figuras geradas em: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("eda_*.png")):
        logging.info(f"  {f.name}")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
