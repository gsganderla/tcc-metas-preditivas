#!/usr/bin/env python3
"""
07b_quality_control.py
======================
Controle de qualidade do painel analítico antes do treinamento.

Regras aplicadas (nesta ordem):
  1. remover_duplicados()  — remove linhas duplicadas por (cnpj8, periodo_str)
  2. tratar_nulos()        — interpolação OU exclusão, critério: % nulos por variável
  3. detectar_outliers()   — método IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  4. tratar_outliers()     — winsorização OU exclusão, critério: assimetria da distribuição

Critérios de nulos (densidade_variavel):
  <= 5%  : interpolação forward/backward fill dentro de cada cooperativa
  5–20%  : interpolação linear temporal dentro de cada cooperativa
  20–50% : exclusão da variável (feature removida do painel)
  > 50%  : exclusão da variável

Critérios de outliers (analise_distribuicao):
  |skewness| < 2 : winsorização nos fences IQR (Q1-1.5*IQR, Q3+1.5*IQR)
  |skewness| >= 2 : winsorização nos percentis 1%–99% (distribuição muito assimétrica)

Variáveis ID e targets são excluídas do tratamento de outliers.

Entrada : data/processed/panel_features.parquet
Saída   : data/processed/panel_features_clean.parquet / .csv
Relatório: data/processed/quality_report.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "data" / "processed" / "panel_features.parquet"
OUT_CSV  = ROOT / "data" / "processed" / "panel_features_clean.csv"
OUT_PAR  = ROOT / "data" / "processed" / "panel_features_clean.parquet"
RPT_FILE = ROOT / "data" / "processed" / "quality_report.csv"
LOG_DIR  = ROOT / "logs"

# Colunas que NÃO são tratadas como features numéricas
ID_COLS = {
    "cnpj8", "nome", "uf", "municipio", "codigo_ibge_7d",
    "periodo_str", "periodo", "ano", "trimestre", "data_base",
    "segmento", "segmento_num", "ano_ref_pib",
}
TARGET_COLS = {"vol_credito_rs_mil", "captacao_rs_mil", "carteira_credito_rs_mil"}

# Thresholds de densidade de nulos
THRESH_FFILL   = 0.05   # <= 5% : forward/backward fill
THRESH_INTERP  = 0.20   # <= 20%: interpolação linear
THRESH_EXCLUDE = 0.50   # > 50% : excluir variável

# Threshold de assimetria para escolha do método de outlier
SKEW_THRESH = 2.0


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "quality_control.log", encoding="utf-8"),
        ],
    )


# ── 1. Remover duplicados ──────────────────────────────────────────────────────
def remover_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove linhas duplicadas por (cnpj8, periodo_str)."""
    n_antes = len(df)
    df = df.drop_duplicates(subset=["cnpj8", "periodo_str"], keep="first")
    n_rem = n_antes - len(df)
    logging.info(f"  Duplicados removidos: {n_rem} (de {n_antes})")
    return df.reset_index(drop=True), {"duplicados_removidos": n_rem}


# ── 2. Tratar nulos ────────────────────────────────────────────────────────────
def tratar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Aplica tratamento de nulos coluna a coluna, com base na densidade de nulos.
    Interpolação é feita dentro de cada cooperativa (cnpj8), respeitando a ordem temporal.
    """
    relatorio = []
    feat_cols = [c for c in df.columns if c not in ID_COLS and c not in TARGET_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]

    cols_excluir = []
    for col in feat_cols:
        pct = df[col].isna().mean()
        n_nulos = df[col].isna().sum()

        if n_nulos == 0:
            metodo = "nenhum"
            n_tratados = 0
        elif pct <= THRESH_FFILL:
            # Forward fill + backward fill dentro de cada cooperativa
            df[col] = (
                df.sort_values(["cnpj8", "ano", "trimestre"])
                  .groupby("cnpj8")[col]
                  .transform(lambda s: s.ffill().bfill())
            )
            metodo = "ffill_bfill"
            n_tratados = n_nulos - df[col].isna().sum()
        elif pct <= THRESH_INTERP:
            # Interpolação linear dentro de cada cooperativa
            df[col] = (
                df.sort_values(["cnpj8", "ano", "trimestre"])
                  .groupby("cnpj8")[col]
                  .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )
            metodo = "interpolacao_linear"
            n_tratados = n_nulos - df[col].isna().sum()
        elif pct <= THRESH_EXCLUDE:
            # Entre 20% e 50%: excluir variável
            cols_excluir.append(col)
            metodo = "exclusao_variavel"
            n_tratados = 0
        else:
            # Mais de 50%: excluir variável
            cols_excluir.append(col)
            metodo = "exclusao_variavel_alta_densidade"
            n_tratados = 0

        relatorio.append({
            "coluna": col,
            "pct_nulos_original": round(pct * 100, 2),
            "n_nulos_original": n_nulos,
            "metodo_nulos": metodo,
            "n_tratados": n_tratados,
        })

    if cols_excluir:
        logging.info(f"  Variáveis excluídas por alta % de nulos ({len(cols_excluir)}): "
                     f"{cols_excluir[:5]}{'...' if len(cols_excluir) > 5 else ''}")
        df = df.drop(columns=cols_excluir)

    n_interp = sum(1 for r in relatorio if "interpolacao" in r["metodo_nulos"])
    n_ffill  = sum(1 for r in relatorio if r["metodo_nulos"] == "ffill_bfill")
    logging.info(f"  Nulos: {n_ffill} colunas ffill, {n_interp} colunas interpoladas, "
                 f"{len(cols_excluir)} excluídas")
    return df, relatorio


# ── 3. Detectar outliers (IQR) ─────────────────────────────────────────────────
def detectar_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Detecta outliers em todas as features numéricas usando o método IQR.
    Retorna o DataFrame com flag booleana por coluna e relatório.
    """
    feat_cols = [c for c in df.columns if c not in ID_COLS and c not in TARGET_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]

    relatorio = []
    for col in feat_cols:
        serie = df[col].dropna()
        if len(serie) < 4:
            continue
        Q1  = serie.quantile(0.25)
        Q3  = serie.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        fence_low  = Q1 - 1.5 * IQR
        fence_high = Q3 + 1.5 * IQR
        mask_out   = (df[col] < fence_low) | (df[col] > fence_high)
        n_out      = mask_out.sum()
        sk         = float(skew(serie, nan_policy="omit"))

        relatorio.append({
            "coluna":      col,
            "Q1":          round(Q1, 4),
            "Q3":          round(Q3, 4),
            "IQR":         round(IQR, 4),
            "fence_low":   round(fence_low, 4),
            "fence_high":  round(fence_high, 4),
            "n_outliers":  int(n_out),
            "pct_outliers": round(n_out / len(df) * 100, 2),
            "skewness":    round(sk, 4),
        })

    total_out = sum(r["n_outliers"] for r in relatorio)
    n_cols_c_out = sum(1 for r in relatorio if r["n_outliers"] > 0)
    logging.info(f"  Outliers detectados: {total_out} valores em {n_cols_c_out} colunas")
    return df, relatorio


# ── 4. Tratar outliers ─────────────────────────────────────────────────────────
def tratar_outliers(df: pd.DataFrame,
                    relatorio_iqr: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Trata outliers detectados pelo IQR:
      - |skewness| < SKEW_THRESH : winsorização nos fences IQR
      - |skewness| >= SKEW_THRESH: winsorização nos percentis 1%-99%
    """
    relatorio = []
    iqr_map = {r["coluna"]: r for r in relatorio_iqr if r["n_outliers"] > 0}

    for col, info in iqr_map.items():
        if col not in df.columns:
            continue

        sk  = abs(info["skewness"])
        n_antes = df[col].isna().sum()

        if sk < SKEW_THRESH:
            # Winsorização nos fences IQR
            acao = "winsorizacao_IQR"
            low, high = info["fence_low"], info["fence_high"]
        else:
            # Winsorização nos percentis 1%–99% (distribuição muito assimétrica)
            acao = "winsorizacao_P1_P99"
            low  = df[col].quantile(0.01)
            high = df[col].quantile(0.99)

        n_clamp_low  = (df[col] < low).sum()
        n_clamp_high = (df[col] > high).sum()
        df[col] = df[col].clip(lower=low, upper=high)

        relatorio.append({
            "coluna":        col,
            "acao":          acao,
            "skewness":      info["skewness"],
            "limite_baixo":  round(low, 4),
            "limite_alto":   round(high, 4),
            "n_clamp_baixo": int(n_clamp_low),
            "n_clamp_alto":  int(n_clamp_high),
            "n_total_ajustados": int(n_clamp_low + n_clamp_high),
        })

    n_wins = sum(1 for r in relatorio if "IQR" in r["acao"])
    n_p99  = sum(1 for r in relatorio if "P1" in r["acao"])
    total_ajust = sum(r["n_total_ajustados"] for r in relatorio)
    logging.info(f"  Outliers tratados: {total_ajust} valores "
                 f"({n_wins} cols winsorização IQR, {n_p99} cols winsorização P1-P99)")
    return df, relatorio


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    logging.info("=== Controle de Qualidade – Painel Cresol ===")

    df = pd.read_parquet(IN_FILE)
    logging.info(f"Painel entrada: {df.shape[0]} linhas × {df.shape[1]} colunas")

    rpt_rows = []

    # 1. Duplicados
    logging.info("\n[1] Removendo duplicados…")
    df, info_dup = remover_duplicados(df)
    rpt_rows.append({"etapa": "duplicados", **info_dup, "acao": "remocao"})

    # 2. Nulos
    logging.info("\n[2] Tratando nulos (critério: densidade da variável)…")
    df, info_nulos = tratar_nulos(df)
    for r in info_nulos:
        rpt_rows.append({"etapa": "nulos", **r})

    # 3. Detectar outliers
    logging.info("\n[3] Detectando outliers (método IQR)…")
    df, info_iqr = detectar_outliers(df)
    for r in info_iqr:
        rpt_rows.append({"etapa": "deteccao_outliers", **r})

    # 4. Tratar outliers
    logging.info("\n[4] Tratando outliers (critério: assimetria da distribuição)…")
    df, info_out = tratar_outliers(df, info_iqr)
    for r in info_out:
        rpt_rows.append({"etapa": "tratamento_outliers", **r})

    # Salvar painel limpo
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.to_parquet(OUT_PAR, index=False)

    # Salvar relatório
    df_rpt = pd.DataFrame(rpt_rows)
    df_rpt.to_csv(RPT_FILE, index=False, encoding="utf-8-sig")

    # Resumo final
    logging.info(f"\n{'='*60}")
    logging.info(f"Painel limpo: {df.shape[0]} linhas × {df.shape[1]} colunas")
    logging.info(f"Targets (verificação de integridade):")
    for t in ["vol_credito_rs_mil", "captacao_rs_mil", "carteira_credito_rs_mil"]:
        if t in df.columns:
            pct = df[t].isna().mean() * 100
            logging.info(f"  {t}: {df[t].notna().sum()} obs, {pct:.1f}% nulos")

    logging.info(f"\nArquivos salvos:")
    logging.info(f"  {OUT_CSV.name}")
    logging.info(f"  {OUT_PAR.name}")
    logging.info(f"  {RPT_FILE.name}")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
