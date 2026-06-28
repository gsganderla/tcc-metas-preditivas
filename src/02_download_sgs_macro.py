#!/usr/bin/env python3
"""
02_download_sgs_macro.py
========================
Baixa indicadores macroeconômicos via API SGS BACEN e agrega para frequência
trimestral (2019T4–2024T4, incluindo uma janela de lag para features defasadas).

Séries baixadas:
  432:  Taxa SELIC Over (% a.a.) — juros básicos
  433:  IPCA (% a.m.)            — inflação geral
   12:  Câmbio BRL/USD PTAX venda (fim de período)
 7812:  IBC-Br (índice)          — proxy atividade econômica
20714:  INPC (% a.m.)            — inflação popular
 1839:  IPA-DI Agropecuária      — preços ao produtor agrícola
 1838:  IPA-OG (geral)           — preços ao produtor geral
 4189:  Crédito Total PF+PJ — concessões (R$ milhões)

Agregação trimestral:
  Selic / câmbio  → média aritmética do trimestre
  IPCA / INPC     → taxa acumulada composta: ∏(1+r_i/100) − 1 em %
  IBC-Br / IPAs   → média do índice no trimestre
  Crédito         → soma das concessões mensais

Arquivo de saída: data/raw/bcb_sgs/indicadores_macro_trimestral.{csv,parquet}
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "bcb_sgs"
LOG_DIR  = ROOT / "logs"

# ── SGS API ────────────────────────────────────────────────────────────────────
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

# Período de coleta: 2019-10 a 2024-12 (janela extra para calcular lags futuros)
DATA_INICIAL = "01/10/2019"
DATA_FINAL   = "31/12/2024"

# Séries e configurações de agregação
SERIES = {
    432:   {"nome": "selic_aa",        "agg": "mean",     "desc": "Taxa SELIC Over (% a.a.)"},
    433:   {"nome": "ipca_acum_trim",   "agg": "compound", "desc": "IPCA acumulado no trimestre (%)"},
    12:    {"nome": "cambio_brl_usd",   "agg": "last",     "desc": "BRL/USD PTAX venda (fim de trim.)"},
    7812:  {"nome": "ibc_br",          "agg": "mean",     "desc": "IBC-Br (índice, média trimestral)"},
    20714: {"nome": "inpc_acum_trim",   "agg": "compound", "desc": "INPC acumulado no trimestre (%)"},
    1839:  {"nome": "ipa_agro",         "agg": "mean",     "desc": "IPA-DI Agropecuária (média trim.)"},
    1838:  {"nome": "ipa_og",           "agg": "mean",     "desc": "IPA-OG geral (média trim.)"},
    4189:  {"nome": "concessoes_cred",  "agg": "sum",      "desc": "Concessões de crédito total (R$ mi)"},
}

REQUEST_DELAY = 1.0
MAX_RETRIES   = 4


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_sgs.log", encoding="utf-8"),
        ],
    )


# ── Download SGS ───────────────────────────────────────────────────────────────
def baixar_serie(codigo: int) -> pd.DataFrame:
    """
    Baixa uma série do SGS BACEN.

    Returns:
        DataFrame com colunas 'data' (datetime) e 'valor' (float).
    """
    url = SGS_URL.format(codigo=codigo)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params={"dataInicial": DATA_INICIAL, "dataFinal": DATA_FINAL, "formato": "json"},
                timeout=30,
            )
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            dados = r.json()
            if not dados:
                logging.warning(f"  Série {codigo}: retornou lista vazia")
                return pd.DataFrame(columns=["data", "valor"])

            df = pd.DataFrame(dados)
            # "data" pode ser "dd/MM/yyyy"
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            # "valor" pode ter vírgula como separador decimal
            df["valor"] = (
                df["valor"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace("", None)
                .pipe(pd.to_numeric, errors="coerce")
            )
            df = df.dropna(subset=["data", "valor"]).sort_values("data").reset_index(drop=True)
            logging.info(f"  Série {codigo}: {len(df)} observações mensais")
            return df[["data", "valor"]]

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logging.error(f"  Série {codigo}: falha após {MAX_RETRIES} tentativas — {exc}")
                return pd.DataFrame(columns=["data", "valor"])
            wait = 2 ** attempt
            logging.warning(f"  Série {codigo}: tentativa {attempt} falhou. Aguardando {wait}s…")
            time.sleep(wait)

    return pd.DataFrame(columns=["data", "valor"])


# ── Agregação Trimestral ───────────────────────────────────────────────────────
def _compound_rate(rates: pd.Series) -> float:
    """Acumula taxas mensais percentuais: ∏(1 + r/100) − 1 em %."""
    return ((1 + rates / 100).prod() - 1) * 100


def agregar_trimestral(df: pd.DataFrame, agg: str, nome_col: str) -> pd.DataFrame:
    """
    Agrega série mensal para trimestral.

    Args:
        agg: 'mean', 'last', 'sum', 'compound'
    """
    df = df.copy()
    df["periodo"] = df["data"].dt.to_period("Q")

    if agg == "compound":
        grp = df.groupby("periodo")["valor"].apply(_compound_rate).reset_index()
    elif agg == "last":
        grp = df.groupby("periodo")["valor"].last().reset_index()
    elif agg == "sum":
        grp = df.groupby("periodo")["valor"].sum().reset_index()
    else:  # mean
        grp = df.groupby("periodo")["valor"].mean().reset_index()

    grp.columns = ["periodo", nome_col]
    return grp


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download SGS BACEN – Indicadores Macroeconômicos ===")

    # Baixar e agregar cada série
    df_macro = None
    for codigo, cfg in SERIES.items():
        logging.info(f"Baixando série {codigo}: {cfg['desc']}")
        df_serie = baixar_serie(codigo)
        if df_serie.empty:
            logging.warning(f"  Série {codigo} vazia — será omitida")
            continue

        df_trim = agregar_trimestral(df_serie, cfg["agg"], cfg["nome"])

        if df_macro is None:
            df_macro = df_trim
        else:
            df_macro = df_macro.merge(df_trim, on="periodo", how="outer")

    if df_macro is None or df_macro.empty:
        logging.error("Nenhuma série baixada com sucesso.")
        sys.exit(1)

    # Adicionar colunas auxiliares
    df_macro = df_macro.sort_values("periodo").reset_index(drop=True)
    df_macro["ano"]      = df_macro["periodo"].dt.year
    df_macro["trimestre"] = df_macro["periodo"].dt.quarter
    df_macro["periodo_str"] = df_macro["periodo"].astype(str).str.replace("Q", "Q")

    # Filtrar 2020T1–2024T4 para o dataset final (manter 2019T4–2019T4 como contexto)
    df_macro["periodo_dt"] = df_macro["periodo"].dt.to_timestamp()
    df_alvo = df_macro[df_macro["ano"].between(2019, 2024)].copy()

    # Salvar
    out_csv     = DATA_DIR / "indicadores_macro_trimestral.csv"
    out_parquet = DATA_DIR / "indicadores_macro_trimestral.parquet"
    df_alvo.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_alvo.to_parquet(out_parquet, index=False)

    logging.info("─" * 60)
    logging.info(f"Períodos: {df_alvo['periodo_str'].min()} a {df_alvo['periodo_str'].max()}")
    logging.info(f"Colunas:  {list(df_alvo.columns)}")
    logging.info("\n" + df_alvo.tail(8).to_string(index=False))
    logging.info(f"CSV     → {out_csv}")
    logging.info(f"Parquet → {out_parquet}")
    logging.info("Concluído.")


if __name__ == "__main__":
    main()
