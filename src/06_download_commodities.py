#!/usr/bin/env python3
"""
06_download_commodities.py
==========================
Baixa preços de commodities agrícolas relevantes para o Sistema Cresol
(cooperativas majoritariamente no Sul e Centro-Oeste).

Produtos alvo: soja, milho, leite (principais culturas da base associada Cresol)

Fontes (tentadas em ordem):
  Fonte 1 — CEPEA/ESALQ (Indicadores de Preços Agropecuários)
    URL padrão: https://www.cepea.esalq.usp.br/br/indicador/{produto}.aspx
    Dados: preços diários/semanais → agregados para mensais → trimestrais
    Extrai link de download Excel do HTML da página de indicador

  Fonte 2 — SGS BACEN (fallback: índices de preços agrícolas)
    Série 1839: IPA-DI Agropecuária (já no script 02, aqui com mais granularidade)
    Série 7828: Soja — price index (se disponível)
    Série 7829: Milho — price index (se disponível)

Agregação: preços diários → média mensal → média trimestral

Saídas:
  data/raw/commodities/precos_commodities_mensal.{csv,parquet}
  data/raw/commodities/precos_commodities_trimestral.{csv,parquet}
"""

import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "commodities"
LOG_DIR  = ROOT / "logs"

# ── CEPEA ─────────────────────────────────────────────────────────────────────
CEPEA_BASE = "https://www.cepea.esalq.usp.br"

# Produtos CEPEA: chave usada na URL, coluna de saída, nome para log
CEPEA_PRODUTOS = {
    "soja": {
        "url_indicador": f"{CEPEA_BASE}/br/indicador/soja.aspx",
        "coluna": "soja_rs_saca60kg",
        "desc":   "Soja Paraná (R$/saca 60 kg)",
    },
    "milho": {
        "url_indicador": f"{CEPEA_BASE}/br/indicador/milho.aspx",
        "coluna": "milho_rs_saca60kg",
        "desc":   "Milho Paraná (R$/saca 60 kg)",
    },
    "leite": {
        "url_indicador": f"{CEPEA_BASE}/br/indicador/leite.aspx",
        "coluna": "leite_rs_litro",
        "desc":   "Leite Brasil (R$/litro)",
    },
}

# Período de interesse
ANO_INICIAL = 2019
ANO_FINAL   = 2024

# ── SGS BACEN (fallback) ───────────────────────────────────────────────────────
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

SGS_COMMODITIES = {
    # Índices de preços agropecuários — confirmados no SGS BACEN
    1839:  "ipa_agro_idx",        # IPA-DI Agropecuária (índice, mensal)
    28521: "soja_rs_60kg",        # Preço soja Paraná (R$/sc 60 kg) — IGP-FGV
    28558: "milho_rs_60kg",       # Preço milho Paraná (R$/sc 60 kg)
    28565: "boi_gordo_rs_arroba", # Preço boi gordo (R$/@) — SP
    28503: "leite_rs_litro",      # Preço leite ao produtor (R$/litro)
}

DATA_INICIAL_SGS = "01/10/2019"
DATA_FINAL_SGS   = "31/12/2024"

REQUEST_DELAY = 1.5
MAX_RETRIES   = 2    # falhas rápidas — séries inexistentes são ignoradas

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_commodities.log", encoding="utf-8"),
        ],
    )


# ── Fonte 1: CEPEA ────────────────────────────────────────────────────────────
def _extrair_link_excel(html: str, base_url: str) -> str | None:
    """
    Extrai o primeiro link de download .xls/.xlsx da página CEPEA.
    """
    padrao = re.compile(
        r'href=["\']([^"\']*\.(xls|xlsx)[^"\']*)["\']',
        re.IGNORECASE
    )
    for m in padrao.finditer(html):
        href = m.group(1).strip()
        if not href.startswith("http"):
            href = base_url.rstrip("/") + "/" + href.lstrip("/")
        return href
    return None


def _parse_excel_cepea(conteudo: bytes, produto: str) -> pd.DataFrame:
    """
    Tenta ler o Excel CEPEA. A estrutura varia por produto, por isso tenta
    múltiplas abordagens: diferentes engines, header rows e colunas.
    """
    for engine in ("openpyxl", "xlrd"):
        try:
            xlsx = BytesIO(conteudo)
            # Ler sem header para inspecionar
            df_raw = pd.read_excel(xlsx, header=None, engine=engine)
            break
        except Exception:
            continue
    else:
        raise ValueError("Nenhum engine Excel disponível (instale openpyxl ou xlrd)")

    # Procurar linha do header buscando colunas com 'data' e 'preço' / 'valor'
    header_row = None
    for i, row in df_raw.iterrows():
        vals = [str(v).strip().upper() for v in row.values]
        if any("DATA" in v for v in vals) and any(
            kw in v for v in vals for kw in ("PRECO", "PREÇO", "VALOR", "R$", "MEDIA")
        ):
            header_row = i
            break

    if header_row is None:
        # Heurística: primeira linha com data parsável
        for i, row in df_raw.iterrows():
            for v in row.values:
                try:
                    pd.to_datetime(str(v), dayfirst=True)
                    header_row = max(0, i - 1)
                    break
                except Exception:
                    pass
            if header_row is not None:
                break

    if header_row is None:
        header_row = 0

    xlsx = BytesIO(conteudo)
    try:
        df = pd.read_excel(xlsx, header=header_row, engine="openpyxl")
    except Exception:
        xlsx = BytesIO(conteudo)
        df = pd.read_excel(xlsx, header=header_row, engine="xlrd")

    df.columns = [str(c).strip() for c in df.columns]

    # Encontrar coluna de data e coluna de preço
    col_data = next(
        (c for c in df.columns if "DATA" in c.upper() or "DATE" in c.upper()), None
    )
    col_preco = next(
        (c for c in df.columns
         if any(kw in c.upper() for kw in ("PRECO", "PREÇO", "VALOR", "MÉDIA", "MEDIA", "R$"))
         and c != col_data),
        None,
    )

    if col_data is None or col_preco is None:
        # Fallback: assumir primeira coluna = data, segunda = preço
        cols = df.columns.tolist()
        col_data  = cols[0] if cols else None
        col_preco = cols[1] if len(cols) > 1 else None

    if col_data is None:
        raise ValueError(f"Colunas data/preço não identificadas no Excel CEPEA ({produto})")

    result = pd.DataFrame()
    result["data"]  = pd.to_datetime(df[col_data], dayfirst=True, errors="coerce")
    result["preco"] = pd.to_numeric(
        df[col_preco].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )
    result = result.dropna(subset=["data", "preco"])
    result = result[
        (result["data"].dt.year >= ANO_INICIAL) &
        (result["data"].dt.year <= ANO_FINAL)
    ]
    return result.sort_values("data").reset_index(drop=True)


def baixar_cepea(produto: str, cfg: dict, s: requests.Session) -> pd.DataFrame:
    """
    Baixa série de preços de um produto CEPEA.

    1. Acessa a página do indicador
    2. Extrai link para arquivo Excel
    3. Baixa e parseia o Excel
    """
    logging.info(f"  CEPEA {produto}: acessando {cfg['url_indicador']}")

    # Passo 1: página do indicador
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.get(cfg["url_indicador"], timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logging.warning(f"    Página {produto} inacessível: {exc}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    html = r.text
    link_excel = _extrair_link_excel(html, CEPEA_BASE)

    if not link_excel:
        logging.warning(f"    Link Excel não encontrado na página {produto}")
        return pd.DataFrame()

    logging.info(f"    Excel encontrado: {link_excel}")

    # Passo 2: baixar Excel
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r2 = s.get(link_excel, timeout=60)
            r2.raise_for_status()
            time.sleep(REQUEST_DELAY)
            break
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logging.warning(f"    Download Excel falhou: {exc}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    # Passo 3: parsear
    try:
        df = _parse_excel_cepea(r2.content, produto)
        df = df.rename(columns={"preco": cfg["coluna"]})
        logging.info(f"    {len(df)} observações de {cfg['desc']}")
        return df
    except Exception as exc:
        logging.warning(f"    Parse Excel falhou: {exc}")
        return pd.DataFrame()


# ── Fonte 2: SGS ──────────────────────────────────────────────────────────────
def _baixar_sgs(codigo: int) -> pd.DataFrame:
    url = SGS_URL.format(codigo=codigo)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params={"dataInicial": DATA_INICIAL_SGS, "dataFinal": DATA_FINAL_SGS, "formato": "json"},
                timeout=20,
            )
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            dados = r.json()
            if not dados:
                return pd.DataFrame()
            df = pd.DataFrame(dados)
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df["valor"] = pd.to_numeric(
                df["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
            return df.dropna(subset=["data", "valor"]).sort_values("data").reset_index(drop=True)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logging.warning(f"  SGS {codigo}: {exc}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def baixar_sgs_commodities() -> pd.DataFrame:
    logging.info("Baixando séries SGS de índices de preços agrícolas…")
    df_join = None
    for codigo, nome in SGS_COMMODITIES.items():
        logging.info(f"  Série {codigo}: {nome}")
        df = _baixar_sgs(codigo)
        if df.empty:
            logging.warning(f"    Série {codigo} não disponível")
            continue
        df = df.rename(columns={"valor": nome})
        if df_join is None:
            df_join = df[["data", nome]]
        else:
            df_join = df_join.merge(df[["data", nome]], on="data", how="outer")

    if df_join is None:
        return pd.DataFrame()
    return df_join.sort_values("data").reset_index(drop=True)


# ── Agregação ─────────────────────────────────────────────────────────────────
def agregar_mensal(df: pd.DataFrame, col_preco: str) -> pd.DataFrame:
    """Agrega preços diários para média mensal."""
    df = df.copy()
    df["ano_mes"] = df["data"].dt.to_period("M")
    return (
        df.groupby("ano_mes")[col_preco]
        .mean()
        .reset_index()
        .rename(columns={"ano_mes": "periodo_M"})
    )


def agregar_trimestral_df(df: pd.DataFrame, cols_val: list[str]) -> pd.DataFrame:
    """Agrega série mensal (coluna periodo_M) para trimestral (média)."""
    df = df.copy()
    if "periodo_M" in df.columns:
        df["periodo_Q"] = df["periodo_M"].dt.to_timestamp().dt.to_period("Q")
    elif "data" in df.columns:
        df["periodo_Q"] = df["data"].dt.to_period("Q")

    df_trim = (
        df.groupby("periodo_Q")[cols_val]
        .mean()
        .reset_index()
    )
    df_trim["ano"]        = df_trim["periodo_Q"].dt.year
    df_trim["trimestre"]  = df_trim["periodo_Q"].dt.quarter
    df_trim["periodo_str"] = df_trim["periodo_Q"].astype(str)
    return df_trim[df_trim["ano"].between(ANO_INICIAL, ANO_FINAL)].reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download Commodities – CEPEA + SGS ===")

    s = requests.Session()
    s.headers.update(HEADERS)

    # ── Fonte 1: CEPEA ────────────────────────────────────────────────────────
    logging.info("── Fonte 1: CEPEA/ESALQ ──")
    frames_cepea = {}
    for produto, cfg in CEPEA_PRODUTOS.items():
        df = baixar_cepea(produto, cfg, s)
        if not df.empty:
            frames_cepea[cfg["coluna"]] = df

    # Consolidar CEPEA em um único DataFrame mensal (outer join por data)
    df_cepea_mensal = None
    for col, df in frames_cepea.items():
        df_m = agregar_mensal(df, col)
        if df_cepea_mensal is None:
            df_cepea_mensal = df_m
        else:
            df_cepea_mensal = df_cepea_mensal.merge(df_m, on="periodo_M", how="outer")

    # ── Fonte 2: SGS ──────────────────────────────────────────────────────────
    logging.info("── Fonte 2: SGS BACEN (índices de preços agrícolas) ──")
    df_sgs = baixar_sgs_commodities()

    # ── Combinar ──────────────────────────────────────────────────────────────
    if df_cepea_mensal is not None:
        # Converter para trimestral
        cols_cepea = list(frames_cepea.keys())
        df_trim_cepea = agregar_trimestral_df(df_cepea_mensal, cols_cepea)
    else:
        df_trim_cepea = pd.DataFrame()
        logging.warning(
            "CEPEA não acessível. Alternativa manual:\n"
            "  Acesse https://www.cepea.esalq.usp.br/br/indicador/ e baixe os Excel\n"
            "  para soja, milho e leite. Coloque em data/raw/commodities/cepea_manual/."
        )

    if not df_sgs.empty:
        df_sgs["periodo_Q"] = df_sgs["data"].dt.to_period("Q")
        cols_sgs = [c for c in SGS_COMMODITIES.values() if c in df_sgs.columns]
        df_trim_sgs = agregar_trimestral_df(df_sgs, cols_sgs)
    else:
        df_trim_sgs = pd.DataFrame()

    # Juntar CEPEA + SGS por periodo_str
    if not df_trim_cepea.empty and not df_trim_sgs.empty:
        df_final = df_trim_cepea.merge(
            df_trim_sgs.drop(columns=["ano", "trimestre"], errors="ignore"),
            on=["periodo_Q", "periodo_str"],
            how="outer",
        )
    elif not df_trim_cepea.empty:
        df_final = df_trim_cepea
    elif not df_trim_sgs.empty:
        df_final = df_trim_sgs
    else:
        logging.error("Nenhuma fonte de commodities disponível.")
        sys.exit(1)

    df_final = df_final.sort_values("periodo_str").reset_index(drop=True)

    # Salvar
    out_csv     = DATA_DIR / "precos_commodities_trimestral.csv"
    out_parquet = DATA_DIR / "precos_commodities_trimestral.parquet"
    df_final.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_final.to_parquet(out_parquet, index=False)

    logging.info("─" * 60)
    logging.info(f"Colunas: {list(df_final.columns)}")
    logging.info(f"Períodos: {df_final['periodo_str'].min()} a {df_final['periodo_str'].max()}")
    logging.info("\n" + df_final.tail(8).to_string(index=False))
    logging.info(f"CSV     → {out_csv}")
    logging.info(f"Parquet → {out_parquet}")
    logging.info("Concluído.")


if __name__ == "__main__":
    main()
