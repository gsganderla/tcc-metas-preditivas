#!/usr/bin/env python3
"""
04_download_n_associados.py
===========================
Coleta dados de número de cooperados/associados do Sistema Cresol via
fontes alternativas ao IF.data (onde esse campo não está disponível).

ESTRATÉGIA:
  Fonte 1 — BACEN "Dados sobre cooperativas de crédito" (Excel publicado no site)
    URL base: https://www.bcb.gov.br/estabilidadefinanceira/cooperativascredito
    O BACEN publica planilhas com dados por cooperativa, incluindo n_associados.
    Download: tenta baixar o último relatório publicado.

  Fonte 2 — SGS BACEN séries relativas ao cooperativismo de crédito
    Série 12682: Cooperativas de crédito — número de associados (sistema total)
    Série 12684: Cooperativas de crédito — número de cooperativas
    Estas séries são AGREGADAS (sistema todo), não por cooperativa.
    Uso: feature macroeconômica de setor, não por instituição.

  Fonte 3 (manual) — Relatórios anuais Cresol/OCB
    Se as fontes automáticas não cobrirem todos os períodos, o usuário deve
    complementar manualmente. O script salva um template para preenchimento.

Saídas:
  data/raw/n_associados/n_assoc_sistema_sgs.{csv,parquet}   ← séries SGS (agregado)
  data/raw/n_associados/n_assoc_por_cooperativa.{csv,parquet} ← dados BACEN Excel
  data/raw/n_associados/template_preenchimento_manual.csv   ← template para complementar
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
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "raw" / "n_associados"
LOG_DIR      = ROOT / "logs"
CRESOL_CSV   = ROOT / "data" / "raw" / "ifdata" / "cresol_resumo_trimestral.csv"

# ── SGS API ────────────────────────────────────────────────────────────────────
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

# Séries SGS relacionadas ao cooperativismo de crédito
SGS_SERIES_COOP = {
    12682: "n_associados_total_sistema",   # total de associados em todas cooperativas de crédito BR
    12684: "n_cooperativas_total",         # número total de cooperativas de crédito ativas
    12683: "n_pontos_atendimento",         # pontos de atendimento das cooperativas
}

DATA_INICIAL_SGS = "01/01/2019"
DATA_FINAL_SGS   = "31/12/2024"

# ── BACEN cooperativismo (Excel) ───────────────────────────────────────────────
# O BACEN publica relatórios periódicos em:
# https://www.bcb.gov.br/estabilidadefinanceira/cooperativascredito
# A URL exata do arquivo muda a cada publicação; tentamos padrões conhecidos.
BACEN_COOP_URLS = [
    "https://www.bcb.gov.br/content/estabilidadefinanceira/cooperativascredito/DadosCooperativasCreditoBCB.xlsx",
    "https://www.bcb.gov.br/content/estabilidadefinanceira/cooperativascredito/informacoes_cooperativas_credito.xlsx",
]

REQUEST_DELAY = 1.0
MAX_RETRIES   = 3


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_n_associados.log", encoding="utf-8"),
        ],
    )


# ── Fonte 2: SGS BACEN (agregado) ─────────────────────────────────────────────
def _baixar_sgs(codigo: int, nome_col: str) -> pd.DataFrame:
    url = SGS_URL.format(codigo=codigo)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params={"dataInicial": DATA_INICIAL_SGS, "dataFinal": DATA_FINAL_SGS, "formato": "json"},
                timeout=30,
            )
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            dados = r.json()
            if not dados:
                return pd.DataFrame()
            df = pd.DataFrame(dados)
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df["valor"] = pd.to_numeric(
                df["valor"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            df = df.dropna(subset=["data", "valor"]).rename(columns={"valor": nome_col})
            df["periodo_Q"] = df["data"].dt.to_period("Q")
            return df[["data", "periodo_Q", nome_col]]
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                logging.error(f"  SGS {codigo}: falha — {exc}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def baixar_sgs_coop() -> pd.DataFrame:
    """Baixa séries SGS de cooperativismo e agrega para trimestral."""
    logging.info("Baixando séries SGS de cooperativismo de crédito…")
    frames = []
    for codigo, nome in SGS_SERIES_COOP.items():
        logging.info(f"  Série {codigo}: {nome}")
        df = _baixar_sgs(codigo, nome)
        if not df.empty:
            frames.append(df.set_index(["data", "periodo_Q"]))

    if not frames:
        logging.warning("  Nenhuma série SGS de cooperativismo baixada.")
        return pd.DataFrame()

    df_join = frames[0]
    for f in frames[1:]:
        df_join = df_join.join(f, how="outer")
    df_join = df_join.reset_index()

    # Agregar para trimestral (last do trimestre)
    cols_val = list(SGS_SERIES_COOP.values())
    df_trim = (
        df_join
        .groupby("periodo_Q")[cols_val]
        .last()
        .reset_index()
    )
    df_trim["ano"]       = df_trim["periodo_Q"].dt.year
    df_trim["trimestre"] = df_trim["periodo_Q"].dt.quarter
    df_trim["periodo_str"] = df_trim["periodo_Q"].astype(str)
    return df_trim[df_trim["ano"].between(2019, 2024)].reset_index(drop=True)


# ── Fonte 1: BACEN Excel por cooperativa ──────────────────────────────────────
def baixar_excel_bacen() -> pd.DataFrame:
    """
    Tenta baixar o Excel de dados de cooperativas de crédito do BACEN.
    Procura por aba com 'cooperad' ou 'associad' e colunas relevantes.
    """
    logging.info("Tentando baixar Excel BACEN cooperativismo…")
    for url in BACEN_COOP_URLS:
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=60,
            )
            if r.status_code != 200:
                logging.warning(f"  {url}: status {r.status_code}")
                continue
            content_type = r.headers.get("content-type", "")
            if "html" in content_type.lower():
                logging.warning(f"  {url}: retornou HTML, não Excel")
                continue

            logging.info(f"  Excel baixado de: {url}")
            excel_bytes = BytesIO(r.content)

            # Detectar aba relevante
            xls = pd.ExcelFile(excel_bytes)
            abas_relevantes = [
                s for s in xls.sheet_names
                if any(kw in s.lower() for kw in ["cooperad", "associad", "resumo", "dados"])
            ]
            if not abas_relevantes:
                abas_relevantes = xls.sheet_names[:3]

            frames = []
            for aba in abas_relevantes:
                try:
                    df = pd.read_excel(xls, sheet_name=aba, header=None)
                    # Localizar linha do header buscando por 'CNPJ' ou 'cooperativa'
                    header_row = None
                    for i, row in df.iterrows():
                        row_vals = row.astype(str).str.upper().tolist()
                        if any("CNPJ" in v or "COOPERATIVA" in v for v in row_vals):
                            header_row = i
                            break
                    if header_row is None:
                        continue
                    df.columns = df.iloc[header_row]
                    df = df.iloc[header_row + 1:].reset_index(drop=True)
                    df.columns = [str(c).strip() for c in df.columns]
                    frames.append(df)
                    logging.info(f"    Aba '{aba}': {len(df)} linhas, colunas: {list(df.columns[:8])}")
                except Exception as exc:
                    logging.warning(f"    Aba '{aba}' ignorada: {exc}")

            if frames:
                return pd.concat(frames, ignore_index=True)

        except requests.RequestException as exc:
            logging.warning(f"  {url}: {exc}")

    logging.warning("  Excel BACEN não baixado. Verifique manualmente.")
    return pd.DataFrame()


def filtrar_cresol_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra linhas do Excel BACEN que correspondem a cooperativas Cresol.
    Busca 'CRESOL' em qualquer coluna de texto.
    """
    if df.empty:
        return df
    mask = df.apply(lambda col: col.astype(str).str.upper().str.contains("CRESOL")).any(axis=1)
    df_cresol = df[mask].copy()
    logging.info(f"  Cresol no Excel BACEN: {len(df_cresol)} linhas")
    return df_cresol


# ── Template manual ───────────────────────────────────────────────────────────
def gerar_template_manual() -> None:
    """
    Gera template CSV para preenchimento manual de n_associados por cooperativa.
    Pré-popula com os CNPJs e nomes do dataset IF.data.
    """
    template_path = DATA_DIR / "template_preenchimento_manual.csv"
    if template_path.exists():
        logging.info(f"Template manual já existe: {template_path}")
        return

    if not CRESOL_CSV.exists():
        logging.warning("Dataset Cresol (script 01) não encontrado. Template genérico criado.")
        df = pd.DataFrame({
            "cnpj8":       ["00000000"],
            "nome":        ["EXEMPLO COOPERATIVA CRESOL"],
            "uf":          ["PR"],
            "municipio":   ["FRANCISCO BELTRAO"],
        })
    else:
        df = pd.read_csv(CRESOL_CSV, encoding="utf-8-sig")
        df = df[["cnpj8", "nome", "uf", "municipio"]].drop_duplicates()

    periodos = [
        f"{ano}Q{tri}"
        for ano in range(2020, 2025)
        for tri in range(1, 5)
    ]

    # Criar painel (cooperativa × período)
    df["_key"] = 1
    df_per = pd.DataFrame({"periodo_str": periodos, "_key": 1})
    df_template = df.merge(df_per, on="_key").drop(columns="_key")
    df_template["n_associados"] = ""
    df_template["fonte"] = ""   # ex: "Relatorio_Anual_2022", "BACEN_Cooperativismo_2023"

    df_template.to_csv(template_path, index=False, encoding="utf-8-sig")
    logging.info(
        f"Template manual gerado: {template_path}\n"
        f"  Preencha a coluna 'n_associados' com os dados dos relatórios Cresol/BACEN.\n"
        f"  Fontes sugeridas:\n"
        f"    - Relatórios Anuais Cresol: https://cresol.com.br/relatorio-anual\n"
        f"    - BACEN Cooperativismo: https://www.bcb.gov.br/estabilidadefinanceira/cooperativascredito\n"
        f"    - BACEN Panorama: https://www.bcb.gov.br/publicacoes/panoramasfn"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download N_Associados – Fontes Alternativas ===")

    # ── Fonte 2: SGS (agregado sistema) ───────────────────────────────────────
    df_sgs = baixar_sgs_coop()
    if not df_sgs.empty:
        sgs_csv     = DATA_DIR / "n_assoc_sistema_sgs.csv"
        sgs_parquet = DATA_DIR / "n_assoc_sistema_sgs.parquet"
        df_sgs.to_csv(sgs_csv, index=False, encoding="utf-8-sig")
        df_sgs.to_parquet(sgs_parquet, index=False)
        logging.info(f"SGS (agregado) salvo: {sgs_csv}")
        logging.info("\n" + df_sgs.to_string(index=False))
    else:
        logging.warning(
            "Séries SGS de cooperativismo não disponíveis.\n"
            "Verifique os códigos 12682-12684 em https://www3.bcb.gov.br/sgspub/"
        )

    # ── Fonte 1: Excel BACEN por cooperativa ──────────────────────────────────
    df_excel = baixar_excel_bacen()
    if not df_excel.empty:
        df_cresol_excel = filtrar_cresol_excel(df_excel)
        if not df_cresol_excel.empty:
            excl_csv = DATA_DIR / "n_assoc_por_cooperativa_raw.csv"
            df_cresol_excel.to_csv(excl_csv, index=False, encoding="utf-8-sig")
            logging.info(f"Excel BACEN (Cresol) salvo: {excl_csv}")
            logging.info(
                "ATENÇÃO: verifique manualmente as colunas e normalize\n"
                "  cnpj8, periodo, n_associados antes de usar no modelo."
            )

    # ── Template para preenchimento manual ────────────────────────────────────
    gerar_template_manual()

    logging.info("─" * 60)
    logging.info(
        "RESUMO:\n"
        f"  Dados SGS (sistema):   {'OK' if not df_sgs.empty else 'NÃO BAIXADO'}\n"
        f"  Excel BACEN:           {'OK' if not df_excel.empty else 'NÃO BAIXADO'}\n"
        f"  Template manual:       {DATA_DIR / 'template_preenchimento_manual.csv'}\n"
        "\n"
        "PRÓXIMO PASSO para n_associados por cooperativa:\n"
        "  1. Preencher template_preenchimento_manual.csv com dados dos relatórios.\n"
        "  2. OU usar n_assoc_sistema_sgs.csv como feature de setor (não por cooperativa).\n"
        "  3. Fontes: https://cresol.com.br/relatorio-anual\n"
        "             https://www.bcb.gov.br/estabilidadefinanceira/cooperativascredito"
    )


if __name__ == "__main__":
    main()
