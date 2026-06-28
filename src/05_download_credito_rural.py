#!/usr/bin/env python3
"""
05_download_credito_rural.py
============================
Baixa dados de crédito rural por município via BACEN SICOR (Olinda OData)
e séries agregadas via SGS BACEN.

Fontes:
  Fonte 1 — BACEN SICOR Olinda API
    URL: https://olinda.bcb.gov.br/olinda/servico/SICOR/versao/v1/odata/
    Entidade: RecursosLiberadosMunicipios (descobre do $metadata se nome mudar)
    Dados: valor de crédito rural liberado por município × mês
    Filtro: municípios com cooperativa Cresol + 2019-2024

  Fonte 2 — SGS BACEN (agregado nacional, usado como feature macro)
    Séries de concessões de crédito rural total e por modalidade.
    Estas séries são nacionais — complementam a visão municipal do SICOR.

Agregação: mensal → trimestral (soma das concessões no trimestre)

Saídas:
  data/raw/credito_rural/credito_rural_municipal.{csv,parquet}  ← SICOR por município
  data/raw/credito_rural/credito_rural_agregado_sgs.{csv,parquet} ← SGS nacional
"""

import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "raw" / "credito_rural"
LOG_DIR    = ROOT / "logs"
CRESOL_CSV = ROOT / "data" / "raw" / "ifdata" / "cresol_resumo_trimestral.csv"

# ── SICOR Olinda ───────────────────────────────────────────────────────────────
SICOR_BASE     = "https://olinda.bcb.gov.br/olinda/servico/SICOR/versao/v1/odata"
SICOR_METADATA = f"{SICOR_BASE}/$metadata"

# Candidatos a nomes de entidade (tenta na ordem até encontrar uma que funcione)
SICOR_ENTITY_CANDIDATES = [
    "RecursosLiberadosMunicipios",
    "RecursosLiberadosMunicipio",
    "RecursosLiberados",
    "OperacoesMunicipio",
]

# Campos esperados (fallback se nomes diferirem)
CAMPO_MUNICIPIO  = None   # detectado do metadata
CAMPO_UF         = None
CAMPO_ANO        = None
CAMPO_MES        = None
CAMPO_VALOR      = None

# ── SGS BACEN ─────────────────────────────────────────────────────────────────
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

SGS_CREDITO_RURAL = {
    # Séries SGS confirmadas para crédito rural (concessões mensais, R$ milhões)
    # Fonte: BACEN Nota de Crédito Rural — verificar em www3.bcb.gov.br/sgspub/
    20597: "cr_total_rs_mi",           # Crédito rural - Total - Concessões (R$ mi)
    20598: "cr_custeio_rs_mi",         # Custeio agrícola - Concessões (R$ mi)
    20599: "cr_investimento_rs_mi",    # Investimento - Concessões (R$ mi)
    20600: "cr_comercializacao_rs_mi", # Comercialização - Concessões (R$ mi)
}

DATA_INICIAL_SGS = "01/10/2019"
DATA_FINAL_SGS   = "31/12/2024"

REQUEST_DELAY = 1.0
MAX_RETRIES   = 2    # reduzido — séries inexistentes falham rápido
TIMEOUT_SGS   = 20   # segundos — aborta rápido se série não existe
ODATA_TOP     = 10000   # linhas por página SICOR

# Estados de atuação Cresol (filtro SICOR para reduzir volume)
UFS_CRESOL = {"PR", "SC", "RS", "MG", "SP", "ES", "BA", "GO", "MT", "RO"}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_credito_rural.log", encoding="utf-8"),
        ],
    )


# ── Descoberta da entidade SICOR ───────────────────────────────────────────────
def descobrir_entidade_sicor(s: requests.Session) -> tuple[str, list[str]]:
    """
    Busca no $metadata do SICOR o nome da entidade de recursos por município
    e lista os campos disponíveis.

    Returns:
        (nome_entidade, [lista_de_campos]) — ou ("", []) se não encontrar.
    """
    logging.info("Consultando $metadata do SICOR…")
    try:
        r = s.get(SICOR_METADATA, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        logging.warning(f"  Metadata SICOR indisponível: {exc}")
        return "", []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        logging.warning("  Metadata SICOR não é XML válido")
        return "", []

    # Suporta OData v3 e v4 (namespaces diferentes)
    ns_candidates = [
        "http://schemas.microsoft.com/ado/2009/11/edm",   # OData v3
        "http://docs.oasis-open.org/odata/ns/edm",         # OData v4
    ]
    entity_sets = []
    ns_used = ""
    for ns in ns_candidates:
        entity_sets = root.findall(f".//{{{ns}}}EntitySet")
        if entity_sets:
            ns_used = ns
            break

    # Fallback: busca sem namespace (cobre variações)
    if not entity_sets:
        entity_sets = root.findall(".//EntitySet")

    entity_names = [e.get("Name", "") for e in entity_sets]
    logging.info(f"  Entidades SICOR ({ns_used or 'sem-ns'}): {entity_names}")

    for candidato in SICOR_ENTITY_CANDIDATES:
        for e_elem in entity_sets:
            e = e_elem.get("Name", "")
            if candidato.lower() in e.lower():
                et_name = e_elem.get("EntityType", "").split(".")[-1]
                campos = []
                if et_name and ns_used:
                    entity_type = root.find(f".//{{{ns_used}}}EntityType[@Name='{et_name}']")
                    if entity_type is not None:
                        campos = [p.get("Name", "") for p in
                                  entity_type.findall(f"{{{ns_used}}}Property")]
                logging.info(f"  Entidade selecionada: {e}, campos: {campos}")
                return e, campos

    if entity_names:
        return entity_names[0], []
    return "", []


def _inferir_campos(campos: list[str]) -> dict:
    """
    Mapeia campos detectados no metadata para os conceitos municipio/uf/ano/mes/valor.
    Retorna dict vazio se os campos não puderem ser mapeados.
    """
    mapa = {}
    for campo in campos:
        cu = campo.upper()
        if "MUNIC" in cu and "municipio" not in mapa:
            mapa["municipio"] = campo
        elif ("SG_UF" in cu or cu in ("UF", "SG_UF")) and "uf" not in mapa:
            mapa["uf"] = campo
        elif "ANO" in cu and "ano" not in mapa:
            mapa["ano"] = campo
        elif "MES" in cu and "mes" not in mapa:
            mapa["mes"] = campo
        elif any(kw in cu for kw in ("VALOR", "VL_", "CREDITO", "LIBERADO")) and "valor" not in mapa:
            mapa["valor"] = campo
    return mapa


# ── Download SICOR por município ───────────────────────────────────────────────
def baixar_sicor_municipal(
    s: requests.Session,
    entidade: str,
    campos_mapa: dict,
    municipios_cresol: set[str],
    anos: list[int],
) -> pd.DataFrame:
    """
    Faz paginação OData no SICOR para baixar crédito rural por município.
    Filtra por UFs Cresol e anos de interesse.
    """
    if not entidade:
        logging.warning("Entidade SICOR não identificada — pulando download municipal.")
        return pd.DataFrame()

    url = f"{SICOR_BASE}/{entidade}"
    campos_sel = list(campos_mapa.values())
    anos_filter = " or ".join(f"{campos_mapa.get('ano','ANO_CRED')} eq {a}" for a in anos)
    ufs_filter  = " or ".join(f"{campos_mapa.get('uf','SG_UF')} eq '{uf}'" for uf in sorted(UFS_CRESOL))
    filtro = f"({anos_filter}) and ({ufs_filter})"

    frames = []
    skip = 0
    pagina = 1
    while True:
        params = {
            "$top":    ODATA_TOP,
            "$skip":   skip,
            "$filter": filtro,
            "$format": "json",
        }
        if campos_sel:
            params["$select"] = ",".join(campos_sel)

        logging.info(f"  SICOR página {pagina} (skip={skip})…")
        try:
            r = s.get(url, params=params, timeout=120)
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
        except requests.RequestException as exc:
            logging.error(f"  SICOR falhou na página {pagina}: {exc}")
            break

        dados = r.json().get("value", [])
        if not dados:
            break
        frames.append(pd.DataFrame(dados))
        logging.info(f"    {len(dados)} registros")

        if len(dados) < ODATA_TOP:
            break
        skip += ODATA_TOP
        pagina += 1

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Renomear colunas para nomes padronizados
    rename = {v: k for k, v in campos_mapa.items()}
    df = df.rename(columns=rename)
    logging.info(f"  SICOR total: {len(df)} registros, {df.get('municipio', pd.Series()).nunique()} municípios")
    return df


def agregar_sicor_trimestral(df: pd.DataFrame, campos_mapa: dict) -> pd.DataFrame:
    """
    Agrega crédito rural SICOR para trimestral por município.
    """
    if df.empty:
        return df

    df = df.copy()
    # Colunas após renaming são: municipio, uf, ano, mes, valor
    for col in ("ano", "mes", "valor"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ano", "mes", "valor"])
    df["trimestre"] = ((df["mes"] - 1) // 3 + 1).astype(int)
    df["periodo_str"] = df["ano"].astype(int).astype(str) + "Q" + df["trimestre"].astype(str)

    grp_cols = [c for c in ["municipio", "uf", "ano", "trimestre", "periodo_str"] if c in df.columns]
    df_trim = (
        df.groupby(grp_cols)["valor"]
        .sum()
        .reset_index()
        .rename(columns={"valor": "credito_rural_rs_mil"})
    )
    return df_trim


# ── Download SGS (agregado) ────────────────────────────────────────────────────
def _baixar_sgs(codigo: int) -> pd.DataFrame:
    url = SGS_URL.format(codigo=codigo)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params={"dataInicial": DATA_INICIAL_SGS, "dataFinal": DATA_FINAL_SGS, "formato": "json"},
                timeout=TIMEOUT_SGS,
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


def baixar_sgs_credito_rural() -> pd.DataFrame:
    logging.info("Baixando séries SGS de crédito rural…")
    df_macro = None
    for codigo, nome in SGS_CREDITO_RURAL.items():
        logging.info(f"  Série {codigo}: {nome}")
        df = _baixar_sgs(codigo)
        if df.empty:
            logging.warning(f"    Série {codigo} não disponível — ignorada")
            continue
        df["periodo_Q"] = df["data"].dt.to_period("Q")
        df_trim = (
            df.groupby("periodo_Q")["valor"]
            .sum()
            .reset_index()
            .rename(columns={"valor": nome})
        )
        if df_macro is None:
            df_macro = df_trim
        else:
            df_macro = df_macro.merge(df_trim, on="periodo_Q", how="outer")

    if df_macro is None:
        return pd.DataFrame()

    df_macro["ano"]        = df_macro["periodo_Q"].dt.year
    df_macro["trimestre"]  = df_macro["periodo_Q"].dt.quarter
    df_macro["periodo_str"] = df_macro["periodo_Q"].astype(str)
    return df_macro[df_macro["ano"].between(2019, 2024)].reset_index(drop=True)


# ── Municípios Cresol ──────────────────────────────────────────────────────────
def carregar_municipios_cresol() -> set[str]:
    if not CRESOL_CSV.exists():
        logging.warning("Dataset Cresol (script 01) não encontrado. Sem filtro municipal.")
        return set()
    df = pd.read_csv(CRESOL_CSV, encoding="utf-8-sig")
    return set(df["municipio"].str.upper().unique())


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download Crédito Rural – BACEN SICOR + SGS ===")

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TCC-Research/1.0)"})

    municipios_cresol = carregar_municipios_cresol()
    anos_alvo = list(range(2019, 2025))

    # ── Fonte 1: SICOR municipal ───────────────────────────────────────────────
    logging.info("── Fonte 1: SICOR Olinda (crédito rural por município) ──")
    entidade, campos_raw = descobrir_entidade_sicor(s)
    campos_mapa = _inferir_campos(campos_raw)
    logging.info(f"  Mapeamento de campos: {campos_mapa}")

    if entidade and campos_mapa:
        df_sicor = baixar_sicor_municipal(s, entidade, campos_mapa, municipios_cresol, anos_alvo)
        df_sicor_trim = agregar_sicor_trimestral(df_sicor, campos_mapa)

        if not df_sicor_trim.empty:
            out_mun_csv = DATA_DIR / "credito_rural_municipal.csv"
            out_mun_pq  = DATA_DIR / "credito_rural_municipal.parquet"
            df_sicor_trim.to_csv(out_mun_csv, index=False, encoding="utf-8-sig")
            df_sicor_trim.to_parquet(out_mun_pq, index=False)
            logging.info(f"  Municipal salvo: {out_mun_csv} ({len(df_sicor_trim)} linhas)")
        else:
            logging.warning("  SICOR municipal: sem dados após agregação")
    else:
        logging.warning(
            "  SICOR não acessível ou campos não mapeados.\n"
            "  Alternativa manual: https://www.bcb.gov.br/estabilidadefinanceira/creditorural"
        )

    # ── Fonte 2: SGS agregado ──────────────────────────────────────────────────
    logging.info("── Fonte 2: SGS BACEN (crédito rural agregado nacional) ──")
    df_sgs = baixar_sgs_credito_rural()

    if not df_sgs.empty:
        out_sgs_csv = DATA_DIR / "credito_rural_agregado_sgs.csv"
        out_sgs_pq  = DATA_DIR / "credito_rural_agregado_sgs.parquet"
        df_sgs.to_csv(out_sgs_csv, index=False, encoding="utf-8-sig")
        df_sgs.to_parquet(out_sgs_pq, index=False)
        logging.info(f"  SGS agregado salvo: {out_sgs_csv}")
        logging.info("\n" + df_sgs.tail(8).to_string(index=False))
    else:
        logging.warning(
            "  Nenhuma série SGS de crédito rural disponível.\n"
            "  Verifique os códigos 11971/12720-12723 em https://www3.bcb.gov.br/sgspub/"
        )

    logging.info("Concluído.")


if __name__ == "__main__":
    main()
