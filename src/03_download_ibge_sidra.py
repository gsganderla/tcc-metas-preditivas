#!/usr/bin/env python3
"""
03_download_ibge_sidra.py
=========================
Baixa dados de PIB e valor adicionado por setor para todos os municípios
brasileiros via IBGE SIDRA (tabela 5938) e extrai apenas os municípios onde
há cooperativas Cresol.

Tabela 5938 — Produto Interno Bruto dos Municípios:
  Variável 37:  PIB a preços correntes (R$ mil)
  Variável 498: PIB per capita (R$ 1,00)
  Variável 543: Valor adicionado bruto — Agropecuária (R$ mil)
  Variável 544: Valor adicionado bruto — Indústria (R$ mil)
  Variável 545: Valor adicionado bruto — Serviços e adm. pública (R$ mil)

Disponibilidade: dados anuais, normalmente até penúltimo ano (2022 no início de 2024).
Frequência: anual → propagada para trimestres via forward-fill.

Fluxo:
  1. Lê lista de municípios Cresol do output do script 01
     (data/raw/ifdata/cresol_resumo_trimestral.csv)
  2. Baixa tabela 5938 para TODOS os municípios (UF×município) por variável
  3. Filtra pelos municípios Cresol (match via nome + UF)
  4. Converte para painel trimestral com propagação anual → trimestral
  5. Salva data/raw/ibge_sidra/pib_municipal_cresol.{csv,parquet}

Chave de integração:
  codigo_ibge_7d (7 dígitos = IBGE municipality code)
  O prefixo 0 em alguns estados (ex: 0500100 para AC) é preservado.

IBGE SIDRA API v3:
  GET https://servicodados.ibge.gov.br/api/v3/agregados/{tabela}/periodos/{periodos}
      /variaveis/{variaveis}?localidades=N6[all]
"""

import logging
import sys
import time
from pathlib import Path
from unicodedata import normalize

import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ibge_sidra"
LOG_DIR  = ROOT / "logs"
CRESOL_CSV = ROOT / "data" / "raw" / "ifdata" / "cresol_resumo_trimestral.csv"

# ── SIDRA API ──────────────────────────────────────────────────────────────────
SIDRA_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
TABELA     = 5938

# ── Tabela de população ────────────────────────────────────────────────────────
# Tabela 6579: Estimativas da população residente (IBGE, publicação anual)
# Variável 9324: Pessoas residentes estimadas (unidades)
TABELA_POP = 6579
VAR_POP    = 9324
ANO_POP    = 2021   # último ano publicado com cobertura completa

# Variáveis da tabela 5938
VARIAVEIS = {
    37:  "pib_corrente_rs_mil",
    498: "pib_per_capita_rs",
    543: "vab_agro_rs_mil",
    544: "vab_industria_rs_mil",
    545: "vab_servicos_rs_mil",
}

# Anos disponíveis (tabela 5938 publica com ~2 anos de lag; ajuste se necessário)
ANOS = list(range(2018, 2024))   # 2018–2023

REQUEST_DELAY = 1.5
MAX_RETRIES   = 4


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_sidra.log", encoding="utf-8"),
        ],
    )


# ── Normalização de nomes ──────────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Normaliza nome de município: remove acentos, upper, strip."""
    s = str(s).strip().upper()
    return normalize("NFD", s).encode("ascii", "ignore").decode()


# ── Download SIDRA ─────────────────────────────────────────────────────────────
def _sidra_get(url_completa: str) -> list:
    """
    Baixa uma URL do SIDRA sem encoding de caracteres especiais (|, [, ]).
    Retorna o JSON como lista, ou lança exceção após MAX_RETRIES.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url_completa, timeout=120)
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            logging.warning(f"    Tentativa {attempt} falhou. Aguardando {wait}s…")
            time.sleep(wait)
    return []


def baixar_tabela_sidra(anos: list[int], variaveis: list[int]) -> pd.DataFrame:
    """
    Baixa tabela 5938 para todos os municípios (N6[all]), uma variável por vez
    para evitar timeout/500 do servidor SIDRA.

    A URL é construída manualmente para preservar | e [all] sem encoding.
    """
    periodos_str = "|".join(str(a) for a in anos)
    frames = []

    for var in variaveis:
        url = (
            f"{SIDRA_BASE}/{TABELA}/periodos/{periodos_str}/variaveis/{var}"
            f"?localidades=N6[all]"
        )
        logging.info(f"  Variável {var} — {VARIAVEIS.get(var, var)}…")
        try:
            data = _sidra_get(url)
            df_var = _parsear_sidra_json(data, {var: VARIAVEIS[var]})
            if not df_var.empty:
                frames.append(df_var)
                logging.info(f"    {df_var['codigo_ibge_7d'].nunique()} municípios")
        except Exception as exc:
            logging.error(f"    Variável {var} falhou: {exc}")

    if not frames:
        return pd.DataFrame()

    # Juntar todas as variáveis por (código, nome, uf, ano)
    df = frames[0]
    for f in frames[1:]:
        merge_keys = ["codigo_ibge_7d", "nome_ibge", "uf_ibge", "ano"]
        df = df.merge(f, on=merge_keys, how="outer")
    return df


def _parsear_sidra_json(data: list, variaveis_map: dict) -> pd.DataFrame:
    """
    Converte a resposta JSON da SIDRA para DataFrame long com colunas:
      codigo_ibge_7d, nome_ibge, uf, ano, <variável_1>, <variável_2>, ...
    """
    # Construir dict inverso: id_variavel → nome_coluna
    var_nome = {str(v): nome for v, nome in variaveis_map.items()}

    registros = []
    for bloco in data:
        var_id = str(bloco.get("id", ""))
        col_nome = var_nome.get(var_id)
        if not col_nome:
            continue

        for resultado in bloco.get("resultados", []):
            for serie_info in resultado.get("series", []):
                loc = serie_info.get("localidade", {})
                cod_ibge = str(loc.get("id", "")).zfill(7)
                nome_full = loc.get("nome", "")  # ex: "Alta Floresta D'Oeste - RO"

                # Extrair nome e UF do padrão "Nome - UF"
                if " - " in nome_full:
                    partes = nome_full.rsplit(" - ", 1)
                    nome_mun = partes[0].strip()
                    uf_ibge  = partes[1].strip()
                else:
                    nome_mun = nome_full
                    uf_ibge  = ""

                serie = serie_info.get("serie", {})
                for ano_str, valor_str in serie.items():
                    try:
                        ano = int(ano_str)
                        # SIDRA usa "-" ou ".." para dado não disponível
                        if str(valor_str).strip() in ("-", "..", ""):
                            valor = None
                        else:
                            valor = float(str(valor_str).replace(",", "."))
                    except (ValueError, TypeError):
                        valor = None

                    registros.append({
                        "codigo_ibge_7d": cod_ibge,
                        "nome_ibge":      nome_mun,
                        "uf_ibge":        uf_ibge,
                        "ano":            ano,
                        "variavel":       col_nome,
                        "valor":          valor,
                    })

    if not registros:
        return pd.DataFrame()

    df_long = pd.DataFrame(registros)

    # Pivotar para wide: uma linha por (municipio × ano)
    df_wide = df_long.pivot_table(
        index=["codigo_ibge_7d", "nome_ibge", "uf_ibge", "ano"],
        columns="variavel",
        values="valor",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None
    return df_wide


# ── Filtro por municípios Cresol ───────────────────────────────────────────────
def carregar_municipios_cresol() -> pd.DataFrame:
    """
    Lê o dataset de saída do script 01 e extrai pares únicos (municipio, uf).
    Cria chave normalizada para matching.
    """
    if not CRESOL_CSV.exists():
        logging.warning(
            f"Arquivo Cresol não encontrado: {CRESOL_CSV}\n"
            "Execute o script 01 primeiro. Usando todos os municípios do SIDRA."
        )
        return pd.DataFrame()

    df = pd.read_csv(CRESOL_CSV, encoding="utf-8-sig")
    muns = df[["municipio", "uf"]].drop_duplicates().copy()
    muns["_chave"] = muns["municipio"].map(_norm) + "_" + muns["uf"].str.strip().str.upper()
    logging.info(f"Municípios Cresol distintos: {len(muns)}")
    return muns


def _match_municipios(df_sidra: pd.DataFrame, muns_cresol: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra df_sidra pelos municípios Cresol via matching normalizado nome+UF.
    """
    if muns_cresol.empty:
        return df_sidra  # retorna tudo se não há lista Cresol

    chaves_cresol = set(muns_cresol["_chave"])
    df_sidra = df_sidra.copy()
    df_sidra["_chave"] = df_sidra["nome_ibge"].map(_norm) + "_" + df_sidra["uf_ibge"].str.upper()
    df_filtrado = df_sidra[df_sidra["_chave"].isin(chaves_cresol)].drop(columns=["_chave"])

    municipios_encontrados = df_filtrado["nome_ibge"].nunique()
    municipios_esperados   = len(muns_cresol)
    logging.info(
        f"Match municípios: {municipios_encontrados}/{municipios_esperados} "
        f"encontrados no SIDRA"
    )

    # Alertar sobre municípios não encontrados
    df_sidra_temp = df_sidra.copy()
    chaves_encontradas = set(df_sidra_temp[df_sidra_temp["_chave"].isin(chaves_cresol)]["_chave"])
    nao_encontrados = [
        row["_chave"]
        for _, row in muns_cresol.iterrows()
        if row["_chave"] not in chaves_encontradas
    ]
    if nao_encontrados:
        logging.warning(f"Municípios Cresol NÃO encontrados no SIDRA: {nao_encontrados}")

    return df_filtrado


# ── Propagação anual → trimestral (vetorizada) ────────────────────────────────
def propagar_para_trimestres(df_anual: pd.DataFrame) -> pd.DataFrame:
    """
    Expande dados anuais para trimestral com forward-fill vetorizado.

    Para cada município, usa o dado do último ano disponível ≤ ano-alvo.
    Abordagem: cross-join município × quarter, depois merge_asof por ano.
    """
    var_cols = [c for c in df_anual.columns if c not in
                ("codigo_ibge_7d", "nome_ibge", "uf_ibge", "ano")]

    # Forward-fill nulos dentro de cada município (ex: 2022 pode ter PIB total
    # mas não per capita/VAB — usa o último ano disponível para cada coluna)
    df_anual = df_anual.sort_values(["codigo_ibge_7d", "ano"]).copy()
    for col in var_cols:
        df_anual[col] = df_anual.groupby("codigo_ibge_7d")[col].ffill()

    # Grade de trimestres-alvo 2020–2024
    grades = pd.DataFrame(
        [(ano, tri) for ano in range(2020, 2025) for tri in range(1, 5)],
        columns=["ano", "trimestre"],
    )
    grades["periodo_str"] = grades["ano"].astype(str) + "Q" + grades["trimestre"].astype(str)

    # Municípios únicos
    muns = df_anual[["codigo_ibge_7d", "nome_ibge", "uf_ibge"]].drop_duplicates().copy()
    muns["_k"] = 1
    grades["_k"] = 1
    painel = muns.merge(grades, on="_k").drop(columns="_k")  # cross join

    # Para cada (municipio, ano_alvo): encontrar o maior ano_disponivel <= ano_alvo
    df_anual_sorted = df_anual.sort_values(["codigo_ibge_7d", "ano"])

    # merge: cada linha do painel pode casar com vários anos do histórico
    merged = painel.merge(
        df_anual_sorted[["codigo_ibge_7d", "ano"] + var_cols].rename(columns={"ano": "ano_ref_pib"}),
        on="codigo_ibge_7d",
        how="left",
    )
    # Manter apenas anos_ref <= ano_alvo
    merged = merged[merged["ano_ref_pib"] <= merged["ano"]]
    # Para cada (municipio, ano, trimestre): manter apenas o maior ano_ref_pib
    idx = merged.groupby(["codigo_ibge_7d", "ano", "trimestre"])["ano_ref_pib"].idxmax()
    df_trim = merged.loc[idx].reset_index(drop=True)
    df_trim = df_trim.sort_values(["codigo_ibge_7d", "ano", "trimestre"]).reset_index(drop=True)
    return df_trim


# ── Download de população ─────────────────────────────────────────────────────
def baixar_populacao_municipios(muns_cresol: pd.DataFrame) -> pd.DataFrame:
    """
    Baixa estimativas populacionais (tabela 6579, variável 9324, ano 2021)
    do IBGE SIDRA para os municípios Cresol e salva em cache.

    Retorna DataFrame com colunas:
      codigo_ibge_7d, nome_ibge, uf_ibge, ano_pop, populacao_residente
    """
    cache = DATA_DIR / "populacao_municipios_cresol.parquet"
    if cache.exists():
        logging.info(f"  População: usando cache ({cache.name})")
        return pd.read_parquet(cache, dtype={"codigo_ibge_7d": str})

    # Baixa para todos os municípios (N6[all]) — ~5570 registros, ~2 s
    url = (
        f"{SIDRA_BASE}/{TABELA_POP}/periodos/{ANO_POP}"
        f"/variaveis/{VAR_POP}?localidades=N6[all]"
    )
    logging.info(f"  Baixando tabela {TABELA_POP} var {VAR_POP} ({ANO_POP})…")
    data = _sidra_get(url)

    df_all = _parsear_sidra_json(data, {VAR_POP: "populacao_residente"})
    if df_all.empty:
        logging.warning("  Tabela de população retornou vazia")
        return pd.DataFrame()

    # Filtra municípios Cresol
    df_pop = _match_municipios(df_all, muns_cresol)
    if df_pop.empty:
        logging.warning("  Nenhum município Cresol encontrado na tabela de população")
        return pd.DataFrame()

    # Mantém apenas o ano mais recente por município (por garantia)
    df_pop = (df_pop.sort_values("ano", ascending=False)
              .drop_duplicates("codigo_ibge_7d")
              .rename(columns={"ano": "ano_pop"})
              .reset_index(drop=True))

    df_pop.to_parquet(cache, index=False)
    logging.info(f"  População salva: {len(df_pop)} municípios → {cache.name}")
    return df_pop


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download IBGE SIDRA – PIB Municipal (Tabela 5938) ===")

    # 1. Baixar tabela completa (ou usar cache se já existir)
    raw_csv = DATA_DIR / "pib_municipal_brasil_raw.csv"
    if raw_csv.exists():
        logging.info(f"Usando cache: {raw_csv}")
        df_wide = pd.read_csv(raw_csv, encoding="utf-8-sig", dtype={"codigo_ibge_7d": str})
    else:
        df_wide = baixar_tabela_sidra(ANOS, list(VARIAVEIS.keys()))
        if df_wide.empty:
            logging.error("Nenhum dado retornado pelo SIDRA.")
            sys.exit(1)
        df_wide.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        logging.info(f"Dados brutos salvos: {raw_csv}")

    logging.info(f"Municípios no SIDRA: {df_wide['codigo_ibge_7d'].nunique()}")

    # 2. Filtrar municípios Cresol
    muns_cresol = carregar_municipios_cresol()

    if muns_cresol.empty:
        logging.warning(
            "CSV Cresol não encontrado — execute script 01 primeiro.\n"
            "Dados brutos SIDRA já salvos. Reexecute este script após o 01."
        )
        sys.exit(0)   # saída limpa, dados brutos já estão salvos

    df_cresol = _match_municipios(df_wide, muns_cresol)

    if df_cresol.empty:
        logging.error("Nenhum município Cresol encontrado no SIDRA.")
        sys.exit(1)

    # 3. Propagar para trimestres (vetorizado)
    logging.info(f"Gerando painel trimestral para {df_cresol['codigo_ibge_7d'].nunique()} municípios…")
    df_trim = propagar_para_trimestres(df_cresol)

    if df_trim.empty:
        logging.error("Propagação trimestral gerou DataFrame vazio.")
        sys.exit(1)

    # 4. Salvar
    out_csv     = DATA_DIR / "pib_municipal_cresol.csv"
    out_parquet = DATA_DIR / "pib_municipal_cresol.parquet"
    df_trim.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_trim.to_parquet(out_parquet, index=False)

    logging.info("─" * 60)
    logging.info(f"Municípios Cresol com PIB: {df_trim['codigo_ibge_7d'].nunique()}")
    logging.info(f"Trimestres: {df_trim['periodo_str'].min()} a {df_trim['periodo_str'].max()}")
    logging.info(f"Linhas totais: {len(df_trim)}")
    logging.info("\n" + df_trim[["periodo_str","nome_ibge","uf_ibge","pib_corrente_rs_mil","vab_agro_rs_mil"]].head(8).to_string(index=False))
    logging.info(f"CSV     → {out_csv}")
    logging.info(f"Parquet → {out_parquet}")

    # 5. Baixar estimativas populacionais (tabela 6579) — necessário para porte_municipio
    logging.info("\nBaixando estimativas populacionais (tabela 6579)…")
    df_pop = baixar_populacao_municipios(muns_cresol)
    if not df_pop.empty:
        logging.info(
            f"  Municípios com população: {len(df_pop)}"
            f" | min={df_pop['populacao_residente'].min():,.0f}"
            f" | max={df_pop['populacao_residente'].max():,.0f}"
            f" | mediana={df_pop['populacao_residente'].median():,.0f}"
        )
    logging.info("Concluído.")


if __name__ == "__main__":
    main()
