#!/usr/bin/env python3
"""
07_feature_engineering.py
=========================
Constrói o painel analítico (cooperativa × trimestre) integrando todas as
fontes de dados e derivando features para os modelos preditivos.

Fontes de entrada:
  data/raw/ifdata/cresol_resumo_trimestral.csv        — dados financeiros Cresol
  data/raw/bcb_sgs/indicadores_macro_trimestral.csv   — macro (Selic, IPCA, …)
  data/raw/ibge_sidra/pib_municipal_cresol.csv        — PIB municipal
  data/raw/credito_rural/credito_rural_agregado_sgs.csv — crédito rural nacional
  data/raw/commodities/precos_commodities_trimestral.csv — preços agro
  data/raw/n_associados/template_preenchimento_manual.csv — n_associados (opcional)

Saídas:
  data/processed/panel_features.csv / .parquet   — painel completo pronto p/ modelo
  data/processed/feature_metadata.csv            — dicionário de features (grupo, lag, descrição)

Targets:
  vol_credito_rs_mil    — saldo de Operações de Crédito (R$ mil)
  captacao_rs_mil       — Captações totais (R$ mil)
  n_associados          — número de cooperados (se template preenchido)

Metodologia:
  - Features derivadas sempre com dados de t-1 em relação ao target (sem data leakage)
  - Lag máximo = 4 trimestres → 2020Q1-2020Q4 viram features para 2021Q1-2024Q4
  - Dados anuais (PIB) usados via forward-fill já aplicado no script 03
"""

import json
import logging
import sys
import time
from pathlib import Path
from unicodedata import normalize

import numpy as np
import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW      = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "processed"
LOG_DIR  = ROOT / "logs"

# Arquivos de entrada
IFDATA_CSV   = RAW / "ifdata"  / "cresol_resumo_trimestral.csv"
MACRO_CSV    = RAW / "bcb_sgs" / "indicadores_macro_trimestral.csv"
PIB_CSV      = RAW / "ibge_sidra" / "pib_municipal_cresol.csv"
CR_CSV       = RAW / "credito_rural" / "credito_rural_agregado_sgs.csv"
COMOD_CSV    = RAW / "commodities"   / "precos_commodities_trimestral.csv"
NASSOC_CSV   = RAW / "n_associados"  / "template_preenchimento_manual.csv"
POP_PARQUET  = RAW / "ibge_sidra"   / "populacao_municipios_cresol.parquet"

# Lags a calcular para variáveis financeiras internas
LAGS = [1, 2, 4]

# Variáveis financeiras da IF.data (a partir das quais criar lags e taxas)
FINS = ["vol_credito_rs_mil", "captacao_rs_mil", "ativo_total_rs_mil",
        "patrimonio_liq_rs_mil", "carteira_credito_rs_mil"]

# Variáveis macro (join por periodo_str)
MACRO_COLS = ["selic_aa", "ipca_acum_trim", "cambio_brl_usd",
              "ibc_br", "inpc_acum_trim", "ipa_agro", "concessoes_cred"]

# Variáveis PIB (join por municipio + periodo_str)
PIB_COLS = ["pib_corrente_rs_mil", "pib_per_capita_rs",
            "vab_agro_rs_mil", "vab_industria_rs_mil", "vab_servicos_rs_mil"]

# Variáveis crédito rural e commodities
CR_COLS    = ["cr_total_rs_mi", "cr_custeio_rs_mi",
              "cr_investimento_rs_mi", "cr_comercializacao_rs_mi"]
COMOD_COLS = ["ipa_agro_idx", "milho_rs_60kg", "boi_gordo_rs_arroba", "leite_rs_litro"]

# UFs da região Sul — ciclo agrícola safra verão/inverno mais pronunciado
SUL = {"PR", "RS", "SC"}

# Nomes das dummies de safra criadas em adicionar_dummies_safra()
SAFRA_COLS = [
    "dummy_plantio_verao",      # Q4 out–dez: plantio soja/milho (pico crédito)
    "dummy_colheita_verao",     # Q1 jan–mar: colheita soja/milho (liquidações)
    "dummy_plantio_inv",        # Q2 abr–jun: plantio trigo/aveia/cevada
    "dummy_colheita_inv",       # Q3 jul–set: colheita inverno
    "dummy_sul",                # cooperativa em PR/SC/RS
    "dummy_plantio_verao_sul",  # interação: plantio verão × Sul
    "dummy_colheita_verao_sul", # interação: colheita verão × Sul
]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "feature_engineering.log", encoding="utf-8"),
        ],
    )


# ── Utilitários ────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = str(s).strip().upper()
    return normalize("NFD", s).encode("ascii", "ignore").decode()


def _periodo_sort_key(df: pd.DataFrame) -> pd.Series:
    """Converte 'ANOQtri' em inteiro para ordenação correta."""
    return df["periodo_str"].str.replace("Q", "").astype(int)


def _pct_change(a: pd.Series, b: pd.Series) -> pd.Series:
    """Variação percentual segura: (a - b) / |b|, NaN se b = 0."""
    return np.where(b.abs() > 0, (a - b) / b.abs() * 100, np.nan)


# ── 1. Carregar bases ──────────────────────────────────────────────────────────
def carregar_ifdata() -> pd.DataFrame:
    df = pd.read_csv(IFDATA_CSV, encoding="utf-8-sig")
    df["data_base"] = pd.to_datetime(df["data_base"])
    # Ordenar para garantir lag correto dentro de cada cooperativa
    df = df.sort_values(["cnpj8", "ano", "trimestre"]).reset_index(drop=True)
    logging.info(f"IF.data: {len(df)} linhas, {df['cnpj8'].nunique()} cooperativas")
    return df


def carregar_macro() -> pd.DataFrame:
    df = pd.read_csv(MACRO_CSV, encoding="utf-8-sig")
    # Coluna periodo_str já existe; renomear periodo_Q se vier do Parquet
    if "periodo_Q" in df.columns and "periodo_str" not in df.columns:
        df["periodo_str"] = df["periodo_Q"].astype(str)
    cols = ["periodo_str"] + [c for c in MACRO_COLS if c in df.columns]
    df = df[cols].drop_duplicates("periodo_str")
    logging.info(f"Macro SGS: {len(df)} trimestres, colunas: {list(df.columns)}")
    return df


def carregar_pib() -> pd.DataFrame:
    df = pd.read_csv(PIB_CSV, encoding="utf-8-sig", dtype={"codigo_ibge_7d": str})
    # Mantém nome_ibge e uf_ibge para o lookup codigo_ibge_7d em adicionar_pib()
    cols = ["nome_ibge", "uf_ibge", "codigo_ibge_7d", "periodo_str", "ano_ref_pib"] + \
           [c for c in PIB_COLS if c in df.columns]
    df = df[cols].drop_duplicates(["codigo_ibge_7d", "periodo_str"])
    logging.info(f"PIB SIDRA: {len(df)} linhas, {df['codigo_ibge_7d'].nunique()} municípios")
    return df


def carregar_credito_rural() -> pd.DataFrame:
    df = pd.read_csv(CR_CSV, encoding="utf-8-sig")
    if "periodo_Q" in df.columns and "periodo_str" not in df.columns:
        df["periodo_str"] = df["periodo_Q"].astype(str)
    cols = ["periodo_str"] + [c for c in CR_COLS if c in df.columns]
    df = df[cols].drop_duplicates("periodo_str")
    logging.info(f"Crédito Rural: {len(df)} trimestres")
    return df


def carregar_commodities() -> pd.DataFrame:
    df = pd.read_csv(COMOD_CSV, encoding="utf-8-sig")
    if "periodo_Q" in df.columns and "periodo_str" not in df.columns:
        df["periodo_str"] = df["periodo_Q"].astype(str)
    cols = ["periodo_str"] + [c for c in COMOD_COLS if c in df.columns]
    df = df[cols].drop_duplicates("periodo_str")
    logging.info(f"Commodities: {len(df)} trimestres")
    return df


def carregar_n_associados() -> pd.DataFrame | None:
    """
    Carrega n_associados do template manual SE alguma célula foi preenchida.
    Retorna None se o template estiver em branco.
    """
    if not NASSOC_CSV.exists():
        return None
    df = pd.read_csv(NASSOC_CSV, encoding="utf-8-sig", dtype={"cnpj8": str})
    df["n_associados"] = pd.to_numeric(df["n_associados"], errors="coerce")
    preenchidos = df["n_associados"].notna().sum()
    if preenchidos == 0:
        logging.info("Template n_associados em branco — coluna omitida do painel")
        return None
    logging.info(f"n_associados: {preenchidos} células preenchidas de {len(df)}")
    return df[["cnpj8", "periodo_str", "n_associados"]].dropna(subset=["n_associados"])


# ── 2. Features financeiras (lags e crescimento) ───────────────────────────────
def criar_features_financeiras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada cooperativa, calcula:
      L{n}_{variavel} : valor defasado em n trimestres
      {variavel}_qoq  : variação percentual trimestral (t vs t-1)
      {variavel}_yoy  : variação percentual anual (t vs t-4)
      ratios          : participações relativas ao ativo total
    """
    df = df.copy()

    # Garantir ordem correta dentro de cada cooperativa
    df = df.sort_values(["cnpj8", "ano", "trimestre"]).reset_index(drop=True)

    for var in FINS:
        if var not in df.columns:
            continue
        for lag in LAGS:
            df[f"L{lag}_{var}"] = df.groupby("cnpj8")[var].shift(lag)

        # Taxas de crescimento
        L1 = df.groupby("cnpj8")[var].shift(1)
        L4 = df.groupby("cnpj8")[var].shift(4)
        df[f"{var}_qoq"] = _pct_change(df[var], L1)
        df[f"{var}_yoy"] = _pct_change(df[var], L4)

    # Ratios financeiros (calculados com valores correntes)
    ativo = df["ativo_total_rs_mil"].replace(0, np.nan)
    if "vol_credito_rs_mil" in df.columns:
        df["credito_sobre_ativo"]   = df["vol_credito_rs_mil"]   / ativo
    if "captacao_rs_mil" in df.columns:
        df["captacao_sobre_ativo"]  = df["captacao_rs_mil"]      / ativo
    if "patrimonio_liq_rs_mil" in df.columns:
        df["pl_sobre_ativo"]        = df["patrimonio_liq_rs_mil"] / ativo

    # Indicador de segmento (encoding simples)
    if "segmento" in df.columns:
        df["segmento_num"] = df["segmento"].astype("category").cat.codes

    return df


# ── 3. Features PIB (join por município) ──────────────────────────────────────
def adicionar_pib(df_base: pd.DataFrame, df_pib: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IF.data × IBGE por codigo_ibge_7d + periodo_str.

    O código IBGE é obtido via lookup (nome normalizado → codigo_ibge_7d) a partir
    do próprio arquivo SIDRA, eliminando dependência de matching textual no join.
    Gera dataset unificado por cooperativa (cnpj8) e período (periodo_str).
    """
    df_base = df_base.copy()

    # 1. Lookup: chave normalizada → codigo_ibge_7d (a partir do SIDRA)
    uniq = (df_pib[["nome_ibge", "uf_ibge", "codigo_ibge_7d"]]
            .drop_duplicates()
            .assign(_chave=lambda d: d["nome_ibge"].map(_norm) + "_" +
                                     d["uf_ibge"].str.strip().str.upper()))
    lookup_code = uniq.set_index("_chave")["codigo_ibge_7d"].to_dict()

    # 2. Adicionar codigo_ibge_7d ao painel IF.data via lookup
    df_base["_chave"] = (df_base["municipio"].map(_norm) + "_" +
                         df_base["uf"].str.strip().str.upper())
    df_base["codigo_ibge_7d"] = df_base["_chave"].map(lookup_code)

    sem_cod = df_base["codigo_ibge_7d"].isna().sum()
    if sem_cod:
        logging.warning(f"  {sem_cod} linhas sem codigo_ibge_7d — verifique nomes de município")

    # 3. Join por codigo_ibge_7d + periodo_str (chave estável, sem matching textual)
    pib_join_cols = (["codigo_ibge_7d", "periodo_str", "ano_ref_pib"] +
                     [c for c in PIB_COLS if c in df_pib.columns])
    df_pib_j = df_pib[pib_join_cols].drop_duplicates(["codigo_ibge_7d", "periodo_str"])

    merged = df_base.merge(df_pib_j, on=["codigo_ibge_7d", "periodo_str"], how="left")

    # 4. Share agropecuária do município
    pib = merged["pib_corrente_rs_mil"].replace(0, np.nan)
    if "vab_agro_rs_mil" in merged.columns:
        merged["share_agro_mun"] = merged["vab_agro_rs_mil"] / pib

    n_sem_pib = merged["pib_corrente_rs_mil"].isna().sum()
    if n_sem_pib:
        logging.warning(f"  PIB não encontrado: {n_sem_pib} linhas ({n_sem_pib/len(merged)*100:.1f}%)")
    else:
        logging.info(f"  Join IF.data × IBGE: {len(merged)} linhas, 0 nulos — chave: codigo_ibge_7d")

    return merged.drop(columns=["_chave"])


# ── 4. Lags para variáveis externas ───────────────────────────────────────────
def lag_externas(df: pd.DataFrame, cols: list[str], lags: list[int] = (1,)) -> pd.DataFrame:
    """
    Cria lags de variáveis externas (macro, crédito rural, commodities).
    Como essas séries são nacionais (não por cooperativa), o lag é simplesmente
    um shift no calendário — mapeia periodo_str → periodo_str deslocado.
    """
    # Constrói um mapeamento periodo_str → valor para cada coluna
    for col in cols:
        if col not in df.columns:
            continue
        # Série temporal única por periodo_str (já é a mesma para todas as coops)
        serie = df.drop_duplicates("periodo_str").set_index("periodo_str")[col].sort_index()
        serie_shift = serie.shift(lag := 1)  # lag=1 para variáveis externas
        df[f"L1_{col}"] = df["periodo_str"].map(serie_shift)
    return df


# ── 5. Dummies de safra agrícola ─────────────────────────────────────────────
def adicionar_dummies_safra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria indicadores binários do calendário agrícola do Sul brasileiro.

    Ciclo safra verão (soja/milho): plantio Q4 (out–dez) → colheita Q1 (jan–mar)
    Ciclo safra inverno (trigo):    plantio Q2 (abr–jun) → colheita Q3 (jul–set)

    As interações *_sul capturam o efeito específico das cooperativas de PR/SC/RS,
    onde o calendário da safra tem maior influência nas metas financeiras.
    A dummy do trimestre atual não gera data leakage: o calendário é conhecido
    no momento da previsão.
    """
    df = df.copy()
    tri = df["trimestre"]
    sul = df["uf"].isin(SUL).astype(int)

    df["dummy_plantio_verao"]  = (tri == 4).astype(int)
    df["dummy_colheita_verao"] = (tri == 1).astype(int)
    df["dummy_plantio_inv"]    = (tri == 2).astype(int)
    df["dummy_colheita_inv"]   = (tri == 3).astype(int)
    df["dummy_sul"]            = sul

    # Interações: sazonalidade × região Sul
    df["dummy_plantio_verao_sul"]  = df["dummy_plantio_verao"]  * sul
    df["dummy_colheita_verao_sul"] = df["dummy_colheita_verao"] * sul

    logging.info(f"  Dummies safra criadas: {SAFRA_COLS}")
    return df


# ── 6. Porte do município por população IBGE ──────────────────────────────────
def _baixar_populacao_ibge(codigos_ibge: list[str]) -> pd.DataFrame:
    """
    Baixa estimativas populacionais do IBGE SIDRA (tabela 6579, variável 9324,
    ano 2021) filtrando pelos municípios Cresol.

    Usa cache em POP_PARQUET; faz o download apenas na primeira execução.
    Retorna DataFrame com (codigo_ibge_7d, populacao_residente).
    """
    if POP_PARQUET.exists():
        df = pd.read_parquet(POP_PARQUET)
        logging.info(f"  Pop. IBGE: cache carregado ({len(df)} municípios)")
        df["codigo_ibge_7d"] = df["codigo_ibge_7d"].astype(str).str.zfill(7)
        return df[["codigo_ibge_7d", "populacao_residente"]].drop_duplicates("codigo_ibge_7d")

    # Primeira execução — baixa via SIDRA filtrando pelos códigos conhecidos
    SIDRA_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
    codigos_str = "|".join(sorted(set(codigos_ibge)))
    url = (
        f"{SIDRA_BASE}/6579/periodos/2021/variaveis/9324"
        f"?localidades=N6[{codigos_str}]"
    )
    logging.info(f"  Baixando estimativas populacionais (IBGE SIDRA tab 6579)…")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        time.sleep(1.0)
        data = r.json()
    except Exception as exc:
        logging.warning(f"  Download de população falhou: {exc}")
        return pd.DataFrame(columns=["codigo_ibge_7d", "populacao_residente"])

    registros = []
    for bloco in data:
        if str(bloco.get("id")) != "9324":
            continue
        for res in bloco.get("resultados", []):
            for serie_info in res.get("series", []):
                loc = serie_info["localidade"]
                cod = str(loc.get("id", "")).zfill(7)
                for ano_str, val_str in serie_info.get("serie", {}).items():
                    try:
                        pop = float(str(val_str).replace(",", "."))
                        if pop > 0:
                            registros.append({"codigo_ibge_7d": cod,
                                              "populacao_residente": pop})
                    except (ValueError, TypeError):
                        pass

    if not registros:
        logging.warning("  Resposta SIDRA vazia para tabela de população")
        return pd.DataFrame(columns=["codigo_ibge_7d", "populacao_residente"])

    df_pop = (pd.DataFrame(registros)
              .sort_values("populacao_residente", ascending=False)
              .drop_duplicates("codigo_ibge_7d")
              .reset_index(drop=True))

    df_pop.to_parquet(POP_PARQUET, index=False)
    logging.info(f"  Pop. IBGE: {len(df_pop)} municípios salvos em cache")
    return df_pop[["codigo_ibge_7d", "populacao_residente"]]


def adicionar_porte_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável categórica ``porte_municipio`` e sua codificação ordinal
    ``porte_num``, a partir da população residente estimada pelo IBGE (2021).

    Faixas (baseadas na população residente):
      'pequeno'       — menos de 40.000 habitantes
      'intermediario' — de 40.000 (inclusive) a menos de 100.000 habitantes
      'grande'        — 100.000 habitantes ou mais

    Codificação ordinal para uso direto nos modelos:
      pequeno = 0 | intermediario = 1 | grande = 2

    Chamada logo após o join IBGE (codigo_ibge_7d já disponível no painel).
    """
    codigos = df["codigo_ibge_7d"].dropna().unique().tolist()
    df_pop  = _baixar_populacao_ibge(codigos)

    if df_pop.empty:
        logging.warning("  porte_municipio não criado: dados de população indisponíveis")
        return df

    df = df.merge(df_pop, on="codigo_ibge_7d", how="left")

    def _classificar(pop: float) -> str:
        if pd.isna(pop):
            return "desconhecido"
        if pop < 40_000:
            return "pequeno"
        if pop < 100_000:
            return "intermediario"
        return "grande"

    df["porte_municipio"] = df["populacao_residente"].map(_classificar).astype("category")

    PORTE_ORD = {"pequeno": 0, "intermediario": 1, "grande": 2, "desconhecido": -1}
    df["porte_num"] = df["porte_municipio"].map(PORTE_ORD).astype("Int8")

    # Remove coluna auxiliar de população (mantém apenas porte)
    df = df.drop(columns=["populacao_residente"])

    n_desc = (df["porte_municipio"] == "desconhecido").sum()
    if n_desc:
        logging.warning(f"  {n_desc} obs sem classificação de porte (código IBGE sem match)")

    contagem = df["porte_municipio"].value_counts().to_dict()
    logging.info(f"  porte_municipio criado: {contagem}")
    return df


# ── 7. n_associados ───────────────────────────────────────────────────────────
def adicionar_n_associados(df: pd.DataFrame, df_nassoc: pd.DataFrame | None) -> pd.DataFrame:
    if df_nassoc is None:
        return df
    df = df.merge(df_nassoc, on=["cnpj8", "periodo_str"], how="left")
    # Lag do n_associados
    df = df.sort_values(["cnpj8", "ano", "trimestre"]).reset_index(drop=True)
    df["L1_n_associados"] = df.groupby("cnpj8")["n_associados"].shift(1)
    logging.info(f"n_associados adicionado: {df['n_associados'].notna().sum()} valores")
    return df


# ── 6. Metadados das features ─────────────────────────────────────────────────
def gerar_metadata(df_feat: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Gera dicionário das features: nome, grupo, lag, descrição."""
    grupos = {
        "id":     {"cnpj8","nome","uf","municipio","codigo_ibge_7d","periodo_str",
                   "periodo","ano","trimestre","data_base","segmento","ano_ref_pib"},
        "target": set(targets),
        "fin_raw": set(FINS),
        "fin_lag": {f"L{l}_{v}" for l in LAGS for v in FINS},
        "fin_growth": {f"{v}_{s}" for v in FINS for s in ("qoq","yoy")},
        "fin_ratio": {"credito_sobre_ativo","captacao_sobre_ativo","pl_sobre_ativo","segmento_num"},
        "macro":   set(MACRO_COLS) | {f"L1_{c}" for c in MACRO_COLS},
        "pib":     set(PIB_COLS) | {"share_agro_mun"},
        "cr_rural": set(CR_COLS) | {f"L1_{c}" for c in CR_COLS},
        "commodity": set(COMOD_COLS) | {f"L1_{c}" for c in COMOD_COLS},
        "safra":   set(SAFRA_COLS),
        "porte":   {"porte_municipio", "porte_num"},
        "n_assoc": {"n_associados","L1_n_associados"},
    }

    records = []
    for col in df_feat.columns:
        grupo = "outro"
        for g, cols in grupos.items():
            if col in cols:
                grupo = g
                break
        records.append({"feature": col, "grupo": grupo,
                        "dtype": str(df_feat[col].dtype),
                        "n_nulos": int(df_feat[col].isna().sum()),
                        "pct_nulos": round(df_feat[col].isna().mean() * 100, 1)})
    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Feature Engineering – Painel Cresol ===")

    # 1. Carregar dados
    df        = carregar_ifdata()
    df_macro  = carregar_macro()
    df_pib    = carregar_pib()
    df_cr     = carregar_credito_rural()
    df_comod  = carregar_commodities()
    df_nassoc = carregar_n_associados()

    # 2. Features financeiras (lags e ratios) dentro de cada cooperativa
    logging.info("Calculando features financeiras (lags, crescimento, ratios)…")
    df = criar_features_financeiras(df)

    # 3. Join macro (chave: periodo_str)
    logging.info("Adicionando features macro…")
    df = df.merge(df_macro, on="periodo_str", how="left")

    # 4. Join PIB municipal (chave: municipio+uf+periodo_str)
    logging.info("Adicionando PIB municipal…")
    df = adicionar_pib(df, df_pib)

    # 4.5. Porte do município por população IBGE (logo após join — codigo_ibge_7d disponível)
    logging.info("Adicionando porte_municipio…")
    df = adicionar_porte_municipio(df)

    # 5. Join crédito rural (chave: periodo_str)
    logging.info("Adicionando crédito rural…")
    df = df.merge(df_cr, on="periodo_str", how="left")

    # 6. Join commodities (chave: periodo_str)
    logging.info("Adicionando preços de commodities…")
    df = df.merge(df_comod, on="periodo_str", how="left")

    # 7. Lags de variáveis externas (L1 do trimestre anterior)
    logging.info("Calculando L1 de variáveis externas…")
    df = lag_externas(df, MACRO_COLS + CR_COLS + COMOD_COLS)

    # 8. Dummies de sazonalidade agrícola (safra verão/inverno × região Sul)
    logging.info("Adicionando dummies de safra…")
    df = adicionar_dummies_safra(df)

    # 9. n_associados (opcional)
    df = adicionar_n_associados(df, df_nassoc)

    # 9. Remover período 2020Q1 (sem lag L4 — todos NaN nas financeiras)
    #    e manter apenas registros com pelo menos L1 disponível
    n_antes = len(df)
    df_modelo = df[df["L1_vol_credito_rs_mil"].notna()].copy()
    logging.info(f"Filtro L1 disponível: {n_antes} → {len(df_modelo)} linhas")

    # 10. Ordenar e salvar
    df_modelo = df_modelo.sort_values(["cnpj8", "ano", "trimestre"]).reset_index(drop=True)

    out_csv     = OUT_DIR / "panel_features.csv"
    out_parquet = OUT_DIR / "panel_features.parquet"
    df_modelo.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_modelo.to_parquet(out_parquet, index=False)

    # 11. Metadados
    targets = ["vol_credito_rs_mil", "captacao_rs_mil", "carteira_credito_rs_mil"]
    if "n_associados" in df_modelo.columns:
        targets.append("n_associados")
    df_meta = gerar_metadata(df_modelo, targets)
    df_meta.to_csv(OUT_DIR / "feature_metadata.csv", index=False, encoding="utf-8-sig")

    # 12. Relatório
    logging.info("=" * 60)
    logging.info(f"Painel final: {df_modelo.shape[0]} linhas x {df_modelo.shape[1]} colunas")
    logging.info(f"Cooperativas: {df_modelo['cnpj8'].nunique()}")
    logging.info(f"Períodos:     {df_modelo['periodo_str'].min()} a {df_modelo['periodo_str'].max()}")
    logging.info(f"Targets:")
    for t in targets:
        if t in df_modelo.columns:
            pct_null = df_modelo[t].isna().mean() * 100
            logging.info(f"  {t}: {df_modelo[t].notna().sum()} obs, {pct_null:.1f}% nulos")

    logging.info("\nFeatures por grupo:")
    for grp, cnt in df_meta[df_meta["grupo"] != "id"].groupby("grupo")["feature"].count().items():
        logging.info(f"  {grp:12}: {cnt} features")

    logging.info(f"\nTop 10 features com mais nulos:")
    top_nulos = df_meta[df_meta["grupo"] != "id"].nlargest(10, "pct_nulos")[["feature","pct_nulos"]]
    logging.info("\n" + top_nulos.to_string(index=False))

    logging.info(f"\nCSV     -> {out_csv}")
    logging.info(f"Parquet -> {out_parquet}")
    logging.info(f"Metadata -> {OUT_DIR / 'feature_metadata.csv'}")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
