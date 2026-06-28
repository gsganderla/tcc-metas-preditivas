#!/usr/bin/env python3
"""
01_download_ifdata_cresol.py
============================
Baixa dados financeiros trimestrais das cooperativas singulares do Sistema
Cresol via portal IF.data do BACEN (REST/JSON interno).

Variáveis-alvo extraídas:
  vol_credito_rs        — Operações de Crédito (R$, coluna 78191)
  carteira_credito_rs   — Carteira de Crédito Classificada (R$, coluna 78183)
  captacao_rs           — Captações totais (R$, coluna 78185)
  ativo_total_rs        — Ativo Total (R$, coluna 78182)
  patrimonio_liq_rs     — Patrimônio Líquido (R$, coluna 78186)

Período: 2020T1 – 2024T4 (trimestral, 20 períodos)

NOTA sobre unidades:
  Os valores retornados pela API estão em R$ (reais). O portal exibe "em R$ mil".
  A conversão ÷ 1000 é feita no output final para consistência com o portal.

NOTA sobre número de cooperados (n_associados):
  Esse campo NÃO está disponível nos dados do IF.data (relatório Resumo, tipo 1006).
  Fontes alternativas: SGS BACEN série 12682 (agregado), ou relatórios OCB/Cresol.

Fonte: https://www3.bcb.gov.br/ifdata/

Estrutura da API interna:
  GET /ifdata/rest/arquivos?nomeArquivo=ifdata/{YYYYMM}/cadastro{YYYYMM}_1006.json
  GET /ifdata/rest/arquivos?nomeArquivo=ifdata/{YYYYMM}/dados{YYYYMM}_1.json
  (periodo 2025+: prefixo ifdata_2025_2030/ em vez de ifdata/)
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ifdata"
LOG_DIR  = ROOT / "logs"

# ── URLs e endereços ───────────────────────────────────────────────────────────
PORTAL_URL    = "https://www3.bcb.gov.br/ifdata/"
ARQUIVOS_URL  = "https://www3.bcb.gov.br/ifdata/rest/arquivos"
INDEX_URL     = "https://www3.bcb.gov.br/ifdata/rest/relatorios"

# Prefix para acessar arquivos (2020-2024 usa 'ifdata/', 2025+ usa 'ifdata_2025_2030/')
FILE_PREFIX_PRE2025  = "ifdata/"
FILE_PREFIX_POST2025 = "ifdata_2025_2030/"

# Tipo de instituição = 1006 (Instituições Individuais)
TIPO_INDIVIDUAL = 1006

# ── Mapeamento de colunas do Resumo (dados_1.json, tipo 1006) ─────────────────
# Confirmado via inspeção do arquivo info{YYYYMM}.json
COLUNAS_RESUMO = {
    78182: "ativo_total_rs",
    78183: "carteira_credito_rs",   # Carteira de Crédito Classificada
    78185: "captacao_rs",           # Captações totais
    78186: "patrimonio_liq_rs",
    78191: "vol_credito_rs",        # Operações de Crédito (d1) — antes de provisão
}

# ── Parâmetros de download ─────────────────────────────────────────────────────
REQUEST_DELAY = 3.0   # segundos entre chamadas
MAX_RETRIES   = 5
TIMEOUT       = 120   # segundos (arquivos grandes ~7-15 MB)
CACHE_ENABLED = True  # salvar cada período em CSV parcial para retomada

# Períodos 2020-2024 (YYYYMM, fins de trimestre)
PERIODOS = [
    202003, 202006, 202009, 202012,
    202103, 202106, 202109, 202112,
    202203, 202206, 202209, 202212,
    202303, 202306, 202309, 202312,
    202403, 202406, 202409, 202412,
]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "download_ifdata.log", encoding="utf-8"),
        ],
    )


# ── Sessão HTTP ────────────────────────────────────────────────────────────────
def criar_sessao() -> requests.Session:
    """
    Cria sessão autenticada no portal IF.data.
    O JSESSIONID obtido na página principal é necessário para acessar os arquivos.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
        "Referer": PORTAL_URL,
        "Accept": "application/json, */*",
    })
    # Obter cookie de sessão
    logging.info("Obtendo cookie de sessão do portal IF.data…")
    s.get(PORTAL_URL, timeout=30)
    return s


def _get_arquivo(
    s: requests.Session,
    nome_arquivo: str,
    retries: int = MAX_RETRIES,
) -> dict | list:
    """
    Baixa um arquivo JSON do portal IF.data via endpoint /rest/arquivos.

    Args:
        nome_arquivo: caminho relativo, ex. 'ifdata/202012/dados202012_1.json'
    """
    for attempt in range(1, retries + 1):
        try:
            r = s.get(
                ARQUIVOS_URL,
                params={"nomeArquivo": nome_arquivo},
                timeout=TIMEOUT,
            )
            # O portal retorna 200 com texto 'Erro interno' quando o arquivo não existe
            if r.content == b"Erro interno - Internal error":
                raise FileNotFoundError(f"Arquivo não encontrado: {nome_arquivo}")
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()

        except FileNotFoundError:
            raise
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logging.warning(f"Tentativa {attempt} falhou ({exc}). Aguardando {wait}s…")
            time.sleep(wait)
            # Renovar sessão em caso de erro de conexão
            try:
                s.get(PORTAL_URL, timeout=30)
            except Exception:
                pass

    raise RuntimeError(f"Todas as {retries} tentativas falharam para {nome_arquivo}")


def _prefixo_periodo(periodo: int) -> str:
    """Retorna o prefixo correto para o período."""
    return FILE_PREFIX_POST2025 if periodo >= 202503 else FILE_PREFIX_PRE2025


# ── Funções de download ────────────────────────────────────────────────────────
def baixar_cadastro(s: requests.Session, periodo: int) -> list[dict]:
    """
    Baixa o arquivo cadastro (tipo 1006 = Instituições Individuais) para um período.
    Filtra apenas cooperativas singulares (c3='b3S') com 'CRESOL' no nome.

    Returns:
        Lista de dicts com campos renomeados para legibilidade.
    """
    pref = _prefixo_periodo(periodo)
    nome = f"{pref}{periodo}/cadastro{periodo}_{TIPO_INDIVIDUAL}.json"
    logging.debug(f"Baixando cadastro {periodo}…")
    dados_brutos = _get_arquivo(s, nome)

    resultado = []
    for inst in dados_brutos:
        if inst.get("c3", "").lower() != "b3s":
            continue
        if "CRESOL" not in inst.get("c2", "").upper():
            continue
        resultado.append({
            "cnpj8":       str(inst["c0"]).zfill(8),
            "nome":        inst.get("c2", ""),
            "tcb":         inst.get("c3", ""),
            "uf":          inst.get("c10", ""),
            "municipio":   inst.get("c11", ""),
            "segmento":    inst.get("c12", ""),
            "_c0":         inst["c0"],  # chave para cruzamento com dados
        })

    logging.info(f"  {periodo}: {len(resultado)} cooperativas Cresol singulares")
    return resultado


def baixar_resumo(s: requests.Session, periodo: int) -> list[dict]:
    """
    Baixa o Relatório Resumo (dados_1) para todas as instituições individuais.

    Returns:
        Lista de dicts: {entity_id: str, col_id: float, ...}
    """
    pref = _prefixo_periodo(periodo)
    nome = f"{pref}{periodo}/dados{periodo}_1.json"
    logging.debug(f"Baixando resumo {periodo}…")
    dados_brutos = _get_arquivo(s, nome)

    resultado = []
    for entrada in dados_brutos.get("values", []):
        eid = str(entrada["e"])
        row = {"_c0": eid}
        for v in entrada.get("v", []):
            col_id = v["i"]
            if col_id in COLUNAS_RESUMO:
                row[COLUNAS_RESUMO[col_id]] = v["v"]
        resultado.append(row)

    return resultado


# ── Processamento de um período ────────────────────────────────────────────────
def processar_periodo(
    s: requests.Session,
    periodo: int,
) -> pd.DataFrame | None:
    """
    Baixa cadastro + resumo para um período e retorna DataFrame das Cresol.
    """
    logging.info(f"Processando {periodo}…")
    try:
        cadastro = baixar_cadastro(s, periodo)
    except Exception as exc:
        logging.error(f"  Falha no cadastro de {periodo}: {exc}")
        return None

    if not cadastro:
        logging.warning(f"  Nenhuma Cresol singular encontrada em {periodo}")
        return None

    try:
        resumo = baixar_resumo(s, periodo)
    except Exception as exc:
        logging.error(f"  Falha no resumo de {periodo}: {exc}")
        return None

    # Indexar resumo por c0
    resumo_idx = {str(r["_c0"]): r for r in resumo}

    # Cruzar cadastro com dados financeiros
    registros = []
    for inst in cadastro:
        c0 = str(inst["_c0"])
        dados_fin = resumo_idx.get(c0, {})
        registros.append({
            "periodo":     periodo,
            "cnpj8":       inst["cnpj8"],
            "nome":        inst["nome"],
            "uf":          inst["uf"],
            "municipio":   inst["municipio"],
            "segmento":    inst["segmento"],
            # Valores em R$ mil (÷1000 porque API retorna em R$)
            "vol_credito_rs_mil":      round((dados_fin.get("vol_credito_rs", 0) or 0) / 1000, 2),
            "carteira_credito_rs_mil": round((dados_fin.get("carteira_credito_rs", 0) or 0) / 1000, 2),
            "captacao_rs_mil":         round((dados_fin.get("captacao_rs", 0) or 0) / 1000, 2),
            "ativo_total_rs_mil":      round((dados_fin.get("ativo_total_rs", 0) or 0) / 1000, 2),
            "patrimonio_liq_rs_mil":   round((dados_fin.get("patrimonio_liq_rs", 0) or 0) / 1000, 2),
        })

    return pd.DataFrame(registros)


def adicionar_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de ano, trimestre e período textual (ex: 2020Q1)."""
    df = df.copy()
    # YYYYMM → ano e mês
    df["ano"]   = df["periodo"] // 100
    df["mes"]   = df["periodo"] % 100
    # Mês 3→Q1, 6→Q2, 9→Q3, 12→Q4
    df["trimestre"] = (df["mes"] - 1) // 3 + 1
    df["periodo_str"] = df["ano"].astype(str) + "Q" + df["trimestre"].astype(str)
    # Data de referência (último dia do trimestre)
    fim_mes = {3: "31", 6: "30", 9: "30", 12: "31"}
    df["data_base"] = (
        df["ano"].astype(str)
        + "-"
        + df["mes"].astype(str).str.zfill(2)
        + "-"
        + df["mes"].map(fim_mes)
    )
    df["data_base"] = pd.to_datetime(df["data_base"])
    return df.drop(columns=["mes"])


def validar_output(df: pd.DataFrame) -> None:
    """Diagnóstico básico do dataset final."""
    logging.info("─" * 60)
    logging.info("VALIDAÇÃO DO DATASET FINAL")
    logging.info(f"  Linhas:          {len(df)}")
    logging.info(f"  Cooperativas:    {df['cnpj8'].nunique()}")
    logging.info(f"  Períodos:        {sorted(df['periodo_str'].unique())}")

    for col in ["vol_credito_rs_mil", "captacao_rs_mil", "ativo_total_rs_mil"]:
        nulos = df[col].isna().sum()
        zeros = (df[col] == 0).sum()
        if nulos > 0:
            logging.warning(f"  ATENÇÃO: {nulos} nulos em '{col}'")
        if zeros > len(df) * 0.1:
            logging.warning(f"  ATENÇÃO: {zeros} zeros ({zeros/len(df):.0%}) em '{col}'")

    logging.info("─" * 60)
    logging.info("Amostra (primeiras 5 linhas):")
    colunas_display = ["periodo_str", "cnpj8", "nome", "uf", "vol_credito_rs_mil", "captacao_rs_mil"]
    logging.info("\n" + df[colunas_display].head(5).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def verificar_servidor(s: requests.Session) -> bool:
    """Verifica se o portal IF.data está disponível antes de iniciar."""
    try:
        r = s.get(PORTAL_URL, timeout=15)
        return r.status_code < 500
    except Exception:
        return False


def _cache_path(periodo: int) -> Path:
    return DATA_DIR / f"_cache_{periodo}.csv"


def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Download IF.data – Cooperativas Cresol ===")

    s = criar_sessao()

    if not verificar_servidor(s):
        logging.error(
            "Portal IF.data (www3.bcb.gov.br) indisponível (503/timeout).\n"
            "Aguarde alguns minutos e execute novamente. O download retomará\n"
            "automaticamente os períodos já salvos em cache."
        )
        sys.exit(1)

    todos_frames: list[pd.DataFrame] = []
    periodos_pendentes = []

    # Carregar cache de períodos já baixados
    for periodo in PERIODOS:
        cache = _cache_path(periodo)
        if CACHE_ENABLED and cache.exists():
            df_cache = pd.read_csv(cache, encoding="utf-8-sig")
            todos_frames.append(df_cache)
            logging.info(f"  {periodo}: carregado do cache ({len(df_cache)} registros)")
        else:
            periodos_pendentes.append(periodo)

    logging.info(f"{len(todos_frames)} períodos em cache, {len(periodos_pendentes)} pendentes")

    for i, periodo in enumerate(periodos_pendentes, 1):
        logging.info(f"[{i}/{len(periodos_pendentes)}] Período {periodo}")
        df_periodo = processar_periodo(s, periodo)
        if df_periodo is not None and not df_periodo.empty:
            if CACHE_ENABLED:
                df_periodo.to_csv(_cache_path(periodo), index=False, encoding="utf-8-sig")
            todos_frames.append(df_periodo)
        # Pausa extra entre períodos para evitar rate limiting
        time.sleep(REQUEST_DELAY)

    if not todos_frames:
        logging.error(
            "Nenhum dado baixado.\n"
            "  • Verifique se o portal IF.data está acessível.\n"
            "  • Execute novamente: períodos já baixados serão lidos do cache."
        )
        sys.exit(1)

    # Consolidar
    df_final = pd.concat(todos_frames, ignore_index=True)
    df_final = adicionar_labels(df_final)

    # Salvar
    out_csv     = DATA_DIR / "cresol_resumo_trimestral.csv"
    out_parquet = DATA_DIR / "cresol_resumo_trimestral.parquet"
    df_final.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_final.to_parquet(out_parquet, index=False)

    validar_output(df_final)
    logging.info(f"CSV     → {out_csv}")
    logging.info(f"Parquet → {out_parquet}")
    logging.info("Download concluído.")


if __name__ == "__main__":
    main()
