#!/usr/bin/env python3
"""
08_train_models.py
==================
Treina e avalia modelos preditivos para metas financeiras das cooperativas Cresol.

Modelos  : Ridge (linear), RandomForest, XGBoost, LightGBM
Validação: Time Series Split com janela expansível (expanding window) por período
Métricas : R², RMSE, MAE (out-of-sample)
Targets  : vol_credito_rs_mil, captacao_rs_mil

Features usadas (sem data leakage — todas referentes a t-1 ou antes):
  - Lags financeiros L1/L2/L4 dos 5 indicadores Cresol  (15 features)
  - L1 de variáveis macro SGS (Selic, IPCA, câmbio…)     ( 7 features)
  - PIB municipal anual IBGE (com lag natural de ~1 ano)  ( 6 features)
  - L1 de preços de commodities agrícolas                 ( 4 features)
  - L1 de crédito rural agregado                          ( 4 features)
  - Segmento da cooperativa (encoding ordinal)            ( 1 feature)
  Total: até 37 preditores por target
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ── Caminhos ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PANEL    = ROOT / "data" / "processed" / "panel_features_clean.parquet"
OUT_DIR  = ROOT / "data" / "processed"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR  = ROOT / "logs"

# ── Configuração ───────────────────────────────────────────────────────────────
TARGETS = ["vol_credito_rs_mil", "captacao_rs_mil"]

FINS = ["vol_credito_rs_mil", "captacao_rs_mil", "ativo_total_rs_mil",
        "patrimonio_liq_rs_mil", "carteira_credito_rs_mil"]

MACRO_COLS = ["selic_aa", "ipca_acum_trim", "cambio_brl_usd",
              "ibc_br", "inpc_acum_trim", "ipa_agro", "concessoes_cred"]

PIB_COLS = ["pib_corrente_rs_mil", "pib_per_capita_rs",
            "vab_agro_rs_mil", "vab_industria_rs_mil", "vab_servicos_rs_mil",
            "share_agro_mun"]

COMOD_COLS = ["ipa_agro_idx", "milho_rs_60kg", "boi_gordo_rs_arroba", "leite_rs_litro"]

CR_COLS = ["cr_total_rs_mi", "cr_custeio_rs_mi",
           "cr_investimento_rs_mi", "cr_comercializacao_rs_mi"]

# Todas as features (sem data leakage)
FEAT_FIN_LAG   = [f"L{l}_{v}" for l in [1, 2, 4] for v in FINS]
FEAT_MACRO_L1  = [f"L1_{c}" for c in MACRO_COLS]
FEAT_PIB       = [c for c in PIB_COLS]
FEAT_COMOD_L1  = [f"L1_{c}" for c in COMOD_COLS]
FEAT_CR_L1     = [f"L1_{c}" for c in CR_COLS]
FEAT_CAT       = ["segmento_num", "porte_num"]

# Dummies de sazonalidade agrícola (calendário é conhecido no momento da previsão)
FEAT_SAFRA = [
    "dummy_plantio_verao",      # Q4: plantio soja/milho
    "dummy_colheita_verao",     # Q1: colheita soja/milho
    "dummy_plantio_inv",        # Q2: plantio trigo/aveia
    "dummy_colheita_inv",       # Q3: colheita inverno
    "dummy_sul",                # cooperativa em PR/SC/RS
    "dummy_plantio_verao_sul",  # interação plantio verão × Sul
    "dummy_colheita_verao_sul", # interação colheita verão × Sul
]

ALL_FEATURES = (FEAT_FIN_LAG + FEAT_MACRO_L1 + FEAT_PIB +
                FEAT_COMOD_L1 + FEAT_CR_L1 + FEAT_CAT + FEAT_SAFRA)

# ── Modelos ────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    """Retorna dicionário {nome: estimator} com os 4 modelos candidatos."""
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  MinMaxScaler()),   # Min-Max: linear é sensível à magnitude
            ("model",   Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            # Sem scaler: árvores são invariantes à escala das features
            ("model",   RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbosity=0,
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbose=-1,
            )),
        ]),
    }


# ── Time Series Split ──────────────────────────────────────────────────────────
def ts_splits(df: pd.DataFrame, n_splits: int = 5, min_train_periods: int = 8):
    """
    Gera (train_idx, test_idx) com janela expansível por período temporal.

    Ordena períodos cronologicamente e divide de forma que:
      - Treino: todos os dados até o ponto de corte
      - Teste:  os ~2 períodos seguintes
    """
    periodos = sorted(df["periodo_str"].unique())
    n = len(periodos)
    test_size = max(1, (n - min_train_periods) // n_splits)

    for i in range(n_splits):
        split = min_train_periods + i * test_size
        if split >= n:
            break
        train_p = periodos[:split]
        test_p  = periodos[split: split + test_size]
        if not test_p:
            break
        train_idx = df.index[df["periodo_str"].isin(train_p)]
        test_idx  = df.index[df["periodo_str"].isin(test_p)]
        yield i + 1, train_p, test_p, train_idx, test_idx


# ── Métricas ───────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


# ── Treino e avaliação ─────────────────────────────────────────────────────────
def cross_validate_models(df: pd.DataFrame, target: str,
                          features: list[str]) -> pd.DataFrame:
    """
    Executa Time Series CV para todos os modelos em um target.
    Retorna DataFrame com métricas por fold.
    """
    feats_avail = [f for f in features if f in df.columns and f != target]
    logging.info(f"  Features disponíveis: {len(feats_avail)}/{len(features)}")

    records = []
    for modelo_nome, modelo in build_models().items():
        logging.info(f"  Modelo: {modelo_nome}")
        for fold, train_p, test_p, tr_idx, te_idx in ts_splits(df):
            X_tr = df.loc[tr_idx, feats_avail]
            y_tr = df.loc[tr_idx, target]
            X_te = df.loc[te_idx, feats_avail]
            y_te = df.loc[te_idx, target]

            # Remover linhas com target nulo
            mask_tr = y_tr.notna()
            mask_te = y_te.notna()
            X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
            X_te, y_te = X_te[mask_te], y_te[mask_te]

            if len(X_tr) < 10 or len(X_te) < 1:
                continue

            modelo.fit(X_tr, y_tr)
            y_pred = modelo.predict(X_te)

            m = calc_metrics(y_te.values, y_pred)
            records.append({
                "target":  target,
                "modelo":  modelo_nome,
                "fold":    fold,
                "train_inicio": train_p[0],
                "train_fim":    train_p[-1],
                "test_inicio":  test_p[0],
                "test_fim":     test_p[-1],
                "n_train":  len(y_tr),
                "n_test":   len(y_te),
                **m,
            })
            logging.info(
                f"    Fold {fold} [{test_p[0]}-{test_p[-1]}] "
                f"R²={m['r2']:.3f}  RMSE={m['rmse']:,.0f}  MAE={m['mae']:,.0f}"
            )

    return pd.DataFrame(records)


def treinar_modelo_final(df: pd.DataFrame, target: str,
                         features: list[str], modelo_nome: str):
    """
    Treina o melhor modelo em TODOS os dados disponíveis (sem CV).
    Retorna pipeline treinado.
    """
    feats_avail = [f for f in features if f in df.columns and f != target]
    modelos = build_models()
    modelo = modelos[modelo_nome]

    mask = df[target].notna()
    X = df.loc[mask, feats_avail]
    y = df.loc[mask, target]
    modelo.fit(X, y)
    return modelo, feats_avail


def extrair_importancia(modelo_pipeline, feats: list[str],
                        modelo_nome: str) -> pd.DataFrame | None:
    """Extrai feature importance do modelo (RF, XGB, LGBM) ou coeficientes (Ridge)."""
    step = modelo_pipeline.named_steps.get("model")
    if step is None:
        return None

    if hasattr(step, "feature_importances_"):
        imp = step.feature_importances_
    elif hasattr(step, "coef_"):
        imp = np.abs(step.coef_)
    else:
        return None

    df_imp = pd.DataFrame({"feature": feats, "importance": imp})
    df_imp["modelo"] = modelo_nome
    return df_imp.sort_values("importance", ascending=False).reset_index(drop=True)


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "model_training.log", encoding="utf-8"),
        ],
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=== Treinamento de Modelos – Cresol TCC ===")

    # 1. Carregar painel
    df = pd.read_parquet(PANEL)
    logging.info(f"Painel: {df.shape[0]} linhas × {df.shape[1]} colunas")
    logging.info(f"Períodos: {df['periodo_str'].min()} a {df['periodo_str'].max()}")
    logging.info(f"Cooperativas: {df['cnpj8'].nunique()}")

    all_cv    = []
    all_imp   = []
    best_info = {}

    for target in TARGETS:
        logging.info(f"\n{'='*60}")
        logging.info(f"Target: {target}")
        logging.info(f"{'='*60}")

        # Features: excluir o próprio target e outros targets atuais
        outros_targets = [t for t in TARGETS if t != target]
        feats = [f for f in ALL_FEATURES if f not in [target] + outros_targets]

        # 2. Cross-validation
        df_cv = cross_validate_models(df, target, feats)
        df_cv.to_csv(OUT_DIR / f"cv_results_{target}.csv", index=False, encoding="utf-8-sig")
        all_cv.append(df_cv)

        # 3. Resumo por modelo
        logging.info(f"\nResumo CV — {target}:")
        resumo = (
            df_cv.groupby("modelo")[["r2", "rmse", "mae"]]
            .agg(["mean", "std"])
            .round(4)
        )
        logging.info("\n" + resumo.to_string())

        # 4. Eleger melhor modelo (maior R² médio)
        media_r2 = df_cv.groupby("modelo")["r2"].mean()
        melhor   = media_r2.idxmax()
        logging.info(f"\nMelhor modelo para {target}: {melhor} (R²={media_r2[melhor]:.3f})")
        best_info[target] = {"modelo": melhor, "r2_cv": media_r2[melhor]}

        # 5. Re-treinar melhor modelo em todos os dados
        modelo_final, feats_usadas = treinar_modelo_final(df, target, feats, melhor)
        caminho = MODEL_DIR / f"best_{target}_{melhor.lower()}.joblib"
        joblib.dump({"pipeline": modelo_final, "features": feats_usadas}, caminho)
        logging.info(f"Modelo salvo: {caminho.name}")

        # 6. Feature importance
        df_imp = extrair_importancia(modelo_final, feats_usadas, melhor)
        if df_imp is not None:
            df_imp["target"] = target
            all_imp.append(df_imp)
            logging.info(f"\nTop 10 features ({melhor}):")
            logging.info("\n" + df_imp.head(10)[["feature", "importance"]].to_string(index=False))

    # 7. Consolidar e salvar resultados
    df_all_cv = pd.concat(all_cv, ignore_index=True)
    df_all_cv.to_csv(OUT_DIR / "cv_results_all.csv", index=False, encoding="utf-8-sig")

    if all_imp:
        df_all_imp = pd.concat(all_imp, ignore_index=True)
        df_all_imp.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 8. Relatório final
    logging.info(f"\n{'='*60}")
    logging.info("RESUMO FINAL:")
    for target, info in best_info.items():
        r2_cv = info["r2_cv"]
        logging.info(f"  {target}: melhor={info['modelo']}, R²_cv={r2_cv:.3f}")

    logging.info(f"\nArquivos salvos em {OUT_DIR}:")
    logging.info(f"  cv_results_all.csv, feature_importance.csv")
    logging.info(f"  models/ → pipeline treinado no dataset completo")
    logging.info("Concluido.")


if __name__ == "__main__":
    main()
