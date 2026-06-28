# Previsão de Metas Financeiras — Sistema Cresol

TCC — Modelagem preditiva de Volume de Crédito e Captação de Recursos para cooperativas singulares do Sistema Cresol, com análise por porte municipal e conversão em metas comerciais.

## Requisitos

- Python 3.11+
- Ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Dependências de dados externos

Alguns dados não estão disponíveis via API e exigem obtenção manual:

| Script | Dado | Ação necessária |
|:--- |:--- |:--- |
| `04_n_associados.py` | Número de associados por cooperativa | Preencher `data/raw/n_associados/` manualmente |
| `06_commodities.py` | Preço da soja (CEPEA) | Download em <https://www.cepea.esalq.usp.br/br/indicador/soja.aspx> |

## Ordem de execução

Execute os scripts na seguinte sequência:

| # | Script | Descrição |
|:--- |:--- |:--- |
| 1 | `src/01_download_ifdata.py` | Download de dados contábeis das cooperativas (IF.data/BACEN) |
| 2 | `src/02_download_sgs.py` | Séries macroeconômicas do SGS/BCB (Selic, IPCA, IBC-Br, câmbio, etc.) |
| 3 | `src/03_download_ibge.py` | PIB municipal e estimativas populacionais (IBGE SIDRA) |
| 4 | `src/04_n_associados.py` | Template de número de associados (preenchimento manual) |
| 5 | `src/05_credito_rural.py` | Crédito rural por modalidade (SGS/BCB) |
| 6 | `src/06_commodities.py` | Preços de commodities agrícolas (SGS/BCB + CEPEA) |
| 7 | `src/07_feature_engineering.py` | Construção do painel: lags, dummies sazonais, porte municipal |
| 8 | `src/07b_quality_control.py` | Controle de qualidade: duplicados, nulos, outliers (winsorização) |
| 9 | `src/08_train_initial.py` | Treinamento inicial (RF, XGBoost, LightGBM, Ridge — hiperparâmetros fixos) |
| 10 | `src/09_shap_initial.py` | SHAP values e importância de variáveis (modelos iniciais) |
| 11 | `src/10_eda.py` | Análise exploratória: distribuições, correlações, sazonalidade |
| 12 | `src/11_train_tuned.py` | Tuning de hiperparâmetros (GridSearch RF, RandomSearch XGB/LGBM) + OLS |
| 13 | `src/12_model_evaluation.py` | Avaliação OOS: R², RMSE, MAE, MAPE — global e por porte |
| 14 | `src/13_interpretability.py` | Interpretabilidade: Feature/Permutation Importance e SHAP por porte |
| 15 | `src/14_metas_comerciais.py` | Conversão de previsões em metas comerciais por porte |
| 16 | `src/15_relatorio.py` | Geração do relatório de resultados em Markdown |

## Saídas principais

| Arquivo | Descrição |
|:--- |:--- |
| `data/processed/panel_features_clean.parquet` | Painel final (1.227 obs × 91 variáveis) |
| `data/processed/models/tuned/` | Modelos ajustados (`.joblib`) |
| `reports/evaluation_metrics.csv` | Métricas globais OOS por modelo e target |
| `reports/metricas_por_porte.csv` | Métricas OOS estratificadas por porte |
| `reports/metas_comerciais.csv` | Previsões técnicas e metas estratégicas |
| `reports/relatorio_resultados.md` | Relatório acadêmico de resultados |
| `reports/figures/` | Visualizações (SHAP, importâncias, métricas, metas) |

## Reprodutibilidade

Todas as sementes aleatórias estão fixadas em `RANDOM_STATE = 42`. As divisões temporais (Time Series Split expansível, 5 folds) são determinísticas por ordenação dos períodos. O subsampling de background para SHAP LinearExplainer é pré-amostrado com `numpy.random.default_rng(42)` antes de ser passado ao masker.
