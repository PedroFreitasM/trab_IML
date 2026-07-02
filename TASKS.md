# TASKS.md — Divisão de trabalho (3 pessoas, em paralelo)

Complementa o `PLAN.md`. Objetivo: **paralelismo real** com (1) um dono por arquivo
(zero conflito de merge) e (2) **2 contratos congelados** para ninguém ficar
bloqueado esperando o código do outro.

## Contratos congelados

### Contrato 1 — Interface do `backend/preprocessamento.py`

Já existe como stub funcional. Assinaturas:

```python
DATA_DIR, MODELS_DIR, ARQUIVOS, COLUNAS_VAZAMENTO, MAPA_FAMILIAS
carregar_dados(arquivos=None, columns=None) -> pd.DataFrame
limpar(df) -> pd.DataFrame                       # headers, inf->NaN, dropna
criar_targets(df) -> pd.DataFrame                # add 'target_bin' e 'target_tipo'
preparar_features(df, alvo) -> (X, y)            # remove targets/leakage
split(X, y, val=.15, teste=.15, seed=42) -> dict # X_train/X_val/X_test/y_*
```

### Contrato 2 — Bundle de modelo (`models/*.joblib`)

Cada etapa salva um dict auto-suficiente (helpers `salvar_bundle`/`carregar_bundle`):

```python
{"modelo": clf, "scaler": scaler_ou_None, "colunas": [...], "classes": [...], "limiar": 0.5}
```

> **Escalonamento, filtro de variância, SMOTE e tuning NÃO ficam no preprocessamento** —
> pertencem ao Pipeline de treino (Track B) e são ajustados **só no treino** (anti-leakage).

---

## 👤 Track A — Dados & Pré-processamento (fundação · caminho crítico)

**Dono:** `backend/preprocessamento.py`, `backend/inspecionar_dados.py`, `requirements.txt`

- **A1** Finalizar ambiente (`uv` + Python 3.12), validar imports, documentar setup. _(Fase 0)_
- **A2** Script de inspeção → **valores reais de `Label`**, se já está normalizado, NaN/inf,
  nº de features. Preencher o GATE 1.A do PLAN.md e **fechar o `MAPA_FAMILIAS`**. _(Fase 1.A)_
- **A3** Concluir `preprocessamento.py` (substituir TODO(A2) pelo mapa real). _(Fase 1)_
- **A4** Anti-leakage: helper de filtro de variância **no treino**; revisar `COLUNAS_VAZAMENTO`. _(Fase 1)_
- **A5** Gerar `data/amostra.parquet` (10–50k linhas estratificadas) p/ B e C testarem rápido.

## 👤 Track B — Modelagem (2 etapas + validação)

**Dono:** `backend/etapa1_deteccao.py`, `backend/etapa2_identificacao.py`, `backend/avaliacao.py`

- **B1** Etapa 1 binária: split → DT/RF/LogReg c/ `class_weight` + under-sampling no pipeline →
  métricas na val → **ajuste de limiar** p/ Recall → salva `models/rf_etapa1.joblib`. _(Fase 2)_
- **B2** Etapa 2 multiclasse: filtra ataques → **GridSearchCV** → SMOTE-no-Pipeline →
  **macro-F1** → salva `models/rf_etapa2.joblib`. _(Fase 3)_
- **B3** Tuning: `RandomizedSearchCV` via **`imblearn.Pipeline`** (sem vazamento) + importância. _(Fase 4)_
- **B4** **Avaliação cascata** fim-a-fim (teste por Etapa1→Etapa2, incl. "não detectado"). _(Fase 4)_
- **B5** **Modelo de Interpretabilidade (DT dedicada):** script `dt_interpretabilidade.py` para treinar árvore de decisão rasa nas Etapas 1 e 2, salvar `models/dt_etapa1.joblib` e `models/dt_etapa2.joblib`, exportar regras lógicas de negócio e gráficos de estrutura. _(Fase 4)_

Começa contra o Contrato 1 + `amostra.parquet`. Usa `visualizacao.plotar_matriz` (Track C).

## 👤 Track C — Dashboard, Visualização & Entrega

**Dono:** `frontend/app.py`, `backend/visualizacao.py`, `README.md`

- **C1** Esqueleto Streamlit + `file_uploader` + **normalizar headers** + **reindexar p/
  `colunas` salvas** + `carregar_bundle`. _(Fase 5)_
- **C2** Fluxo 2 etapas + confiança (`predict_proba`) + alertas **vermelho/verde** + KPIs. _(Fase 5)_
- **C3** Gráficos dinâmicos (volume, distribuição por tipo) + nota de **supervisão humana**. _(Fase 5)_
- **C4** `visualizacao.py` (heatmap reutilizável, **reusado por B**) +
  README "como rodar" + doc de download dos dados.
- **C5** `@st.cache_resource`, polimento, figuras para o relatório. _(Fase 5)_
- **C6** **Dashboard de Interpretabilidade:** estender a interface Streamlit para exibir regras textuais lógicas e visualização gráfica de caminhos lógicos quando o modelo "Árvore de Decisão" for selecionado. _(Fase 5)_

Começa com dados dummy; integra com modelos reais no M3.
