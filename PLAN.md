# Plano — Detecção e Identificação de Anomalias em Tráfego de Rede

Dataset CIC-IDS2017 — https://www.kaggle.com/datasets/dhoogla/cicids2017/data

> **Mudança de escopo (vs. 1ª apresentação):** o trabalho deixa de focar só em DDoS e
> passa a cobrir **todas as famílias de ataque** do CICIDS2017, num pipeline de
> **duas etapas**: (1) **Detecção** binária — é ataque ou normal? e, quando for ataque,
> (2) **Identificação** multiclasse — qual o tipo (DDoS, DoS, PortScan, Botnet, Bruteforce,
> Infiltration, WebAttacks).

## Contexto

O projeto entrega um sistema de ML que analisa fluxos de tráfego de rede, **detecta**
comportamento malicioso e **identifica o tipo de ataque**, com um dashboard de alertas.
Base nos pontos já apresentados pelo grupo (1ª apresentação): dataset CICIDS2017 (77 features,
>2 milhões de registros, já pré-processado no Kaggle — sem nulos/infinitos, sem IP/timestamp,
valores normalizados); modelos Decision Tree, Random Forest e Regressão Logística; estratégia
70/15/15 com ajuste de hiperparâmetros (Grid/Random Search); métricas Precisão, Recall e F1;
e reflexão ética (privacidade, falsos positivos, supervisão humana).

Hoje existe: os dados (`data/*.parquet`, 8 famílias), um script de modelo binário funcional
até a matriz de confusão (`backend/analise_matriz.py`) e a imagem gerada (`images/mat_confusao.png`).

**Bloqueio crítico:** o `.venv` (Python 3.14.6) só tem `pip` — nenhuma biblioteca instalada,
então **nenhum script roda atualmente**. Não há `requirements.txt`. O `backend/tratamento.py`
está quebrado (erro de sintaxe na linha 3, usa `ficheiros_alvo` inexistente, grafia errada
`DDos`) e os scripts leem os parquet do diretório atual, mas os arquivos estão em `data/`.

Objetivo do plano: levar o projeto de "modelo binário isolado que não roda" até "pipeline
reprodutível de 2 etapas (detecção + identificação), com métricas completas e dashboard funcional".

## Estado atual (arquivos)

- `data/*.parquet` — 8 famílias: Benign, DDoS, DoS, PortScan, Botnet, Bruteforce, Infiltration, WebAttacks. Nomes "no-metadata".
- `backend/analise_matriz.py` — pipeline RF binário até matriz de confusão (caminhos errados, sem métricas).
- `backend/tratamento.py` — **quebrado**, será substituído.
- `images/mat_confusao.png` — saída já gerada uma vez.

---

## Fase 0 — Ambiente (bloqueante)

Sem isto nada roda.

1. Criar `requirements.txt` na raiz:
   `pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, pyarrow, joblib, streamlit`.
2. Instalar no venv: `.venv/Scripts/python.exe -m pip install -r requirements.txt`.
3. **Risco:** Python 3.14 é muito novo; se faltar wheel (ex.: imbalanced-learn), fallback é
   fixar Python 3.12 no venv. Validar com um `import` de teste de cada lib.

## Fase 1 — Pré-processamento reutilizável (`backend/preprocessamento.py`)

Substitui o `tratamento.py` quebrado por funções importáveis, reaproveitadas pelas duas etapas
de treino e pelo dashboard. **Carrega as 8 famílias.**

- `DATA_DIR = Path(__file__).parent.parent / "data"` — corrige o bug de caminho.
- `carregar_dados()` — lê e concatena **os 8 parquet**.
- `limpar(df)` — `inf`→`NaN` + `dropna` (a apresentação diz que os dados já vêm limpos do Kaggle;
  manter como rede de segurança). Remover colunas de vazamento com filtro defensivo
  `[c for c in cols if c in df.columns]` — como são "no-metadata", IP/timestamp já não existem.
- `criar_targets(df)` — gera **dois rótulos** a partir de `Label`:
  - `target_bin` → 0 se `Label.strip().upper() == 'BENIGN'`, senão 1 (lógica já validada em `analise_matriz.py:35`).
  - `target_tipo` → família do ataque (consolidar sub-rótulos: ex. `DoS Hulk`/`DoS GoldenEye`→`DoS`,
    `FTP-Patator`/`SSH-Patator`→`Bruteforce`, `Web Attack *`→`WebAttacks`). **Verificar os valores reais
    de `Label`** assim que o pandas estiver instalado (não deu pra inspecionar: lib ausente).
- `preparar_features(df)` — separa X dos targets, remove colunas de variância 0 **só de X** (já corrigido em `analise_matriz.py:52`).

## Fase 2 — Etapa 1: Detecção binária (`backend/etapa1_deteccao.py`)

"É ataque ou normal?" — treinado com **todas as 8 famílias** (todo ataque vira classe 1).

1. Carregar → limpar → targets → features (Fase 1).
2. Split estratificado **70% treino / 15% validação / 15% teste** (conforme apresentação).
3. **SMOTE só no treino** (após o split). O dataset é grande (>2M); se ficar pesado, usar
   `class_weight='balanced'` ou subamostrar Benign. Decidir conforme tempo de execução.
4. `StandardScaler` ajustado **só no treino** (necessário p/ Regressão Logística; árvores dispensam).
5. Treinar e comparar os 3 modelos da apresentação: **Decision Tree, Random Forest, Regressão Logística**
   (RF é o principal — reaproveitar config de `analise_matriz.py:61`).
6. **Métricas (Seção 4):** `classification_report` (Precisão/Recall/F1) no conjunto de teste,
   `confusion_matrix` + heatmap em `images/mat_confusao_deteccao.png`, e importância das features (top-N do RF).
7. Salvar artefatos com `joblib` em `models/` (modelo de detecção, scaler, lista de colunas).

## Fase 3 — Etapa 2: Identificação do tipo (`backend/etapa2_identificacao.py`)

Multiclasse, treinado **apenas no tráfego malicioso** (filtra `target_bin == 1`) usando `target_tipo`.

1. Reusar a Fase 1; filtrar só os ataques.
2. Mesmo split 70/15/15 estratificado por tipo.
3. SMOTE só no treino — atenção ao forte desbalanceamento entre famílias (Infiltration é minúscula vs. DDoS/PortScan).
4. Treinar e comparar DT / RF / Regressão Logística (multinomial).
5. **Métricas:** `classification_report` por classe + `confusion_matrix` **NxN** em `images/mat_confusao_tipo.png`.
6. Salvar o modelo de identificação em `models/`.

## Fase 4 — Validação e ajuste de hiperparâmetros

Conforme a apresentação (Grid Search / Random Search), nas duas etapas:

- Usar o conjunto de **validação (15%)** para seleção; reportar o desempenho final só no **teste (15%)**.
- `GridSearchCV` ou `RandomizedSearchCV` para os hiperparâmetros principais (RF: `n_estimators`,
  `max_depth`; LogReg: `C`). Pode rodar em subamostra para caber no tempo.

## Fase 5 — Dashboard Streamlit (`frontend/app.py`)

Front-end refletindo o pipeline de 2 etapas:

- `st.file_uploader` para CSV → mesmo pré-processamento (Fase 1) → carrega os 2 modelos (`joblib.load`).
- **Fluxo:** Etapa 1 marca cada fluxo como Normal/Ataque; para os marcados como Ataque, Etapa 2 informa o **tipo**.
- Painel de alertas **vermelho/verde** por fluxo; quando vermelho, exibir o tipo de ataque identificado.
- Gráficos dinâmicos do volume de tráfego e distribuição por tipo de ataque.
- Métricas de confiança via `predict_proba`.
- Nota de **supervisão humana** no resultado (reflexão ética da apresentação).
- Rodar com `.venv/Scripts/python.exe -m streamlit run frontend/app.py`.

---

## Verificação (end-to-end)

1. `.venv/Scripts/python.exe -c "import pandas, sklearn, imblearn, streamlit, seaborn"` — sem erro.
2. `.venv/Scripts/python.exe backend/etapa1_deteccao.py` — imprime `classification_report` (Precisão/Recall/F1),
   gera `images/mat_confusao_deteccao.png` + importância, e salva o modelo de detecção em `models/`.
3. `.venv/Scripts/python.exe backend/etapa2_identificacao.py` — `classification_report` por tipo + matriz NxN em `images/mat_confusao_tipo.png`.
4. Conferir que o **Recall da classe Ataque** (Etapa 1) é alto (evitar falsos negativos — prioridade da apresentação).
5. `streamlit run frontend/app.py` → subir CSV de teste → ver detecção + tipo + painel de alertas funcionando.

## Observações / decisões em aberto

- `backend/analise_matriz.py`: será absorvido pela Etapa 1. Manter como referência ou **deletar** — confirmar na execução.
- Volume (>2M registros) pode exigir subamostragem para treino/Grid Search rodarem em tempo razoável.
- **Verificar os valores reais de `Label`** para montar o mapeamento de famílias da Etapa 2 (pendente: pandas não estava instalado).
- Estrutura de pastas: `backend/` (lógica), `frontend/` (Streamlit), `models/` (artefatos), `requirements.txt` (raiz).
