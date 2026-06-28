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
>2 milhões de registros, supostamente já pré-processado no Kaggle — sem nulos/infinitos, sem
IP/timestamp, valores normalizados — **a confirmar, ver Fase 1**); modelos Decision Tree,
Random Forest e Regressão Logística; estratégia 70/15/15 com ajuste de hiperparâmetros
(Grid/Random Search); métricas Precisão, Recall e F1; e reflexão ética (privacidade, falsos
positivos, supervisão humana).

Hoje existe: os dados (`data/*.parquet`, 8 famílias), um script de modelo binário funcional
até a matriz de confusão (`backend/analise_matriz.py`) e a imagem gerada (`images/mat_confusao.png`).

**Bloqueio crítico (ambiente):** a única versão instalada na máquina é **Python 3.14**
(`py -3.12/-3.11` → "No suitable Python runtime found"), e o `.venv` (Python 3.14.6) só tem
`pip` — nenhuma biblioteca, então **nenhum script roda**. Não há `requirements.txt` versionado
de forma utilizável em 3.14 (numpy 1.26/sklearn 1.5 não têm wheel p/ 3.14). **`uv` está instalado**
(`AppData\Roaming\uv`) e resolve isso baixando o Python 3.12 — ver Fase 0.

**Bloqueio crítico (código):** o `backend/tratamento.py` está quebrado e **nunca executou**:
`from sklearn,model_selection` (vírgula, linha 3), itera sobre `ficheiros_alvo` inexistente
(`NameError`, linha 13), grafia errada `DDos` (linha 7) e o `concat`+`print` estão **dentro do
loop** (indentação, linha 21). Os scripts leem os parquet do diretório atual, mas os arquivos
estão em `data/`.

Objetivo do plano: levar o projeto de "modelo binário isolado que não roda" até "pipeline
reprodutível de 2 etapas (detecção + identificação), com métricas completas e dashboard funcional".

## Estado atual (arquivos)

- `data/*.parquet` — 8 famílias: Benign, DDoS, DoS, PortScan, Botnet, Bruteforce, Infiltration, WebAttacks. Nomes "no-metadata". **`data/` está em `.gitignore` → dados não versionados** (rodar Fase 0.0 para baixá-los).
- `backend/analise_matriz.py` — pipeline RF binário até matriz de confusão (caminhos relativos errados, salva PNG no diretório atual, sem métricas).
- `backend/tratamento.py` — **quebrado**, será substituído (ver bugs no Contexto).
- `images/mat_confusao.png` — saída já gerada uma vez.
- `models/` — **ainda não existe**; será criado na Fase 2 para os artefatos `joblib`.

---

## Fase 0 — Ambiente (bloqueante)

Sem isto nada roda. Como **só há Python 3.14** (incompatível com as wheels de numpy/sklearn
fixadas) e o **`uv` já está disponível**, usar `uv` para instalar o 3.12 e criar o venv.

**0.0 — Dados (não versionados):** garantir que `data/` contém os 8 parquet (baixar do link do
Kaggle acima, se necessário). `data/` é gitignored de propósito (arquivos grandes).

**0.1 — `requirements.txt`** na raiz, com versões fixadas para Python 3.12 (já criado).

**0.2 — Recriar o venv em 3.12 com uv** (caminho recomendado):
```powershell
uv python install 3.12
Remove-Item -Recurse -Force .venv          # remove o venv 3.14.6 antigo
uv venv --python 3.12 .venv
uv pip install -r requirements.txt         # instala no .venv
```
*Fallback sem uv:* instalar Python 3.12 (python.org) e
`py -3.12 -m venv .venv ; .venv\Scripts\python.exe -m pip install -r requirements.txt`.

**0.3 — Validar:**
`.venv\Scripts\python.exe -c "import pandas, sklearn, imblearn, streamlit, seaborn, pyarrow"`
sem erro. **Risco:** se faltar wheel para alguma lib, é sintoma de versão de Python errada
(confirmar que o venv é 3.12, não 3.14).

## Fase 1 — Pré-processamento reutilizável (`backend/preprocessamento.py`)

Substitui o `tratamento.py` quebrado por funções importáveis, reaproveitadas pelas duas etapas
de treino e pelo dashboard. **Carrega as 8 famílias.**

- `DATA_DIR = Path(__file__).parent.parent / "data"` — corrige o bug de caminho.
- `carregar_dados()` — lê e concatena **os 8 parquet** (lista única, `pd.concat` **após** o loop).
- `limpar(df)` — `inf`→`NaN` + `dropna` (rede de segurança, mesmo que o Kaggle já venha limpo).
  Remover colunas de vazamento com filtro defensivo `[c for c in cols if c in df.columns]`.
- `criar_targets(df)` — gera **dois rótulos** a partir de `Label`:
  - `target_bin` → 0 se `Label.strip().upper() == 'BENIGN'`, senão 1 (lógica validada em `analise_matriz.py:35`).
  - `target_tipo` → família do ataque (consolidar sub-rótulos: ex. `DoS Hulk`/`DoS GoldenEye`→`DoS`,
    `FTP-Patator`/`SSH-Patator`→`Bruteforce`, `Web Attack *`→`WebAttacks`).
- `preparar_features(df)` — separa X dos targets; remoção de colunas de variância 0 **ajustada no
  treino** (não no dataset inteiro) — fazer dentro do split/Pipeline para não vazar do teste
  (o `analise_matriz.py:52` faz no conjunto todo; aqui corrigir).

### 1.A — GATE de verificação dos dados (fazer assim que o pandas estiver instalado)

Resolve premissas que mudam o resto do pipeline. Rodar um script de inspeção e **anotar o
resultado aqui**:
- **Valores reais de `Label`** por arquivo (`value_counts`) → fecha o mapeamento de `target_tipo`.
- **Já está normalizado?** `df.describe()` (min/max/mean). Se sim, **o `StandardScaler` é
  dispensável p/ árvores e possivelmente p/ LogReg** — decidir aqui, não escalar "por reflexo".
- **Há `NaN`/`inf` de fato?** (ex.: `Flow Bytes/s`) → confirma se o `dropna` é mesmo necessário.
- **IP/Timestamp/Flow ID ainda existem?** ("no-metadata" sugere que não) → enxuga a lista de leakage.
- **Nº de features e total de linhas** → confirma os "77 features / >2M" e dimensiona a subamostragem.

#### RESULTADOS (verificado em 2026-06-28, contra os 8 parquet)

- **`Label` (15 valores) → `MAPA_FAMILIAS` COMPLETO** (todos mapeiam, 0 falhas). Os rótulos de
  Web Attack contêm de fato o byte `U+FFFD` (`'Web Attack � Brute Force'`); as chaves do mapa estão
  corretas. `_mapear_familia` ganhou um guard por prefixo (`startswith("WEB ATTACK")`) como rede de segurança.
- **NÃO está normalizado** — 59/77 colunas com valores fora de [0,1]. ⇒ **`StandardScaler` é
  OBRIGATÓRIO para a Regressão Logística (Track B)**; árvores dispensam. (Corrige a premissa antiga
  de "valores normalizados", que estava **errada**.)
- **Sem `NaN`/`Inf`** no arquivo testado (PortScan) → `limpar()` permanece como rede de segurança.
- **Metadados ausentes:** das colunas de vazamento, só `Label` existe (sem IP/Timestamp/Flow ID/Dest Port);
  `Protocol` está presente e é mantido como feature.
- **78 colunas (77 features + `Label`); 2.313.810 linhas; BENIGN = 85,5%.** Distribuição por família:

| Família | Linhas | % | Observação |
|---|--:|--:|---|
| Benign | 1.977.318 | 85,46% | majoritária → subamostrar (Etapa 1) |
| DoS | 193.756 | 8,37% | inclui Heartbleed (11) |
| DDoS | 128.014 | 5,53% | |
| Bruteforce | 9.150 | 0,40% | FTP/SSH-Patator |
| WebAttacks | 2.143 | 0,093% | BF/XSS/SQLi |
| PortScan | 1.956 | 0,085% | |
| Botnet | 1.437 | 0,062% | |
| Infiltration | **36** | 0,0016% | **ultra-raro** |

> **Aviso à Track B:** (1) dados **não normalizados** → `StandardScaler` no Pipeline (fit só no
> treino) para a Regressão Logística; (2) **Infiltration (36 linhas)** inviabiliza holdout único na
> Etapa 2 → usar **StratifiedKFold** + **macro-F1** e reportar *support*; SMOTE apenas no treino.

## Fase 2 — Etapa 1: Detecção binária (`backend/etapa1_deteccao.py`)

"É ataque ou normal?" — treinado com **todas as 8 famílias** (todo ataque vira classe 1).

1. Carregar → limpar → targets → features (Fase 1).
2. **Subamostrar BENIGN** (teto p/ caber em RAM/tempo; >2M é inviável inteiro) **antes** do split.
3. Split estratificado **70% treino / 15% validação / 15% teste** (conforme apresentação).
4. **Balanceamento da Etapa 1:** preferir `class_weight='balanced'` (DT/RF/LogReg) **+ subamostragem**.
   **Não usar SMOTE aqui** (oversampling sintético contra milhões de BENIGN é inviável).
5. `StandardScaler` ajustado **só no treino** — **obrigatório p/ LogReg** (dados NÃO normalizados,
   confirmado no GATE 1.A); árvores (DT/RF) dispensam.
6. Treinar e comparar os 3 modelos: **Decision Tree, Random Forest, Regressão Logística**
   (RF é o principal — reaproveitar config de `analise_matriz.py:61`).
7. **Seleção:** comparar no conjunto de **validação**; **prioridade = Recall da classe Ataque**
   (minimizar falsos negativos). Considerar **ajustar o limiar** do `predict_proba` (não fixar 0.5)
   na validação para atingir um Recall-alvo.
8. **Métricas (no teste):** `classification_report` (Precisão/Recall/F1), `confusion_matrix` +
   heatmap em `images/mat_confusao_deteccao.png`, e importância das features (top-N do RF).
9. Salvar artefatos com `joblib` em `models/` (modelo, scaler — se usado, e **lista de colunas de treino**).

## Fase 3 — Etapa 2: Identificação do tipo (`backend/etapa2_identificacao.py`)

Multiclasse, treinado **apenas no tráfego malicioso** (filtra `target_bin == 1`) usando `target_tipo`.

1. Reusar a Fase 1; filtrar só os ataques.
2. **Atenção às classes ultra-raras** (ex.: Infiltration). Um holdout único de 15% deixa pouquíssimos
   exemplos no teste → preferir **StratifiedKFold** para avaliação, e/ou reportar sempre o *support*
   por classe. Avaliar fundir classes minúsculas se inviável estatisticamente.
3. **SMOTE só no treino** (base menor, faz sentido aqui) para as famílias raras; alternativa
   `class_weight='balanced'`. Ver Fase 4 sobre SMOTE dentro do CV.
4. Treinar e comparar DT / RF / Regressão Logística (multinomial). **Métrica de seleção = macro-F1**
   (dá peso igual às classes raras).
5. **Métricas:** `classification_report` por classe + `confusion_matrix` **NxN** em `images/mat_confusao_tipo.png`.
6. Salvar o modelo e o **mapa de classes** em `models/`.

## Fase 4 — Validação e ajuste de hiperparâmetros

Conforme a apresentação (Grid/Random Search), nas duas etapas:

- **Evitar vazamento no tuning:** montar um **`imblearn.pipeline.Pipeline([scaler, smote, clf])`**
  e passar o Pipeline ao search, para que **scaler e SMOTE sejam reajustados em cada fold** do CV.
  (Aplicar SMOTE uma vez antes do `GridSearchCV` vaza amostras sintéticas para os folds de validação.)
- Usar **`RandomizedSearchCV`** (mais barato que GridSearch no volume deste dataset), em **subamostra**
  se necessário. Espaço de busca enxuto: RF (`n_estimators`, `max_depth`, `max_features`),
  LogReg (`C`), DT (`max_depth`, `min_samples_leaf`).
- Papéis dos conjuntos: **CV interno** (no treino) p/ hiperparâmetros; **validação (15%)** p/ comparar
  modelos/limiar; **teste (15%)** intocado p/ o número final.
- **Avaliação cascata fim-a-fim (reflete o dashboard):** rodar o **teste** por Etapa 1 → (se ataque)
  Etapa 2 e reportar a performance combinada por tipo, **incluindo os ataques perdidos pela Etapa 1**
  (bucket "não detectado"). É a métrica honesta do sistema de 2 estágios.

## Fase 5 — Dashboard Streamlit (`frontend/app.py`)

Front-end refletindo o pipeline de 2 etapas:

- `st.file_uploader` para CSV → **normalizar nomes de colunas** (CICIDS real tem espaços nos headers,
  ex. `' Flow Duration'`) → mesmo pré-processamento da Fase 1 → **reindexar para a lista de colunas
  salva no treino** (tratar colunas ausentes/extras) → aplicar o `scaler` salvo (se usado) → carregar
  os 2 modelos (`joblib.load`).
- **Fluxo:** Etapa 1 marca cada fluxo como Normal/Ataque; para os marcados como Ataque, Etapa 2 informa o **tipo**.
- Painel de alertas **vermelho/verde** por fluxo; quando vermelho, exibir o tipo de ataque identificado.
- Gráficos dinâmicos do volume de tráfego e distribuição por tipo de ataque.
- Métricas de confiança via `predict_proba`.
- Nota de **supervisão humana** no resultado (reflexão ética da apresentação).
- `@st.cache_resource` para carregar os modelos uma única vez.
- Rodar com `.venv\Scripts\python.exe -m streamlit run frontend/app.py`.

---

## Verificação (end-to-end)

1. `.venv\Scripts\python.exe -c "import pandas, sklearn, imblearn, streamlit, seaborn, pyarrow"` — sem erro.
2. Rodar a inspeção da **Fase 1.A** e anotar `Label`/normalização/NaN/nº de features no plano.
3. `.venv\Scripts\python.exe backend/etapa1_deteccao.py` — imprime `classification_report`,
   gera `images/mat_confusao_deteccao.png` + importância, e salva modelo/colunas em `models/`.
4. `.venv\Scripts\python.exe backend/etapa2_identificacao.py` — `classification_report` por tipo
   (com *support*) + matriz NxN em `images/mat_confusao_tipo.png`.
5. Conferir que o **Recall da classe Ataque** (Etapa 1) é alto (evitar falsos negativos — prioridade).
6. **Cascata fim-a-fim** (Fase 4): rodar o teste por Etapa1→Etapa2 e conferir a matriz combinada.
7. `streamlit run frontend/app.py` → subir CSV de teste → ver detecção + tipo + painel de alertas.

## Observações / decisões

- `backend/analise_matriz.py`: **extrair** as partes reutilizáveis (lógica do Target, lista de leakage,
  heatmap) para `preprocessamento.py`/utilidades e então **aposentar** o script (mover p/ `legacy/` ou
  deletar) para não haver duas fontes de verdade.
- `data/` é gitignored → documentar no README de onde baixar (link do Kaggle). `models/` precisa existir
  (criar com `.gitkeep` ou via script).
- Volume (>2M) exige subamostragem para treino/Search rodarem em tempo razoável.
- Decisão de **escalar ou não** depende da Fase 1.A (não escalar por reflexo se já vier normalizado).

## Histórico de revisão

- **2026-06-28 (revisão):** corrigida a Fase 0 (só há Python 3.14; usar `uv` para o 3.12);
  adicionado GATE de verificação de dados (Fase 1.A); SMOTE movido p/ dentro do Pipeline no CV e
  restrito à Etapa 2; balanceamento da Etapa 1 via subamostragem+`class_weight`; classes raras e
  macro-F1 na Etapa 2; avaliação cascata fim-a-fim; normalização de headers/reindex no dashboard;
  notas sobre dados não versionados e destino do `analise_matriz.py`.
