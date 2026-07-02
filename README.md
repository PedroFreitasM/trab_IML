# 🛡️ Sistema de Detecção e Identificação de Intrusões em Tráfego de Rede

**Projeto da disciplina de Introdução ao Aprendizado de Máquina (IML)**

Sistema baseado em Machine Learning que analisa fluxos de tráfego de rede em **duas etapas**: (1) **Detecção** binária — é ataque ou normal? e (2) **Identificação** multiclasse — qual o tipo de ataque (DDoS, DoS, PortScan, Botnet, Bruteforce, WebAttacks).

---

## 📑 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Rodar](#-como-rodar)
- [Base de Dados](#-base-de-dados-cicids2017)
- [Modelos e Métricas](#-modelos-e-métricas)
- [Dashboard](#-dashboard-streamlit)
- [Captura de Tráfego Real](#-captura-de-tráfego-real)
- [Equipe](#-equipe)

---

## 🎯 Visão Geral

O aumento de ataques de rede (DDoS, DoS, Bruteforce, etc.) exige mecanismos rápidos e automáticos de detecção. Este projeto implementa um **pipeline de ML em cascata** com dois estágios complementares:

| Etapa | Objetivo | Modelo Principal | Métrica Chave |
|:---:|:---|:---|:---|
| **Etapa 1** | Detectar se o tráfego é Normal ou Ataque | Random Forest (com limiar otimizado) | Recall ≥ 98% |
| **Etapa 2** | Identificar o tipo de ataque | Random Forest + SMOTE (via `imblearn.Pipeline`) | Macro-F1 ≥ 96% |

O sistema inclui um **dashboard interativo em Streamlit** para upload de arquivos CSV, visualização de alertas em tempo real e análise da confiança das predições.

---

## 🏗️ Arquitetura do Pipeline

```
CSV de Tráfego
      │
      ▼
┌─────────────────────┐
│  Pré-processamento  │  ← preprocessamento.py
│  (limpar, targets)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│   ETAPA 1           │     │   ETAPA 2                    │
│   Detecção Binária  │────▶│   Identificação Multiclasse  │
│   (Normal/Ataque)   │     │   (DDoS, DoS, Botnet, ...)   │
│   Limiar: 0.60      │     │   SMOTE + StratifiedKFold    │
└─────────────────────┘     └──────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────┐
│           Dashboard Streamlit               │
│  Alertas (🟢/🔴) + Gráficos + Confiança    │
└─────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```
trab_IML/
├── backend/                          # Pipeline de ML
│   ├── preprocessamento.py           # Contrato 1: funções reutilizáveis de dados
│   ├── etapa1_deteccao.py            # Treino da detecção binária (RF)
│   ├── etapa2_identificacao.py       # Treino da identificação multiclasse (RF + SMOTE)
│   ├── regressao_logistica.py        # Treino da detecção binária (LogReg)
│   ├── regressao_logistica_multiclasse.py  # Treino da identificação (LogReg)
│   ├── avaliacao.py                  # Avaliação em cascata fim-a-fim
│   ├── visualizacao.py               # Funções de plotagem (matrizes de confusão)
│   ├── inspecionar_dados.py          # Script de inspeção do dataset (GATE 1.A)
│   ├── gerar_amostra.py              # Gerador de subamostra estratificada (50k linhas)
│   ├── gerar_bundle_falso.py         # Bundle mock para desenvolvimento do frontend
│   └── test_*.py                     # Testes unitários
│
├── frontend/
│   └── app.py                        # Dashboard Streamlit (Track C)
│
├── captura_wifi.py                   # Captura de pacotes de rede via Scapy
├── conversor.py                      # Converte .pcap → CSV (formato CICFlowMeter)
│
├── data/                             # Dados (não versionados, ver instruções abaixo)
│   ├── *.parquet                     # 8 arquivos parquet do CICIDS2017
│   └── amostra.parquet               # Subamostra estratificada (50k linhas)
│
├── models/                           # Modelos treinados (não versionados)
│   ├── rf_etapa1.joblib              # Bundle Random Forest (Detecção)
│   ├── rf_etapa2.joblib              # Bundle Random Forest (Identificação)
│   ├── lr_etapa1.joblib              # Bundle Regressão Logística (Detecção)
│   └── lr_etapa2.joblib              # Bundle Regressão Logística (Identificação)
│
├── images/                           # Gráficos gerados
│   ├── mat_confusao_deteccao.png     # Matriz de confusão (Etapa 1)
│   ├── mat_confusao_tipo.png         # Matriz de confusão multiclasse (Etapa 2)
│   └── mat_confusao_cascata.png      # Matriz de confusão fim-a-fim
│
├── requirements.txt                  # Dependências Python
├── PLAN.md                           # Plano detalhado de implementação
├── TASKS.md                          # Divisão de tarefas por tracks
└── README.md                         # Este arquivo
```

---

## 🚀 Como Rodar

### 1. Pré-requisitos

- **Python 3.12+**
- **pip** ou **uv** (recomendado)

### 2. Clonar o Repositório

```bash
git clone https://github.com/PedroFreitasM/trab_IML.git
cd trab_IML
```

### 3. Criar o Ambiente Virtual e Instalar Dependências

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install plotly                # Necessário para os gráficos do dashboard
```

### 4. Baixar os Dados

Os dados **não estão versionados** por serem muito grandes. Baixe o dataset CICIDS2017 no formato Parquet:

👉 [CICIDS2017 no Kaggle](https://www.kaggle.com/datasets/dhoogla/cicids2017/data)

Coloque os 8 arquivos `.parquet` na pasta `data/`.

### 5. Gerar a Amostra Estratificada

```bash
python backend/gerar_amostra.py
```

Isso cria o arquivo `data/amostra.parquet` com 50.000 linhas estratificadas, usado para treino e testes rápidos.

### 6. Treinar os Modelos

**Random Forest (modelo principal):**
```bash
python backend/etapa1_deteccao.py
python backend/etapa2_identificacao.py
```

**Regressão Logística (modelo comparativo):**
```bash
python backend/regressao_logistica.py
python backend/regressao_logistica_multiclasse.py
```

### 7. Avaliação em Cascata

```bash
python backend/avaliacao.py
```

### 8. Rodar o Dashboard

```bash
streamlit run frontend/app.py
```

Acesse `http://localhost:8501` no navegador, selecione o algoritmo desejado (Random Forest ou Regressão Logística) e faça upload de um CSV de tráfego de rede.

### 9. Rodar os Testes

```bash
python -m unittest discover -s backend
```

---

## 📊 Base de Dados: CICIDS2017

O projeto utiliza a base pública **CICIDS2017**, com **2.313.810 registros** e **77 features** estatísticas de fluxos de rede.

| Família | Linhas | % do Total |
|:---|---:|---:|
| Benign | 1.977.318 | 85,46% |
| DoS | 193.756 | 8,37% |
| DDoS | 128.014 | 5,53% |
| Bruteforce | 9.150 | 0,40% |
| WebAttacks | 2.143 | 0,09% |
| PortScan | 1.956 | 0,08% |
| Botnet | 1.437 | 0,06% |
| Infiltration | 36 | 0,002% |

> **Nota:** A classe `Infiltration` foi removida da modelagem por possuir amostras insuficientes para treino e validação cruzada.

---

## 🤖 Modelos e Métricas

### Modelos Comparados

Em cada etapa, treinamos e comparamos **3 modelos**:
- **Decision Tree** — baseline interpretável
- **Random Forest** — modelo principal (melhor desempenho)
- **Regressão Logística** — baseline linear com `StandardScaler`

### Resultados no Conjunto de Teste

#### Etapa 1 — Detecção Binária (Limiar otimizado: 0.60)

| Classe | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Normal (0) | 0.99 | 1.00 | 1.00 | 3.272 |
| Ataque (1) | 0.99 | 0.98 | 0.99 | 1.091 |
| **Acurácia** | | | **0.99** | **4.363** |

#### Etapa 2 — Identificação Multiclasse (Random Forest + SMOTE)

| Classe | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Botnet | 0.83 | 1.00 | 0.91 | 5 |
| Bruteforce | 1.00 | 0.97 | 0.98 | 30 |
| DDoS | 1.00 | 1.00 | 1.00 | 415 |
| DoS | 1.00 | 1.00 | 1.00 | 628 |
| PortScan | 1.00 | 1.00 | 1.00 | 6 |
| WebAttacks | 1.00 | 1.00 | 1.00 | 7 |
| **Macro Avg** | **0.97** | **0.99** | **0.98** | **1.091** |

#### Cascata Fim-a-Fim

- **Ataques não detectados (Falsos Negativos):** 17 de 1.091 (**1.56%**)
- **Macro-F1 combinado:** **0.92**
- **Acurácia geral:** **99.77%**

### Anti-Leakage

- `StandardScaler` ajustado **apenas no treino**
- `SMOTE` aplicado **dentro do pipeline** (`imblearn.Pipeline`), refeito em cada fold do CV
- Filtro de variância zero calculado **apenas no treino**
- Divisão estratificada **70/15/15** (treino/validação/teste)

---

## 🖥️ Dashboard Streamlit

O dashboard permite ao operador de rede:

- 📤 **Upload** de arquivos CSV de tráfego
- 🔀 **Seleção** do algoritmo (Random Forest ou Regressão Logística)
- 🟢🔴 **Painel de alertas** colorido por fluxo (Normal = verde, Ataque = vermelho)
- 📊 **Gráficos interativos** de distribuição e tipologia de ataques (Plotly)
- 📈 **Métricas de confiança** via `predict_proba`
- ⚠️ **Aviso de supervisão humana** (requisito ético)

---

## 📡 Captura de Tráfego Real

O projeto inclui ferramentas para capturar e converter tráfego de rede real:

1. **`captura_wifi.py`** — Captura pacotes de uma interface de rede usando Scapy e salva em `.pcap` no formato `capturas/<nome>_<timestamp>.pcap`.
2. **`conversor.py`** — Converte arquivos `.pcap` para CSV mapeando as colunas curtas do CICFlowMeter para os nomes longos esperados pelos modelos na inferência.

```bash
# Capturar 30 segundos de tráfego na interface Wi-Fi (salva como capturas/wifi_capture_YYYYMMDD_HHMMSS.pcap)
sudo python captura_wifi.py -i wlp2s0 -d 30 --nome wifi_capture

# Converter o .pcap para CSV (substitua o timestamp pelo gerado no comando anterior)
python conversor.py capturas/wifi_capture_20260701_213000.pcap trafego_real.csv

# Fazer upload do CSV no dashboard para análise
streamlit run frontend/app.py
```

---

## 👥 Equipe

| Membro | Track | Responsabilidade |
|:---|:---:|:---|
| Jurbas | A | Dados, pré-processamento, inspeção e geração de amostras |
| Pedro | B | Modelagem (RF), avaliação em cascata e integração geral |
| Nicholas | B | Modelagem (LogReg), captura Wi-Fi e conversor PCAP |
| Arthur | B | Modelagem (RF alternativa) |
| Igor | C | Dashboard Streamlit e visualizações |

---

## 📄 Licença

Projeto acadêmico — uso restrito à disciplina de IML.
