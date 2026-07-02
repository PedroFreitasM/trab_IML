# Detecção e Visualização de Anomalias em Tráfego de Rede (DDoS)

## 1. Introdução e Objetivo

O aumento de ataques de Negação de Serviço Distribuída (DDoS) exige mecanismos rápidos e automáticos de detecção. O objetivo do projeto é desenvolver um sistema baseado em Aprendizado de Máquina para analisar fluxos de tráfego de rede, classificar padrões como "Normais" ou "Maliciosos" e gerar alertas visuais para o administrador da rede.

## 2. Base de Dados: CICIDS2017

O projeto utiliza a base pública **CICIDS2017** , composta por tráfego benigno e famílias de ataques (DDoS, DoS, PortScan) dividido em diferentes dias e horários. Contém de 80 a 85 colunas com características do tráfego (como duração do fluxo e tamanho dos pacotes) e pesa até 1 GB em CSV.

O modelo focará nas variáveis mais importantes para o escopo, como `Flow Duration`, `Flow Bytes/s`, `SYN Flag Count` e `Packet Length Variance`.

## 3. Pipeline de Aprendizado de Máquina (Back-end)

* **Pré-processamento:** Tratamento de valores nulos e infinitos , codificação de variáveis categóricas e normalização de features numéricas.


* **Modelagem:**
1. *Random Forest:* Método principal que lida com alta dimensionalidade, captura relações não lineares e oferece índice de importância de variáveis.
2. *Regressão Logística:* Modelo de *baseline* linear com velocidade de inferência superior.
3. *Árvore de Decisão (Interpretabilidade):* Árvore rasa dedicada à explicabilidade, permitindo a extração de regras textuais de decisão e plotagem visual das ramificações.

* **Validação e Teste:** Divisão dos dados em 70% para treino, 15% para validação e 15% para testes (ou K-Fold). Os hiperparâmetros são otimizados via `GridSearchCV` dentro de um pipeline anti-leakage da biblioteca `imblearn`.

## 4. Avaliação e Métricas

Devido ao desbalanceamento dos dados, a acurácia isolada é inadequada. A avaliação utiliza a Matriz de Confusão e as métricas:
* **Recall:** Fundamental para garantir que os ataques não passem despercebidos.
* **Precision:** Vital para evitar falsos alertas e a fadiga do operador.
* **F1-Score:** Média harmônica ideal para validar a qualidade geral do modelo.

## 5. Interface e Visualização (Front-end)

Consiste em um Dashboard interativo em **Streamlit** que permite a seleção do algoritmo (Random Forest, Regressão Logística ou Árvore de Decisão), o upload de arquivos CSV, exibe gráficos dinâmicos do volume de tráfego e distribuição de tipos de ataques, apresenta painéis com alertas visuais (vermelho/verde), exibe métricas de confiança e, no caso da Árvore de Decisão, expõe regras de negócios e visualização de caminhos.

## 6. Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.12 configurado. Se possuir a ferramenta `uv`, execute:
```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

### Passo 1: Gerar Bundles Falsos (Opcional - Apenas para rodar o Dashboard sem treinar)
```bash
.venv/Scripts/python.exe backend/gerar_bundle_falso.py
```

### Passo 2: Treinar os Modelos Reais
Para treinar os modelos e gerar os bundles em `models/`:
- **Modelos Principais (Random Forest & Regressão Logística):**
  ```bash
  .venv/Scripts/python.exe backend/etapa1_deteccao.py
  .venv/Scripts/python.exe backend/etapa2_identificacao.py
  ```
- **Árvores de Decisão (Interpretabilidade):**
  ```bash
  .venv/Scripts/python.exe backend/dt_interpretabilidade.py
  ```

### Passo 3: Executar a Avaliação Fim-a-Fim
```bash
.venv/Scripts/python.exe backend/avaliacao.py
```

### Passo 4: Executar o Dashboard Streamlit
```bash
.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

### Executar a Suíte de Testes
```bash
.venv/Scripts/python.exe -m unittest discover -s backend
```
