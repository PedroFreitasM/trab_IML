# Detecção e Visualização de Anomalias em Tráfego de Rede (DDoS)

## 1. Introdução e Objetivo

O aumento de ataques de Negação de Serviço Distribuída (DDoS) exige mecanismos rápidos e automáticos de detecção. O objetivo do projeto é desenvolver um sistema baseado em Aprendizado de Máquina para analisar fluxos de tráfego de rede, classificar padrões como "Normais" ou "Maliciosos" e gerar alertas visuais para o administrador da rede.

## 2. Base de Dados: CICIDS2017

O projeto utiliza a base pública **CICIDS2017** , composta por tráfego benigno e famílias de ataques (DDoS, DoS, PortScan) dividido em diferentes dias e horários. Contém de 80 a 85 colunas com características do tráfego (como duração do fluxo e tamanho dos pacotes) e pesa até 1 GB em CSV.

O modelo focará nas variáveis mais importantes para o escopo, como `Flow Duration`, `Flow Bytes/s`, `SYN Flag Count` e `Packet Length Variance`.

## 3. Pipeline de Aprendizado de Máquina (Back-end)

* **Pré-processamento:** Tratamento de valores nulos e infinitos , codificação de variáveis categóricas e normalização de features numéricas. Aplica-se o algoritmo **SMOTE** para balanceamento das classes.


* **Modelagem:**
1. 
*Random Forest:* Método principal que lida com alta dimensionalidade, captura relações não lineares e oferece índice de importância de variáveis.


2. 
*Regressão Logística:* Modelo de *baseline* linear com velocidade de inferência superior.


3. 
*Isolation Forest (Opcional):* Abordagem não supervisionada que sinaliza desvios do padrão normal.




* **Validação e Teste:** Divisão dos dados em 70% para treino, 15% para validação e 15% para testes (ou K-Fold). Os hiperparâmetros serão otimizados via `GridSearchCV` ou `RandomizedSearchCV`.



## 4. Avaliação e Métricas

Devido ao desbalanceamento dos dados, a acurácia isolada é inadequada. A avaliação utilizará a Matriz de Confusão  e as métricas:

* **Recall:** Fundamental para garantir que os ataques não passem despercebidos.


* **Precision:** Vital para evitar falsos alertas e a fadiga do operador.


* **F1-Score:** Média harmônica ideal para validar a qualidade geral do modelo.



## 5. Interface e Visualização (Front-end)

Consiste em um Dashboard interativo em **Streamlit** que permite o upload de arquivos CSV , exibe gráficos dinâmicos do volume de tráfego , apresenta painéis com alertas visuais (vermelho/verde) para DDoS e mostra as métricas de confiança das predições.
