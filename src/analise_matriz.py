import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento e Modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ==============================================================
# 1. CARREGAMENTO DOS ARQUIVOS
# ==============================================================
print("1. Carregando os arquivos Parquet...")
ficheiros = [ 
    'Benign-Monday-no-metadata.parquet',
    'DDoS-Friday-no-metadata.parquet',
    'Portscan-Friday-no-metadata.parquet'
]

# Lê e concatena todos os arquivos em um único DataFrame
df_completo = pd.concat([pd.read_parquet(f) for f in ficheiros], ignore_index=True)
print(f"-> Arquivos carregados. Total: {df_completo.shape[0]} linhas.")

# ==============================================================
# 2. PRÉ-PROCESSAMENTO E LIMPEZA
# ==============================================================
print("2. Limpando os dados...")
# Remove valores infinitos e nulos
df_completo.replace([np.inf, -np.inf], np.nan, inplace=True)
df_completo.dropna(inplace=True)

# CORREÇÃO 1: Limpeza da string (strip e upper) antes de criar o Target
# Isso garante que ' BENIGN ', 'benign' ou 'BENIGN' sejam lidos corretamente
df_completo['Target'] = df_completo['Label'].astype(str).str.strip().str.upper().apply(lambda x: 0 if x == 'BENIGN' else 1)

# Remove colunas que causam vazamento de dados
colunas_vazamento = ['Source IP', 'Destination IP', 'Source Port', 'Timestamp', 'Flow ID', 'Label']
df_completo.drop(columns=[c for c in colunas_vazamento if c in df_completo.columns], inplace=True)

# ==============================================================
# 3. SEPARAÇÃO EM TREINO E TESTE
# ==============================================================
print("3. Separando os dados para o modelo...")

# Separamos X e y ANTES de aplicar o filtro de variância
X = df_completo.drop(columns=['Target'])
y = df_completo['Target']

# CORREÇÃO 2: Removemos colunas de variância zero apenas das variáveis (X)
# Assim, nosso Target (y) fica blindado contra exclusão acidental
X = X.loc[:, X.var() != 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==============================================================
# 4. TREINAMENTO E PREVISÃO (y_pred)
# ==============================================================
print("4. Treinando o modelo (Random Forest)...")
# Usamos max_depth para acelerar o processo e n_jobs=-1 para usar todo o processador
modelo_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

print("5. Gerando as previsões (y_pred)...")
# y_pred é a decisão do modelo (o que ele achou que era)
y_pred = modelo_rf.predict(X_test)

# ==============================================================
# 6. PLOTAGEM DA MATRIZ DE CONFUSÃO
# ==============================================================
print("6. Desenhando a Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Previsto: Normal (0)', 'Previsto: Ataque (1)'],
            yticklabels=['Real: Normal (0)', 'Real: Ataque (1)'])

plt.title('Matriz de Confusão - Tráfego de Rede', pad=15, fontsize=14)
plt.ylabel('Rótulo Verdadeiro (Realidade do Dataset)', labelpad=10, fontsize=11)
plt.xlabel('Previsão do Algoritmo (O que o modelo classificou)', labelpad=10, fontsize=11)

plt.tight_layout()
nome_img = 'mat_confusao.png'
plt.savefig(nome_img, dpi=300, bbox_inches='tight')
