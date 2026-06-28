import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

dir_data = '../data/'

# ==========================================
# 1. CARREGAMENTO E LIMPEZA
# ==========================================
print("Iniciando Regressão Logística...")
ficheiros = [ 
    os.path.join(dir_data, 'Benign-Monday-no-metadata.parquet'),
    os.path.join(dir_data, 'DDoS-Friday-no-metadata.parquet'),
    os.path.join(dir_data, 'Portscan-Friday-no-metadata.parquet')
]

df = pd.concat([pd.read_parquet(f) for f in ficheiros], ignore_index=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Higienização da coluna Label
df['Target'] = df['Label'].astype(str).str.strip().str.upper().apply(lambda x: 0 if x == 'BENIGN' else 1)

# Remoção de colunas de vazamento
colunas_vazamento = ['Source IP', 'Destination IP', 'Source Port', 'Timestamp', 'Flow ID', 'Label']
df.drop(columns=[c for c in colunas_vazamento if c in df.columns], inplace=True)

# ==========================================
# 2. SEPARAÇÃO E NORMALIZAÇÃO (CRUCIAL AQUI)
# ==========================================
X = df.drop(columns=['Target'])
y = df['Target']
X = X.loc[:, X.var() != 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Normalizando os dados...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. TREINAMENTO E AVALIAÇÃO
# ==========================================
print("Treinando o modelo...")
modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
modelo_lr.fit(X_train_scaled, y_train)

y_pred_lr = modelo_lr.predict(X_test_scaled)

print("\n--- RESULTADOS: REGRESSÃO LOGÍSTICA ---")
print(classification_report(y_test, y_pred_lr, target_names=['Normal (0)', 'Ataque (1)']))

# ==========================================
# 4. SALVAR MATRIZ DE CONFUSÃO
# ==========================================
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Previsto: Normal', 'Previsto: Ataque'],
            yticklabels=['Real: Normal', 'Real: Ataque'])
plt.title('Matriz de Confusão - Regressão Logística', pad=15)
plt.tight_layout()
plt.savefig('matriz_logistica.png', dpi=300, bbox_inches='tight')
print("-> Gráfico salvo como 'matriz_logistica.png'")
