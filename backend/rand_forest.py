import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

dir_data = '../data/'

# ==========================================
# 1. CARREGAMENTO E LIMPEZA
# ==========================================
print("Iniciando Random Forest...")
ficheiros = [ 
    os.path.join(dir_data, 'Benign-Monday-no-metadata.parquet'),
    os.path,join(dir_data, 'DDoS-Friday-no-metadata.parquet'),
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
# 2. SEPARAÇÃO DOS DADOS
# ==========================================
X = df.drop(columns=['Target'])
y = df['Target']
X = X.loc[:, X.var() != 0]

# Não usamos StandardScaler aqui, pois árvores lidam bem com diferentes escalas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# 3. TREINAMENTO E AVALIAÇÃO
# ==========================================
print("Treinando o modelo (Isso pode levar alguns segundos)...")
modelo_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)

print("\n--- RESULTADOS: RANDOM FOREST ---")
print(classification_report(y_test, y_pred_rf, target_names=['Normal (0)', 'Ataque (1)']))

# ==========================================
# 4. SALVAR MATRIZ DE CONFUSÃO
# ==========================================
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Previsto: Normal', 'Previsto: Ataque'],
            yticklabels=['Real: Normal', 'Real: Ataque'])
plt.title('Matriz de Confusão - Random Forest', pad=15)
plt.tight_layout()
plt.savefig('matriz_floresta.png', dpi=300, bbox_inches='tight')
print("-> Gráfico salvo como 'matriz_floresta.png'")
