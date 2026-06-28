import os
import gc
import time
import pandas as pd
import numpy as np
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score
)

# ==========================================
# ETAPA 1: FUNÇÕES AUXILIARES (IDÊNTICAS À RF)
# ==========================================
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any(): continue
        val_min, val_max = df[col].min(), df[col].max()
        tipo = df[col].dtype
        if np.issubdtype(tipo, np.integer):
            if val_min > np.iinfo(np.int8).min and val_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
            elif val_min > np.iinfo(np.int16).min and val_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
            elif val_min > np.iinfo(np.int32).min and val_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
        elif np.issubdtype(tipo, np.floating):
            if val_min > np.finfo(np.float32).min and val_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
    return df

def clean_and_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# ==========================================
# ETAPA 2: CARREGAMENTO E PREPARAÇÃO
# ==========================================
print("1. CARREGAMENTO E UNIÃO DOS DADOS")
pasta_dados = '../data/'

ficheiros = [
    os.path.join(pasta_dados, 'Benign-Monday-no-metadata.parquet'),
    os.path.join(pasta_dados, 'DDoS-Friday-no-metadata.parquet'),
    os.path.join(pasta_dados, 'Portscan-Friday-no-metadata.parquet')
]

df_full = pd.concat([pd.read_parquet(f) for f in ficheiros], ignore_index=True)
df_full = optimize_memory(df_full)
df_full = clean_and_sanitize(df_full)

df_full['Label'] = df_full['Label'].astype(str).str.strip().str.upper()
df_full['Label_Binary'] = df_full['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
y = df_full['Label_Binary']

colunas_vazamento = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp', 'Flow ID', 'Label', 'Label_Binary']
X = df_full.drop(columns=[c for c in colunas_vazamento if c in df_full.columns])
X = X.select_dtypes(include=[np.number])
X = X.loc[:, X.var() != 0]
del df_full; gc.collect()

# ==========================================
# ETAPA 3: DIVISÃO DOS DADOS
# ==========================================
print("\n2. DIVISÃO DOS DADOS")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
del X, y; gc.collect()
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_temp, y_temp; gc.collect()

# ==========================================
# ETAPA 4: BALANCEAMENTO (SMOTE) E NORMALIZAÇÃO
# ==========================================
print("\n3. BALANCEAMENTO E NORMALIZAÇÃO (Escala Métrica)")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
del X_train, y_train; gc.collect()

# A Regressão Logística exige dados na mesma escala (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test) # O teste usa o transform para evitar data leakage
del X_train_res; gc.collect()

# ==========================================
# ETAPA 5: TREINAMENTO (REGRESSÃO LOGÍSTICA)
# ==========================================
print("\n4. TREINANDO A REGRESSÃO LOGÍSTICA")
# max_iter alto garante que a matemática linear convirja sem dar avisos de erro
modelo_lr = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)

inicio_treino = time.time()
modelo_lr.fit(X_train_scaled, y_train_res)
fim_treino = time.time()

print(f"-> Tempo de Treinamento: {(fim_treino - inicio_treino):.2f} segundos")

# ==========================================
# ETAPA 6: AVALIAÇÃO E MEDIÇÃO DE VELOCIDADE
# ==========================================
print("\n5. AVALIAÇÃO NO TESTE FINAL")

inicio_inferencia = time.time()
y_predicao = modelo_lr.predict(X_test_scaled)
fim_inferencia = time.time()

tempo_inferencia_ms = (fim_inferencia - inicio_inferencia) * 1000
print(f"-> Tempo para analisar {len(X_test_scaled)} pacotes: {tempo_inferencia_ms:.2f} milissegundos")

matriz_confusao = confusion_matrix(y_test, y_predicao)
tn, fp, fn, tp = matriz_confusao.ravel()

print(f"\n--- RELATÓRIO: REGRESSÃO LOGÍSTICA ---")
print(f" Verdadeiros Positivos (Bloqueados) : {tp}")
print(f" Verdadeiros Negativos (Soltos)     : {tn}")
print(f" Falsos Positivos (Alarmes Falsos)  : {fp}")
print(f" Falsos Negativos (Não Detectados)  : {fn}")

print(f"\nRecall (Taxa de Captura)   : {recall_score(y_test, y_predicao):.4f}")
print(f"Precision (Confiabilidade) : {precision_score(y_test, y_predicao):.4f}")
print(f"F1-Score (Qualidade Geral) : {f1_score(y_test, y_predicao):.4f}")

# ==========================================
# ETAPA 7: EXPORTAÇÃO DO MODELO E SCALER
# ==========================================
print("\n6. EXPORTAÇÃO DO MODELO")
os.makedirs('../models', exist_ok=True)

# Exportamos tanto o modelo quanto o scaler (o Streamlit vai precisar do scaler para novos arquivos)
joblib.dump(modelo_lr, '../models/modelo_lr_baseline.pkl')
joblib.dump(scaler, '../models/scaler_lr.pkl')

print("-> Sucesso! Modelo Linear e Scaler exportados para a pasta 'models/'")
