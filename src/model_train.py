import os
import gc
import pandas as pd
import numpy as np
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score, classification_report
)

# ==========================================
# ETAPA 1: FUNÇÕES AUXILIARES (MEMÓRIA E LIMPEZA)
# ==========================================

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """ Reduz o tamanho que os dados numéricos ocupam na memória RAM. """
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            continue
        valor_min = df[col].min()
        valor_max = df[col].max()
        tipo_coluna = df[col].dtype

        if np.issubdtype(tipo_coluna, np.integer):
            if valor_min > np.iinfo(np.int8).min and valor_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif valor_min > np.iinfo(np.int16).min and valor_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif valor_min > np.iinfo(np.int32).min and valor_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif np.issubdtype(tipo_coluna, np.floating):
            if valor_min > np.finfo(np.float32).min and valor_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def clean_and_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove erros matemáticos, nulos e linhas duplicadas (vazamento de dados). """
    total_antes = len(df)
    
    # 1. Ajusta o nome das colunas
    df.columns = df.columns.str.strip()
    
    # 2. Remove infinitos e nulos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 3. CORREÇÃO CRÍTICA: Remove fluxos duplicados para evitar o F1 de 100% artificial
    df.drop_duplicates(inplace=True)
    
    print(f"   [Sanitização] Linhas limpas/duplicadas removidas: {total_antes - len(df)}")
    return df

# ==========================================
# ETAPA 2: CARREGAMENTO E UNIÃO DOS DADOS
# ==========================================

print("1. CARREGAMENTO E UNIÃO DOS DADOS")
pasta_dados = '../data/' # Caminho relativo padrão do projeto

df_benign   = pd.read_parquet(os.path.join(pasta_dados, 'Benign-Monday-no-metadata.parquet'))
df_ddos     = pd.read_parquet(os.path.join(pasta_dados, 'DDoS-Friday-no-metadata.parquet'))
df_portscan = pd.read_parquet(os.path.join(pasta_dados, 'Portscan-Friday-no-metadata.parquet'))

df_full = pd.concat([df_benign, df_ddos, df_portscan], ignore_index=True)

del df_benign, df_ddos, df_portscan; gc.collect()

print("   Otimizando e limpando base unificada...")
df_full = optimize_memory(df_full)
df_full = clean_and_sanitize(df_full)

# Criação do Alvo Binário Seguro
df_full['Label'] = df_full['Label'].astype(str).str.strip().str.upper()
df_full['Label_Binary'] = df_full['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

y = df_full['Label_Binary']

# CORREÇÃO CRÍTICA: Remoção explícita de colunas identificadoras e portas (evita Data Leakage)
colunas_vazamento = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp', 'Flow ID', 'Label', 'Label_Binary']
X = df_full.drop(columns=[c for c in colunas_vazamento if c in df_full.columns])
X = X.select_dtypes(include=[np.number])

# Remove colunas que não variam (constantes)
X = X.loc[:, X.var() != 0]

feature_names = X.columns.tolist()

del df_full; gc.collect()

# ==========================================
# ETAPA 3: DIVISÃO DOS DADOS (70/15/15)
# ==========================================

print("\n2. DIVISÃO DOS DADOS (70/15/15)")
# Separa 70% para treino e guarda 30% temporários
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
del X, y; gc.collect()

# Divide os 30% temporários entre Validação (15%) e Teste (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_temp, y_temp; gc.collect()

# ==========================================
# ETAPA 4: BALANCEAMENTO COM SMOTE (CONFORME README)
# ==========================================

print("\n3. BALANCEAMENTO COM SMOTE")
# Cumpre o requisito do SMOTE do seu planejamento, sem jogar dados fora
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

del X_train, y_train; gc.collect()

# ==========================================
# ETAPA 5: OTIMIZAÇÃO DE HIPERPARAMÊTROS
# ==========================================

print("\n4. TREINAMENTO E OTIMIZAÇÃO (RANDOM FOREST)")
# Mudamos para RandomizedSearchCV para o SMOTE rodar em tempo viável sem travar a máquina
parametros_para_testar = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15],  # Árvores podadas para evitar o overfitting perfeito
    'min_samples_split': [5, 10]
}

otimizador = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=parametros_para_testar,
    n_iter=4, # Testará 4 combinações aleatórias inteligentes
    cv=3,     # K-Fold de 3 partes conforme planejado
    scoring='f1',
    n_jobs=1, # Mantém 1 para segurança de RAM, mas a RF interna usa todos os núcleos devido ao n_jobs=-1 acima
    random_state=42,
    verbose=1
)

otimizador.fit(X_train_res, y_train_res)
melhor_modelo = otimizador.best_estimator_

print(f"-> Melhores hiperparâmetros: {otimizador.best_params_}")
del X_train_res, y_train_res; gc.collect()

# ==========================================
# ETAPA 6: AVALIAÇÃO FINAL NO TESTE
# ==========================================

print("\n5. AVALIAÇÃO NO TESTE FINAL")
y_predicao = melhor_modelo.predict(X_test)

matriz_confusao = confusion_matrix(y_test, y_predicao)
tn, fp, fn, tp = matriz_confusao.ravel()

print(f"\n--- RELATÓRIO DO MODELO NO MUNDO REAL ---")
print(f" Verdadeiros Positivos (Ataques bloqueados)  : {tp}")
print(f" Verdadeiros Negativos (Tráfego normal solto): {tn}")
print(f" Falsos Positivos (Alarmes Falsos)           : {fp}")
print(f" Falsos Negativos (Ataques Não Detectados)   : {fn}")

print(f"\nRecall (Taxa de Captura)      : {recall_score(y_test, y_predicao):.4f}")
print(f"Precision (Confiabilidade)    : {precision_score(y_test, y_predicao):.4f}")
print(f"F1-Score (Qualidade Geral)    : {f1_score(y_test, y_predicao):.4f}")

# ==========================================
# ETAPA 7: EXPORTAÇÃO DO MODELO (PARA O STREAMLIT)
# ==========================================

print("\n6. EXPORTAÇÃO DO MODELO")
os.makedirs('../models', exist_ok=True)
caminho_saida = '../models/modelo_rf_ddos.pkl'
joblib.dump(melhor_modelo, caminho_saida)
print(f"-> Sucesso! Modelo exportado para o Front-end em: '{caminho_saida}'")
