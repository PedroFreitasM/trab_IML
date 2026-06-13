import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler


# 1. FUNÇÕES AUXILIARES E PRÉ-PROCESSAMENTO


def optimize_memory(df):
    """Aplica downcasting nos tipos numéricos para economizar memória RAM."""
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def clean_data(df):
    """Limpa nomes de colunas, remove valores infinitos e nulos."""
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS


print("Carregando os arquivos Parquet...")
df_benign = pd.read_parquet('Benign-Monday-no-metadata.parquet')
df_ddos = pd.read_parquet('DDoS-Friday-no-metadata.parquet')
df_portscan = pd.read_parquet('Portscan-Friday-no-metadata.parquet')

df_full = pd.concat([df_benign, df_ddos, df_portscan], ignore_index=True)
del df_benign, df_ddos, df_portscan # Libera RAM imediatamente
gc.collect() # Força a limpeza

print("Otimizando e limpando os dados...")
df_full = optimize_memory(df_full)
df_full = clean_data(df_full)

# Binarização Robusta da Variável Alvo
if 'Label' in df_full.columns:
    df_full['Label'] = df_full['Label'].astype(str).str.strip().str.upper()
    df_full['Label_Binary'] = df_full['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    print("\n[DEBUG] Distribuição da variável binária (0 = Normal, 1 = Ataque):")
    print(df_full['Label_Binary'].value_counts())

    X = df_full.drop(columns=['Label', 'Label_Binary'])
    X = X.select_dtypes(include=[np.number]) 
    y = df_full['Label_Binary']
    
    del df_full # Deleta o dataset principal gigante, mantemos apenas X e y
    gc.collect()
else:
    raise ValueError("A coluna 'Label' não foi encontrada no dataset.")


# 3. PARTICIONAMENTO DOS DADOS E LIMPEZA DE RAM


print("\nDividindo os dados em Treino, Validação e Teste...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

del X, y # Já dividimos, não precisamos mais dos originais
gc.collect()

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

del X_temp, y_temp # Libera os dados temporários
gc.collect()


# 4. PADRONIZAÇÃO PROTEGIDA CONTRA OOM


print("Padronizando as features (com casting para float32)...")
scaler = StandardScaler()
# StandardScaler transforma tudo em float64. Convertendo de volta para float32 cortamos a RAM pela metade!
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

print("Aplicando RandomUnderSampler...")
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)

del X_train, y_train, X_train_scaled # Limpeza agressiva antes do treino pesado
gc.collect()


# 5. MODELAGEM (SEM PARALELISMO)


print(f"Treinando modelo com {len(X_train_resampled)} amostras balanceadas...")
param_grid = {
    'C': [0.1, 1, 10], 
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500] 
}

# n_jobs=1: Crucial para não estourar a memória (Impede o scikit-learn de fazer cópias do dataset)
log_reg = LogisticRegression(random_state=42, n_jobs=1)
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=3, scoring='f1', n_jobs=1, verbose=1)

grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")


# 6. VALIDAÇÃO, AVALIAÇÃO E MATRIZ DE CONFUSÃO VISUAL


print("\n--- Avaliação no Conjunto de Teste Final ---")
y_test_pred = best_model.predict(X_test_scaled)
cm_test = confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred, target_names=['Normal (0)', 'Ataque (1)']))

recall_val = recall_score(y_test, y_test_pred)
precision_val = precision_score(y_test, y_test_pred)
f1_val = f1_score(y_test, y_test_pred)

print("\n--- Resumo das Métricas Críticas ---")
print(f"Falsos Positivos (Tráfego normal bloqueado incorretamente): {cm_test[0][1]}")
print(f"Falsos Negativos (Ataques que passaram despercebidos): {cm_test[1][0]}")
print(f"Recall: {recall_val:.4f} | Precision: {precision_val:.4f} | F1-Score: {f1_val:.4f}")

# --- GERAÇÃO VISUAL DA MATRIZ DE CONFUSÃO ---
print("\nGerando e salvando a Matriz de Confusão como imagem...")

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Benigno (Normal)', 'Ataque (Malicioso)'])
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

plt.title('Matriz de Confusão - Regressão Logística', fontsize=14, pad=15)
plt.xlabel('Rótulo Predito (Pelo Modelo)', fontsize=12)
plt.ylabel('Rótulo Verdadeiro (Realidade)', fontsize=12)
plt.tight_layout()

plt.savefig('matriz_confusao_LR.png', dpi=300)
print("Sucesso! A imagem da Matriz de Confusão foi salva como 'matriz_confusao_LR.png'.")