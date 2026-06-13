import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    ConfusionMatrixDisplay
)

from imblearn.under_sampling import RandomUnderSampler

# 1. FUNÇÕES AUXILIARES E PRÉ-PROCESSAMENTO

def optimize_memory(df):
    """Aplica downcasting nos tipos numéricos para economizar memória."""
    for col in df.columns:
        col_type = df[col].dtype
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
    """Remove infinitos e valores ausentes."""
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# 2. CARREGAMENTO DOS DADOS

print("Carregando os arquivos Parquet...")
df_benign = pd.read_parquet("Benign-Monday-no-metadata.parquet")
df_ddos = pd.read_parquet("DDoS-Friday-no-metadata.parquet")
df_portscan = pd.read_parquet("Portscan-Friday-no-metadata.parquet")

df_full = pd.concat([df_benign, df_ddos, df_portscan], ignore_index=True)
del df_benign, df_ddos, df_portscan
gc.collect() # Libera a RAM imediatamente

print("Otimizando memória...")
df_full = optimize_memory(df_full)

print("Limpando dados...")
df_full = clean_data(df_full)

# 3. CRIAÇÃO DA VARIÁVEL ALVO

if "Label" not in df_full.columns:
    raise ValueError("Coluna 'Label' não encontrada.")

df_full["Label"] = df_full["Label"].astype(str).str.strip().str.upper()
df_full["Label_Binary"] = df_full["Label"].apply(lambda x: 0 if "BENIGN" in x else 1)

print("\nDistribuição das classes:")
print(df_full["Label_Binary"].value_counts())

X = df_full.drop(columns=["Label", "Label_Binary"])
X = X.select_dtypes(include=[np.number])
y = df_full["Label_Binary"]

del df_full 
gc.collect()

# 4. DIVISÃO TREINO / VALIDAÇÃO / TESTE

print("\nDividindo os dados...")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
del X, y 
gc.collect()

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_temp, y_temp 
gc.collect()

# 5. BALANCEAMENTO

print("Aplicando RandomUnderSampler...")
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

del X_train, y_train # Limpa os dados desbalanceados
gc.collect()

print(f"Amostras após balanceamento: {len(X_train_resampled)}")

# 6. RANDOM FOREST + GRID SEARCH

print("Treinando Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20], 
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=1, 
    verbose=2
)

grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

print("\nMelhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# 7. AVALIAÇÃO FINAL

print("\n--- Avaliação no conjunto de teste ---")
y_test_pred = best_model.predict(X_test)
cm_test = confusion_matrix(y_test, y_test_pred)

print(classification_report(y_test, y_test_pred, target_names=["Normal (0)", "Ataque (1)"]))

recall_val = recall_score(y_test, y_test_pred)
precision_val = precision_score(y_test, y_test_pred)
f1_val = f1_score(y_test, y_test_pred)

print("\n--- Resumo das Métricas Críticas ---")
print(f"Falsos Positivos: {cm_test[0][1]}")
print(f"Falsos Negativos: {cm_test[1][0]}")
print(f"Recall: {recall_val:.4f} | Precision: {precision_val:.4f} | F1-Score: {f1_val:.4f}")

# 8. MATRIZ DE CONFUSÃO

print("\nGerando matriz de confusão visual...")
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Benigno (0)", "Ataque (1)"])
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

plt.title("Matriz de Confusão - Random Forest", fontsize=14, pad=15)
plt.xlabel("Rótulo Predito (Pelo Modelo)")
plt.ylabel("Rótulo Realidade")
plt.tight_layout()

plt.savefig("matriz_confusao_RF.png", dpi=300)
print("Sucesso! Imagem salva como 'matriz_confusao_RF.png'")

# 9. IMPORTÂNCIA DAS FEATURES

print("\nAnalisando Importância das Features...")
importances = pd.DataFrame({
    'Feature': X_val.columns, 
    'Importance': best_model.feature_importances_
})

importances = importances.sort_values(by='Importance', ascending=False)
print("\nTop 20 Features Mais Importantes:")
print(importances.head(20))

top_20_features = importances.head(20).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10,8))
top_20_features.plot(x='Feature', y='Importance', kind='barh', color='skyblue', legend=False)

plt.title("Top 20 Atributos de Rede Mais Relevantes para a Detecção", fontsize=14, pad=15)
plt.xlabel("Grau de Importância (Gini)", fontsize=12)
plt.ylabel("Atributo (Feature)", fontsize=12)
plt.tight_layout()

plt.savefig("feature_importance_RF.png", dpi=300)
print("Sucesso! Imagem salva como 'feature_importance_RF.png'")