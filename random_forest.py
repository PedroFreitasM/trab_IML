import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
)

# ETAPA 1: FUNÇÕES AUXILIARES (PREPARAÇÃO)

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o tamanho que os dados numéricos ocupam na memória RAM.
    """
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove linhas com erros matemáticos (Infinito) ou vazias (NaN). """
    total_antes = len(df)

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    total_depois = len(df)
    print(f"   [Limpeza] Linhas removidas: {total_antes - total_depois}")
    return df

# ETAPA 2: CARREGAMENTO E UNIÃO DOS DADOS

print("CARREGAMENTO DOS DADOS")

arquivos_parquet = [
    'data/Benign-Monday-no-metadata.parquet',
    'data/DDoS-Friday-no-metadata.parquet',
    'data/Portscan-Friday-no-metadata.parquet',
    'data/Botnet-Friday-no-metadata.parquet',
    'data/Infiltration-Thursday-no-metadata.parquet',
    'data/Bruteforce-Tuesday-no-metadata.parquet',
    'data/WebAttacks-Thursday-no-metadata.parquet',
    'data/DoS-Wednesday-no-metadata.parquet',
]

chunks = []
for caminho in arquivos_parquet:
    nome_arquivo = caminho.split('/')[-1]
    print(f"  Carregando: {nome_arquivo}")
    df_chunk = pd.read_parquet(caminho)
    df_chunk = optimize_memory(df_chunk)
    df_chunk = clean_data(df_chunk)
    chunks.append(df_chunk)
    del df_chunk
    gc.collect()

print("\nUnindo todos os arquivos...")
df_full = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

# 0 = Tráfego Normal (Benigno) ou 1 = Ataque Cibernético
df_full['Label'] = df_full['Label'].astype(str).str.strip().str.upper()
df_full['Label_Binary'] = df_full['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

print(f"\nDistribuição das classes após união:")
print(df_full['Label'].value_counts())
print(f"\nTotal de registros: {len(df_full):,}")

y = df_full['Label_Binary']
X = df_full.drop(columns=['Label', 'Label_Binary'])
X = X.select_dtypes(include=[np.number])

del df_full
gc.collect()

# ETAPA 3: DIVISÃO DOS DADOS (TREINO, VALIDAÇÃO E TESTE)

print("\nDIVISÃO DOS DADOS")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
del X, y; gc.collect()

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_temp, y_temp; gc.collect()

# ETAPA 4: PADRONIZAÇÃO E BALANCEAMENTO

print("\nPADRONIZAÇÃO E BALANCEAMENTO")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

feature_names = X_train.columns.tolist()

del X_train, X_val; gc.collect()

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)

del X_train_scaled, y_train; gc.collect()

# ETAPA 5: TREINAMENTO DO MODELO E BUSCA PELOS MELHORES PARÂMETROS

print("\nTREINAMENTO (RANDOM FOREST)")

modelo_base = RandomForestClassifier(random_state=42, n_jobs=-1)

parametros_para_testar = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

otimizador = GridSearchCV(
    estimator=modelo_base,
    param_grid=parametros_para_testar,
    cv=3,
    scoring='f1',
    n_jobs=1,
    verbose=2
)

otimizador.fit(X_train_res, y_train_res)

melhor_modelo = otimizador.best_estimator_
print(f"Melhor configuração encontrada: {otimizador.best_params_}")

del X_train_res, y_train_res; gc.collect()

# ETAPA 6: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE

print("\n--- AVALIAÇÃO NO TESTE FINAL ---")

y_predicao = melhor_modelo.predict(X_test_scaled)

matriz_confusao = confusion_matrix(y_test, y_predicao)
tn, fp, fn, tp = matriz_confusao.ravel()

print(f" Verdadeiros Positivos (Ataques bloqueados)  : {tp}")
print(f" Verdadeiros Negativos (Tráfego normal solto): {tn}")
print(f" Falsos Positivos (Tráfego normal bloqueado) : {fp} <- ALARME FALSO")
print(f" Falsos Negativos (Ataques não detectados)   : {fn} <- FALHA DE SEGURANÇA")

print(f"\nRecall (Taxa de captura de ataques): {recall_score(y_test, y_predicao):.4f}")
print(f"Precision (Precisão dos alarmes)   : {precision_score(y_test, y_predicao):.4f}")
print(f"F1-Score (Média harmônica)         : {f1_score(y_test, y_predicao):.4f}")

# ETAPA 7: VISUALIZAÇÃO GRÁFICA

print("\nGerando gráficos...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Desempenho da Detecção de Ataques Cibernéticos', fontsize=14, fontweight='bold')

disp_abs = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=['Benigno', 'Ataque'])
disp_abs.plot(cmap='Blues', ax=axes[0], values_format='d', colorbar=False)
axes[0].set_title('Contagens Absolutas')

matriz_normalizada = confusion_matrix(y_test, y_predicao, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=matriz_normalizada, display_labels=['Benigno', 'Ataque'])
disp_norm.plot(cmap='Oranges', ax=axes[1], values_format='.2%', colorbar=False)
axes[1].set_title('Proporções (%)')

plt.tight_layout()
plt.savefig('matriz_confusao_RF.png', dpi=300)
print("Sucesso! Imagem salva como 'matriz_confusao_RF.png'")

# ETAPA 8: IMPORTÂNCIA DAS FEATURES

print("\nIMPORTÂNCIA DAS FEATURES")

importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': melhor_modelo.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 Features Mais Importantes:")
print(importances.head(20).to_string(index=False))

top_20 = importances.head(20).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(top_20['Feature'], top_20['Importance'], color='skyblue')
plt.title('Top 20 Atributos de Rede Mais Relevantes para a Detecção', fontsize=14, pad=15)
plt.xlabel('Grau de Importância (Gini)', fontsize=12)
plt.ylabel('Atributo (Feature)', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_RF.png', dpi=300)
print("Sucesso! Imagem salva como 'feature_importance_RF.png'")