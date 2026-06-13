import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
)

# ETAPA 1: FUNÇÕES AUXILIARES (PREPARAÇÃO)

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o tamanho que os dados numéricos ocupam na memória RAM.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        # Se tiver dados vazios (NaN), ignora-se
        if df[col].isnull().any():
            continue
            
        valor_min = df[col].min()
        valor_max = df[col].max()
        tipo_coluna = df[col].dtype

        # Se for um número inteiro 
        if np.issubdtype(tipo_coluna, np.integer):
            if valor_min > np.iinfo(np.int8).min and valor_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif valor_min > np.iinfo(np.int16).min and valor_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif valor_min > np.iinfo(np.int32).min and valor_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
        # Se for um número decimal
        elif np.issubdtype(tipo_coluna, np.floating):
            if valor_min > np.finfo(np.float32).min and valor_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove linhas com erros matemáticos (Infinito) ou vazias (NaN). """
    total_antes = len(df)
    
    df.columns = df.columns.str.strip() # Remove espaços extras no nome das colunas
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df.dropna(inplace=True) 
    
    total_depois = len(df)
    print(f"   [Limpeza] Linhas removidas: {total_antes - total_depois}")
    return df

# ETAPA 2: CARREGAMENTO E UNIÃO DOS DADOS

print("CARREGAMENTO DOS DADOS")

# Carrega os 3 arquivos separados
df_benign   = pd.read_parquet('data/Benign-Monday-no-metadata.parquet')
df_ddos     = pd.read_parquet('data/DDoS-Friday-no-metadata.parquet')
df_portscan = pd.read_parquet('data/Portscan-Friday-no-metadata.parquet')

# Une os 3 arquivos em um dataframe
df_full = pd.concat([df_benign, df_ddos, df_portscan], ignore_index=True)

# Libera espaço na RAM
del df_benign, df_ddos, df_portscan
gc.collect() 

print("Otimizando e limpando os dados unidos...")
df_full = optimize_memory(df_full)
df_full = clean_data(df_full)

# 0 = Tráfego Normal (Benigno) ou 1 = Ataque Cibernético
df_full['Label'] = df_full['Label'].astype(str).str.strip().str.upper()
df_full['Label_Binary'] = df_full['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

y = df_full['Label_Binary']
X = df_full.drop(columns=['Label', 'Label_Binary'])
X = X.select_dtypes(include=[np.number]) 

del df_full
gc.collect()

# ETAPA 3: DIVISÃO DOS DADOS (TREINO, VALIDAÇÃO E TESTE)

print("\nDIVISÃO DOS DADOS")

# Separa 70% para Treino e guarda 30% como Temporário
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
del X, y; gc.collect()

# Pega-se os 30% temporários e divide-se ao meio (15% Validação, 15% Teste)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_temp, y_temp; gc.collect()

# ETAPA 4: PADRONIZAÇÃO E BALANCEAMENTO

print("\nPADRONIZAÇÃO E BALANCEAMENTO")

# Coloca-se todas as grandezas na mesma escala (para treino, validação e teste)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32) 
X_val_scaled   = scaler.transform(X_val).astype(np.float32)       
X_test_scaled  = scaler.transform(X_test).astype(np.float32)      

del X_train, X_val; gc.collect()

# Reduz os dados normais para ficarem na mesma quantidade dos ataques
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)

del X_train_scaled, y_train; gc.collect()

# ETAPA 5: TREINAMENTO DO MODELO E BUSCA PELOS MELHORES PARÂMETROS

print("\nTREINAMENTO (REGRESSÃO LOGÍSTICA)")

# Modelo linear a ser usado
modelo_base = LogisticRegression(random_state=42, n_jobs=1, penalty='l2')

# Diferentes hiperparâmetros para o computador testar
parametros_para_testar = {
    'C': [0.01, 0.1, 1, 10], 
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000]
}

# GridSearchCV: Vai testar todas as combinações de parâmetros acima para achar a melhor
otimizador = GridSearchCV(
    estimator=modelo_base,
    param_grid=parametros_para_testar,
    cv=3,            
    scoring='f1',    # A métrica que ele vai tentar maximizar é o F1-Score
    n_jobs=1         # Usa 1 núcleo do processador para evitar estourar a memória RAM
)

# Treina o modelo com os dados balanceados e padronizados
otimizador.fit(X_train_res, y_train_res) 

# Melhor modelo encontrado pelo GridSearchCV
melhor_modelo = otimizador.best_estimator_ 
print(f"Melhor configuração encontrada: {otimizador.best_params_}")

del X_train_res, y_train_res; gc.collect()

# ETAPA 6: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE

print("\n--- AVALIAÇÃO NO TESTE FINAL ---")

# Realiza-se a predição usando o melhor modelo encontrado e os dados de teste
y_predicao = melhor_modelo.predict(X_test_scaled)

# Calculam-se as métricas de acerto
matriz_confusao = confusion_matrix(y_test, y_predicao)
tn, fp, fn, tp = matriz_confusao.ravel() # Verdadeiros/Falsos Positivos/Negativos

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

# Gráfico 1: Quantidade Absoluta
disp_abs = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=['Benigno', 'Ataque'])
disp_abs.plot(cmap='Blues', ax=axes[0], values_format='d', colorbar=False)
axes[0].set_title('Contagens Absolutas')

# Gráfico 2: Porcentagem 
matriz_normalizada = confusion_matrix(y_test, y_predicao, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=matriz_normalizada, display_labels=['Benigno', 'Ataque'])
disp_norm.plot(cmap='Oranges', ax=axes[1], values_format='.2%', colorbar=False)
axes[1].set_title('Proporções (%)')

plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=300)