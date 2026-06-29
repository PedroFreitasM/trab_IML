import sys
from pathlib import Path

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Garante que o import funciona tanto rodando direto quanto via módulo
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from backend.preprocessamento import (
    carregar_dados, limpar, criar_targets, preparar_features,
    split, filtrar_variancia_zero, salvar_bundle, MODELS_DIR
)

TETO_BENIGN = 300_000

# Diretório de saída para imagens
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

SEED = 42


# ── Etapa 1: Carregamento e pré-processamento

print("=" * 60)
print("ETAPA 1 — DETECÇÃO BINÁRIA")
print("=" * 60)

print("\n[1/9] Carregando e limpando os 8 parquet...")
df = criar_targets(limpar(carregar_dados()))
print(f"      Shape total: {df.shape}")
print(f"\n      Distribuição target_bin:\n{df['target_bin'].value_counts()}")


# ── Etapa 2: Subamostrar BENIGN antes do split

print(f"\n[2/9] Subamostrar BENIGN para no máximo {TETO_BENIGN:,} linhas...")
df_ataques = df[df["target_bin"] == 1]
df_benign  = df[df["target_bin"] == 0]

if len(df_benign) > TETO_BENIGN:
    df_benign = df_benign.sample(n=TETO_BENIGN, random_state=SEED)

df = pd.concat([df_benign, df_ataques], ignore_index=True)
del df_benign, df_ataques
gc.collect()

print(f"      Shape após subamostragem: {df.shape}")
print(f"      Distribuição:\n{df['target_bin'].value_counts()}")


# ── Etapa 3: Separar features e target, depois split

print("\n[3/9] Separando features e realizando split 70/15/15...")
X, y = preparar_features(df, alvo="target_bin")
del df
gc.collect()

particoes = split(X, y, val=0.15, teste=0.15, seed=SEED)
del X, y
gc.collect()

X_train = particoes["X_train"]
X_val   = particoes["X_val"]
X_test  = particoes["X_test"]
y_train = particoes["y_train"]
y_val   = particoes["y_val"]
y_test  = particoes["y_test"]

print(f"      Treino : {X_train.shape[0]:,} linhas")
print(f"      Val    : {X_val.shape[0]:,} linhas")
print(f"      Teste  : {X_test.shape[0]:,} linhas")


# ── Etapa 4: Remover features de variância zero (ajustada só no treino)

print("\n[4/9] Removendo colunas de variância zero...")
X_train, X_val, X_test = filtrar_variancia_zero(X_train, X_val, X_test)
feature_names = X_train.columns.tolist()
print(f"      Features restantes: {len(feature_names)}")


# ── Etapa 5: StandardScaler (fit só no treino)

print("\n[5/9] Padronizando features (StandardScaler)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_val_sc   = scaler.transform(X_val).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

del X_train, X_val
gc.collect()


# ── Etapa 6: Treino com RandomizedSearchCV 

print("\n[6/9] Treinando Random Forest com RandomizedSearchCV...")

modelo_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

espaco_busca = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":     ["sqrt", "log2"],
}

otimizador = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=espaco_busca,
    n_iter=20,          # 20 combinações aleatórias (ajuste se tiver mais tempo)
    cv=3,
    scoring="f1",
    random_state=SEED,
    n_jobs=1,           # evita conflito com n_jobs=-1 do RF interno
    verbose=2,
)

otimizador.fit(X_train_sc, y_train)
melhor_modelo = otimizador.best_estimator_
print(f"\n      Melhor configuração: {otimizador.best_params_}")

del X_train_sc
gc.collect()


# ── Etapa 7: Ajuste de limiar via conjunto de validação 

print("\n[7/9] Ajustando limiar de decisão via validação (prioridade = Recall)...")

probas_val = melhor_modelo.predict_proba(X_val_sc)[:, 1]

melhor_limiar = 0.5
melhor_f1_val = 0.0

resultados_limiar = []
for limiar in np.arange(0.20, 0.71, 0.05):
    y_pred_val = (probas_val >= limiar).astype(int)
    rec  = recall_score(y_val, y_pred_val, zero_division=0)
    prec = precision_score(y_val, y_pred_val, zero_division=0)
    f1   = f1_score(y_val, y_pred_val, zero_division=0)
    resultados_limiar.append((round(limiar, 2), rec, prec, f1))
    # Critério: maximizar F1 com Recall mínimo de 0.97
    if rec >= 0.97 and f1 > melhor_f1_val:
        melhor_f1_val = f1
        melhor_limiar = round(limiar, 2)

print(f"\n      {'Limiar':>8} | {'Recall':>8} | {'Precision':>9} | {'F1':>8}")
print(f"      {'-'*42}")
for lim, rec, prec, f1 in resultados_limiar:
    marca = " <-- SELECIONADO" if lim == melhor_limiar else ""
    print(f"      {lim:>8.2f} | {rec:>8.4f} | {prec:>9.4f} | {f1:>8.4f}{marca}")

print(f"\n      Limiar selecionado: {melhor_limiar}")

del X_val_sc
gc.collect()


# ── Etapa 8: Avaliação final no conjunto de teste

print("\n[8/9] Avaliando no conjunto de TESTE (limiar = {})...".format(melhor_limiar))

probas_test = melhor_modelo.predict_proba(X_test_sc)[:, 1]
y_pred = (probas_test >= melhor_limiar).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n--- RESULTADOS FINAIS ---")
print(f"  Verdadeiros Positivos (Ataques detectados)    : {tp:,}")
print(f"  Verdadeiros Negativos (Tráfego normal liberado): {tn:,}")
print(f"  Falsos Positivos (Normal bloqueado)            : {fp:,}  <- ALARME FALSO")
print(f"  Falsos Negativos (Ataques não detectados)      : {fn:,}  <- FALHA DE SEGURANÇA")

print("\n" + classification_report(y_test, y_pred, target_names=["Benigno", "Ataque"]))


# ── Etapa 8a: Gráficos 

print("      Gerando gráficos...")

# Matriz de confusão dupla (absoluta + normalizada)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Desempenho da Detecção Binária de Ataques Cibernéticos",
             fontsize=14, fontweight="bold")

ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Benigno", "Ataque"]).plot(
    cmap="Blues", ax=axes[0], values_format="d", colorbar=False)
axes[0].set_title("Contagens Absolutas")

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                       display_labels=["Benigno", "Ataque"]).plot(
    cmap="Oranges", ax=axes[1], values_format=".2%", colorbar=False)
axes[1].set_title("Proporções (%)")

plt.tight_layout()
caminho_cm = IMAGES_DIR / "mat_confusao_deteccao.png"
plt.savefig(caminho_cm, dpi=300)
plt.close()
print(f"      Matriz de confusão salva em: {caminho_cm}")

# Importância das features (top 20)
importancias = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": melhor_modelo.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\n      Top 20 features mais importantes:")
print(importancias.head(20).to_string(index=False))

top20 = importancias.head(20).sort_values("Importance", ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(top20["Feature"], top20["Importance"], color="skyblue")
plt.title("Top 20 Atributos Mais Relevantes para Detecção Binária", fontsize=14, pad=15)
plt.xlabel("Grau de Importância (Gini)", fontsize=12)
plt.ylabel("Atributo (Feature)", fontsize=12)
plt.tight_layout()
caminho_fi = IMAGES_DIR / "feature_importance_deteccao.png"
plt.savefig(caminho_fi, dpi=300)
plt.close()
print(f"      Importância de features salva em: {caminho_fi}")

del X_test_sc
gc.collect()


# ── Etapa 9: Salvar bundle (Contrato 2 do TASKS.md)

print("\n[9/9] Salvando bundle em models/etapa1.joblib...")

salvar_bundle(
    caminho=MODELS_DIR / "etapa1.joblib",
    modelo=melhor_modelo,
    colunas=feature_names,
    scaler=scaler,
    classes=[0, 1],
    limiar=melhor_limiar,
)

print(f"      Bundle salvo em: {MODELS_DIR / 'etapa1.joblib'}")
print("\n[CONCLUÍDO] Etapa 1 — Detecção Binária finalizada com sucesso.")