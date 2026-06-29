import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from backend.preprocessamento import (
    ARQUIVOS,
    DATA_DIR,
    MODELS_DIR,
    criar_targets,
    filtrar_variancia_zero,
    limpar,
    preparar_features,
    salvar_bundle,
    split,
)

#  Diretório de imagens 
IMAGES_DIR = ROOT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

#  CONFIGURAÇÃO GLOBAL ─

RANDOM_STATE = 42

# Fração de amostragem estratificada POR ARQUIVO, antes do concat.
# Mantém o volume total gerenciável com 8 GB de RAM.
# None = todos os dados (pode estourar RAM). 0.3 = 30% por arquivo.
SAMPLE_FRAC = 0.3


#  FUNÇÃO AUXILIAR DE BALANCEAMENTO 

def undersample_inplace(
    X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE
) -> tuple:
    """Undersample da classe majoritária sem duplicar X na memória."""
    rng = np.random.default_rng(random_state)
    idx_majority = np.where(y == 0)[0]
    idx_minority = np.where(y == 1)[0]
    idx_sampled = rng.choice(idx_majority, size=len(idx_minority), replace=False)
    idx_balanced = np.concatenate([idx_sampled, idx_minority])
    rng.shuffle(idx_balanced)
    return X[idx_balanced], y[idx_balanced]


#  ETAPA 2: CARREGAMENTO ARQUIVO POR ARQUIVO ─
#
# POR QUE NÃO USAR carregar_dados() AQUI:
# carregar_dados() lê os 8 parquet inteiros e concatena tudo em RAM de uma
# vez. Com 2,3M linhas em float64 isso ultrapassa facilmente 8 GB e o
# processo é encerrado pelo SO ("Killed").
#
# SOLUÇÃO: ler, limpar, binarizar e amostrar cada arquivo individualmente
# — descartando-o da RAM antes de carregar o próximo. Ainda assim reutiliza
# limpar() e criar_targets() do preprocessamento.py.

print("=" * 60)
print("ETAPA 2 — CARREGAMENTO DOS DADOS (arquivo por arquivo)")
print("=" * 60)

partes: list[pd.DataFrame] = []

for nome_arquivo in ARQUIVOS:
    caminho = DATA_DIR / nome_arquivo
    print(f"   Carregando: {nome_arquivo} ...", end=" ", flush=True)

    # 1. Leitura
    df_parte = pd.read_parquet(caminho, engine="pyarrow")

    # 2. Limpeza via preprocessamento.py (normaliza colunas, remove inf/NaN)
    df_parte = limpar(df_parte)

    # 3. Criação de targets via preprocessamento.py
    df_parte = criar_targets(df_parte)

    # 4. Descarta colunas de texto que não serão usadas no treino
    #    (Label já foi consumida por criar_targets; target_tipo não é usado aqui)
    colunas_descartar = [c for c in ["Label", "target_tipo"] if c in df_parte.columns]
    df_parte.drop(columns=colunas_descartar, inplace=True)

    # 5. Converte numéricas para float32 ANTES de qualquer outra operação
    #    (reduz RAM ~50% vs float64 padrão do pandas)
    for col in df_parte.select_dtypes(include=[np.number]).columns:
        if col == "target_bin":
            continue
        df_parte[col] = df_parte[col].astype(np.float32)

    # 6. Amostragem estratificada por arquivo (mantém proporção benigno/ataque)
    if SAMPLE_FRAC is not None and SAMPLE_FRAC < 1.0:
        df_parte = (
            df_parte
            .groupby("target_bin", group_keys=False)
            .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )

    n_ben = int((df_parte["target_bin"] == 0).sum())
    n_atk = int((df_parte["target_bin"] == 1).sum())
    print(f"{len(df_parte):,} linhas  (benigno={n_ben:,}  ataque={n_atk:,})")

    partes.append(df_parte)
    del df_parte
    gc.collect()

print("\nUnindo todos os arquivos...")
df = pd.concat(partes, ignore_index=True)
del partes
gc.collect()

print(f"Total após concat: {len(df):,} linhas  |  {df.shape[1]} colunas")
print(f"Benigno (0): {(df['target_bin'] == 0).sum():,}")
print(f"Ataque  (1): {(df['target_bin'] == 1).sum():,}")


#  ETAPA 3: SEPARAÇÃO DE FEATURES E SPLIT 70 / 15 / 15 

print("\n" + "=" * 60)
print("ETAPA 3 — SEPARAÇÃO DE FEATURES E SPLIT 70 / 15 / 15")
print("=" * 60)

# preparar_features remove colunas de vazamento e separa X de y
X, y = preparar_features(df, alvo="target_bin")
del df
gc.collect()

particoes = split(X, y, val=0.15, teste=0.15, seed=RANDOM_STATE)
del X, y
gc.collect()

X_train = particoes["X_train"]
X_val   = particoes["X_val"]
X_test  = particoes["X_test"]
y_train = particoes["y_train"]
y_val   = particoes["y_val"]
y_test  = particoes["y_test"]

print(f"Treino    : {len(X_train):,} amostras")
print(f"Validação : {len(X_val):,} amostras")
print(f"Teste     : {len(X_test):,} amostras")


#  ETAPA 4: VARIÂNCIA ZERO + BALANCEAMENTO + PADRONIZAÇÃO 

print("\n" + "=" * 60)
print("ETAPA 4 — FILTRO DE VARIÂNCIA ZERO / BALANCEAMENTO / PADRONIZAÇÃO")
print("=" * 60)

print("Removendo colunas de variância zero (calculado só no treino)...")
X_train, X_val, X_test = filtrar_variancia_zero(X_train, X_val, X_test)
feature_names = X_train.columns.tolist()
print(f"Features restantes: {len(feature_names)}")

# Converte para numpy float32 (pandas já está em float32, mas garante homogeneidade)
print("Convertendo para arrays numpy...")
X_train_np = X_train.values.astype(np.float32); del X_train; gc.collect()
X_val_np   = X_val.values.astype(np.float32);   del X_val;   gc.collect()
X_test_np  = X_test.values.astype(np.float32);  del X_test;  gc.collect()
y_train_np = y_train.values.astype(np.int8);    del y_train; gc.collect()
y_test_np  = y_test.values.astype(np.int8);     del y_test;  gc.collect()
y_val_np   = y_val.values.astype(np.int8);      del y_val;   gc.collect()

print("Balanceando classes no treino (undersample da maioria)...")
X_train_bal, y_train_bal = undersample_inplace(X_train_np, y_train_np, RANDOM_STATE)
del X_train_np, y_train_np
gc.collect()
print(f"  Benigno após balanceamento: {(y_train_bal == 0).sum():,}")
print(f"  Ataque  após balanceamento: {(y_train_bal == 1).sum():,}")

print("Padronizando features (StandardScaler — fit APENAS no treino)...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal).astype(np.float32)
X_val_scaled   = scaler.transform(X_val_np).astype(np.float32);  del X_val_np;  gc.collect()
X_test_scaled  = scaler.transform(X_test_np).astype(np.float32); del X_test_np; gc.collect()
del X_train_bal
gc.collect()


#  ETAPA 5: TREINAMENTO (REGRESSÃO LOGÍSTICA + GRIDSEARCHCV) ─

print("\n" + "=" * 60)
print("ETAPA 5 — TREINAMENTO (REGRESSÃO LOGÍSTICA + GRIDSEARCHCV)")
print("=" * 60)

modelo_base = LogisticRegression(
    random_state=RANDOM_STATE,
    n_jobs=2,
    penalty="l2",
    solver="saga",
    max_iter=500,
)

parametros = {
    "C"       : [0.1, 1, 10],
    "max_iter": [500],
}

print("Iniciando GridSearchCV (cv=3, scoring=f1, n_jobs=-1)...")
otimizador = GridSearchCV(
    estimator  = modelo_base,
    param_grid = parametros,
    cv         = 3,
    scoring    = "f1",
    n_jobs     = 2,
    verbose    = 1,
    refit      = True,
)
otimizador.fit(X_train_scaled, y_train_bal)

melhor_modelo = otimizador.best_estimator_
print(f"\nMelhor configuração: {otimizador.best_params_}")

del X_train_scaled, y_train_bal
gc.collect()


#  ETAPA 6: AVALIAÇÃO NA VALIDAÇÃO ─

print("\n" + "=" * 60)
print("ETAPA 6 — AVALIAÇÃO NA VALIDAÇÃO")
print("=" * 60)

y_val_pred = melhor_modelo.predict(X_val_scaled)
print(f"  Recall    (val): {recall_score(y_val_np, y_val_pred):.4f}")
print(f"  Precision (val): {precision_score(y_val_np, y_val_pred):.4f}")
print(f"  F1        (val): {f1_score(y_val_np, y_val_pred):.4f}")
del X_val_scaled, y_val_pred, y_val_np
gc.collect()


#  ETAPA 7: AVALIAÇÃO NO TESTE FINAL ─

print("\n" + "=" * 60)
print("ETAPA 7 — AVALIAÇÃO NO TESTE FINAL")
print("=" * 60)

y_pred      = melhor_modelo.predict(X_test_scaled)
matriz_conf = confusion_matrix(y_test_np, y_pred)
tn, fp, fn, tp = matriz_conf.ravel()

print(f"  Verdadeiros Positivos (Ataques detectados)   : {tp:,}")
print(f"  Verdadeiros Negativos (Tráfego normal solto) : {tn:,}")
print(f"  Falsos Positivos (Tráfego normal bloqueado)  : {fp:,}  ← ALARME FALSO")
print(f"  Falsos Negativos (Ataques não detectados)    : {fn:,}  ← FALHA DE SEGURANÇA")

recall_val    = recall_score(y_test_np, y_pred)
precision_val = precision_score(y_test_np, y_pred)
f1_val        = f1_score(y_test_np, y_pred)

print(f"\n  Recall    : {recall_val:.4f}")
print(f"  Precision : {precision_val:.4f}")
print(f"  F1-Score  : {f1_val:.4f}")


#  ETAPA 8: SALVAR BUNDLE DE INFERÊNCIA 

print("\n" + "=" * 60)
print("ETAPA 8 — SALVANDO BUNDLE (models/logreg_etapa1.joblib)")
print("=" * 60)

bundle_path = MODELS_DIR / "logreg_etapa1.joblib"
salvar_bundle(
    caminho = bundle_path,
    modelo  = melhor_modelo,
    colunas = feature_names,
    scaler  = scaler,
    classes = [0, 1],
    limiar  = 0.5,
)
print(f"Bundle salvo em: {bundle_path}")

#  ETAPA 9: TOP 10 FEATURES MAIS IMPORTANTES ─

print("\n" + "=" * 60)
print("ETAPA 9 — TOP 10 FEATURES MAIS IMPORTANTES")
print("=" * 60)

coeficientes   = melhor_modelo.coef_[0]
importancia_df = pd.DataFrame({
    "Feature"    : feature_names,
    "Coeficiente": coeficientes,
    "Abs_Coef"   : np.abs(coeficientes),
}).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

top10 = importancia_df.head(10)
print(f"\n  {'#':>2}  {'Feature':<40} {'Coeficiente':>12}  Direção")
print("  " + "-" * 70)
for i, row in top10.iterrows():
    sinal = "▲ Indica ataque" if row["Coeficiente"] > 0 else "▼ Indica benigno"
    print(f"  {i+1:>2}. {row['Feature']:<40} {row['Coeficiente']:>+12.4f}  {sinal}")

csv_path = IMAGES_DIR / "feature_importance_logreg.csv"
importancia_df.to_csv(csv_path, index=False)
print(f"\nRanking completo salvo em: {csv_path}")


#  ETAPA 10: VISUALIZAÇÃO GRÁFICA 

print("\nGerando gráficos...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle(
    "Detecção de Ataques Cibernéticos — CICIDS-2017 (8 datasets)\n"
    f"Regressão Logística  |  F1={f1_val:.4f}  |  "
    f"Recall={recall_val:.4f}  |  Precision={precision_val:.4f}",
    fontsize=13,
    fontweight="bold",
)

ConfusionMatrixDisplay(matriz_conf, display_labels=["Benigno", "Ataque"]).plot(
    cmap="Blues", ax=axes[0], values_format="d", colorbar=False
)
axes[0].set_title("Matriz de Confusão\n(Contagens Absolutas)")

matriz_norm = confusion_matrix(y_test_np, y_pred, normalize="true")
ConfusionMatrixDisplay(matriz_norm, display_labels=["Benigno", "Ataque"]).plot(
    cmap="Oranges", ax=axes[1], values_format=".2%", colorbar=False
)
axes[1].set_title("Matriz de Confusão\n(Proporções por Classe Real)")

cores     = ["#d73027" if c > 0 else "#4575b4" for c in top10["Coeficiente"]]
feat_inv  = top10["Feature"].values[::-1]
coefs_inv = top10["Coeficiente"].values[::-1]
cores_inv = cores[::-1]
bars = axes[2].barh(feat_inv, coefs_inv, color=cores_inv, edgecolor="white", linewidth=0.5)
axes[2].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_xlabel("Coeficiente (escala padronizada)", fontsize=10)
axes[2].set_title(
    "Top 10 Features mais Importantes\n"
    "(vermelho = indica ataque | azul = indica benigno)"
)
axes[2].tick_params(axis="y", labelsize=8)

max_abs = top10["Abs_Coef"].max()
for bar, val in zip(bars, coefs_inv):
    offset = 0.02 * max_abs
    ha, x_pos = ("left", val + offset) if val >= 0 else ("right", val - offset)
    axes[2].text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}",
        va="center",
        ha=ha,
        fontsize=8,
    )

plt.tight_layout()
png_path = IMAGES_DIR / "resultado_regressao_logistica.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {png_path}")
print("\nConcluído com sucesso!")