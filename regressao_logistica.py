import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    ConfusionMatrixDisplay,
)

# CONFIGURAÇÃO GLOBAL

RANDOM_STATE   = 42
# Fração de amostragem POR ARQUIVO antes do concat.
# None = usar todos os dados (comportamento original).
# 0.3  = usar 30% de cada arquivo → reduz RAM e tempo ~3×.
# Recomendado para testes: 0.1 – 0.3; para treino final: None.
SAMPLE_FRAC    = 0.3

# ETAPA 1: FUNÇÕES AUXILIARES

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com inf ou NaN e normaliza nomes de colunas."""
    total_antes = len(df)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"   [Limpeza] Linhas removidas: {total_antes - len(df):,}")
    return df


def undersample_inplace(X: np.ndarray, y: np.ndarray,
                        random_state: int = RANDOM_STATE) -> tuple:
    """
    Undersample da classe majoritária de forma in-place:
    constrói apenas o array de índices balanceados, sem duplicar X.
    """
    rng = np.random.default_rng(random_state)
    idx_majority  = np.where(y == 0)[0]
    idx_minority  = np.where(y == 1)[0]
    n_minority    = len(idx_minority)
    idx_majority_sampled = rng.choice(idx_majority, size=n_minority, replace=False)
    idx_balanced  = np.concatenate([idx_majority_sampled, idx_minority])
    rng.shuffle(idx_balanced)
    # Fatia direta — sem cópia adicional se X for C-contiguous
    return X[idx_balanced], y[idx_balanced]


# ETAPA 2: LEITURA OTIMIZADA DOS 8 DATASETS

print("=" * 60)
print("ETAPA 2 — CARREGAMENTO DOS DADOS (8 arquivos)")
print("=" * 60)

arquivos = {
    "Benign (Monday)"        : "data/Benign-Monday-no-metadata.parquet",
    "Botnet (Friday)"        : "data/Botnet-Friday-no-metadata.parquet",
    "Bruteforce (Tuesday)"   : "data/Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS (Friday)"          : "data/DDoS-Friday-no-metadata.parquet",
    "DoS (Wednesday)"        : "data/DoS-Wednesday-no-metadata.parquet",
    "Infiltration (Thursday)": "data/Infiltration-Thursday-no-metadata.parquet",
    "Portscan (Friday)"      : "data/Portscan-Friday-no-metadata.parquet",
    "WebAttacks (Thursday)"  : "data/WebAttacks-Thursday-no-metadata.parquet",
}


# Detectar colunas disponíveis no primeiro arquivo para leitura seletiva.

_colunas_sample = pd.read_parquet(list(arquivos.values())[0], engine="pyarrow").columns.tolist()
_colunas_sample = [c.strip() for c in _colunas_sample]

# Mantém 'Label' + todas as numéricas (descobre a lista real antes do concat)
# O filtro definitivo de dtype ocorre após a leitura por arquivo.
LABEL_COL = "Label"

partes = []
for nome, caminho in arquivos.items():
    print(f"   Carregando: {nome} ...", end=" ", flush=True)

    # ── Leitura seletiva (pyarrow lê só colunas pedidas do parquet) ──
    df_parte = pd.read_parquet(caminho, engine="pyarrow")
    df_parte.columns = df_parte.columns.str.strip()

    # ── Binarização antecipada (mantém Label_Binary, descarta Label texto) ──
    df_parte[LABEL_COL] = df_parte[LABEL_COL].astype(str).str.strip().str.upper()
    df_parte["Label_Binary"] = (df_parte[LABEL_COL] != "BENIGN").astype(np.int8)
    df_parte.drop(columns=[LABEL_COL], inplace=True)

    # ── Conversão para float32 ANTES de qualquer operação pesada ──
    for col in df_parte.select_dtypes(include=[np.number]).columns:
        if col == "Label_Binary":
            continue
        if not df_parte[col].isnull().any():
            df_parte[col] = df_parte[col].astype(np.float32)

    # ── Limpeza por arquivo (antes de concatenar — mais eficiente) ──
    df_parte.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_parte.dropna(inplace=True)

    # ── Amostragem estratificada por arquivo (reduz volume antes do concat) ──
    if SAMPLE_FRAC is not None and SAMPLE_FRAC < 1.0:
        df_parte = df_parte.groupby("Label_Binary", group_keys=False).apply(
            lambda g: g.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        )

    print(f"{len(df_parte):,} linhas  "
          f"(benign={int((df_parte['Label_Binary']==0).sum()):,}  "
          f"ataque={int((df_parte['Label_Binary']==1).sum()):,})")
    partes.append(df_parte)
    del df_parte
    gc.collect()

print("\nUnindo todos os arquivos...")
df_full = pd.concat(partes, ignore_index=True)
del partes
gc.collect()

print(f"Total após concat: {len(df_full):,} linhas  |  {df_full.shape[1]} colunas")
print(f"Benigno (0): {(df_full['Label_Binary']==0).sum():,}")
print(f"Ataque  (1): {(df_full['Label_Binary']==1).sum():,}")

y_all         = df_full["Label_Binary"].values.astype(np.int8)
X_df          = df_full.drop(columns=["Label_Binary"]).select_dtypes(include=[np.number])
feature_names = X_df.columns.tolist()
X_all         = X_df.values.astype(np.float32)   # garante float32 homogêneo
del df_full, X_df
gc.collect()

# ETAPA 3: DIVISÃO DOS DADOS (70% Treino | 15% Validação | 15% Teste)

print("\n" + "=" * 60)
print("ETAPA 3 — DIVISÃO DOS DADOS")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all
)
del X_all, y_all
gc.collect()

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
del X_temp, y_temp
gc.collect()

print(f"Treino    : {len(X_train):,} amostras")
print(f"Validação : {len(X_val):,} amostras")
print(f"Teste     : {len(X_test):,} amostras")

# ETAPA 4: BALANCEAMENTO → PADRONIZAÇÃO

print("\n" + "=" * 60)
print("ETAPA 4 — BALANCEAMENTO E PADRONIZAÇÃO")
print("=" * 60)

print("Balanceando classes no treino (undersample)...")
X_train_bal, y_train_bal = undersample_inplace(X_train, y_train, RANDOM_STATE)
del X_train, y_train
gc.collect()

print(f"  Benigno após balanceamento: {(y_train_bal == 0).sum():,}")
print(f"  Ataque  após balanceamento: {(y_train_bal == 1).sum():,}")

print("Padronizando features (StandardScaler)...")
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train_bal).astype(np.float32)
del X_train_bal
gc.collect()

X_val_scaled  = scaler.transform(X_val).astype(np.float32);  del X_val;  gc.collect()
X_test_scaled = scaler.transform(X_test).astype(np.float32); del X_test; gc.collect()

# ETAPA 5: TREINAMENTO (REGRESSÃO LOGÍSTICA + GRIDSEARCHCV OTIMIZADO)

print("\n" + "=" * 60)
print("ETAPA 5 — TREINAMENTO (REGRESSÃO LOGÍSTICA)")
print("=" * 60)

modelo_base = LogisticRegression(
    random_state=RANDOM_STATE,
    n_jobs=-1,          # usa todos os núcleos no fit individual também
    penalty="l2",
    solver="saga",      # ← muito mais rápido que lbfgs/liblinear em datasets grandes
    max_iter=500,       # saga converge mais cedo
)

# Grid reduzido: 3 × 1 × 3 folds = 9 fits (era 4 × 2 × 3 = 24)
parametros_para_testar = {
    "C"       : [0.1, 1, 10],
    "max_iter": [500],
}

print("Iniciando GridSearchCV (cv=3, scoring=f1, n_jobs=-1)...")
otimizador = GridSearchCV(
    estimator  = modelo_base,
    param_grid = parametros_para_testar,
    cv         = 3,
    scoring    = "f1",
    n_jobs     = -1,    # ← paraleliza os folds: 9 fits rodam ao mesmo tempo
    verbose    = 1,
    refit      = True,
)
otimizador.fit(X_train_scaled, y_train_bal)

melhor_modelo = otimizador.best_estimator_
print(f"\nMelhor configuração encontrada: {otimizador.best_params_}")

del X_train_scaled, y_train_bal
gc.collect()

# ETAPA 6: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE

print("\n" + "=" * 60)
print("ETAPA 6 — AVALIAÇÃO NO TESTE FINAL")
print("=" * 60)

y_predicao    = melhor_modelo.predict(X_test_scaled)
matriz_conf   = confusion_matrix(y_test, y_predicao)
tn, fp, fn, tp = matriz_conf.ravel()

print(f"  Verdadeiros Positivos (Ataques detectados)   : {tp:,}")
print(f"  Verdadeiros Negativos (Tráfego normal solto) : {tn:,}")
print(f"  Falsos Positivos (Tráfego normal bloqueado)  : {fp:,}  ← ALARME FALSO")
print(f"  Falsos Negativos (Ataques não detectados)    : {fn:,}  ← FALHA DE SEGURANÇA")

recall_val    = recall_score(y_test, y_predicao)
precision_val = precision_score(y_test, y_predicao)
f1_val        = f1_score(y_test, y_predicao)

print(f"\n  Recall    : {recall_val:.4f}")
print(f"  Precision : {precision_val:.4f}")
print(f"  F1-Score  : {f1_val:.4f}")

# ETAPA 7: FEATURES MAIS IMPORTANTES

print("\n" + "=" * 60)
print("ETAPA 7 — TOP 10 FEATURES MAIS IMPORTANTES")
print("=" * 60)

coeficientes = melhor_modelo.coef_[0]
importancia_df = pd.DataFrame({
    "Feature"    : feature_names,
    "Coeficiente": coeficientes,
    "Abs_Coef"   : np.abs(coeficientes),
}).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

top10 = importancia_df.head(10)
print("\nTop 10 features com maior impacto:\n")
print(f"  {'#':>2}  {'Feature':<40} {'Coeficiente':>12}  Direção")
print("  " + "-" * 70)
for i, row in top10.iterrows():
    sinal = "▲ Indica ataque" if row["Coeficiente"] > 0 else "▼ Indica benigno"
    print(f"  {i+1:>2}. {row['Feature']:<40} {row['Coeficiente']:>+12.4f}  {sinal}")

importancia_df.to_csv("feature_importance.csv", index=False)
print("\nRanking completo salvo em: feature_importance.csv")

# ETAPA 8: VISUALIZAÇÃO GRÁFICA

print("\nGerando gráficos...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle(
    "Detecção de Ataques Cibernéticos — CICIDS-2017 (8 datasets)\n"
    f"Regressão Logística  |  F1={f1_val:.4f}  |  "
    f"Recall={recall_val:.4f}  |  Precision={precision_val:.4f}",
    fontsize=13, fontweight="bold",
)

ConfusionMatrixDisplay(matriz_conf, display_labels=["Benigno","Ataque"]).plot(
    cmap="Blues", ax=axes[0], values_format="d", colorbar=False
)
axes[0].set_title("Matriz de Confusão\n(Contagens Absolutas)")

matriz_norm = confusion_matrix(y_test, y_predicao, normalize="true")
ConfusionMatrixDisplay(matriz_norm, display_labels=["Benigno","Ataque"]).plot(
    cmap="Oranges", ax=axes[1], values_format=".2%", colorbar=False
)
axes[1].set_title("Matriz de Confusão\n(Proporções por Classe Real)")

cores       = ["#d73027" if c > 0 else "#4575b4" for c in top10["Coeficiente"]]
feat_inv    = top10["Feature"].values[::-1]
coefs_inv   = top10["Coeficiente"].values[::-1]
cores_inv   = cores[::-1]
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
    axes[2].text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}", va="center", ha=ha, fontsize=8)

plt.tight_layout()
plt.savefig("resultado_cicids2017.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo em: resultado_cicids2017.png")
print("\nConcluído com sucesso!")