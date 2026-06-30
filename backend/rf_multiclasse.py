import sys
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

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

# ── Caminhos de saída

PDF_PATH = ROOT_DIR / "images" / "rf_multiclasse.pdf"
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Configuração global 

RANDOM_STATE = 42
SAMPLE_FRAC  = 0.3   

PALETTE = {
    "Benign":       "#2196F3",
    "DDoS":         "#F44336",
    "DoS":          "#FF9800",
    "PortScan":     "#9C27B0",
    "Botnet":       "#009688",
    "Bruteforce":   "#795548",
    "Infiltration": "#607D8B",
    "WebAttacks":   "#E91E63",
}

# ── ETAPA 2 — Carregamento dos dados

print("\nETAPA 2 — CARREGAMENTO DOS DADOS (arquivo por arquivo)\n")

partes: list[pd.DataFrame] = []
for nome_arquivo in ARQUIVOS:
    caminho = DATA_DIR / nome_arquivo
    if not caminho.exists():
        print(f"   [AVISO] Arquivo não encontrado: {nome_arquivo}")
        continue
    print(f"   Carregando: {nome_arquivo} ...", end=" ", flush=True)

    df_parte = pd.read_parquet(caminho, engine="pyarrow")
    df_parte = limpar(df_parte)
    df_parte = criar_targets(df_parte)

    # Mantém apenas target_tipo (multiclasse)
    colunas_descartar = [c for c in ["Label", "target_bin"] if c in df_parte.columns]
    df_parte.drop(columns=colunas_descartar, inplace=True)

    # Reduz precisão para economizar RAM
    for col in df_parte.select_dtypes(include=[np.number]).columns:
        if col != "target_tipo":
            df_parte[col] = df_parte[col].astype(np.float32)

    # Amostragem estratificada por arquivo (igual ao arquivo de Regressão Logística)
    if SAMPLE_FRAC is not None and SAMPLE_FRAC < 1.0:
        df_parte = (
            df_parte.groupby("target_tipo", group_keys=True)
            .apply(
                lambda g: g.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE),
                include_groups=False,
            )
            .reset_index(level=0)
            .reset_index(drop=True)
        )
    print(f"{len(df_parte):,} linhas")
    partes.append(df_parte)
    del df_parte
    gc.collect()

df = pd.concat(partes, ignore_index=True)
del partes
gc.collect()
print(f"\nTotal consolidado: {len(df):,} linhas")
print("\nDistribuição de classes (target_tipo):")
print(df["target_tipo"].value_counts())

# ── ETAPA 3 — Separação de features e split 

print("\nETAPA 3 — SEPARAÇÃO DE FEATURES E SPLIT\n")

X, y = preparar_features(df, alvo="target_tipo")
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

# ── ETAPA 4 — Pré-processamento final

print("\nETAPA 4 — PRÉ-PROCESSAMENTO FINAL\n")

X_train, X_val, X_test = filtrar_variancia_zero(X_train, X_val, X_test)
feature_names = X_train.columns.tolist()
print(f"Features após remoção de variância zero: {len(feature_names)}")

# RF não precisa de StandardScaler (invariante a escala),
# mas codificamos os rótulos para compatibilidade com o LabelEncoder
# (permite recuperar os nomes das classes no bundle e nos gráficos)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.values); del y_train
y_val_enc   = le.transform(y_val.values);       del y_val
y_test_enc  = le.transform(y_test.values);      del y_test
gc.collect()

class_names = list(le.classes_)
n_classes   = len(class_names)
print(f"Classes detectadas ({n_classes}): {class_names}")

# ── ETAPA 5 — Treinamento com RandomizedSearchCV 


print("\nETAPA 5 — TREINAMENTO (RANDOM FOREST MULTICLASSE)\n")

modelo_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

espaco_busca = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

otimizador = RandomizedSearchCV(
    modelo_base,
    espaco_busca,
    n_iter=20,
    cv=3,
    scoring="f1_macro",   # macro = peso igual para classes raras (PLAN.md Fase 3)
    random_state=RANDOM_STATE,
    n_jobs=1,             # evita conflito com n_jobs=-1 do RF interno
    verbose=2,
)
otimizador.fit(X_train.values.astype(np.float32), y_train_enc)

melhor_modelo = otimizador.best_estimator_
print(f"\nMelhor configuração: {otimizador.best_params_}")
print(f"F1-Macro CV (treino): {otimizador.best_score_:.4f}")

del X_train
gc.collect()

# ── ETAPA 6 — Avaliação no conjunto de validação (diagnóstico) 

print("\nETAPA 6 — AVALIAÇÃO NA VALIDAÇÃO\n")

y_val_pred = melhor_modelo.predict(X_val.values.astype(np.float32))
print("Relatório por classe (validação):")
print(classification_report(y_val_enc, y_val_pred, target_names=class_names, zero_division=0))

del X_val, y_val_enc, y_val_pred
gc.collect()

# ── ETAPA 7 — Avaliação final no conjunto de teste 

print("\nETAPA 7 — AVALIAÇÃO FINAL (TESTE)\n")

X_test_np = X_test.values.astype(np.float32)
del X_test
gc.collect()

y_pred = melhor_modelo.predict(X_test_np)

f1_macro  = f1_score(y_test_enc, y_pred, average="macro",    zero_division=0)
f1_weight = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
prec_mac  = precision_score(y_test_enc, y_pred, average="macro",    zero_division=0)
rec_mac   = recall_score(y_test_enc, y_pred, average="macro",       zero_division=0)

print(f"F1-Score  (macro)    : {f1_macro:.4f}")
print(f"F1-Score  (weighted) : {f1_weight:.4f}")
print(f"Precision (macro)    : {prec_mac:.4f}")
print(f"Recall    (macro)    : {rec_mac:.4f}")

print("\nRelatório detalhado por classe:")
report_str = classification_report(y_test_enc, y_pred, target_names=class_names, zero_division=0)
print(report_str)

report_dict = classification_report(
    y_test_enc, y_pred, target_names=class_names,
    zero_division=0, output_dict=True,
)
df_metrics = (
    pd.DataFrame(report_dict)
    .T
    .loc[class_names, ["precision", "recall", "f1-score", "support"]]
    .astype({"support": int})
)

matriz_conf = confusion_matrix(y_test_enc, y_pred)
matriz_norm = confusion_matrix(y_test_enc, y_pred, normalize="true")

# ── Importância de features (Gini — nativo do RF) 


importancias = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": melhor_modelo.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nTop 20 features mais importantes (Gini):")
print(importancias.head(20).to_string(index=False))

top20 = importancias.head(20).copy()

# ── ETAPA 8 — Salvar bundle (Contrato 2 do TASKS.md) ─────────────────────────
# RF não usa scaler, então scaler=None. O dashboard trata isso corretamente.

print("\nETAPA 8 — SALVANDO BUNDLE\n")

bundle_path = MODELS_DIR / "rf_multiclasse.joblib"
salvar_bundle(
    bundle_path,
    melhor_modelo,
    feature_names,
    scaler=None,        # RF não precisa de scaler
    classes=class_names,
    limiar=0.5,         # referência; multiclasse usa argmax(proba)
)
print(f"Bundle salvo em: {bundle_path}")

# ── ETAPA 10 — Visualizações (6 páginas em PDF, espelho do arquivo de LogReg) ─

print("\nETAPA 10 — VISUALIZAÇÃO\n")

supports     = df_metrics["support"].values
colors_cls   = [PALETTE.get(c, "#999999") for c in class_names]
legend_patches = [
    mpatches.Patch(color=PALETTE.get(c, "#999999"), label=c)
    for c in class_names
]

PDF_METADATA = {
    "Title":    "Random Forest Multiclasse — CICIDS2017",
    "Author":   "Pipeline de Detecção de Intrusão",
    "Subject":  "Resultados de treinamento e avaliação do modelo",
    "Keywords": "CICIDS2017, RandomForest, IDS, multiclass",
}

with PdfPages(PDF_PATH, metadata=PDF_METADATA) as pdf:

    # ── Página 1: Métricas por classe (barras agrupadas)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(class_names))
    width = 0.25

    bars_p = ax1.bar(x - width, df_metrics["precision"], width, label="Precision", color="#4575b4")
    bars_r = ax1.bar(x,          df_metrics["recall"],   width, label="Recall",    color="#d73027")
    bars_f = ax1.bar(x + width,  df_metrics["f1-score"], width, label="F1-Score",  color="#1a9850")

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=25, ha="right")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Precision, Recall e F1-Score por Classe\n(Random Forest Multiclasse)")
    ax1.legend(loc="upper right")
    ax1.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    pdf.savefig(fig1, dpi=150)
    plt.show()
    plt.close(fig1)
    print("  [1/6] Métricas por classe — concluído")

    # ── Página 2: Matriz de Confusão Absoluta
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        matriz_conf, display_labels=class_names
    ).plot(cmap="Blues", ax=ax2, values_format="d", colorbar=True, xticks_rotation=30)
    ax2.set_title("Matriz de Confusão — Valores Absolutos\n(Random Forest Multiclasse)")
    plt.tight_layout()
    pdf.savefig(fig2, dpi=150)
    plt.show()
    plt.close(fig2)
    print("  [2/6] Matriz de confusão absoluta — concluído")

    # ── Página 3: Matriz de Confusão Normalizada 
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        matriz_norm, display_labels=class_names
    ).plot(cmap="Oranges", ax=ax3, values_format=".1%", colorbar=True, xticks_rotation=30)
    ax3.set_title(
        "Matriz de Confusão — Normalizada por Linha\n"
        "(Taxa de Acerto por Classe Real · Random Forest)"
    )
    plt.tight_layout()
    pdf.savefig(fig3, dpi=150)
    plt.show()
    plt.close(fig3)
    print("  [3/6] Matriz de confusão normalizada — concluído")

    # ── Página 4: Top 20 Features (Gini)

    top20_plot = top20.sort_values("Importance", ascending=True)
    cores_top20 = [PALETTE.get(c, "#skyblue") for c in top20_plot["Feature"]]

    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.barh(top20_plot["Feature"], top20_plot["Importance"], color="skyblue", edgecolor="white")
    ax4.set_xlabel("Grau de Importância (Gini)")
    ax4.set_ylabel("Atributo (Feature)")
    ax4.set_title(
        "Top 20 Atributos Mais Relevantes para Identificação do Tipo de Ataque\n"
        "(Random Forest — Importância por Redução de Impureza Gini)"
    )
    plt.tight_layout()
    pdf.savefig(fig4, dpi=150)
    plt.show()
    plt.close(fig4)
    print("  [4/6] Top 20 features — concluído")

    # ── Página 5: Importância acumulada (curva)

    imp_sorted = importancias["Importance"].values
    imp_cumsum = np.cumsum(imp_sorted)
    n_features_range = np.arange(1, len(imp_sorted) + 1)

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(n_features_range, imp_cumsum, color="#4575b4", linewidth=2)
    for threshold in [0.80, 0.90, 0.95]:
        n_needed = np.searchsorted(imp_cumsum, threshold) + 1
        ax5.axhline(threshold, color="gray", linewidth=0.8, linestyle="--")
        ax5.axvline(n_needed,  color="gray", linewidth=0.8, linestyle="--")
        ax5.text(n_needed + 1, threshold - 0.02, f"{int(threshold*100)}% com {n_needed} features",
                 fontsize=8, color="gray")
    ax5.set_xlabel("Número de Features (ordenadas por importância decrescente)")
    ax5.set_ylabel("Importância Acumulada")
    ax5.set_title(
        "Importância Acumulada das Features\n"
        "(Quantas features cobrem X% da capacidade discriminativa do modelo)"
    )
    ax5.set_xlim(1, len(imp_sorted))
    ax5.set_ylim(0, 1.05)
    plt.tight_layout()
    pdf.savefig(fig5, dpi=150)
    plt.show()
    plt.close(fig5)
    print("  [5/6] Importância acumulada — concluído")

    # ── Página 6: Distribuição de suporte por classe
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    bars6 = ax6.bar(class_names, supports, color=colors_cls, edgecolor="white")
    ax6.set_yscale("log")
    ax6.set_ylabel("Nº de amostras (escala log)")
    ax6.set_title("Distribuição de Amostras no Conjunto de Teste por Classe")
    ax6.set_xticklabels(class_names, rotation=25, ha="right")
    for bar, val in zip(bars6, supports):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:,}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    pdf.savefig(fig6, dpi=150)
    plt.show()
    plt.close(fig6)
    print("  [6/6] Distribuição de suporte — concluído")

print(f"\nPDF salvo em: {PDF_PATH}")
print(f"     Total de páginas: 6")
print("\n[RESUMO FINAL]")
print(f"  Classes:          {class_names}")
print(f"  F1 Macro (teste): {f1_macro:.4f}")
print(f"  F1 Weighted:      {f1_weight:.4f}")
print(f"  Precision Macro:  {prec_mac:.4f}")
print(f"  Recall Macro:     {rec_mac:.4f}")
print(f"\n[CONCLUÍDO] Etapa 2 — Identificação do Tipo de Ataque (Random Forest) finalizada.")