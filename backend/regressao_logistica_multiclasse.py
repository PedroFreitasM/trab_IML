import sys
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Caminho do PDF consolidado de saída
PDF_PATH = ROOT_DIR / "images" / "reg_log_multiclasse.pdf"
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

# CONFIGURAÇÃO GLOBAL

RANDOM_STATE = 42
SAMPLE_FRAC = 0.3   # Reduz RAM; ajuste conforme disponibilidade

# Paleta de cores por classe (8 famílias + Benign)
PALETTE = {
    "Benign":        "#2196F3",
    "DDoS":          "#F44336",
    "DoS":           "#FF9800",
    "PortScan":      "#9C27B0",
    "Botnet":        "#009688",
    "Bruteforce":    "#795548",
    "Infiltration":  "#607D8B",
    "WebAttacks":    "#E91E63",
}

# FUNÇÃO AUXILIAR: undersample por classe

def undersample_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
    ratio: float = 5.0,
) -> tuple:
    """
    Balanceamento suave para multiclasse.

    Estratégia:
      - Calcula o tamanho da classe MINORITÁRIA.
      - Limita cada classe a no máximo `ratio * minority_size` amostras.
      - Mantém TODAS as amostras das classes com < ratio * minority_size exemplos.
    Isso evita descartar completamente classes raras (ex.: Infiltration) e ao
    mesmo tempo reduz o domínio de Benign/DDoS que costumam ser gigantes.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_size = counts.min()
    cap = int(ratio * minority_size)

    indices_selecionados = []
    for cls in classes:
        idx_cls = np.where(y == cls)[0]
        if len(idx_cls) > cap:
            idx_cls = rng.choice(idx_cls, size=cap, replace=False)
        indices_selecionados.append(idx_cls)

    idx_final = np.concatenate(indices_selecionados)
    rng.shuffle(idx_final)
    return X[idx_final], y[idx_final]

# ETAPA 2 — CARREGAMENTO DOS DADOS

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

    # Reduz precisão das features numéricas para economizar RAM
    for col in df_parte.select_dtypes(include=[np.number]).columns:
        if col != "target_tipo":
            df_parte[col] = df_parte[col].astype(np.float32)

    # Amostragem estratificada por arquivo
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

# ETAPA 3 — SEPARAÇÃO DE FEATURES E SPLIT

print("\nETAPA 3 — SEPARAÇÃO DE FEATURES E SPLIT\n")

# alvo agora é target_tipo (multiclasse: Benign, DDoS, DoS, PortScan…)
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

# ETAPA 4 — PRÉ-PROCESSAMENTO FINAL

print("\nETAPA 4 — PRÉ-PROCESSAMENTO FINAL\n")

X_train, X_val, X_test = filtrar_variancia_zero(X_train, X_val, X_test)
feature_names = X_train.columns.tolist()
print(f"Features após remoção de variância zero: {len(feature_names)}")

# Converte para numpy
X_train_np = X_train.values.astype(np.float32); del X_train
X_val_np   = X_val.values.astype(np.float32);   del X_val
X_test_np  = X_test.values.astype(np.float32);  del X_test

# Codifica os rótulos de string → inteiro (LabelEncoder preserva os nomes)
le = LabelEncoder()
y_train_np = le.fit_transform(y_train.values); del y_train
y_val_np   = le.transform(y_val.values);       del y_val
y_test_np  = le.transform(y_test.values);      del y_test
gc.collect()

class_names = list(le.classes_)   # ex.: ['Benign','Botnet','Bruteforce',...]
n_classes   = len(class_names)
print(f"Classes detectadas ({n_classes}): {class_names}")

# BALANCEAMENTO MULTICLASSE (undersample suave)

X_train_bal, y_train_bal = undersample_multiclass(
    X_train_np, y_train_np, random_state=RANDOM_STATE, ratio=5.0
)
del X_train_np, y_train_np
gc.collect()

print("\nDistribuição pós-balanceamento (treino):")
for cls_id, cls_name in enumerate(class_names):
    cnt = np.sum(y_train_bal == cls_id)
    print(f"   {cls_name:15s}: {cnt:,}")

# NORMALIZAÇÃO (fit APENAS no treino → evita leakage)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal).astype(np.float32)
X_val_scaled   = scaler.transform(X_val_np).astype(np.float32)
X_test_scaled  = scaler.transform(X_test_np).astype(np.float32)

# ETAPA 5 — TREINAMENTO (MULTICLASSE)

print("\nETAPA 5 — TREINAMENTO\n")

#   multi_class='multinomial' + solver='saga' → Softmax Regression
#   class_weight=None porque já balanceamos manualmente
modelo_base = LogisticRegression(
    random_state=RANDOM_STATE,
    n_jobs=2,
    solver="saga",
    multi_class="multinomial",   # softmax generalizado
    max_iter=500,
)

# GridSearch: busca o melhor C com F1 macro (justo para classes desbalanceadas)
otimizador = GridSearchCV(
    modelo_base,
    {"C": [0.1, 1, 10]},
    cv=3,
    scoring="f1_macro",          # macro = média simples entre classes
    n_jobs=2,
    verbose=1,
)
otimizador.fit(X_train_scaled, y_train_bal)

melhor_modelo = otimizador.best_estimator_
print(f"\nMelhor C encontrado: {otimizador.best_params_['C']}")
print(f"F1-Macro CV (treino): {otimizador.best_score_:.4f}")

# ETAPA 6 — SELEÇÃO DE LIMIAR VIA CONJUNTO DE VALIDAÇÃO

# Para multiclasse pura, não há um "limiar" único como no binário,
# mas avaliamos o modelo no conjunto de validação para diagnóstico.
y_val_pred = melhor_modelo.predict(X_val_scaled)
print("\nRelatório por classe:")
print(
    classification_report(
        y_val_np,
        y_val_pred,
        target_names=class_names,
        zero_division=0,
    )
)

# ETAPA 7 — AVALIAÇÃO FINAL NO CONJUNTO DE TESTE

print("\nETAPA 7 — AVALIAÇÃO FINAL (TESTE)\n")

y_pred = melhor_modelo.predict(X_test_scaled)

# Métricas globais
f1_macro  = f1_score(y_test_np, y_pred, average="macro",    zero_division=0)
f1_weight = f1_score(y_test_np, y_pred, average="weighted", zero_division=0)
prec_mac  = precision_score(y_test_np, y_pred, average="macro",    zero_division=0)
rec_mac   = recall_score(y_test_np, y_pred, average="macro",       zero_division=0)

print(f"F1-Score  (macro)    : {f1_macro:.4f}")
print(f"F1-Score  (weighted) : {f1_weight:.4f}")
print(f"Precision (macro)    : {prec_mac:.4f}")
print(f"Recall    (macro)    : {rec_mac:.4f}")

# Relatório completo por classe
print("\nRelatório detalhado por classe:")
report_str = classification_report(
    y_test_np, y_pred, target_names=class_names, zero_division=0
)
print(report_str)

# Métricas por classe em DataFrame 
report_dict = classification_report(
    y_test_np, y_pred, target_names=class_names,
    zero_division=0, output_dict=True
)
df_metrics = (
    pd.DataFrame(report_dict)
    .T
    .loc[class_names, ["precision", "recall", "f1-score", "support"]]
    .astype({"support": int})
)

# Matriz de confusão
matriz_conf = confusion_matrix(y_test_np, y_pred)

# IMPORTÂNCIA DE FEATURES (coeficientes do modelo multiclasse)

# coef_ tem shape (n_classes, n_features)
# Para cada feature, usa-se a MAGNITUDE MÁXIMA entre as classes como proxy
# de importância global, e guarda-se QUAL classe ela discrimina mais.
coef_matrix = melhor_modelo.coef_           # (n_classes, n_features)
max_abs_per_feature = np.max(np.abs(coef_matrix), axis=0)
dominant_class_idx  = np.argmax(np.abs(coef_matrix), axis=0)

# Top 15 features por magnitude máxima
top_n = 15
top_idx = np.argsort(max_abs_per_feature)[-top_n:][::-1]
top_feat_names  = [feature_names[i] for i in top_idx]
top_feat_vals   = max_abs_per_feature[top_idx]
top_feat_class  = [class_names[dominant_class_idx[i]] for i in top_idx]
top_feat_colors = [PALETTE.get(c, "#999999") for c in top_feat_class]

print("\nTop 15 Features (importância global):")
for rank, (name, val, cls) in enumerate(
    zip(top_feat_names, top_feat_vals, top_feat_class), 1
):
    print(f"  {rank:2d}. {name:40s} |coef|={val:.4f}  → mais discriminante para: {cls}")

bundle_path = MODELS_DIR / "logreg_multiclasse.joblib"
salvar_bundle(
    bundle_path,
    melhor_modelo,
    feature_names,
    scaler,
    class_names,
    limiar=0.5,    # referência; para multiclasse usa argmax(proba)
)
print(f"\nBundle salvo em: {bundle_path}")

# ETAPA 10 — VISUALIZAÇÕES (todas salvas em um único PDF)

print("\nETAPA 10 — VISUALIZAÇÃO\n")

# Pré-cálculos compartilhados entre figuras
matriz_norm  = confusion_matrix(y_test_np, y_pred, normalize="true")
top20_idx    = np.argsort(max_abs_per_feature)[-20:][::-1]
top20_names  = [feature_names[i] for i in top20_idx]
coef_heatmap = coef_matrix[:, top20_idx]          # (n_classes, 20)
supports     = df_metrics["support"].values
colors_sup   = [PALETTE.get(c, "#999999") for c in class_names]
legend_patches = [
    mpatches.Patch(color=PALETTE.get(c, "#999999"), label=c)
    for c in class_names
]

# Metadados do PDF (visíveis em leitores como Acrobat / Evince)
PDF_METADATA = {
    "Title":   "Regressão Logística Multiclasse — CICIDS2017",
    "Author":  "Pipeline de Detecção de Intrusão",
    "Subject": "Resultados de treinamento e avaliação do modelo",
    "Keywords": "CICIDS2017, LogisticRegression, IDS, multiclass",
}

with PdfPages(PDF_PATH, metadata=PDF_METADATA) as pdf:

    # ── Página 1: Métricas por classe (barras agrupadas) ─────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(class_names))
    width = 0.25

    bars_p = ax1.bar(x - width, df_metrics["precision"], width, label="Precision", color="#4575b4")
    bars_r = ax1.bar(x,          df_metrics["recall"],    width, label="Recall",    color="#d73027")
    bars_f = ax1.bar(x + width,  df_metrics["f1-score"],  width, label="F1-Score",  color="#1a9850")

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=25, ha="right")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Precision, Recall e F1-Score por Classe\n(Regressão Logística Multiclasse)")
    ax1.legend(loc="upper right")
    ax1.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    pdf.savefig(fig1, dpi=150)
    plt.show()
    plt.close(fig1)
    print("  [1/6] Métricas por classe — concluído")

    # ── Página 2: Matriz de Confusão Absoluta ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        matriz_conf, display_labels=class_names
    ).plot(cmap="Blues", ax=ax2, values_format="d", colorbar=True, xticks_rotation=30)
    ax2.set_title("Matriz de Confusão — Valores Absolutos")
    plt.tight_layout()
    pdf.savefig(fig2, dpi=150)
    plt.show()
    plt.close(fig2)
    print("  [2/6] Matriz de confusão absoluta — concluído")

    # ── Página 3: Matriz de Confusão Normalizada ─────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        matriz_norm, display_labels=class_names
    ).plot(cmap="Oranges", ax=ax3, values_format=".1%", colorbar=True, xticks_rotation=30)
    ax3.set_title("Matriz de Confusão — Normalizada por Linha\n(Taxa de Acerto por Classe Real)")
    plt.tight_layout()
    pdf.savefig(fig3, dpi=150)
    plt.show()
    plt.close(fig3)
    print("  [3/6] Matriz de confusão normalizada — concluído")

    # ── Página 4: Top 15 Features ─────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(top_n)[::-1]
    ax4.barh(y_pos, top_feat_vals, color=top_feat_colors, edgecolor="white")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(
        [f"{n}  [{c}]" for n, c in zip(top_feat_names, top_feat_class)],
        fontsize=9,
    )
    ax4.set_xlabel("|Coeficiente| máximo entre classes")
    ax4.set_title(
        "Top 15 Features — Importância Global\n"
        "(Cor indica a classe que mais discrimina)"
    )
    ax4.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Classe dominante")
    plt.tight_layout()
    pdf.savefig(fig4, dpi=150)
    plt.show()
    plt.close(fig4)
    print("  [4/6] Top 15 features — concluído")

    # ── Página 5: Heatmap de coeficientes ────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(16, 5))
    im = ax5.imshow(coef_heatmap, aspect="auto", cmap="RdBu_r")
    ax5.set_xticks(range(len(top20_names)))
    ax5.set_xticklabels(top20_names, rotation=45, ha="right", fontsize=8)
    ax5.set_yticks(range(n_classes))
    ax5.set_yticklabels(class_names, fontsize=9)
    ax5.set_title(
        "Heatmap de Coeficientes — Top 20 Features × Classes\n"
        "(Azul = contribui para BENIGNO  |  Vermelho = contribui para ATAQUE)"
    )
    plt.colorbar(im, ax=ax5, label="Coeficiente")

    for i in range(n_classes):
        for j in range(len(top20_names)):
            val = coef_heatmap[i, j]
            ax5.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6,
                color="white" if abs(val) >= 1.5 else "black",
            )

    plt.tight_layout()
    pdf.savefig(fig5, dpi=150)
    plt.show()
    plt.close(fig5)
    print("  [5/6] Heatmap de coeficientes — concluído")

    # ── Página 6: Distribuição de suporte por classe ──────────────────────
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    bars6 = ax6.bar(class_names, supports, color=colors_sup, edgecolor="white")
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