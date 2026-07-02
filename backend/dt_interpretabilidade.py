#!/usr/bin/env python3
"""
Detecção e Classificação com Árvores de Decisão para Interpretabilidade.
Parte do pipeline CICIDS2017.

Treina árvores de decisão rasas para ambas as etapas do pipeline, permitindo
a visualização da estrutura da árvore, regras textuais e importância de features,
além de salvar os bundles servíveis em models/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configura backend não interativo antes de importar pyplot
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Adiciona o diretório raiz ao python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import (
    carregar_dados,
    limpar,
    criar_targets,
    preparar_features,
    split,
    salvar_bundle,
    carregar_holdout_canonico,
    DATA_DIR,
    MODELS_DIR
)
from backend.visualizacao import plotar_matriz_confusao

def otimizar_limiar(y_val, probas_val, recall_alvo=0.98):
    """Encontra o maior limiar que atinge pelo menos o recall_alvo."""
    limiares = np.linspace(0.01, 0.99, 99)
    melhor_limiar = 0.5
    melhor_f1 = 0.0
    atingiu_alvo = False
    
    print(f"\nOtimizando limiar para Recall Alvo >= {recall_alvo:.2f}...")
    historico = []
    
    for lim in limiares:
        preds = (probas_val >= lim).astype(int)
        rec = recall_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        historico.append((lim, rec, prec, f1))
        
        if rec >= recall_alvo:
            atingiu_alvo = True
            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_limiar = lim
                
    if not atingiu_alvo:
        historico_sort = sorted(historico, key=lambda x: x[1], reverse=True)
        melhor_limiar = historico_sort[0][0]
        print(f"Aviso: Não foi possível atingir o Recall alvo de {recall_alvo:.2f}.")
        print(f"Usando limiar de fallback com maior Recall obtido: {melhor_limiar:.4f}")
        
    return melhor_limiar

def plotar_recursos_interpretabilidade(clf, colunas_var, prefixo_salvar, classes, max_depth_plot=4):
    """Gera gráficos de importância de features, plot da árvore e regras textuais."""
    images_dir = root_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 1. Importância de features
    importancias = pd.Series(clf.feature_importances_, index=colunas_var)
    top_10 = importancias.sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    top_10.plot(kind='barh', color='skyblue').invert_yaxis()
    plt.title(f"Top 10 Features Mais Importantes - {prefixo_salvar.upper()}")
    plt.xlabel("Importância")
    plt.tight_layout()
    caminho_imp = images_dir / f"dt_{prefixo_salvar}_importancias.png"
    plt.savefig(caminho_imp, dpi=150)
    plt.close()
    print(f"Gráfico de importância salvo em: {caminho_imp}")
    
    # 2. Exportar regras textuais
    regras = export_text(clf, feature_names=list(colunas_var))
    caminho_txt = images_dir / f"dt_{prefixo_salvar}_regras.txt"
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(regras)
    print(f"Regras textuais salvas em: {caminho_txt}")
    
    # 3. Plotar estrutura da árvore (limitado para melhor legibilidade)
    plt.figure(figsize=(24, 12))
    plot_tree(
        clf,
        max_depth=max_depth_plot,
        feature_names=list(colunas_var),
        class_names=[str(c) for c in classes],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Estrutura da Árvore de Decisão ({prefixo_salvar.upper()}) - Profundidade máx visualizada: {max_depth_plot}")
    plt.tight_layout()
    caminho_tree = images_dir / f"dt_{prefixo_salvar}_arvore.png"
    plt.savefig(caminho_tree, dpi=150)
    plt.close()
    print(f"Visualização da árvore salva em: {caminho_tree}")

def main():
    # 1. Carregamento dos dados
    caminho_amostra = DATA_DIR / "amostra.parquet"
    if caminho_amostra.exists():
        print(f"Carregando dados da amostra: {caminho_amostra}")
        df = pd.read_parquet(caminho_amostra)
    else:
        print("Amostra não encontrada. Carregando dataset completo (pode demorar)...")
        df = carregar_dados()
        df = limpar(df)
        df = criar_targets(df)
        
    # Carregar holdout canônico existente
    idx_teste = carregar_holdout_canonico()
    df_treino_val = df.loc[~df.index.isin(idx_teste)].copy()
    print(f"Dados para treino/validação (excl. holdout): {len(df_treino_val)}")

    # =========================================================================
    # TREINAMENTO - ETAPA 1 (DETECÇÃO BINÁRIA)
    # =========================================================================
    print("\n--- INICIANDO TREINAMENTO DA ETAPA 1 (ÁRVORE DE DECISÃO BINÁRIA) ---")
    X1, y1 = preparar_features(df_treino_val, alvo="target_bin")
    
    particoes1 = split(X1, y1, val=0.15, teste=0.15, seed=42)
    X_train1, X_val1, X_test1 = particoes1["X_train"], particoes1["X_val"], particoes1["X_test"]
    y_train1, y_val1, y_test1 = particoes1["y_train"], particoes1["y_val"], particoes1["y_test"]
    
    pipeline1 = Pipeline([
        ("variance", VarianceThreshold()),
        ("sampler", RandomUnderSampler(sampling_strategy=0.333, random_state=42)),
        ("model", DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ])
    
    param_grid1 = {
        'model__max_depth': [3, 5, 7],
        'model__min_samples_leaf': [10, 50, 100]
    }
    
    grid1 = GridSearchCV(pipeline1, param_grid1, scoring='f1', cv=5, n_jobs=-1)
    grid1.fit(X_train1, y_train1)
    print(f"Melhores parâmetros Etapa 1: {grid1.best_params_}")
    
    best_pipeline1 = grid1.best_estimator_
    probas_val1 = best_pipeline1.predict_proba(X_val1)[:, 1]
    limiar_otimo = otimizar_limiar(y_val1, probas_val1, recall_alvo=0.98)
    
    # Avaliação no Teste
    probas_test1 = best_pipeline1.predict_proba(X_test1)[:, 1]
    preds_test1 = (probas_test1 >= limiar_otimo).astype(int)
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ETAPA 1 (TESTE) ===")
    print(classification_report(y_test1, preds_test1, target_names=["Normal", "Ataque"]))
    
    cm1 = confusion_matrix(y_test1, preds_test1)
    plotar_matriz_confusao(
        cm1,
        classes=["Normal", "Ataque"],
        caminho_salvar=root_dir / "images" / "mat_confusao_dt_deteccao.png",
        titulo=f"Matriz de Confusão (DT Detecção) - Limiar: {limiar_otimo:.3f}"
    )
    
    # Artefatos
    clf1 = best_pipeline1.named_steps["model"]
    colunas_var1 = X_train1.columns[best_pipeline1.named_steps["variance"].get_support()]
    plotar_recursos_interpretabilidade(clf1, colunas_var1, "etapa1", classes=["Normal", "Ataque"], max_depth_plot=4)
    
    # Salvar bundle
    salvar_bundle(
        caminho=MODELS_DIR / "dt_etapa1.joblib",
        modelo=best_pipeline1,
        colunas=X_train1.columns.tolist(),
        scaler=None,
        classes=[0, 1],
        limiar=limiar_otimo
    )
    print("Bundle dt_etapa1.joblib salvo com sucesso!")

    # =========================================================================
    # TREINAMENTO - ETAPA 2 (CLASSIFICAÇÃO MULTICLASSE)
    # =========================================================================
    print("\n--- INICIANDO TREINAMENTO DA ETAPA 2 (ÁRVORE DE DECISÃO MULTICLASSE) ---")
    df_ataques = df_treino_val[df_treino_val["target_bin"] == 1].copy()
    
    # Limpeza de classes ultra-raras (< 10)
    contagem = df_ataques["target_tipo"].value_counts()
    classes_raras = contagem[contagem < 10].index.tolist()
    if classes_raras:
        print(f"Removendo classes raras: {classes_raras}")
        df_ataques = df_ataques[~df_ataques["target_tipo"].isin(classes_raras)].copy()
        
    X2, y2 = preparar_features(df_ataques, alvo="target_tipo")
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.15, random_state=42, stratify=y2
    )
    
    min_samples = y_train2.value_counts().min()
    k_neighbors = min(5, min_samples - 1)
    print(f"Usando k_neighbors={k_neighbors} para o SMOTE na Etapa 2.")
    
    pipeline2 = Pipeline([
        ("variance", VarianceThreshold()),
        ("smote", SMOTE(k_neighbors=k_neighbors, random_state=42)),
        ("model", DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ])
    
    param_grid2 = {
        'model__max_depth': [5, 8, 10],
        'model__min_samples_leaf': [5, 20, 50]
    }
    
    grid2 = GridSearchCV(pipeline2, param_grid2, scoring='f1_macro', cv=5, n_jobs=-1)
    grid2.fit(X_train2, y_train2)
    print(f"Melhores parâmetros Etapa 2: {grid2.best_params_}")
    
    best_pipeline2 = grid2.best_estimator_
    preds_test2 = best_pipeline2.predict(X_test2)
    classes_ordenadas = sorted(y_test2.unique())
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ETAPA 2 (TESTE) ===")
    print(classification_report(y_test2, preds_test2))
    
    cm2 = confusion_matrix(y_test2, preds_test2, labels=classes_ordenadas)
    plotar_matriz_confusao(
        cm2,
        classes=classes_ordenadas,
        caminho_salvar=root_dir / "images" / "mat_confusao_dt_tipo.png",
        titulo="Matriz de Confusão (DT Classificação de Ataques)"
    )
    
    # Artefatos
    clf2 = best_pipeline2.named_steps["model"]
    colunas_var2 = X_train2.columns[best_pipeline2.named_steps["variance"].get_support()]
    plotar_recursos_interpretabilidade(clf2, colunas_var2, "etapa2", classes=classes_ordenadas, max_depth_plot=5)
    
    # Salvar bundle
    salvar_bundle(
        caminho=MODELS_DIR / "dt_etapa2.joblib",
        modelo=best_pipeline2,
        colunas=X_train2.columns.tolist(),
        scaler=None,
        classes=classes_ordenadas,
        limiar=0.5
    )
    print("Bundle dt_etapa2.joblib salvo com sucesso!")
    print("\n--- TREINAMENTO DAS ÁRVORES DE DECISÃO CONCLUÍDO ---")

if __name__ == "__main__":
    main()
