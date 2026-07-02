#!/usr/bin/env python3
"""
Etapa 1: Detecção Binária (Ataque vs. Normal)
Parte do Track B do pipeline CICIDS2017.

Carrega a amostra, separa holdout canônico, divide em Treino/Validação/Teste,
compara Decision Tree, Random Forest e Regressão Logística via GridSearchCV,
rebalanceia apenas nos folds de treino, ajusta o limiar para priorizar o Recall
de ataques e salva o bundle do modelo.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

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
    gerar_holdout_canonico,
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
    
    # Lista para exibir progresso
    historico = []
    
    for lim in limiares:
        preds = (probas_val >= lim).astype(int)
        rec = recall_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        historico.append((lim, rec, prec, f1))
        
        # Queremos o limiar que maximize o F1-score contanto que o Recall seja >= recall_alvo.
        # Caso nenhum limiar atinja o recall_alvo, buscamos o que dá o maior recall possível.
        if rec >= recall_alvo:
            atingiu_alvo = True
            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_limiar = lim
                
    if not atingiu_alvo:
        # Fallback: pega o limiar que dá o maior recall
        historico_sort = sorted(historico, key=lambda x: x[1], reverse=True)
        melhor_limiar = historico_sort[0][0]
        print(f"Aviso: Não foi possível atingir o Recall alvo de {recall_alvo:.2f}.")
        print(f"Usando limiar de fallback com maior Recall obtido: {melhor_limiar:.4f}")
    
    # Mostrar alguns limiares de exemplo
    print(f"{'Limiar':<8} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<8}")
    print("-" * 45)
    mostrados = [0.1, 0.2, 0.3, 0.4, 0.5, melhor_limiar]
    for lim, rec, prec, f1 in historico:
        if any(abs(lim - m) < 0.005 for m in mostrados):
            is_best = "*" if abs(lim - melhor_limiar) < 0.005 else " "
            print(f"{lim:.2f}{is_best:<5} | {rec:.4f} | {prec:.4f}    | {f1:.4f}")
            
    return melhor_limiar

def main():
    print("=== ETAPA 1: DETECÇÃO BINÁRIA ===")
    
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
        
    # 2. Holdout canônico: separa teste ANTES de qualquer subamostragem
    idx_teste = gerar_holdout_canonico(df)
    df_treino_val = df.loc[~df.index.isin(idx_teste)].copy()
    print(f"Dados restantes para treino/validação (excl. holdout): {len(df_treino_val)}")

    # 3. Preparação de features e split; rebalanceamento ocorre apenas nos folds de treino do pipeline.
    X, y = preparar_features(df_treino_val, alvo="target_bin")
    print(f"Total de features originais: {X.shape[1]}")
    
    particoes = split(X, y, val=0.15, teste=0.15, seed=42)
    X_train, X_val, X_test = particoes["X_train"], particoes["X_val"], particoes["X_test"]
    y_train, y_val, y_test = particoes["y_train"], particoes["y_val"], particoes["y_test"]
    
    print(f"Shapes das partições:")
    print(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")
    print(f"Distribuição de classes no treino original: {dict(y_train.value_counts())}")
    
    # 4. Definição do Pipeline e busca de hiperparâmetros
    pipeline = Pipeline([
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("sampler", RandomUnderSampler(sampling_strategy=0.333, random_state=42)),
        ("model", RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    param_grid = [
        {
            'model': [RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')],
            'model__n_estimators': [50, 100],
            'model__max_depth': [8, 12],
            'scaler': ['passthrough']
        },
        {
            'model': [DecisionTreeClassifier(random_state=42, class_weight='balanced')],
            'model__max_depth': [8, 12],
            'scaler': ['passthrough']
        },
        {
            'model': [LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')],
            'model__C': [0.1, 1.0, 10.0],
            'scaler': [StandardScaler()]
        }
    ]
    
    print("\nExecutando hyperparameter tuning com GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score F1 (validação cruzada): {grid_search.best_score_:.4f}")
    
    best_pipeline = grid_search.best_estimator_
    best_model_obj = best_pipeline.named_steps["model"]
    best_model_name = type(best_model_obj).__name__
    print(f"\nMelhor classificador selecionado: {best_model_name}")
    
    # 5. Otimizar limiar para priorizar Recall no conjunto de validação
    probas_val = best_pipeline.predict_proba(X_val)[:, 1]
    limiar_otimo = otimizar_limiar(y_val, probas_val, recall_alvo=0.98)
    print(f"-> Limiar ótimo selecionado: {limiar_otimo:.4f}")
    
    # 6. Avaliação final no conjunto de TESTE
    probas_test = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_teste = (probas_test >= limiar_otimo).astype(int)
    
    print(f"\n=== RELATÓRIO DE CLASSIFICAÇÃO NO TESTE (Limiar: {limiar_otimo:.4f}) ===")
    print(classification_report(y_test, y_pred_teste, target_names=["Normal (0)", "Ataque (1)"]))
    
    # Imprimir matriz de confusão no terminal
    cm = confusion_matrix(y_test, y_pred_teste)
    print("Matriz de Confusão:")
    print(cm)
    
    # 7. Gerar heatmap e salvar
    caminho_imagem = root_dir / "images" / "mat_confusao_deteccao.png"
    plotar_matriz_confusao(
        cm, 
        classes=["Normal (0)", "Ataque (1)"], 
        caminho_salvar=caminho_imagem,
        titulo=f"Matriz de Confusão (Detecção) - {best_model_name}\nLimiar: {limiar_otimo:.3f}"
    )
    
    # 8. Importância de features (ou coeficientes) para o melhor classificador
    if hasattr(best_model_obj, "feature_importances_"):
        support = best_pipeline.named_steps["variance"].get_support()
        colunas_apos_var = X_train.columns[support]
        importancias = pd.Series(best_model_obj.feature_importances_, index=colunas_apos_var)
        top_10 = importancias.sort_values(ascending=False).head(10)
        print("\n=== TOP 10 FEATURES MAIS IMPORTANTES ===")
        for col, imp in top_10.items():
            print(f"{col:<35}: {imp:.4f}")
    elif hasattr(best_model_obj, "coef_"):
        support = best_pipeline.named_steps["variance"].get_support()
        colunas_apos_var = X_train.columns[support]
        importancias = pd.Series(np.abs(best_model_obj.coef_[0]), index=colunas_apos_var)
        top_10 = importancias.sort_values(ascending=False).head(10)
        print("\n=== TOP 10 FEATURES MAIS IMPORTANTES (Coefs Absolutos) ===")
        for col, imp in top_10.items():
            print(f"{col:<35}: {imp:.4f}")
            
    # 9. Salvar o bundle
    caminho_bundle = MODELS_DIR / "rf_etapa1.joblib"
    print(f"\nSalvando o bundle do modelo em: {caminho_bundle}")
    salvar_bundle(
        caminho=caminho_bundle,
        modelo=best_pipeline,
        colunas=X_train.columns.tolist(),
        scaler=None,
        classes=[0, 1],
        limiar=limiar_otimo
    )
    print("Fase 2 (B1) concluída com sucesso!")

if __name__ == "__main__":
    main()
