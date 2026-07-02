#!/usr/bin/env python3
"""
Etapa 2: Identificação do Tipo de Ataque (Multiclasse)
Parte do Track B do pipeline CICIDS2017.

Filtra apenas registros de ataque, limpa classes ultra-raras para viabilizar o SMOTE/CV,
compara Decision Tree, Random Forest e Regressão Logística usando GridSearchCV,
aplica SMOTE e normalização dentro do pipeline para evitar vazamento de dados,
avalia a performance com macro-F1, gera gráficos e salva o bundle em models/rf_etapa2.joblib.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Adiciona o diretório raiz ao python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import (
    carregar_dados,
    limpar,
    criar_targets,
    preparar_features,
    salvar_bundle,
    carregar_holdout_canonico,
    DATA_DIR,
    MODELS_DIR
)
from backend.visualizacao import plotar_matriz_confusao

CV_SPLITS = 5
SMOTE_K_NEIGHBORS = 5
MIN_SAMPLES_POR_CLASSE = 10

def filtrar_e_preparar_ataques(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas os fluxos de ataque (target_bin == 1)."""
    df_ataques = df[df["target_bin"] == 1].copy()
    print(f"-> Filtrado apenas ataques. Total de registros: {len(df_ataques)}")
    return df_ataques

def limpar_classes_raras(df: pd.DataFrame, min_samples: int = MIN_SAMPLES_POR_CLASSE) -> pd.DataFrame:
    """Remove classes com contagem de amostras inferior a min_samples.
    
    O valor padrão garante que, após o split 85/15, cada classe mantenha
    amostras suficientes nos folds de treino para o SMOTE(k_neighbors=5).
    """
    contagem = df["target_tipo"].value_counts()
    classes_raras = contagem[contagem < min_samples].index.tolist()
    
    if classes_raras:
        print(f"-> AVISO: As seguintes classes possuem menos de {min_samples} amostras e serão removidas da modelagem:")
        for cls in classes_raras:
            print(f"   - {cls} (Contagem: {contagem[cls]})")
        df = df[~df["target_tipo"].isin(classes_raras)].copy()
        print(f"-> Registros após remoção de classes raras: {len(df)}")
    return df

def main():
    print("=== ETAPA 2: IDENTIFICAÇÃO MULTICLASSE DE ATAQUES ===")
    
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
        
    # 2. Excluir holdout canônico (dados reservados para teste fim-a-fim)
    try:
        idx_teste = carregar_holdout_canonico()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Execute etapa1_deteccao.py primeiro para gerar o holdout canônico.")
        return
    df = df.loc[~df.index.isin(idx_teste)].copy()
    print(f"Holdout canônico excluído: usando {len(df)} amostras para treino")

    # 3. Filtrar apenas tráfego malicioso e remover classes ultra-raras
    df_ataques = filtrar_e_preparar_ataques(df)
    df_ataques = limpar_classes_raras(df_ataques)
    
    # 3. Preparação de features e split em Treino/Teste
    X, y = preparar_features(df_ataques, alvo="target_tipo")
    
    # Split 85% treino (para CV) e 15% teste final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Partição de Treino: {X_train.shape} | Teste Final: {X_test.shape}")
    print(f"Distribuição de classes no Treino:\n{y_train.value_counts()}")
    
    # 4. Definição do Pipeline e busca de hiperparâmetros
    pipeline = Pipeline([
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=42)),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
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
        scoring="f1_macro",
        cv=CV_SPLITS,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score Macro-F1 (validação cruzada): {grid_search.best_score_:.4f}")
    
    best_pipeline = grid_search.best_estimator_
    best_model_obj = best_pipeline.named_steps["model"]
    best_model_name = type(best_model_obj).__name__
    print(f"\nMelhor classificador selecionado: {best_model_name}")
    
    # 5. Avaliação final no conjunto de TESTE
    y_pred_teste = best_pipeline.predict(X_test)
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO NO TESTE ===")
    print(classification_report(y_test, y_pred_teste))
    
    # 6. Matriz de Confusão NxN
    classes_ordenadas = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_teste, labels=classes_ordenadas)
    
    caminho_imagem = root_dir / "images" / "mat_confusao_tipo.png"
    plotar_matriz_confusao(
        cm, 
        classes=classes_ordenadas, 
        caminho_salvar=caminho_imagem,
        titulo=f"Matriz de Confusão (Identificação de Ataques) - {best_model_name}"
    )
    
    # 7. Importância de features
    if hasattr(best_model_obj, "feature_importances_"):
        colunas_apos_var = X_train.columns[best_pipeline.named_steps["variance"].get_support()]
        importancias = pd.Series(best_model_obj.feature_importances_, index=colunas_apos_var)
        top_10 = importancias.sort_values(ascending=False).head(10)
        print("\n=== TOP 10 FEATURES MAIS IMPORTANTES (Etapa 2) ===")
        for col, imp in top_10.items():
            print(f"{col:<35}: {imp:.4f}")
    elif hasattr(best_model_obj, "coef_"):
        colunas_apos_var = X_train.columns[best_pipeline.named_steps["variance"].get_support()]
        coef = np.asarray(best_model_obj.coef_)
        valores = np.abs(coef) if coef.ndim == 1 else np.abs(coef).mean(axis=0)
        importancias = pd.Series(valores, index=colunas_apos_var)
        top_10 = importancias.sort_values(ascending=False).head(10)
        print("\n=== TOP 10 FEATURES MAIS IMPORTANTES (Média dos Coefs Absolutos por Classe) ===")
        for col, imp in top_10.items():
            print(f"{col:<35}: {imp:.4f}")
        
    # 8. Salvar o bundle
    caminho_bundle = MODELS_DIR / "rf_etapa2.joblib"
    print(f"\nSalvando o bundle do modelo multiclasse em: {caminho_bundle}")
    
    salvar_bundle(
        caminho=caminho_bundle,
        modelo=best_pipeline,
        colunas=X_train.columns.tolist(),
        scaler=None,
        classes=classes_ordenadas,
        limiar=0.5
    )
    print("Fase 3 (B2) concluída com sucesso!")

if __name__ == "__main__":
    main()
