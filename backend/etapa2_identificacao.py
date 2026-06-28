#!/usr/bin/env python3
"""
Etapa 2: Identificação do Tipo de Ataque (Multiclasse)
Parte do Track B do pipeline CICIDS2017.

Filtra apenas registros de ataque, limpa classes ultra-raras para viabilizar o SMOTE/CV,
compara Decision Tree, Random Forest e Regressão Logística usando 5-Fold Stratified CV,
aplica SMOTE e normalização dentro do pipeline para evitar vazamento de dados,
avalia a performance com macro-F1, gera gráficos e salva o bundle em models/etapa2.joblib.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
    DATA_DIR,
    MODELS_DIR
)
from backend.visualizacao import plotar_matriz_confusao

def filtrar_e_preparar_ataques(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas os fluxos de ataque (target_bin == 1)."""
    df_ataques = df[df["target_bin"] == 1].copy()
    print(f"-> Filtrado apenas ataques. Total de registros: {len(df_ataques)}")
    return df_ataques

def limpar_classes_raras(df: pd.DataFrame, min_samples: int = 6) -> pd.DataFrame:
    """Remove classes com contagem de amostras inferior a min_samples.
    
    Necessário para permitir a divisão no StratifiedKFold e funcionamento do SMOTE.
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

def avaliar_cv(X, y):
    """Executa a validação cruzada 5-Fold Stratified para comparar DT, RF e Regressão Logística."""
    print("\n=== VALIDAÇÃO CRUZADA (5-Fold Stratified CV com SMOTE-no-Pipeline) ===")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Definição dos pipelines
    pipelines = {
        "Decision Tree": Pipeline([
            ("variance", VarianceThreshold()),
            ("smote", SMOTE(random_state=42)),
            ("model", DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("variance", VarianceThreshold()),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"))
        ]),
        "Logistic Regression": Pipeline([
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ])
    }
    
    for nome, pip in pipelines.items():
        macro_f1s = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr_fold, X_va_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_fold, y_va_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            pip.fit(X_tr_fold, y_tr_fold)
            preds = pip.predict(X_va_fold)
            
            macro_f1s.append(f1_score(y_va_fold, preds, average="macro"))
            
        mean_macro_f1 = np.mean(macro_f1s)
        print(f"[{nome}] Média Macro-F1: {mean_macro_f1:.4f}")

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
        
    # 2. Filtrar apenas tráfego malicioso e remover classes ultra-raras
    df_ataques = filtrar_e_preparar_ataques(df)
    df_ataques = limpar_classes_raras(df_ataques, min_samples=6)
    
    # 3. Preparação de features e split em Treino/Teste
    X, y = preparar_features(df_ataques, alvo="target_tipo")
    
    # Split 85% treino (para CV) e 15% teste final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Partição de Treino: {X_train.shape} | Teste Final: {X_test.shape}")
    print(f"Distribuição de classes no Treino:\n{y_train.value_counts()}")
    
    # 4. Executar Validação Cruzada para seleção de modelos
    avaliar_cv(X_train, y_train)
    
    # 5. Treinamento do Modelo Principal (Random Forest) no Treino Completo
    # Montamos o pipeline final com Random Forest
    pipeline_final = Pipeline([
        ("variance", VarianceThreshold()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"))
    ])
    
    print("\nTreinando o pipeline Random Forest final...")
    pipeline_final.fit(X_train, y_train)
    
    # 6. Avaliação final no conjunto de TESTE
    y_pred_teste = pipeline_final.predict(X_test)
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO NO TESTE ===")
    print(classification_report(y_test, y_pred_teste))
    
    # 7. Matriz de Confusão NxN
    classes_ordenadas = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_teste, labels=classes_ordenadas)
    
    caminho_imagem = root_dir / "images" / "mat_confusao_tipo.png"
    plotar_matriz_confusao(
        cm, 
        classes=classes_ordenadas, 
        caminho_salvar=caminho_imagem,
        titulo="Matriz de Confusão (Identificação de Ataques) - Random Forest"
    )
    
    # 8. Importância de features
    clf_model = pipeline_final.named_steps["model"]
    # Precisamos das colunas após o VarianceThreshold
    colunas_apos_var = X_train.columns[pipeline_final.named_steps["variance"].get_support()]
    
    importancias = pd.Series(clf_model.feature_importances_, index=colunas_apos_var)
    top_10 = importancias.sort_values(ascending=False).head(10)
    print("\n=== TOP 10 FEATURES MAIS IMPORTANTES (Etapa 2) ===")
    for col, imp in top_10.items():
        print(f"{col:<35}: {imp:.4f}")
        
    # 9. Salvar o bundle
    # Nota: salvamos o pipeline completo como o "modelo", pois ele encapsula a transformação de variância
    caminho_bundle = MODELS_DIR / "etapa2.joblib"
    print(f"\nSalvando o bundle do modelo multiclasse em: {caminho_bundle}")
    
    # Para o contrato, salvamos o pipeline final como o modelo. 
    # O pipeline já cuida do VarianceThreshold e do classificador internamente.
    salvar_bundle(
        caminho=caminho_bundle,
        modelo=pipeline_final,
        colunas=X_train.columns.tolist(), # colunas originais de entrada para o pipeline
        scaler=None, # scaler não é usado por Random Forest (está embutido se fosse LR)
        classes=classes_ordenadas,
        limiar=0.5
    )
    print("Fase 3 (B2) concluída com sucesso!")

if __name__ == "__main__":
    main()
