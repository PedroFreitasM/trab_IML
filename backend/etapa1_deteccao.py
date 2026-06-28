#!/usr/bin/env python3
"""
Etapa 1: Detecção Binária (Ataque vs. Normal)
Parte do Track B do pipeline CICIDS2017.

Carrega a amostra, subamostra a classe BENIGN, divide em Treino/Validação/Teste,
treina Decision Tree, Random Forest e Regressão Logística, compara resultados,
ajusta o limiar para priorizar o Recall de ataques e salva o bundle do modelo.
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

# Adiciona o diretório raiz ao python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import (
    carregar_dados,
    limpar,
    criar_targets,
    preparar_features,
    split,
    filtrar_variancia_zero,
    salvar_bundle,
    DATA_DIR,
    MODELS_DIR
)
from backend.visualizacao import plotar_matriz_confusao

def subamostrar_benign(df: pd.DataFrame, ratio: float = 3.0, seed: int = 42) -> pd.DataFrame:
    """Subamostra a classe BENIGN para ter no máximo ratio * total_ataques."""
    df_benign = df[df["target_bin"] == 0]
    df_ataque = df[df["target_bin"] == 1]
    
    n_ataques = len(df_ataque)
    n_benign_desejado = int(n_ataques * ratio)
    
    if len(df_benign) > n_benign_desejado:
        df_benign_sampled = df_benign.sample(n=n_benign_desejado, random_state=seed)
        df_final = pd.concat([df_benign_sampled, df_ataque], ignore_index=True)
        print(f"-> Subamostragem BENIGN: reduzido de {len(df_benign)} para {len(df_benign_sampled)} (ratio {ratio}:1)")
        return df_final
    return df

def treinar_e_avaliar_modelos(X_train, y_train, X_val, y_val):
    """Treina DT, RF e Regressão Logística, e retorna os modelos e suas métricas."""
    # 1. Preparação dos dados para Regressão Logística (requer normalização)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 2. Definição dos classificadores
    modelos = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    }
    
    resultados = {}
    
    for nome, clf in modelos.items():
        print(f"\nTreinando {nome}...")
        # Usa features normalizadas apenas para Regressão Logística
        if nome == "Logistic Regression":
            clf.fit(X_train_scaled, y_train)
            preds_val = clf.predict(X_val_scaled)
            probas_val = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            preds_val = clf.predict(X_val)
            probas_val = clf.predict_proba(X_val)[:, 1]
            
        rec = recall_score(y_val, preds_val)
        prec = precision_score(y_val, preds_val)
        f1 = f1_score(y_val, preds_val)
        
        print(f"[{nome} - Validação (limiar 0.5)] Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
        
        resultados[nome] = {
            "modelo": clf,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "probas_val": probas_val,
            "scaler": scaler if nome == "Logistic Regression" else None
        }
        
    return resultados

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
        
    # 2. Subamostragem de BENIGN
    df = subamostrar_benign(df, ratio=3.0)
    
    # 3. Preparação de features e split
    X, y = preparar_features(df, alvo="target_bin")
    print(f"Total de features originais: {X.shape[1]}")
    
    particoes = split(X, y, val=0.15, teste=0.15, seed=42)
    
    # 4. Filtrar variância zero (calculado no treino)
    X_train, X_val, X_test = filtrar_variancia_zero(
        particoes["X_train"], particoes["X_val"], particoes["X_test"]
    )
    y_train, y_val, y_test = particoes["y_train"], particoes["y_val"], particoes["y_test"]
    
    print(f"Shapes após filtros de variância:")
    print(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")
    print(f"Distribuição de classes no treino: {dict(y_train.value_counts())}")
    
    # 5. Treinar e comparar modelos
    resultados = treinar_e_avaliar_modelos(X_train, y_train, X_val, y_val)
    
    # 6. Escolher o melhor modelo (principalmente baseado no F1 e Recall na validação)
    # Por padrão do projeto, Random Forest é o modelo principal.
    melhor_nome = "Random Forest"
    melhor_info = resultados[melhor_nome]
    print(f"\nModelo selecionado como principal: {melhor_nome}")
    
    # 7. Otimizar limiar para priorizar Recall
    limiar_otimo = otimizar_limiar(y_val, melhor_info["probas_val"], recall_alvo=0.98)
    print(f"-> Limiar ótimo selecionado: {limiar_otimo:.4f}")
    
    # 8. Avaliação final no conjunto de TESTE
    clf = melhor_info["modelo"]
    scaler = melhor_info["scaler"]
    
    # Preparar features de teste
    if scaler is not None:
        X_test_proc = scaler.transform(X_test)
        probas_test = clf.predict_proba(X_test_proc)[:, 1]
    else:
        X_test_proc = X_test
        probas_test = clf.predict_proba(X_test_proc)[:, 1]
        
    y_pred_teste = (probas_test >= limiar_otimo).astype(int)
    
    print(f"\n=== RELATÓRIO DE CLASSIFICAÇÃO NO TESTE (Limiar: {limiar_otimo:.4f}) ===")
    print(classification_report(y_test, y_pred_teste, target_names=["Normal (0)", "Ataque (1)"]))
    
    # Imprimir matriz de confusão no terminal
    cm = confusion_matrix(y_test, y_pred_teste)
    print("Matriz de Confusão:")
    print(cm)
    
    # 9. Gerar heatmap e salvar
    caminho_imagem = root_dir / "images" / "mat_confusao_deteccao.png"
    plotar_matriz_confusao(
        cm, 
        classes=["Normal (0)", "Ataque (1)"], 
        caminho_salvar=caminho_imagem,
        titulo=f"Matriz de Confusão (Detecção) - {melhor_nome}\nLimiar: {limiar_otimo:.3f}"
    )
    
    # 10. Importância de features (para Random Forest)
    if hasattr(clf, "feature_importances_"):
        importancias = pd.Series(clf.feature_importances_, index=X_train.columns)
        top_10 = importancias.sort_values(ascending=False).head(10)
        print("\n=== TOP 10 FEATURES MAIS IMPORTANTES ===")
        for col, imp in top_10.items():
            print(f"{col:<35}: {imp:.4f}")
            
    # 11. Salvar o bundle
    caminho_bundle = MODELS_DIR / "etapa1.joblib"
    print(f"\nSalvando o bundle do modelo em: {caminho_bundle}")
    salvar_bundle(
        caminho=caminho_bundle,
        modelo=clf,
        colunas=X_train.columns.tolist(),
        scaler=scaler,
        classes=[0, 1],
        limiar=limiar_otimo
    )
    print("Fase 2 (B1) concluída com sucesso!")

if __name__ == "__main__":
    main()
