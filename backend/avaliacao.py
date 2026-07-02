#!/usr/bin/env python3
"""
Avaliação em Cascata Fim-a-Fim
Parte do Track B (B4) do pipeline CICIDS2017.

Simula o comportamento do dashboard e do ambiente de produção:
1. Aplica o modelo da Etapa 1 (Detecção) para classificar o tráfego em Normal/Ataque.
2. Para os fluxos classificados como Ataque, aplica a Etapa 2 (Identificação) para obter o tipo.
3. Mede a performance combinada real, incluindo os ataques não detectados pela Etapa 1.
4. Gera e salva a matriz de confusão da cascata em images/mat_confusao_cascata.png.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Adiciona o diretório raiz ao python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import (
    preparar_features,
    split,
    carregar_bundle,
    carregar_holdout_canonico,
    DATA_DIR,
    MODELS_DIR
)
from backend.visualizacao import plotar_matriz_confusao

def main():
    print("=== AVALIAÇÃO EM CASCATA FIM-A-FIM ===")
    
    # 1. Carregar bundles dos modelos
    caminho_e1 = MODELS_DIR / "rf_etapa1.joblib"
    caminho_e2 = MODELS_DIR / "rf_etapa2.joblib"
    
    if not caminho_e1.exists() or not caminho_e2.exists():
        print("Erro: Os bundles rf_etapa1.joblib e/ou rf_etapa2.joblib não foram encontrados em models/")
        print("Por favor, execute etapa1_deteccao.py e etapa2_identificacao.py primeiro.")
        return
        
    print("Carregando bundles...")
    bundle1 = carregar_bundle(caminho_e1)
    bundle2 = carregar_bundle(caminho_e2)
    
    # 2. Carregar dados de teste a partir do holdout canônico
    #    (garantia de que nenhuma linha foi vista no treino das Etapas 1/2)
    caminho_amostra = DATA_DIR / "amostra.parquet"
    if not caminho_amostra.exists():
        print(f"Erro: Amostra não encontrada em {caminho_amostra}.")
        return
        
    print("Carregando conjunto de teste a partir do holdout canônico...")
    df = pd.read_parquet(caminho_amostra)
    
    try:
        idx_teste = carregar_holdout_canonico()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return
    
    df_teste = df.loc[df.index.isin(idx_teste)].copy()
    
    # Filtrar classes com menos de 2 membros
    contagem = df_teste["target_tipo"].value_counts()
    classes_raras = contagem[contagem < 2].index.tolist()
    if classes_raras:
        print(f"-> AVISO: Removendo classes com menos de 2 membros: {classes_raras}")
        df_teste = df_teste[~df_teste["target_tipo"].isin(classes_raras)].copy()
        
    X_test, y_test_tipo = preparar_features(df_teste, alvo="target_tipo")
    
    print(f"Total de amostras no conjunto de teste: {len(X_test)}")
    
    # 3. Executar Predição da Etapa 1 (Detecção)
    # Alinhar colunas necessárias da Etapa 1
    X_test_e1 = X_test[bundle1["colunas"]]
    if bundle1["scaler"] is not None:
        X_test_e1_proc = bundle1["scaler"].transform(X_test_e1)
    else:
        X_test_e1_proc = X_test_e1
        
    probas_e1 = bundle1["modelo"].predict_proba(X_test_e1_proc)[:, 1]
    preds_e1 = (probas_e1 >= bundle1["limiar"]).astype(int)
    
    # 4. Executar Predição da Etapa 2 (Identificação)
    # Para cada registro, se Etapa 1 previu 0 -> "Benign"
    # Se previu 1 -> tipo predito pelo pipeline da Etapa 2
    preds_cascata = []
    
    # Extrair modelo (pipeline) da Etapa 2
    pipeline_e2 = bundle2["modelo"]
    colunas_e2 = bundle2["colunas"]
    
    print("Processando previsões em cascata...")
    
    # Otimização: processar previsões em lote
    # Criamos um array de 'Benign' e depois atualizamos as posições de ataque
    preds_cascata = np.full(len(X_test), "Benign", dtype=object)
    
    indices_ataque = np.where(preds_e1 == 1)[0]
    if len(indices_ataque) > 0:
        X_test_e2 = X_test.iloc[indices_ataque][colunas_e2]
        preds_e2 = pipeline_e2.predict(X_test_e2)
        preds_cascata[indices_ataque] = preds_e2
        
    # 5. Avaliação
    # Nomes das classes presentes no teste original
    classes_reais = sorted(y_test_tipo.unique())
    # O modelo prediz 'Benign' ou as classes da Etapa 2
    classes_preditas = sorted(np.unique(preds_cascata))
    todas_classes = sorted(list(set(classes_reais) | set(classes_preditas)))
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO DA CASCATA FIM-A-FIM ===")
    print(classification_report(y_test_tipo, preds_cascata, zero_division=0))
    
    # 6. Matriz de Confusão Combinada
    cm = confusion_matrix(y_test_tipo, preds_cascata, labels=todas_classes)
    
    caminho_imagem = root_dir / "images" / "mat_confusao_cascata.png"
    plotar_matriz_confusao(
        cm,
        classes=todas_classes,
        caminho_salvar=caminho_imagem,
        titulo="Matriz de Confusão Cascata Fim-a-Fim\n(Detecção + Identificação)"
    )
    
    # 7. Relatório de Ataques Perdidos (Falsos Negativos da Etapa 1)
    # São registros que eram Ataques na realidade, mas o modelo 1 previu Normal (0/Benign)
    print("\n=== ANÁLISE DE ATAQUES NÃO DETECTADOS ===")
    erros_deteccao = (y_test_tipo != "Benign") & (preds_cascata == "Benign")
    total_erros = erros_deteccao.sum()
    total_ataques = (y_test_tipo != "Benign").sum()
    
    print(f"Total de ataques no teste: {total_ataques}")
    if total_ataques > 0:
        print(f"Ataques não detectados (Falsos Negativos): {total_erros} ({total_erros / total_ataques * 100:.2f}%)")
    else:
        print("Nenhum ataque presente no conjunto de teste.")
    
    if total_erros > 0:
        print("\nDistribuição dos ataques não detectados por família:")
        dist_erros = y_test_tipo[erros_deteccao].value_counts()
        for tipo, qtd in dist_erros.items():
            qtd_total = (y_test_tipo == tipo).sum()
            pct = (qtd / qtd_total * 100) if qtd_total > 0 else 0.0
            print(f"  - {tipo:<15}: {qtd:>3} de {qtd_total:>3} ({pct:.1f}%)")
            
    print("\nAvaliação cascata (B4) concluída com sucesso!")

if __name__ == "__main__":
    main()
