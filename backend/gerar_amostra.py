#!/usr/bin/env python3
"""
Script to generate a stratified subsample of the CICIDS2017 dataset.
Saves the subsample to 'data/amostra.parquet'.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Find workspace root directory (which contains requirements.txt)
root_dir = Path(__file__).resolve().parent
while root_dir.parent != root_dir:
    if (root_dir / "requirements.txt").exists():
        break
    root_dir = root_dir.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import carregar_dados, limpar, criar_targets, DATA_DIR

def gerar_amostra(tamanho_amostra: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Carrega o dataset completo, realiza a amostragem estratificada por target_tipo,
    calcula e imprime a distribuição de classes, e salva o resultado em amostra.parquet.
    """
    print("Carregando o dataset completo...")
    df_completo = carregar_dados()
    
    print("Limpando e criando targets...")
    df_limpo = limpar(df_completo)
    df_targets = criar_targets(df_limpo)
    
    total_linhas = len(df_targets)
    print(f"Total de linhas no dataset completo: {total_linhas}")
    
    # Validação do tamanho solicitado
    if not (10000 <= tamanho_amostra <= 50000):
        raise ValueError(f"O tamanho da amostra ({tamanho_amostra}) deve estar entre 10.000 e 50.000 linhas.")
        
    # Stratified split using train_test_split
    print(f"Gerando amostra estratificada com {tamanho_amostra} linhas...")
    _, df_amostra = train_test_split(
        df_targets,
        test_size=tamanho_amostra,
        stratify=df_targets['target_tipo'],
        random_state=seed
    )
    
    # Calculate distributions
    print("\n=== VERIFICAÇÃO DA DISTRIBUIÇÃO DAS CLASSES ('target_tipo') ===")
    dist_completo = df_targets['target_tipo'].value_counts(normalize=True) * 100
    dist_amostra = df_amostra['target_tipo'].value_counts(normalize=True) * 100
    
    todas_classes = sorted(list(df_targets['target_tipo'].unique()))
    
    validado = True
    print(f"{'Classe':<15} | {'Completo (%)':<15} | {'Amostra (%)':<15} | {'Desvio (%)':<12} | {'Status':<10}")
    print("-" * 65)
    for cls in todas_classes:
        p_comp = dist_completo.get(cls, 0.0)
        p_amostra = dist_amostra.get(cls, 0.0)
        desvio = p_amostra - p_comp
        abs_desvio = abs(desvio)
        
        status = "OK" if abs_desvio <= 2.0 else "ERRO (>2%)"
        if abs_desvio > 2.0:
            validado = False
            
        print(f"{cls:<15} | {p_comp:>13.4f}% | {p_amostra:>13.4f}% | {desvio:>+11.4f}% | {status:<10}")
        
    print("-" * 65)
    if validado:
        print("Sucesso! O desvio de todas as classes está dentro do limite de +-2%.")
    else:
        print("Erro: Alguma classe excedeu o limite de desvio de +-2%.")
        
    # Save the sample
    caminho_amostra = DATA_DIR / "amostra.parquet"
    print(f"\nSalvando a amostra em: {caminho_amostra}")
    df_amostra.to_parquet(caminho_amostra, index=False)
    print("Amostra gerada e salva com sucesso!")
    
    return df_amostra

if __name__ == "__main__":
    gerar_amostra()
