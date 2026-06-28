"""
Script de inspeção de dados para o dataset CICIDS2017.
Este script carrega os 8 arquivos parquet via preprocessamento.py,
coleta estatísticas básicas e salva os resultados em backend/inspecao_resultado.txt.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Adiciona o diretório raiz ao python path para garantir importações corretas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.preprocessamento import carregar_dados, ARQUIVOS


def inspecionar():
    linhas_resultado = []

    def log(msg):
        print(msg)
        linhas_resultado.append(msg)

    log("=== INSPEÇÃO DO DATASET CICIDS2017 ===")

    # 1. Valores únicos da coluna 'Label' por arquivo (lê apenas 'Label' -> rápido, baixa RAM)
    log("\n1. Valores únicos da coluna 'Label' por arquivo (value_counts):")
    total_linhas_calc = 0

    for arquivo in ARQUIVOS:
        log(f"\n--- Arquivo: {arquivo} ---")
        try:
            serie_label = carregar_dados(arquivos=[arquivo], columns=['Label'])['Label']
            total_linhas_calc += len(serie_label)
            for k, v in serie_label.value_counts(dropna=False).items():
                # ascii() evita UnicodeEncodeError ao imprimir a string contendo � no Windows
                log(f"  Label {ascii(k)}: {v}")
        except Exception as e:
            log(f"Erro ao carregar/inspecionar {arquivo}: {e}")

    # 2. Carregar dataset completo (uma única vez) para a análise estatística global
    log("\nCarregando todo o dataset para análise estatística global...")
    df_global = carregar_dados()
    total_linhas_global = len(df_global)
    total_features = len(df_global.columns)

    # Presença de colunas de metadados (a partir das colunas do dataset global)
    log("\n2. Presença de colunas de metadados (IP, Timestamp, Flow ID):")
    colunas_metadados_pesquisa = ['source ip', 'destination ip', 'source port', 'timestamp', 'flow id']
    metadados_encontrados = [c for c in df_global.columns if any(m in c.lower() for m in colunas_metadados_pesquisa)]
    if metadados_encontrados:
        log(f"Metadados detectados: {metadados_encontrados}")
    else:
        log("Nenhuma coluna de metadados (IP, Timestamp, Flow ID) encontrada nos arquivos parquet.")

    log(f"\n3. Estatísticas Globais:")
    log(f"Total de arquivos lidos: {len(ARQUIVOS)}")
    log(f"Total de colunas (features + Label): {total_features}")
    log(f"Total de features (excluindo 'Label'): {total_features - 1}")
    log(f"Total de linhas (soma individual): {total_linhas_calc}")
    log(f"Total de linhas (concatenado): {total_linhas_global}")

    # 4. Verificação de Normalização
    log("\n4. Verificação de Normalização:")
    num_cols = df_global.select_dtypes(include='number').columns
    desc = df_global[num_cols].describe().loc[['min', 'max', 'mean']]
    colunas_fora_01 = ((df_global[num_cols] < 0) | (df_global[num_cols] > 1)).any()
    num_colunas_fora_01 = colunas_fora_01.sum()

    log(f"Total de colunas numéricas: {len(num_cols)}")
    log(f"Colunas numéricas com valores fora de [0, 1]: {num_colunas_fora_01} de {len(num_cols)}")

    if num_colunas_fora_01 > 0:
        log("Conclusão: O dataset NÃO está normalizado. Existem colunas com valores brutos e em escalas muito diferentes.")
        log("\nExemplos de estatísticas de features para demonstração:")
        exemplos_cols = ['Flow Duration', 'Total Fwd Packets', 'Fwd Packet Length Max', 'Flow Bytes/s', 'Flow Packets/s']
        for col in exemplos_cols:
            if col in df_global.columns:
                val_min = desc.iloc[0][col]
                val_max = desc.iloc[1][col]
                val_mean = desc.iloc[2][col]
                log(f"  Feature '{col}': Min = {val_min:.3f}, Max = {val_max:.3f}, Mean = {val_mean:.3f}")
    else:
        log("Conclusão: O dataset parece estar normalizado (todos os valores numéricos estão em [0, 1]).")

    # 5. Presença de valores NaN e Inf
    log("\n5. Presença de valores NaN e Inf (antes da limpeza por limpar()):")
    nans_por_col = df_global.isna().sum()
    total_nans = nans_por_col.sum()
    infs_por_col = np.isinf(df_global[num_cols]).sum()
    total_infs = infs_por_col.sum()

    log(f"Total de valores NaN globais: {total_nans}")
    log(f"Total de valores Inf globais: {total_infs}")
    for c in ['Flow Bytes/s', 'Flow Packets/s']:
        if c in df_global.columns:
            c_nan = df_global[c].isna().sum()
            c_inf = np.isinf(df_global[c]).sum()
            log(f"  Coluna '{c}': NaNs = {c_nan}, Infs = {c_inf}")

    # 6. Adicional: Variância zero
    stds = df_global[num_cols].std()
    zero_var = stds[stds == 0].index.tolist()
    log(f"\n6. Adicional - Colunas de variável constante (variância zero):")
    log(f"Total: {len(zero_var)}")
    log(f"Colunas: {zero_var}")

    # Salvar resultados
    caminho_saida = Path(__file__).resolve().parent / "inspecao_resultado.txt"
    try:
        with open(caminho_saida, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(linhas_resultado))
        log(f"\nResultados gravados com sucesso em: {caminho_saida}")
    except Exception as e:
        print(f"Erro ao salvar arquivo de resultados: {e}")


if __name__ == "__main__":
    inspecionar()
