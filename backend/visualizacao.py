"""
Funções de visualização para o pipeline do CICIDS2017.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plotar_matriz_confusao(cm, classes, caminho_salvar, titulo="Matriz de Confusão"):
    """Plota a matriz de confusão usando Seaborn e a salva em um arquivo.
    
    Args:
        cm: Matriz de confusão (array-like).
        classes: Lista de strings com os nomes das classes para os eixos.
        caminho_salvar: Caminho completo ou Path para salvar o arquivo de imagem.
        titulo: Título do gráfico.
    """
    caminho_salvar = Path(caminho_salvar)
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Previsto: {c}" for c in classes],
                yticklabels=[f"Real: {c}" for c in classes])
    
    plt.title(titulo, pad=15, fontsize=14)
    plt.ylabel('Rótulo Verdadeiro (Realidade do Dataset)', labelpad=10, fontsize=11)
    plt.xlabel('Previsão do Algoritmo (Classificação)', labelpad=10, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {caminho_salvar}")
