import unittest
import sys
from pathlib import Path
import pandas as pd

# Find workspace root directory (which contains requirements.txt)
root_dir = Path(__file__).resolve().parent
while root_dir.parent != root_dir:
    if (root_dir / "requirements.txt").exists():
        break
    root_dir = root_dir.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import carregar_dados, limpar, criar_targets, DATA_DIR

class TestGerarAmostra(unittest.TestCase):
    def setUp(self):
        self.caminho_amostra = DATA_DIR / "amostra.parquet"
        
    def test_amostra_exists_and_valid(self):
        # 1. Check if the file is created
        self.assertTrue(
            self.caminho_amostra.exists(), 
            f"O arquivo {self.caminho_amostra} deve existir. Execute 'backend/gerar_amostra.py' primeiro."
        )
        
        # 2. Load the sample
        df_amostra = pd.read_parquet(self.caminho_amostra)
        n_linhas = len(df_amostra)
        
        # 3. Check number of rows (between 10,000 and 50,000)
        self.assertTrue(
            10000 <= n_linhas <= 50000, 
            f"Número de linhas na amostra ({n_linhas}) deve estar entre 10.000 e 50.000."
        )
        
        # 4. Check presence of target_tipo and target_bin
        self.assertIn("target_tipo", df_amostra.columns, "A coluna 'target_tipo' deve estar presente na amostra")
        self.assertIn("target_bin", df_amostra.columns, "A coluna 'target_bin' deve estar presente na amostra")
        
        # 5. Load full dataset to compare distributions dynamically
        df_completo = carregar_dados()
        df_limpo = limpar(df_completo)
        df_targets = criar_targets(df_limpo)
        
        # Compute normalized distributions (percentages)
        dist_completo = df_targets['target_tipo'].value_counts(normalize=True) * 100
        dist_amostra = df_amostra['target_tipo'].value_counts(normalize=True) * 100
        
        # Check every class in full dataset
        for cls in dist_completo.index:
            p_comp = dist_completo[cls]
            p_amostra = dist_amostra.get(cls, 0.0)
            desvio = abs(p_amostra - p_comp)
            
            # Assert that deviation is within the +-2.0% limit (percentage points)
            self.assertLessEqual(
                desvio, 
                2.0, 
                f"Classe '{cls}' tem desvio de {desvio:.4f}%, que excede o limite de +-2% (Completo: {p_comp:.4f}%, Amostra: {p_amostra:.4f}%)"
            )

if __name__ == '__main__':
    unittest.main()
