import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.preprocessamento import carregar_bundle, MODELS_DIR

class TestModelagemIntegration(unittest.TestCase):
    def test_etapa1_bundle_loading_and_prediction(self):
        caminho_e1 = MODELS_DIR / "rf_etapa1.joblib"
        self.assertTrue(caminho_e1.exists(), f"Bundle {caminho_e1} não encontrado. Execute gerar_bundle_falso.py ou etapa1_deteccao.py primeiro.")
        
        bundle = carregar_bundle(caminho_e1)
        self.assertIn("modelo", bundle)
        self.assertIn("scaler", bundle)
        self.assertIn("colunas", bundle)
        self.assertIn("classes", bundle)
        self.assertIn("limiar", bundle)
        
        # Testar predição com dados dummy
        colunas = bundle["colunas"]
        n_features = len(colunas)
        
        # Cria um DataFrame de teste dummy com 2 amostras
        X_dummy = pd.DataFrame(
            np.random.randn(2, n_features), 
            columns=colunas
        )
        
        # A Etapa 1 usa o pipeline completo diretamente (com VarianceThreshold, StandardScaler e RandomForest)
        pipeline = bundle["modelo"]
        
        # Obter probabilidades
        probas = pipeline.predict_proba(X_dummy)[:, 1]
        self.assertEqual(len(probas), 2)
        self.assertTrue(np.all(probas >= 0) and np.all(probas <= 1))
        
        # Aplicar limiar
        preds = (probas >= bundle["limiar"]).astype(int)
        self.assertEqual(len(preds), 2)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_etapa2_bundle_loading_and_prediction(self):
        caminho_e2 = MODELS_DIR / "rf_etapa2.joblib"
        self.assertTrue(caminho_e2.exists(), f"Bundle {caminho_e2} não encontrado. Execute gerar_bundle_falso.py ou etapa2_identificacao.py primeiro.")
        
        bundle = carregar_bundle(caminho_e2)
        self.assertIn("modelo", bundle)
        self.assertIn("scaler", bundle)
        self.assertIn("colunas", bundle)
        self.assertIn("classes", bundle)
        self.assertIn("limiar", bundle)
        
        colunas = bundle["colunas"]
        n_features = len(colunas)
        
        # Cria um DataFrame de teste dummy com 2 amostras
        X_dummy = pd.DataFrame(
            np.random.randn(2, n_features), 
            columns=colunas
        )
        
        pipeline = bundle["modelo"]
        
        # Obter predições multiclasse
        preds = pipeline.predict(X_dummy)
        self.assertEqual(len(preds), 2)
        
        # As predições devem pertencer às classes do bundle
        for pred in preds:
            self.assertIn(pred, bundle["classes"])

    def test_dt_etapa1_bundle_loading_and_prediction(self):
        caminho_e1 = MODELS_DIR / "dt_etapa1.joblib"
        self.assertTrue(caminho_e1.exists(), f"Bundle {caminho_e1} não encontrado. Execute dt_interpretabilidade.py primeiro.")
        
        bundle = carregar_bundle(caminho_e1)
        self.assertIn("modelo", bundle)
        self.assertIn("scaler", bundle)
        self.assertIn("colunas", bundle)
        self.assertIn("classes", bundle)
        self.assertIn("limiar", bundle)
        
        colunas = bundle["colunas"]
        X_dummy = pd.DataFrame(
            np.random.randn(2, len(colunas)), 
            columns=colunas
        )
        
        pipeline = bundle["modelo"]
        probas = pipeline.predict_proba(X_dummy)[:, 1]
        self.assertEqual(len(probas), 2)
        self.assertTrue(np.all(probas >= 0) and np.all(probas <= 1))
        
        preds = (probas >= bundle["limiar"]).astype(int)
        self.assertEqual(len(preds), 2)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_dt_etapa2_bundle_loading_and_prediction(self):
        caminho_e2 = MODELS_DIR / "dt_etapa2.joblib"
        self.assertTrue(caminho_e2.exists(), f"Bundle {caminho_e2} não encontrado. Execute dt_interpretabilidade.py primeiro.")
        
        bundle = carregar_bundle(caminho_e2)
        self.assertIn("modelo", bundle)
        self.assertIn("scaler", bundle)
        self.assertIn("colunas", bundle)
        self.assertIn("classes", bundle)
        
        colunas = bundle["colunas"]
        X_dummy = pd.DataFrame(
            np.random.randn(2, len(colunas)), 
            columns=colunas
        )
        
        pipeline = bundle["modelo"]
        preds = pipeline.predict(X_dummy)
        self.assertEqual(len(preds), 2)
        for pred in preds:
            self.assertIn(pred, bundle["classes"])

if __name__ == "__main__":
    unittest.main()
