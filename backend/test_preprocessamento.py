import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Adiciona o diretório raiz ao python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.preprocessamento import (
    _mapear_familia,
    limpar,
    criar_targets,
    preparar_features,
    split,
    filtrar_variancia_zero,
    salvar_bundle,
    carregar_bundle,
    MAPA_FAMILIAS
)

class TestPreprocessamento(unittest.TestCase):
    def test_mapear_familia_known(self):
        self.assertEqual(_mapear_familia("BENIGN"), "Benign")
        self.assertEqual(_mapear_familia("DOS HULK"), "DoS")
        self.assertEqual(_mapear_familia("WEB ATTACK \uFFFD BRUTE FORCE"), "WebAttacks")
        self.assertEqual(_mapear_familia(" SSH-PATATOR "), "Bruteforce")

    def test_mapear_familia_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            _mapear_familia("UNKNOWN_LABEL")

    def test_limpar_removes_nan_and_inf_and_strips_columns(self):
        df = pd.DataFrame({
            " Col1 ": [1.0, np.inf, 2.0],
            "Col2  ": [3.0, 4.0, np.nan],
            "Col3": [5.0, 6.0, 7.0]
        })
        cleaned = limpar(df)
        self.assertListEqual(list(cleaned.columns), ["Col1", "Col2", "Col3"])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Col1"], 1.0)

    def test_criar_targets(self):
        df = pd.DataFrame({
            "Label": ["BENIGN", "DOS HULK", "PORTSCAN"]
        })
        res = criar_targets(df)
        self.assertListEqual(list(res["target_bin"]), [0, 1, 1])
        self.assertListEqual(list(res["target_tipo"]), ["Benign", "DoS", "PortScan"])

    def test_preparar_features(self):
        df = pd.DataFrame({
            "Source IP": [1, 2],
            "Label": ["BENIGN", "DOS HULK"],
            "target_bin": [0, 1],
            "target_tipo": ["Benign", "DoS"],
            "Feature1": [10.0, 20.0],
            "Feature2": [0.1, 0.2]
        })
        X, y = preparar_features(df, "target_tipo")
        self.assertNotIn("Source IP", X.columns)
        self.assertNotIn("Label", X.columns)
        self.assertNotIn("target_bin", X.columns)
        self.assertNotIn("target_tipo", X.columns)
        self.assertListEqual(list(X.columns), ["Feature1", "Feature2"])
        self.assertListEqual(list(y), ["Benign", "DoS"])

    def test_split_proportions(self):
        y = pd.Series(["A"] * 80 + ["B"] * 20)
        X = pd.DataFrame({"Feature": range(100)})
        particoes = split(X, y, val=0.15, teste=0.15, seed=42)
        
        self.assertEqual(len(particoes["X_train"]), 69)
        self.assertEqual(len(particoes["X_val"]), 16)
        self.assertEqual(len(particoes["X_test"]), 15)
        
        self.assertEqual((particoes["y_train"] == "A").sum(), 55)
        self.assertEqual((particoes["y_train"] == "B").sum(), 14)

    def test_filtrar_variancia_zero(self):
        X_train = pd.DataFrame({
            "Const": [1.0, 1.0, 1.0],
            "Var": [1.0, 2.0, 3.0]
        })
        X_val = pd.DataFrame({
            "Const": [1.0, 1.0, 1.0],
            "Var": [4.0, 5.0, 6.0]
        })
        X_test = pd.DataFrame({
            "Const": [1.0, 1.0, 1.0],
            "Var": [7.0, 8.0, 9.0]
        })
        
        tr_f, val_f, te_f = filtrar_variancia_zero(X_train, X_val, X_test)
        self.assertNotIn("Const", tr_f.columns)
        self.assertNotIn("Const", val_f.columns)
        self.assertNotIn("Const", te_f.columns)
        self.assertIn("Var", tr_f.columns)
        self.assertIn("Var", val_f.columns)
        self.assertIn("Var", te_f.columns)

    def test_salvar_e_carregar_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            caminho_bundle = Path(tmpdir) / "bundle.joblib"
            modelo = "MOCK_MODEL"
            colunas = ["F1", "F2"]
            salvar_bundle(caminho_bundle, modelo, colunas, scaler="MOCK_SCALER", classes=["A", "B"], limiar=0.4)
            
            bundle = carregar_bundle(caminho_bundle)
            self.assertEqual(bundle["modelo"], "MOCK_MODEL")
            self.assertEqual(bundle["scaler"], "MOCK_SCALER")
            self.assertListEqual(bundle["colunas"], ["F1", "F2"])
            self.assertListEqual(bundle["classes"], ["A", "B"])
            self.assertEqual(bundle["limiar"], 0.4)

if __name__ == '__main__':
    unittest.main()
