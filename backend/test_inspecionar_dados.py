import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.inspecionar_dados import inspecionar

class TestInspecionarDados(unittest.TestCase):
    @patch('backend.inspecionar_dados.carregar_dados')
    @patch('backend.inspecionar_dados.ARQUIVOS', ['mock_file.parquet'])
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_inspecionar_runs_and_writes_file(self, mock_print, mock_file, mock_carregar):
        # Create a mock DataFrame representing standard features and a label
        mock_df = pd.DataFrame({
            'Flow Duration': [100.0, -5.0, 50.0],
            'Total Fwd Packets': [1.0, 2.0, 3.0],
            'Fwd Packet Length Max': [0.5, 0.6, 0.7],
            'Flow Bytes/s': [10.0, 20.0, 30.0],
            'Flow Packets/s': [1.0, 2.0, 3.0],
            'Bwd PSH Flags': [0.0, 0.0, 0.0],  # Constant feature
            'Label': ['Benign', 'DDoS', 'Benign']
        })
        
        mock_carregar.return_value = mock_df
        
        # Run inspection
        inspecionar()
        
        # Verify carregar_dados was called
        self.assertTrue(mock_carregar.called)
        
        # Verify file was written
        mock_file.assert_called_once()
        
        # Check the written content via mock file write calls
        written_data = "".join([call[0][0] for call in mock_file().write.call_args_list])
        
        self.assertIn("INSPE", written_data.upper())
        self.assertIn("CICIDS2017", written_data)
        self.assertIn("Total de colunas (features + Label): 7", written_data)
        self.assertIn("Total de features (excluindo 'Label'): 6", written_data)
        self.assertIn("valores fora de [0, 1]: 4 de 6", written_data)
        self.assertIn("Total de valores NaN globais: 0", written_data)
        self.assertIn("Total de valores Inf globais: 0", written_data)
        self.assertIn("Bwd PSH Flags", written_data)

if __name__ == '__main__':
    unittest.main()
