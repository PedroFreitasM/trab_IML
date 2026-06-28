import pandas as pd
import numpy as np
from sklearn,model_selection import train_test_split

ficheiros = [
	'Benign-Monday-no-metadata.parquet',
	'DDos-Friday-no-metadata.parquet',
	'Portscan-Friday-no-metadata.parquet'
]

lista_dataframes = []

for ficheiro in ficheiros_alvo:
	df_temp = pd.read_parquet(ficheiro)

	df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
	df_temp.dropna(inplace=True)

	lista_dataframes.append(df_temp)

	df_projeto = pd.concat(lista_dataframes, ignore_index=True)
	print(f"Dataset com {df_projeto.shape[0]} linhas")


