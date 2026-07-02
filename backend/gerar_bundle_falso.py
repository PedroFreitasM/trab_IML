"""
Gera bundles de modelo FALSOS em models/ para o Track C (dashboard) comecar sem
depender do Track B. Treina modelos triviais em dados aleatorios, no formato do
Contrato 2 (ver TASKS.md). NAO usar em producao -- as predicoes nao tem valor.

Uso:  python backend/gerar_bundle_falso.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessamento import (ARQUIVOS, COLUNAS_VAZAMENTO, DATA_DIR, MODELS_DIR,
                              salvar_bundle)

FAMILIAS = ["DDoS", "DoS", "PortScan", "Botnet", "Bruteforce", "Infiltration", "WebAttacks"]


def _colunas_referencia(n_sintetico: int = 20) -> list[str]:
    """Tenta nomes reais de colunas de um parquet; senao gera sinteticas f0..fN."""
    descartar = set(COLUNAS_VAZAMENTO)
    for nome in ARQUIVOS:
        caminho = DATA_DIR / nome
        if caminho.exists():
            cols = pd.read_parquet(caminho).columns.str.strip().tolist()
            return [c for c in cols if c not in descartar]
    return [f"f{i}" for i in range(n_sintetico)]


def main() -> None:
    cols = _colunas_referencia()
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((300, len(cols))), columns=cols)

    # Etapa 1 -- binario
    m1 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, rng.integers(0, 2, 300))
    salvar_bundle(MODELS_DIR / "rf_etapa1.joblib", m1, cols, classes=[0, 1], limiar=0.5)

    # Etapa 2 -- multiclasse (familias)
    m2 = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, rng.choice(FAMILIAS, 300))
    salvar_bundle(MODELS_DIR / "rf_etapa2.joblib", m2, cols, classes=FAMILIAS)

    # Etapa 1 -- DT binario (interpretabilidade)
    m3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, rng.integers(0, 2, 300))
    salvar_bundle(MODELS_DIR / "dt_etapa1.joblib", m3, cols, classes=[0, 1], limiar=0.5)

    # Etapa 2 -- DT multiclasse (interpretabilidade)
    m4 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, rng.choice(FAMILIAS, 300))
    salvar_bundle(MODELS_DIR / "dt_etapa2.joblib", m4, cols, classes=FAMILIAS)

    print(f"Bundles falsos criados em {MODELS_DIR} ({len(cols)} colunas).")
    print("ATENCAO: modelos sem valor preditivo -- apenas para o dashboard rodar.")


if __name__ == "__main__":
    main()
