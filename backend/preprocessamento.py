"""
Pre-processamento reutilizavel do pipeline CICIDS2017 (2 etapas).

Define o CONTRATO 1 (ver TASKS.md): funcoes importadas pelos scripts de treino
(Etapa 1 / Etapa 2) e pelo dashboard. Implementacao base pronta; itens marcados
TODO(A2)/TODO(A4) cabem ao Track A finalizar apos inspecionar os dados.

IMPORTANTE (anti-leakage): StandardScaler, filtro de variancia, SMOTE e tuning
NAO ficam aqui -- pertencem ao Pipeline de treino (Track B, Fase 4) e devem ser
ajustados SO no treino.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ARQUIVOS = [
    "Benign-Monday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
]

# Filtro defensivo de vazamento (os arquivos "no-metadata" provavelmente ja nao
# tem estas colunas). Mesma lista de analise_matriz.py.
# TODO(A2): decidir se 'Destination Port'/'Protocol' tambem devem sair.
COLUNAS_VAZAMENTO = [
    "Source IP", "Destination IP", "Source Port", "Timestamp", "Flow ID", "Label",
]

# Mapa PRELIMINAR de Label -> familia (chave em MAIUSCULAS, sem espacos nas bordas).
# TODO(A2): confirmar/fechar com os valores reais de df['Label'].unique().
MAPA_FAMILIAS = {
    "DDOS": "DDoS",
    "DOS HULK": "DoS", "DOS GOLDENEYE": "DoS", "DOS SLOWLORIS": "DoS",
    "DOS SLOWHTTPTEST": "DoS", "HEARTBLEED": "DoS",
    "PORTSCAN": "PortScan",
    "BOT": "Botnet",
    "FTP-PATATOR": "Bruteforce", "SSH-PATATOR": "Bruteforce",
    "WEB ATTACK - BRUTE FORCE": "WebAttacks", "WEB ATTACK - XSS": "WebAttacks",
    "WEB ATTACK - SQL INJECTION": "WebAttacks",
    "INFILTRATION": "Infiltration",
}


def carregar_dados(arquivos: list[str] | None = None,
                   columns: list[str] | None = None) -> pd.DataFrame:
    """Le e concatena os parquet de data/ (todos por padrao).

    columns: subconjunto de colunas para economizar RAM (opcional).
    """
    arquivos = arquivos or ARQUIVOS
    dfs = [pd.read_parquet(DATA_DIR / nome, columns=columns) for nome in arquivos]
    return pd.concat(dfs, ignore_index=True)


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas (CICIDS tem espacos), troca inf->NaN e remove NaN."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _mapear_familia(label: str) -> str:
    chave = str(label).strip().upper()
    if chave in MAPA_FAMILIAS:
        return MAPA_FAMILIAS[chave]
    # Fallback por palavra-chave (robusto a grafias). TODO(A2): trocar por mapa fechado.
    for termo, familia in (
        ("DDOS", "DDoS"), ("DOS", "DoS"), ("PORT", "PortScan"), ("BOT", "Botnet"),
        ("PATATOR", "Bruteforce"), ("BRUTE", "Bruteforce"), ("WEB", "WebAttacks"),
        ("XSS", "WebAttacks"), ("SQL", "WebAttacks"), ("INFIL", "Infiltration"),
        ("HEART", "DoS"),
    ):
        if termo in chave:
            return familia
    return "Outro"


def criar_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona 'target_bin' (0=BENIGN, 1=ataque) e 'target_tipo' (familia do ataque)."""
    df = df.copy()
    rotulo = df["Label"].astype(str).str.strip().str.upper()
    df["target_bin"] = (rotulo != "BENIGN").astype(int)
    df["target_tipo"] = df["Label"].apply(_mapear_familia)
    return df


def preparar_features(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) e y (alvo), removendo targets e colunas de vazamento.

    alvo: 'target_bin' (Etapa 1) ou 'target_tipo' (Etapa 2).
    """
    descartar = set(COLUNAS_VAZAMENTO) | {"target_bin", "target_tipo"}
    X = df.drop(columns=[c for c in descartar if c in df.columns])
    y = df[alvo]
    return X, y


def split(X, y, val: float = 0.15, teste: float = 0.15, seed: int = 42) -> dict:
    """Split estratificado 70/15/15 -> dict com X_train/X_val/X_test/y_train/y_val/y_test."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=teste, random_state=seed, stratify=y)
    val_rel = val / (1.0 - teste)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=seed, stratify=y_tmp)
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test}


# ---- Contrato 2: bundle de modelo (compartilhado por Track B e C) ----
def salvar_bundle(caminho, modelo, colunas, scaler=None, classes=None, limiar=0.5) -> None:
    """Salva um bundle de inferencia auto-suficiente (ver Contrato 2 no TASKS.md)."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"modelo": modelo, "scaler": scaler, "colunas": list(colunas),
         "classes": list(classes) if classes is not None else None, "limiar": limiar},
        caminho,
    )


def carregar_bundle(caminho) -> dict:
    """Carrega um bundle salvo por salvar_bundle()."""
    return joblib.load(caminho)


if __name__ == "__main__":
    # Smoke test / parte do GATE 1.A. Requer venv 3.12 + dados em data/.
    print("Carregando e limpando os 8 parquet...")
    df = criar_targets(limpar(carregar_dados()))
    print("shape:", df.shape)
    print("NaN restantes:", int(df.isna().sum().sum()))
    print("\ntarget_bin:\n", df["target_bin"].value_counts())
    print("\ntarget_tipo:\n", df["target_tipo"].value_counts())
    if (df["target_tipo"] == "Outro").any():
        print("\n[ATENCAO] Ha rotulos nao mapeados (familia 'Outro'). "
              "Ajustar MAPA_FAMILIAS -- TODO(A2).")
