"""
Pre-processamento reutilizavel do pipeline CICIDS2017 (2 etapas).

Define o CONTRATO 1 (ver TASKS.md): funcoes importadas pelos scripts de treino
(Etapa 1 / Etapa 2) e pelo dashboard.

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
INDICES_TESTE_PATH = MODELS_DIR / "indices_teste.joblib"

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
# tem estas colunas).
COLUNAS_VAZAMENTO = [
    "Source IP", "Destination IP", "Source Port", "Timestamp", "Flow ID", "Label",
]

# Mapa COMPLETO de Label -> familia (chave em MAIUSCULAS, sem espacos nas bordas).
MAPA_FAMILIAS = {
    "BENIGN": "Benign",
    "DDOS": "DDoS",
    "DOS HULK": "DoS",
    "DOS GOLDENEYE": "DoS",
    "DOS SLOWLORIS": "DoS",
    "DOS SLOWHTTPTEST": "DoS",
    "HEARTBLEED": "DoS",
    "PORTSCAN": "PortScan",
    "BOT": "Botnet",
    "FTP-PATATOR": "Bruteforce",
    "SSH-PATATOR": "Bruteforce",
    "INFILTRATION": "Infiltration",
    "WEB ATTACK \uFFFD BRUTE FORCE": "WebAttacks",
    "WEB ATTACK \uFFFD XSS": "WebAttacks",
    "WEB ATTACK \uFFFD SQL INJECTION": "WebAttacks",
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
    # Guard robusto: os rotulos de Web Attack contem um byte lido como U+FFFD.
    # Casar pelo prefixo evita quebrar se a codificacao do parquet mudar.
    if chave.startswith("WEB ATTACK"):
        return "WebAttacks"
    raise ValueError(f"Label '{label}' (chave '{chave}') nao esta mapeada no MAPA_FAMILIAS!")


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


def filtrar_variancia_zero(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identifica colunas de variancia zero apenas em X_train e as remove das 3 particoes.

    Evita vazamento de dados (data leakage) aplicando a estatistica calculada apenas no treino.
    """
    variancias = X_train.var()
    colunas_zero = variancias[variancias == 0].index.tolist()
    if colunas_zero:
        X_train = X_train.drop(columns=colunas_zero)
        X_val = X_val.drop(columns=colunas_zero)
        X_test = X_test.drop(columns=colunas_zero)
    return X_train, X_val, X_test


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


def gerar_holdout_canonico(df: pd.DataFrame, teste: float = 0.15, seed: int = 42) -> np.ndarray:
    """Cria e salva um holdout canônico (compartilhado entre Etapa 1, 2 e avaliação).

    Remove classes com < 2 membros (inviáveis para stratify) e salva os
    índices de teste em INDICES_TESTE_PATH para que todos os scripts usem
    exatamente a mesma partição.
    """
    contagem = df["target_tipo"].value_counts()
    classes_raras = contagem[contagem < 2].index.tolist()
    df_valido = df[~df["target_tipo"].isin(classes_raras)] if classes_raras else df

    indices = df_valido.index.to_numpy()
    _, idx_teste = train_test_split(
        indices, test_size=teste, random_state=seed,
        stratify=df_valido.loc[indices, "target_tipo"],
    )
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(idx_teste, INDICES_TESTE_PATH)
    print(f"Holdout canônico salvo ({len(idx_teste)} amostras de teste) em {INDICES_TESTE_PATH}")
    return idx_teste


def carregar_holdout_canonico() -> np.ndarray:
    """Carrega os índices de teste salvos por gerar_holdout_canonico()."""
    if not INDICES_TESTE_PATH.exists():
        raise FileNotFoundError(
            f"Índices de teste não encontrados em {INDICES_TESTE_PATH}. "
            "Execute etapa1_deteccao.py primeiro para gerar o holdout canônico."
        )
    return joblib.load(INDICES_TESTE_PATH)


if __name__ == "__main__":
    # Smoke test / parte do GATE 1.A. Requer venv 3.12 + dados em data/.
    print("Carregando e limpando os 8 parquet...")
    df = criar_targets(limpar(carregar_dados()))
    print("shape:", df.shape)
    print("NaN restantes:", int(df.isna().sum().sum()))
    print("\ntarget_bin:\n", df["target_bin"].value_counts())
    print("\ntarget_tipo:\n", df["target_tipo"].value_counts())

    print("\nExecutando split e filtrando colunas com variancia zero...")
    X, y = preparar_features(df, alvo="target_tipo")
    particoes = split(X, y)
    X_tr_f, X_va_f, X_te_f = filtrar_variancia_zero(particoes["X_train"], particoes["X_val"], particoes["X_test"])
    print(f"Original shape: {particoes['X_train'].shape}, Filtered shape: {X_tr_f.shape}")
    colunas_removidas = set(particoes["X_train"].columns) - set(X_tr_f.columns)
    print(f"Colunas removidas ({len(colunas_removidas)}): {sorted(list(colunas_removidas))}")
