#!/usr/bin/env python3
"""
Treinamento real dos bundles de Regressao Logistica.

Usa o mesmo contrato das demais etapas:
- holdout canonico compartilhado para avaliacao fim-a-fim;
- split interno de treino/validacao/teste para a Etapa 1;
- scaler e balanceamento dentro do pipeline, evitando vazamento;
- remocao de classes ultra-raras na Etapa 2 antes de aplicar SMOTE.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.preprocessamento import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    carregar_dados,
    carregar_holdout_canonico,
    criar_targets,
    gerar_holdout_canonico,
    limpar,
    preparar_features,
    salvar_bundle,
    split,
)

CV_SPLITS = 5
MIN_SAMPLES_POR_CLASSE = 10


def carregar_base() -> pd.DataFrame:
    caminho_amostra = DATA_DIR / "amostra.parquet"
    if caminho_amostra.exists():
        print(f"Carregando dados da amostra: {caminho_amostra}")
        return pd.read_parquet(caminho_amostra)

    print("Amostra nao encontrada. Carregando dataset completo (pode demorar)...")
    return criar_targets(limpar(carregar_dados()))


def otimizar_limiar(y_val, probas_val, recall_alvo=0.98):
    """Escolhe o limiar que maximiza F1 mantendo recall minimo, quando possivel."""
    melhor_limiar = 0.5
    melhor_f1 = 0.0
    historico = []

    for limiar in np.linspace(0.01, 0.99, 99):
        preds = (probas_val >= limiar).astype(int)
        rec = recall_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        historico.append((limiar, rec, prec, f1))

        if rec >= recall_alvo and f1 > melhor_f1:
            melhor_f1 = f1
            melhor_limiar = limiar

    if melhor_f1 == 0.0:
        melhor_limiar = max(historico, key=lambda item: item[1])[0]
        print(f"Aviso: recall alvo {recall_alvo:.2f} nao foi atingido; usando maior recall.")

    return melhor_limiar


def filtrar_classes_raras(df_ataques: pd.DataFrame) -> pd.DataFrame:
    contagem = df_ataques["target_tipo"].value_counts()
    classes_raras = contagem[contagem < MIN_SAMPLES_POR_CLASSE].index.tolist()
    if classes_raras:
        print(f"Removendo classes raras da Etapa 2: {classes_raras}")
        df_ataques = df_ataques[~df_ataques["target_tipo"].isin(classes_raras)].copy()
    return df_ataques


def treinar_etapa1(df_treino_val: pd.DataFrame) -> None:
    print("\n=== LR ETAPA 1: DETECCAO BINARIA ===")
    X, y = preparar_features(df_treino_val, alvo="target_bin")
    particoes = split(X, y, val=0.15, teste=0.15, seed=42)
    X_train, X_val, X_test = particoes["X_train"], particoes["X_val"], particoes["X_test"]
    y_train, y_val, y_test = particoes["y_train"], particoes["y_val"], particoes["y_test"]

    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("sampler", RandomUnderSampler(sampling_strategy=0.333, random_state=42)),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        pipeline,
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1",
        cv=CV_SPLITS,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Melhores parametros: {grid.best_params_}")
    print(f"Melhor F1 CV: {grid.best_score_:.4f}")

    best_pipeline = grid.best_estimator_
    probas_val = best_pipeline.predict_proba(X_val)[:, 1]
    limiar = otimizar_limiar(y_val, probas_val, recall_alvo=0.98)
    print(f"Limiar LR Etapa 1: {limiar:.4f}")

    probas_test = best_pipeline.predict_proba(X_test)[:, 1]
    preds_test = (probas_test >= limiar).astype(int)
    print(classification_report(y_test, preds_test, target_names=["Normal", "Ataque"]))

    salvar_bundle(
        caminho=MODELS_DIR / "lr_etapa1.joblib",
        modelo=best_pipeline,
        colunas=X_train.columns.tolist(),
        scaler=None,
        classes=[0, 1],
        limiar=limiar,
    )
    print(f"Bundle salvo: {MODELS_DIR / 'lr_etapa1.joblib'}")


def treinar_etapa2(df_treino_val: pd.DataFrame) -> None:
    print("\n=== LR ETAPA 2: IDENTIFICACAO MULTICLASSE ===")
    df_ataques = df_treino_val[df_treino_val["target_bin"] == 1].copy()
    print(f"Ataques antes do filtro de raras: {len(df_ataques)}")
    print(df_ataques["target_tipo"].value_counts())
    df_ataques = filtrar_classes_raras(df_ataques)

    X, y = preparar_features(df_ataques, alvo="target_tipo")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    min_samples = y_train.value_counts().min()
    k_neighbors = min(5, max(1, min_samples - 1))
    print(f"Usando SMOTE(k_neighbors={k_neighbors})")

    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(k_neighbors=k_neighbors, random_state=42)),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        pipeline,
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=CV_SPLITS,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Melhores parametros: {grid.best_params_}")
    print(f"Melhor Macro-F1 CV: {grid.best_score_:.4f}")

    best_pipeline = grid.best_estimator_
    preds_test = best_pipeline.predict(X_test)
    classes_ordenadas = sorted(y_test.unique())
    print(classification_report(y_test, preds_test, zero_division=0))

    salvar_bundle(
        caminho=MODELS_DIR / "lr_etapa2.joblib",
        modelo=best_pipeline,
        colunas=X_train.columns.tolist(),
        scaler=None,
        classes=classes_ordenadas,
        limiar=0.5,
    )
    print(f"Bundle salvo: {MODELS_DIR / 'lr_etapa2.joblib'}")


def main() -> None:
    df = carregar_base()

    try:
        idx_teste = carregar_holdout_canonico()
    except FileNotFoundError:
        idx_teste = gerar_holdout_canonico(df)

    df_treino_val = df.loc[~df.index.isin(idx_teste)].copy()
    print(f"Dados de treino/validacao apos excluir holdout canonico: {len(df_treino_val)}")

    treinar_etapa1(df_treino_val)
    treinar_etapa2(df_treino_val)


if __name__ == "__main__":
    main()
