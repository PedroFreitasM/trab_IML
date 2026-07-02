import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuração inicial da página e Estilização
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IDS Dashboard - Detecção de Anomalias",
    page_icon="🛡️",
    layout="wide"
)

# Estilo para os painéis de alerta (verde/vermelho)
st.markdown("""
    <style>
    .alerta-normal { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .alerta-ataque { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Aviso de Supervisão Humana (Requisito Ético - Track C)
# -----------------------------------------------------------------------------
st.info(
    "⚠️ **Supervisão Humana:** "
    "As classificações apresentadas por este sistema são predições estatísticas. "
    "A decisão final sobre ações de bloqueio ou mitigação deve sempre ser validada "
    "por um analista de segurança para evitar interrupções indevidas."
)

st.title("🛡️ Sistema de Detecção e Identificação de Intrusões")
st.markdown("Análise de fluxo de tráfego em duas etapas: **Detecção Binária** e **Classificação de Tipo**.")

# -----------------------------------------------------------------------------
# Funções de Carregamento e Processamento
# -----------------------------------------------------------------------------
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_ATUAL)
DIRETORIO_MODELS = os.path.join(DIRETORIO_RAIZ, "models")

MODELOS_CANDIDATOS = {
    "Random Forest": {
        "etapa1": ["rf_etapa1.joblib", "etapa1.joblib"],
        "etapa2": ["rf_etapa2.joblib", "etapa2.joblib"],
    },
    "Regressão Logística": {
        "etapa1": ["lr_etapa1.joblib", "logreg_etapa1.joblib"],
        "etapa2": ["lr_etapa2.joblib", "logreg_multiclasse.joblib"],
    },
    "Árvore de Decisão": {
        "etapa1": ["dt_etapa1.joblib"],
        "etapa2": ["dt_etapa2.joblib"],
    },
}

def _achar_bundle(diretorio_models, candidatos):
    """Retorna o primeiro candidato que existe em models/ (ou o primeiro nome, para a mensagem de erro)."""
    for nome in candidatos:
        caminho = os.path.join(diretorio_models, nome)
        if os.path.exists(caminho):
            return caminho
    return os.path.join(diretorio_models, candidatos[0])

def modelo_disponivel(tipo_modelo):
    candidatos = MODELOS_CANDIDATOS[tipo_modelo]
    caminho_e1 = _achar_bundle(DIRETORIO_MODELS, candidatos["etapa1"])
    caminho_e2 = _achar_bundle(DIRETORIO_MODELS, candidatos["etapa2"])
    return os.path.exists(caminho_e1) and os.path.exists(caminho_e2)

@st.cache_resource
def carregar_modelos(tipo_modelo):
    """Carrega os bundles baseados na escolha do usuário usando caminhos absolutos."""
    candidatos = MODELOS_CANDIDATOS[tipo_modelo]
    caminho_e1 = _achar_bundle(DIRETORIO_MODELS, candidatos["etapa1"])
    caminho_e2 = _achar_bundle(DIRETORIO_MODELS, candidatos["etapa2"])

    if not os.path.exists(caminho_e1) or not os.path.exists(caminho_e2):
        return None, None, caminho_e1, caminho_e2

    bundle1 = joblib.load(caminho_e1)
    bundle2 = joblib.load(caminho_e2)
    return bundle1, bundle2, caminho_e1, caminho_e2

def normalizar_colunas(df):
    """
    Remove apenas os espaços das pontas dos nomes de coluna (ex: ' Flow Duration'),
    igual ao limpar() do preprocessamento. Os bundles guardam os nomes originais
    do treino ("Flow Duration"), então mudar caixa/underscore aqui faria o
    reindex() não casar nenhuma coluna e zerar todas as features.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def preparar_entrada_modelo(df, bundle):
    """Alinha o CSV ao contrato do bundle e garante matriz numérica para inferência."""
    X = df.reindex(columns=bundle["colunas"], fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = bundle.get("scaler")
    if scaler is not None:
        return scaler.transform(X)
    return X

def classes_do_modelo(modelo):
    """Obtém classes tanto de estimadores diretos quanto de pipelines sklearn/imblearn."""
    classes = getattr(modelo, "classes_", None)
    if classes is None and hasattr(modelo, "named_steps"):
        estimador_final = modelo.named_steps.get("model")
        classes = getattr(estimador_final, "classes_", None)
    return list(classes) if classes is not None else []

def indice_classe(modelo, classe, n_colunas_proba, fallback=1):
    """Resolve o índice de uma classe em predict_proba sem assumir formato do modelo."""
    classes = classes_do_modelo(modelo)
    if classe in classes:
        return classes.index(classe)
    classe_str = str(classe)
    if classe_str in classes:
        return classes.index(classe_str)
    return fallback if fallback < n_colunas_proba else n_colunas_proba - 1

def traduzir_protocolo(valor):
    """Converte números de protocolo IP para nomes conhecidos sem perder o código original."""
    try:
        codigo = int(float(valor))
    except (TypeError, ValueError):
        return valor

    protocolos = {
        1: "ICMP",
        2: "IGMP",
        6: "TCP",
        17: "UDP",
        47: "GRE",
        50: "ESP",
        51: "AH",
        58: "ICMPv6",
        89: "OSPF",
        132: "SCTP",
    }
    nome = protocolos.get(codigo, "Outro")
    return f"{nome} ({codigo})"

# -----------------------------------------------------------------------------
# Configurações da Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.header("Configurações do Modelo")
if st.sidebar.button("Recarregar modelos"):
    carregar_modelos.clear()
    st.rerun()

modelos_disponiveis = [nome for nome in MODELOS_CANDIDATOS if modelo_disponivel(nome)]
modelos_indisponiveis = [nome for nome in MODELOS_CANDIDATOS if nome not in modelos_disponiveis]

if not modelos_disponiveis:
    st.error("Nenhum par de modelos foi encontrado em `models/`.")
    st.stop()

if modelos_indisponiveis:
    st.sidebar.caption(
        "Indisponível: " + ", ".join(modelos_indisponiveis) +
        ". Gere os dois bundles da Etapa 1 e Etapa 2 para habilitar."
    )

modelo_selecionado = st.sidebar.radio(
    "Selecione o Algoritmo:",
    modelos_disponiveis
)

st.sidebar.markdown("---")
st.sidebar.header("Upload de Tráfego")
arquivo_csv = st.sidebar.file_uploader("Envie o tráfego de rede (CSV)", type=["csv"])

# -----------------------------------------------------------------------------
# Lógica Principal da Interface
# -----------------------------------------------------------------------------
# Passa a escolha do usuário para a função carregar os arquivos certos
bundle_etapa1, bundle_etapa2, path_e1, path_e2 = carregar_modelos(modelo_selecionado)

if bundle_etapa1 is None or bundle_etapa2 is None:
    st.error(f"🚨 **Modelos não encontrados para {modelo_selecionado}!**")
    st.warning(f"Certifique-se de que os arquivos `{path_e1}` e `{path_e2}` existem na pasta `models/`.")
    st.stop()

if arquivo_csv is not None:
    # 1. Carregamento e pré-processamento básico
    df_raw = pd.read_csv(arquivo_csv)
    df_clean = normalizar_colunas(df_raw)

    if df_clean.empty:
        st.warning("O arquivo CSV enviado está vazio. Envie um arquivo com pelo menos um fluxo de rede.")
        st.stop()

    st.write("### Resumo da Captura")
    st.write(f"Total de fluxos analisados: **{len(df_clean)}**")

    # DataFrame para consolidar os resultados de exibição
    df_resultados = pd.DataFrame(index=df_clean.index)
    
    # =========================================================================
    # PIPELINE ETAPA 1: Detecção Binária (Ataque vs Normal)
    # =========================================================================
    X1 = preparar_entrada_modelo(df_clean, bundle_etapa1)
    modelo1 = bundle_etapa1["modelo"]
    probs_e1 = modelo1.predict_proba(X1)
    
    idx_classe_ataque = indice_classe(modelo1, 1, probs_e1.shape[1], fallback=1)
    prob_ataque = probs_e1[:, idx_classe_ataque]
    limiar1 = bundle_etapa1.get("limiar", 0.5)
    
    # Classificação com base no limiar otimizado para Recall
    df_resultados["Alerta (Etapa 1)"] = np.where(prob_ataque >= limiar1, "Ataque", "Normal")
    df_resultados["Confiança Resultado (%)"] = np.round(
        np.where(df_resultados["Alerta (Etapa 1)"] == "Ataque", prob_ataque, 1 - prob_ataque) * 100,
        2
    )
    df_resultados["Tipo de Ataque (Etapa 2)"] = "N/A"
    df_resultados["Confiança Tipo (%)"] = np.nan

    # =========================================================================
    # PIPELINE ETAPA 2: Identificação do Tipo (Apenas nos fluxos anômalos)
    # =========================================================================
    indices_ataques = df_resultados[df_resultados["Alerta (Etapa 1)"] == "Ataque"].index
    
    if len(indices_ataques) > 0:
        df_ataques = df_clean.loc[indices_ataques]
        
        X2 = preparar_entrada_modelo(df_ataques, bundle_etapa2)
        modelo2 = bundle_etapa2["modelo"]
        preds_e2 = modelo2.predict(X2)
        if hasattr(modelo2, "predict_proba"):
            probs_e2 = np.max(modelo2.predict_proba(X2), axis=1)
        else:
            probs_e2 = np.ones(len(preds_e2))
        
        # Mapeamento de classes (se existirem) ou uso direto das predições
        if "classes" in bundle_etapa2:
            preds_nome = [
                bundle_etapa2["classes"][p] if isinstance(p, (int, np.integer)) else p 
                for p in preds_e2
            ]
        else:
            preds_nome = preds_e2
            
        df_resultados.loc[indices_ataques, "Tipo de Ataque (Etapa 2)"] = preds_nome
        df_resultados.loc[indices_ataques, "Confiança Tipo (%)"] = np.round(probs_e2 * 100, 2)

    # =========================================================================
    # Visualização de Dados (Métricas e Gráficos)
    # =========================================================================
    total_anomalias = len(indices_ataques)
    total_normais = len(df_resultados) - total_anomalias
    
    # Cards de KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Tráfego Normal", total_normais)
    col2.metric("Ataques Detectados", total_anomalias, delta_color="inverse")
    taxa_anomalia = (total_anomalias / len(df_resultados)) * 100
    col3.metric("Taxa de Anomalia", f"{taxa_anomalia:.1f}%")

    # Gráficos de Distribuição
    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.subheader("Distribuição do Tráfego")
        fig_pie = px.pie(
            df_resultados, 
            names="Alerta (Etapa 1)", 
            color="Alerta (Etapa 1)",
            color_discrete_map={"Normal": "#28a745", "Ataque": "#dc3545"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g_col2:
        st.subheader("Tipologia de Ataques (Etapa 2)")
        if total_anomalias > 0:
            df_tipos = df_resultados[df_resultados["Alerta (Etapa 1)"] == "Ataque"]
            fig_bar = px.histogram(
                df_tipos, 
                y="Tipo de Ataque (Etapa 2)",
                orientation="h",
                color="Tipo de Ataque (Etapa 2)",
                category_orders={"Tipo de Ataque (Etapa 2)": df_tipos["Tipo de Ataque (Etapa 2)"].value_counts().index}
            )
            fig_bar.update_layout(showlegend=False, xaxis_title="Contagem", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("Nenhum ataque detectado para categorizar.")

    # Tabela Detalhada com formatação condicional (Painel de Alertas)
    st.markdown("---")
    st.subheader("Painel de Alertas de Tráfego")
    
    def colorir_alerta(val):
        cor = '#ffcccc' if val == 'Ataque' else '#ccffcc'
        return f'background-color: {cor}'
    
    # Monta a tabela final combinando infos vitais
    colunas_exibicao = [
        "Alerta (Etapa 1)",
        "Confiança Resultado (%)",
        "Tipo de Ataque (Etapa 2)",
        "Confiança Tipo (%)",
    ]
    
    # CORREÇÃO APLICADA: Filtra de forma segura pegando apenas as colunas que realmente foram criadas
    colunas_presentes = [col for col in colunas_exibicao if col in df_resultados.columns]
    df_exibicao = df_resultados[colunas_presentes].copy()
    
    # Se houver metadados originais como IPs e Portas (ainda que no CICIDS sejam filtrados, se existirem na amostra, mostrar)
    colunas_contexto = [c for c in ['Source IP', 'Src IP', 'Destination IP', 'Dst IP',
                                    'Destination Port', 'Dst Port', 'Protocol'] if c in df_clean.columns]
    if colunas_contexto:
        df_contexto = df_clean[colunas_contexto].copy()
        if "Protocol" in df_contexto.columns:
            df_contexto["Protocolo"] = df_contexto["Protocol"].apply(traduzir_protocolo)
            df_contexto = df_contexto.drop(columns=["Protocol"])
        df_exibicao = pd.concat([df_contexto, df_exibicao], axis=1)

    total_linhas = len(df_exibicao)
    opcoes_por_pagina = [25, 50, 100, 200]
    linhas_por_pagina = st.selectbox(
        "Itens por página",
        opcoes_por_pagina,
        index=1,
        key="alertas_itens_por_pagina",
    )
    total_paginas = max(1, int(np.ceil(total_linhas / linhas_por_pagina)))
    pagina = st.number_input(
        "Página",
        min_value=1,
        max_value=total_paginas,
        value=1,
        step=1,
        key="alertas_pagina",
    )
    inicio = (pagina - 1) * linhas_por_pagina
    fim = min(inicio + linhas_por_pagina, total_linhas)
    df_pagina = df_exibicao.iloc[inicio:fim]

    st.caption(f"Exibindo {inicio + 1}-{fim} de {total_linhas} fluxos")
    altura_tabela = min(760, max(300, 38 * (len(df_pagina) + 1)))

    st.dataframe(
        df_pagina.style.map(colorir_alerta, subset=['Alerta (Etapa 1)']),
        use_container_width=True,
        height=altura_tabela
    )

    # =========================================================================
    # Interpretabilidade (apenas para Árvore de Decisão)
    # =========================================================================
    if modelo_selecionado == "Árvore de Decisão":
        st.markdown("---")
        st.subheader("🌳 Interpretabilidade da Árvore de Decisão")
        
        diretorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        with st.expander("Visualização da Árvore (Etapa 1 - Detecção)", expanded=False):
            img_path_e1 = os.path.join(diretorio_raiz, "images", "dt_etapa1_arvore.png")
            if os.path.exists(img_path_e1):
                st.image(img_path_e1, caption="Árvore de Decisão - Detecção Binária", use_container_width=True)
            else:
                st.info("Execute `python backend/dt_interpretabilidade.py` para gerar a visualização.")
            
            regras_path_e1 = os.path.join(diretorio_raiz, "images", "dt_etapa1_regras.txt")
            if os.path.exists(regras_path_e1):
                with open(regras_path_e1, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text")
        
        with st.expander("Visualização da Árvore (Etapa 2 - Classificação)", expanded=False):
            img_path_e2 = os.path.join(diretorio_raiz, "images", "dt_etapa2_arvore.png")
            if os.path.exists(img_path_e2):
                st.image(img_path_e2, caption="Árvore de Decisão - Identificação de Ataques", use_container_width=True)
            else:
                st.info("Execute `python backend/dt_interpretabilidade.py` para gerar a visualização.")
            
            regras_path_e2 = os.path.join(diretorio_raiz, "images", "dt_etapa2_regras.txt")
            if os.path.exists(regras_path_e2):
                with open(regras_path_e2, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text")

else:
    # Estado inicial (sem arquivo)
    st.write("Aguardando arquivo CSV para iniciar a análise...")
