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
# Configurações da Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.header("Configurações do Modelo")
modelo_selecionado = st.sidebar.radio(
    "Selecione o Algoritmo:",
    ("Random Forest", "Regressão Logística")
)

st.sidebar.markdown("---")
st.sidebar.header("Upload de Tráfego")
arquivo_csv = st.sidebar.file_uploader("Envie o tráfego de rede (CSV)", type=["csv"])

# -----------------------------------------------------------------------------
# Funções de Carregamento e Processamento
# -----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelos(tipo_modelo):
    """Carrega os bundles baseados na escolha do usuário usando caminhos absolutos."""
    if tipo_modelo == "Random Forest":
        prefixo = "rf"
    else:
        prefixo = "lr" # Logistic Regression
    
    # Pega o caminho absoluto da pasta onde o app.py está (frontend/)
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    # Volta uma pasta para a raiz do projeto (trab_IML/)
    diretorio_raiz = os.path.dirname(diretorio_atual)
    
    # Constrói o caminho absoluto até a pasta models
    caminho_e1 = os.path.join(diretorio_raiz, "models", f"{prefixo}_etapa1.joblib")
    caminho_e2 = os.path.join(diretorio_raiz, "models", f"{prefixo}_etapa2.joblib")
    
    if not os.path.exists(caminho_e1) or not os.path.exists(caminho_e2):
        return None, None, caminho_e1, caminho_e2
        
    bundle1 = joblib.load(caminho_e1)
    bundle2 = joblib.load(caminho_e2)
    return bundle1, bundle2, caminho_e1, caminho_e2

def normalizar_colunas(df):
    """
    Remove espaços das pontas das colunas e mapeia nomes curtos do CICFlowMeter
    para os nomes longos esperados pelos modelos treinados.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Mapeia nomes curtos do conversor para os nomes longos do parquet
    mapa_nomes = {
        "Tot Fwd Pkts": "Total Fwd Packets",
        "Tot Bwd Pkts": "Total Backward Packets",
        "TotLen Fwd Pkts": "Fwd Packets Length Total",
        "TotLen Bwd Pkts": "Bwd Packets Length Total",
        "Fwd Pkt Len Max": "Fwd Packet Length Max",
        "Fwd Pkt Len Min": "Fwd Packet Length Min",
        "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
        "Fwd Pkt Len Std": "Fwd Packet Length Std",
        "Bwd Pkt Len Max": "Bwd Packet Length Max",
        "Bwd Pkt Len Min": "Bwd Packet Length Min",
        "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
        "Bwd Pkt Len Std": "Bwd Packet Length Std",
        "Flow Byts/s": "Flow Bytes/s",
        "Flow Pkts/s": "Flow Packets/s",
        "Fwd IAT Tot": "Fwd IAT Total",
        "Bwd IAT Tot": "Bwd IAT Total",
        "Fwd Header Len": "Fwd Header Length",
        "Bwd Header Len": "Bwd Header Length",
        "Pkt Len Min": "Packet Length Min",
        "Pkt Len Max": "Packet Length Max",
        "Pkt Len Mean": "Packet Length Mean",
        "Pkt Len Std": "Packet Length Std",
        "Pkt Len Var": "Packet Length Variance",
        "FIN Flag Cnt": "FIN Flag Count",
        "SYN Flag Cnt": "SYN Flag Count",
        "RST Flag Cnt": "RST Flag Count",
        "PSH Flag Cnt": "PSH Flag Count",
        "ACK Flag Cnt": "ACK Flag Count",
        "URG Flag Cnt": "URG Flag Count",
        "ECE Flag Cnt": "ECE Flag Count",
        "Pkt Size Avg": "Avg Packet Size",
        "Fwd Seg Size Avg": "Avg Fwd Segment Size",
        "Bwd Seg Size Avg": "Avg Bwd Segment Size",
        "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
        "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
        "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
        "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
        "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
        "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
        "Subflow Fwd Pkts": "Subflow Fwd Packets",
        "Subflow Fwd Byts": "Subflow Fwd Bytes",
        "Subflow Bwd Pkts": "Subflow Bwd Packets",
        "Subflow Bwd Byts": "Subflow Bwd Bytes",
        "Init Fwd Win Byts": "Init Fwd Win Bytes",
        "Init Bwd Win Byts": "Init Bwd Win Bytes",
        "Fwd Act Data Pkts": "Fwd Act Data Packets",
    }
    df = df.rename(columns=mapa_nomes)
    return df

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
    df_clean = normalizar_colunas(df_raw.copy())
    
    st.write("### Resumo da Captura")
    st.write(f"Total de fluxos analisados: **{len(df_clean)}**")

    # DataFrame para consolidar os resultados de exibição
    df_resultados = pd.DataFrame(index=df_clean.index)
    
    # =========================================================================
    # PIPELINE ETAPA 1: Detecção Binária (Ataque vs Normal)
    # =========================================================================
    # Reindexar garantindo exatamente as colunas usadas no treino da etapa 1
    X1 = df_clean.reindex(columns=bundle_etapa1["colunas"], fill_value=0)
    
    if bundle_etapa1.get("scaler"):
        X1 = bundle_etapa1["scaler"].transform(X1)
        
    modelo1 = bundle_etapa1["modelo"]
    probs_e1 = modelo1.predict_proba(X1)
    
    # Assume que a classe 1 (Ataque) é o índice 1 nas probabilidades
    idx_classe_ataque = list(modelo1.classes_).index(1) if 1 in modelo1.classes_ else 1
    prob_ataque = probs_e1[:, idx_classe_ataque]
    limiar1 = bundle_etapa1.get("limiar", 0.5)
    
    # Classificação com base no limiar otimizado para Recall
    df_resultados["Alerta (Etapa 1)"] = np.where(prob_ataque >= limiar1, "Ataque", "Normal")
    df_resultados["Confiança_Anomalia"] = np.round(prob_ataque * 100, 2)
    df_resultados["Tipo de Ataque (Etapa 2)"] = "N/A"
    df_resultados["Confiança_Tipo"] = 0.0

    # =========================================================================
    # PIPELINE ETAPA 2: Identificação do Tipo (Apenas nos fluxos anômalos)
    # =========================================================================
    indices_ataques = df_resultados[df_resultados["Alerta (Etapa 1)"] == "Ataque"].index
    
    if len(indices_ataques) > 0:
        df_ataques = df_clean.loc[indices_ataques]
        
        # Reindexar para as colunas exatas da etapa 2
        X2 = df_ataques.reindex(columns=bundle_etapa2["colunas"], fill_value=0)
        
        if bundle_etapa2.get("scaler"):
            X2 = bundle_etapa2["scaler"].transform(X2)
            
        modelo2 = bundle_etapa2["modelo"]
        preds_e2 = modelo2.predict(X2)
        probs_e2 = np.max(modelo2.predict_proba(X2), axis=1)
        
        # Mapeamento de classes (se existirem) ou uso direto das predições
        if "classes" in bundle_etapa2:
            preds_nome = [
                bundle_etapa2["classes"][p] if isinstance(p, (int, np.integer)) else p 
                for p in preds_e2
            ]
        else:
            preds_nome = preds_e2
            
        df_resultados.loc[indices_ataques, "Tipo de Ataque (Etapa 2)"] = preds_nome
        df_resultados.loc[indices_ataques, "Confiança_Tipo"] = np.round(probs_e2 * 100, 2)

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
    colunas_exibicao = ["Alerta (Etapa 1)", "Confiança_Anomalia", "Tipo de Ataque (Etapa 2)", "Confiança_Tipo"]
    
    # CORREÇÃO APLICADA: Filtra de forma segura pegando apenas as colunas que realmente foram criadas
    colunas_presentes = [col for col in colunas_exibicao if col in df_resultados.columns]
    df_exibicao = df_resultados[colunas_presentes].copy()
    
    # Se houver metadados originais como IPs e Portas (que são filtrados antes do modelo, mas úteis para exibição)
    mapa_meta = {
        'Source IP': ['Source IP', 'Src IP', 'source_ip', 'src_ip'],
        'Destination IP': ['Destination IP', 'Dst IP', 'destination_ip', 'dst_ip'],
        'Destination Port': ['Destination Port', 'Dst Port', 'destination_port', 'dst_port'],
        'Protocol': ['Protocol', 'protocol']
    }
    
    colunas_contexto_presentes = []
    df_meta = pd.DataFrame(index=df_clean.index)
    for nome_amigavel, possiveis_nomes in mapa_meta.items():
        for nome in possiveis_nomes:
            if nome in df_clean.columns:
                df_meta[nome_amigavel] = df_clean[nome]
                colunas_contexto_presentes.append(nome_amigavel)
                break
                
    if colunas_contexto_presentes:
        df_exibicao = pd.concat([df_meta, df_exibicao], axis=1)

    st.dataframe(
        df_exibicao.style.map(colorir_alerta, subset=['Alerta (Etapa 1)']),
        use_container_width=True,
        height=400
    )

else:
    # Estado inicial (sem arquivo)
    st.write("Aguardando arquivo CSV para iniciar a análise...")