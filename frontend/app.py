import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- CONFIGURAÇÃO ---
MODELS_DIR = Path(__file__).parent.parent / "models"
st.set_page_config(page_title="CIC-IDS2017 | Detecção", layout="wide")

st.title("🛡️ Sistema de Detecção e Identificação de Ataques")

# --- CARREGAMENTO DOS MODELOS ---
@st.cache_resource
def carregar_bundles():
    b1 = joblib.load(MODELS_DIR / "etapa1.joblib")
    b2 = joblib.load(MODELS_DIR / "etapa2.joblib")
    return b1, b2

bundle_e1, bundle_e2 = carregar_bundles()

# --- UPLOAD E PRÉ-PROCESSAMENTO ---
st.sidebar.header("Entrada de Dados")
arquivo_csv = st.sidebar.file_uploader("Upload do tráfego (CSV)", type=["csv"])

if arquivo_csv:
    # 1. Leitura e Limpeza
    df_raw = pd.read_csv(arquivo_csv)
    df_raw.columns = df_raw.columns.str.strip()
    
    # 2. Reindexação (Alinhamento com o modelo)
    colunas_treino = bundle_e1["colunas"]
    df_modelo = df_raw.reindex(columns=colunas_treino, fill_value=0)
    X_pronto = df_modelo.values
    
    # 3. Inferência (Pipeline 2 Etapas)
    pred_bin = bundle_e1["modelo"].predict(X_pronto)
    proba_bin = bundle_e1["modelo"].predict_proba(X_pronto)
    pred_tipo = bundle_e2["modelo"].predict(X_pronto)
    
    # 4. Consolidação
    df_resultado = df_raw.copy()
    df_resultado["Alerta"] = ["Ataque" if p == 1 else "Normal" for p in pred_bin]
    df_resultado["Tipo_Ataque"] = [tipo if p == 1 else "-" for p, tipo in zip(pred_bin, pred_tipo)]
    df_resultado["Confianca"] = proba_bin.max(axis=1).round(4)
    
    st.write("### Resultados da Análise")
    # --- 5. PAINEL DE KPIs (Métricas Totais) ---
    st.write("### 📊 Visão Geral do Tráfego")
    
    total_fluxos = len(df_resultado)
    total_ataques = len(df_resultado[df_resultado["Alerta"] == "Ataque"])
    taxa_ataque = (total_ataques / total_fluxos) * 100 if total_fluxos > 0 else 0
    
    # Cria 3 colunas no topo para as métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Fluxos Analisados", total_fluxos)
    col2.metric("Ameaças Detectadas", total_ataques, delta="Atenção", delta_color="inverse")
    col3.metric("Taxa de Ataque", f"{taxa_ataque:.1f}%")
    
    st.markdown("---")
    
    # --- 6. GRÁFICOS E ALERTAS VISUAIS ---
    col_grafico, col_tabela = st.columns([1, 2]) # A tabela fica mais larga que o gráfico
    
    with col_grafico:
        st.write("#### Distribuição de Ataques")
        # Filtra só os ataques para o gráfico
        df_ataques = df_resultado[df_resultado["Alerta"] == "Ataque"]
        if not df_ataques.empty:
            contagem_ataques = df_ataques["Tipo_Ataque"].value_counts()
            st.bar_chart(contagem_ataques)
        else:
            st.success("Nenhum ataque detectado neste lote!")
            
    with col_tabela:
        st.write("#### Detalhamento de Alertas (Vermelho/Verde)")
        
        # Função para pintar a linha dependendo se é ataque ou normal
        def pintar_alerta(val):
            cor = '#ffcccc' if val == 'Ataque' else '#ccffcc'
            return f'background-color: {cor}'
        
        # Aplica o estilo apenas na coluna 'Alerta'
        tabela_estilizada = df_resultado[["Alerta", "Tipo_Ataque", "Confianca", "Protocol"]].style.map(
            pintar_alerta, subset=['Alerta']
        )
        
        st.dataframe(tabela_estilizada, use_container_width=True, height=300)

else:
    st.info("Aguardando upload do arquivo CSV no menu lateral.")

# --- AVISO ÉTICO ---
st.markdown("---")
st.caption("⚠️ O sistema é um apoio à decisão. Alertas devem ser revisados por um analista de segurança.")