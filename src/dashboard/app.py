import streamlit as st
from kedro.framework.session import get_current_session

# Obtém o objeto session do Kedro
session = get_current_session()

# Obtém o catálogo e o pipeline do Kedro
catalog = session.load_context().catalog
pipeline = session.load_context().pipeline

# Título do dashboard
st.title("Monitoramento da operação")

# Subtítulo
st.markdown("Acompanhe em tempo real as principais métricas da operação.")

# Gráfico de linhas com a evolução do faturamento
st.subheader("Faturamento")
data = catalog.load("faturamento")
st.line_chart(data)

# Gráfico de barras com as vendas por região
st.subheader("Vendas por região")
data = catalog.load("vendas_por_regiao")
st.bar_chart(data)

# Tabela com as últimas vendas realizadas
st.subheader("Últimas vendas")
data = catalog.load("ultimas_vendas")
st.table(data)