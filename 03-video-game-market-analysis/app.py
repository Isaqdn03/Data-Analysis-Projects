import streamlit as st
import pandas as pd
import plotly_express as px

# lendo os dados
car_data = pd.read_csv('vehicles.csv')

st.header('Analisando Vendas e Preços de Carros')

# botão para o histograma
hist_button = st.button('Criar histograma')

# botão para o gráfico de dispersão
scatter_button = st.button('Criar gráfico de dispersão')

# se o botão do histograma for clicado
if hist_button:
    # escrever uma mensagem
    st.write(
        'Criando um histograma para o conjunto de dados de anúncios de vendas de carros')
    # criar um histograma
    fig = px.histogram(car_data, x="odometer")
    # exibir um gráfico Plotly interativo
    st.plotly_chart(fig, use_container_width=True)

# se o botão do gráfico de dispersão for clicado
if scatter_button:
    st.write(
        'Criando um gráfico de dispersão para comparar quilometragem e preço dos carros')
    # criar um gráfico de dispersão
    fig = px.scatter(car_data,
                     x="odometer",
                     y="price",
                     title="Relação entre Quilometragem e Preço",
                     labels={'odometer': 'Quilometragem', 'price': 'Preço'})
    # exibir o gráfico
    st.plotly_chart(fig, use_container_width=True)
