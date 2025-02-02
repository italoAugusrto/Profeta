import pandas as pd
import streamlit as st
from binance.client import Client
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from secrets_1 import api_key, api_secret
from sklearn.ensemble import RandomForestClassifier

# Configurar cliente Binance
client = Client(api_key, api_secret)

# Função para carregar as criptomoedas do CSV
def carregar_lista_criptos():
    path = 'C:/Users/Italo/OneDrive/Documentos/python/cripto_analize/cypto.csv'
    return pd.read_csv(path)

# Função para buscar dados na API da Binance
def obter_dados_binance(symbol, intervalo='1d', inicio='2010-01-01'):
    try:
        inicio_timestamp = int(datetime.strptime(inicio, '%Y-%m-%d').timestamp() * 1000)
        st.write(f"Buscando dados para {symbol}...")
        klines = client.get_historical_klines(symbol, intervalo, inicio_timestamp)

        if not klines:
            st.error(f"Nenhum dado retornado para {symbol}. Verifique o símbolo ou a data inicial.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['Price'] = pd.to_numeric(df['Close'], errors='coerce')

        st.write("Dados obtidos com sucesso!")
        return df[['Date', 'Price', 'Volume']]
    except Exception as e:
        st.error(f"Erro ao obter dados da Binance para {symbol}: {e}")
        return None

# Função para exibir o gráfico de preços históricos
def exibir_grafico(df_valores, nome):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_valores['Date'], y=df_valores['Price'], name=f'{nome}', line_color='green'))
    st.plotly_chart(fig)

# Função para treinar o modelo Prophet
def treinar_modelo(df_valores):
    df_treino = df_valores.rename(columns={'Date': 'ds', 'Price': 'y'})
    modelo = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    modelo.add_country_holidays(country_name='US')
    modelo.fit(df_treino)
    return modelo, df_treino

# Função para gerar previsão
def gerar_previsao(modelo, n_dias):
    futuro = modelo.make_future_dataframe(periods=n_dias, freq='D')
    previsao = modelo.predict(futuro)
    return previsao

# Função para avaliar a previsão
def avaliar_previsao(df_real, previsao):
    df_comparacao = df_real.merge(previsao[['ds', 'yhat']], left_on='Date', right_on='ds')
    mae = mean_absolute_error(df_comparacao['Price'], df_comparacao['yhat'])
    rmse = np.sqrt(mean_squared_error(df_comparacao['Price'], df_comparacao['yhat']))
    return mae, rmse

# Função para gerar insights com IA
def gerar_insights(df_valores, previsao):
    previsao_ajustada = previsao[previsao['ds'].isin(df_valores['Date'])].copy()
    
    df_analise = previsao_ajustada[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted_Price'})
    df_analise['Actual_Price'] = df_valores['Price'].values
    df_analise['Volume'] = df_valores['Volume'].values

    # Converter a coluna 'Volume' para numérico
    df_analise['Volume'] = pd.to_numeric(df_analise['Volume'], errors='coerce')

    # Calcular mudanças percentuais
    df_analise['Price_Change'] = df_analise['Predicted_Price'].pct_change() * 100
    df_analise['Volume_Change'] = df_analise['Volume'].pct_change() * 100

    # Criar sinais de compra/venda
    df_analise['Signal'] = np.where(df_analise['Price_Change'] > 2, 'Buy',
                            np.where(df_analise['Price_Change'] < -2, 'Sell', 'Hold'))

    # Resumo de insights
    st.subheader('Insights de Compra e Venda')
    st.write(df_analise[['Date', 'Predicted_Price', 'Actual_Price', 'Price_Change', 'Signal']].tail(10))
    return df_analise

# Configuração do Streamlit
st.sidebar.header('Escolha a Criptomoeda')

df_criptos = carregar_lista_criptos()
nome_cripto = st.sidebar.selectbox('Selecione a Criptomoeda:', df_criptos['snome'])
cripto_selecionada = df_criptos[df_criptos['snome'] == nome_cripto].iloc[0]

st.title(f'Previsão de Preços - {nome_cripto}')

# Obter dados da Binance
df_valores = obter_dados_binance(cripto_selecionada['symbol'])

if df_valores is not None:
    st.subheader(f'Dados Históricos - {nome_cripto}')
    st.write(df_valores.tail())
    exibir_grafico(df_valores, nome_cripto)

    # Treinar o modelo
    modelo, df_treino = treinar_modelo(df_valores)

    # Gerar previsão
    n_dias = st.slider('Quantidades de dias de previsão', 30, 365)
    previsao = gerar_previsao(modelo, n_dias)

    # Avaliar previsão
    mae, rmse = avaliar_previsao(df_valores, previsao)
    st.subheader('Métricas de Avaliação')
    st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    st.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")

    # Exibir previsão
    st.subheader('Previsão para os Próximos Dias')
    st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

    # Gráfico de previsão
    grafico_previsao = plot_plotly(modelo, previsao)
    st.plotly_chart(grafico_previsao)

    # Componentes do modelo
    st.subheader('Componentes do Modelo')
    componentes = modelo.plot_components(previsao)
    st.write(componentes)

    # Gerar insights
    gerar_insights(df_valores, previsao)
