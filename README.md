# Previsão de Preços de Criptomoedas

Este repositório contém um projeto para prever os preços de criptomoedas utilizando a API da Binance, o modelo Prophet para previsão de séries temporais e o Streamlit para visualização interativa.

## Requisitos

Antes de executar o projeto, certifique-se de ter as seguintes dependências instaladas:

```bash
pip install pandas streamlit binance prophet plotly scikit-learn numpy
```

Também é necessário ter uma conta na Binance e gerar uma chave de API. Crie um arquivo `secrets_1.py` no diretório do projeto e defina suas credenciais:

```python
# secrets_1.py
api_key = "SUA_API_KEY"
api_secret = "SEU_API_SECRET"
```

## Como Executar

1. Clone este repositório:

```bash
git clone https://github.com/seuusuario/seurepositorio.git
```

2. Acesse o diretório do projeto:

```bash
cd seurepositorio
```

3. Execute o aplicativo com o Streamlit:

```bash
streamlit run bin.py
```

4. A interface interativa abrirá no navegador, permitindo selecionar a criptomoeda e visualizar os resultados da previsão.

## Funcionalidades

- Obtenção de dados históricos da Binance
- Treinamento do modelo Prophet para previsão de preços
- Geração de gráficos interativos com Plotly
- Avaliação do modelo com MAE e RMSE
- Recomendações de compra, venda e retenção

## Estrutura do Projeto

```
/
├── bin.py    # Código principal
├── secrets_1.py          # Credenciais da API use as suas chaves 
├── cypto.csv             # Lista de criptomoedas suportadas
├── README.md             # Este arquivo
```

## Licença

Este projeto é de uso livre. Sinta-se à vontade para modificar e melhorar!

