import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
import pandas_ta as ta
from stocknews import StockNews
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
import spacy
from datetime import datetime, timedelta


newsapi_key = '08580393d090448282a4532bbaa1ce1c'
nlp = spacy.load("en_core_web_sm")

ticker_pairs = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "CHFJPY=X", "AUDJPY=X",
    "EURAUD=X", "GBPAUD=X", "GBPCAD=X", "EURCAD=X", "AUDCAD=X", "AUDNZD=X"
]

today = datetime.today()
five_years_ago = today - timedelta(days=5*365)

st.title('Currency Exchange Dashboard')
ticker = st.sidebar.selectbox('Ticker', ticker_pairs, index=0)
# ticker = st.sidebar.text_input('Ticker', value='EURUSD=X')
start_date = st.sidebar.date_input('Start Date', value=five_years_ago)
end_date = st.sidebar.date_input('End Date', value=today)


data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x=data.index, y=data['Adj Close'], title = ticker)
st.plotly_chart(fig)

pricing_data, tech_indicator, news = st.tabs(["Pricing Data", "Technical Analysis", "Sentiment Analysis"])

with pricing_data:
    st.header('📈 Price Movements')
    data2 = data
    data2 = data.drop(columns=['Volume', 'Adj Close'], errors='ignore')
    data2['% Change'] = data['Close'] / data['Close'].shift(1) - 1
    data2.dropna(inplace = True)

    st.dataframe(data2, width=1400)

    # Calcul des métriques
    annual_return = data2['% Change'].mean() * 252 * 100

    stdev = np.std(data2['% Change']) * np.sqrt(252) * 100

    risk_adj_return = annual_return / stdev

    cumulative_return = (data2['Close'][-1] / data2['Close'][0] - 1) * 100

    roll_max = data2['Close'].cummax()
    daily_drawdown = data2['Close'] / roll_max - 1.0
    max_drawdown = daily_drawdown.min() * 100

    # Organisation des résultats dans un tableau
    st.write("### Key Performance Metrics")

    # Utilisation de pandas pour créer un tableau
    metrics_df = pd.DataFrame({
        'Annual Return (%)': [f'{annual_return:.2f}%'],
        'Cumulative Return (%)': [f'{cumulative_return:.2f}%'],
        'Standard Deviation (%)': [f'{stdev:.2f}%'],
        'Max Drawdown (%)': [f'{max_drawdown:.2f}%'],
        'Risk Adjusted Return (Ratio)': [f'{risk_adj_return:.2f}']
    })

    # Affichage du tableau avec une belle mise en page
    st.write(metrics_df.to_html(index=False), unsafe_allow_html=True)


def get_newsapi_articles(query):
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&apiKey={newsapi_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        st.error(f"Error fetching news from NewsAPI: {response.status_code}")
        return []


# Fonction pour analyser le sentiment d'un texte
def analyze_sentiment(text):
    if text.strip() == "":
        return 0.0, 0.0  # Si le texte est vide, retourner des valeurs neutres
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def color(polarity):
    if polarity > 0:
        return f"<span style='color:green'>{polarity:.3f}</span>"
    elif polarity < 0:
        return f"<span style='color:red'>{polarity:.3f}</span>"
    else:
        return f"<span style='color:black'>{polarity:.3f}</span>"

def card_style():
    return """
    <style>
        .card {
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            background-color: #f5f5f5;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .card h3 {
            color: #2c3e50;
            font-weight: bold;
        }
        .card p {
            color: #34495e;
            font-size: 14px;
        }
        .card a {
            color: #2980b9;
            text-decoration: none;
            font-weight: bold;
        }
        .card a:hover {
            text-decoration: underline;
        }
    </style>
    """

with news:
    st.markdown(card_style(), unsafe_allow_html=True)
    st.title('Exchange Rate News and Sentiment Analysis')
    st.subheader('Latest News Related to Exchange Rates')

    currency_pair_for_news = ticker.replace("=X", "")
    articles = get_newsapi_articles(currency_pair_for_news)

    for article in articles:
        title = article['title']
        description = article['description'] or ""
        content = description + " " + (article['content'] or "")

        # Analyser le sentiment du contenu
        polarity, subjectivity = analyze_sentiment(content)

        rounded_polarity = round(polarity, 3)
        rounded_subjectivity = round(subjectivity, 3)

        # Afficher les résultats
        st.markdown(f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <p><strong>Polarity</strong>: {color(rounded_polarity)} &nbsp; <strong>Subjectivity</strong>: {color(rounded_subjectivity)}</p>
                    <p><a href="{article['url']}" target="_blank">Read more</a></p>
                </div>
                """, unsafe_allow_html=True)





with tech_indicator:
    st.subheader('Technical Analysis Dashboard')
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    tech_indicator = st.selectbox('Tech Indicator', options = ind_list)
    method = tech_indicator
    indicator = pd.DataFrame(getattr(ta,method)(low=data['Low'], close=data['Close'], high=data['High'], open=data['Open']))
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)





