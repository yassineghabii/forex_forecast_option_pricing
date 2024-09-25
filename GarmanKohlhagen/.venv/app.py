from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import math
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pandas_datareader import data as pdr
import datetime
import ta
from tensorflow.keras import layers, models
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib.backends.backend_pdf import PdfPages
import schedule
import time
import threading
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import exc, text, create_engine
import io
import psycopg2
from celery import Celery
from celery.schedules import crontab
from celery_module import make_celery

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://atou26ag:qsdffdsq26@finlogik-postgres:5432/finlogikk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration pour Celery
app.config.update(
    broker_url='redis://redis:6379/0',
    result_backend='redis://redis:6379/0'
)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialiser Celery avec Flask
celery = make_celery(app)

# Définir le dossier de téléchargement
DOWNLOAD_FOLDER = 'static/prevision'
UPLOAD_FOLDER = os.path.join('static', 'prevision')

# currency_pairs = [
#      "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
#     "EURGBP=X", "EURJPY=X", "EURCHF=X", "GBPJPY=X", "AUDJPY=X", "AUDNZD=X", "CHFJPY=X",
#     "GBPCHF=X", "EURAUD=X", "USDHKD=X", "EURCAD=X", "GBPCAD=X", "NZDJPY=X"
# ]
currency_pairs = [
     "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "CHFJPY=X", "AUDJPY=X",
    "EURAUD=X", "GBPAUD=X", "GBPCAD=X", "EURCAD=X", "AUDCAD=X", "AUDNZD=X"
]

# Définition des classes et fonctions pour la tarification des options
class _GK_Limites:
    MAX32 = 2147483248.0
    MIN_T = 1.0 / 1000.0
    MIN_K = 0.01
    MIN_S = 0.01
    MIN_V = 0.005
    MAX_V = 1
    MAX_T = 100
    MAX_K = MAX32
    MAX_S = MAX32
    MIN_rf = -1
    MIN_rd = -1
    MAX_rf = 1
    MAX_rd = 1

class GK_CalculationError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

def _test_option_type(option_type):
    if (option_type != "c") and (option_type != "p"):
        raise ValueError("Entrée invalide option_type. Valeurs acceptables: c, p")

def _gk_test_inputs(option_type, s, k, t, rf, rd, v):
    _test_option_type(option_type)
    if (k < _GK_Limites.MIN_K) or (k > _GK_Limites.MAX_K):
        raise ValueError(f"Prix de Strike invalide (K). Intervalle acceptable : {_GK_Limites.MIN_K} à {_GK_Limites.MAX_K}")
    if (s < _GK_Limites.MIN_S) or (s > _GK_Limites.MAX_S):
        raise ValueError(f"Prix de Spot invalide (S). Intervalle acceptable : {_GK_Limites.MIN_S} à {_GK_Limites.MAX_S}")
    if (t < _GK_Limites.MIN_T) or (t > _GK_Limites.MAX_T):
        raise ValueError(f"Temps invalide (T = {t}). Intervalle acceptable : {_GK_Limites.MIN_T} à {_GK_Limites.MAX_T}")
    if (rf < _GK_Limites.MIN_rf) or (rf > _GK_Limites.MAX_rf):
        raise ValueError(f"Foreign rate invalide (rf = {rf}). Intervalle acceptable : {_GK_Limites.MIN_rf} à {_GK_Limites.MAX_rf}")
    if (rd < _GK_Limites.MIN_rd) or (rd > _GK_Limites.MAX_rd):
        raise ValueError(f"Domestic rate invalide (rd = {rd}). Intervalle acceptable : {_GK_Limites.MIN_rd} à {_GK_Limites.MAX_rd}")
    if (v < _GK_Limites.MIN_V) or (v > _GK_Limites.MAX_V):
        raise ValueError(f"Volatilité implicite invalide (V = {v}). Intervalle acceptable : {_GK_Limites.MIN_V} à {_GK_Limites.MAX_V}")

def _gk(option_type, s, k, t, rf, rd, v):
    _gk_test_inputs(option_type, s, k, t, rf, rd, v)
    t__sqrt = math.sqrt(t)
    d1 = (math.log(s / k) + ((rd-rf) + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        value = s * math.exp(-rf * t) * norm.cdf(d1) - k * math.exp(-rd * t) * norm.cdf(d2)
        delta = math.exp(-rf * t) * norm.cdf(d1)
        gamma = math.exp(-rf * t) * norm.pdf(d1) / (s * v * t__sqrt)
        theta = (s * v * math.exp(-rf * t) * norm.pdf(d1)) / (2 * t__sqrt) - rf * s * math.exp(
            -rf * t) * norm.cdf(d1) + rd * k * math.exp(-rd * t) * norm.cdf(d2)
        vega = math.exp(-rf * t) * s * t__sqrt * norm.pdf(d1)
        rho_domestic = k * t * math.exp(-rd * t) * norm.cdf(d2)
        rho_foreign = -s * t * math.exp(-rf * t) * norm.cdf(d1)
        omega = delta * s / value
        vanna = (-math.exp(rf * t) * norm.pdf(d1) * d2 / v)
    else:
        value = k * math.exp(-rd * t) * norm.cdf(-d2) - (s * math.exp(-rf * t) * norm.cdf(-d1))
        delta = -math.exp(-rf * t) * norm.cdf(-d1)
        gamma = math.exp(-rf * t) * norm.pdf(d1) / (s * v * t__sqrt)
        theta = (s * v * math.exp(-rf * t) * norm.pdf(d1)) / (2 * t__sqrt) + rf * s * math.exp(
            -rf * t) * norm.cdf(-d1) - rd * k * math.exp(-rd * t) * norm.cdf(-d2)
        vega = math.exp(-rf * t) * s * t__sqrt * norm.pdf(d1)
        rho_domestic = -k * t * math.exp(-rd * t) * norm.cdf(-d2)
        rho_foreign = s * t * math.exp(-rf * t) * norm.cdf(-d1)
        omega = delta * s / value
        vanna = (-math.exp(rf * t) * norm.pdf(d1) * d2 / v)

    return value, delta, gamma, theta, vega, rho_domestic, rho_foreign, omega, vanna

def garman_kohlhagen(option_type, s, k, t, rf, rd, v):
    gk = _gk(option_type, s, k, t, rf, rd, v)
    return gk

def _vol_implicite_approx(option_type, s, k, t, rf, rd, cp):
    _test_option_type(option_type)

    ebrt = math.exp(-rf * t)
    ert = math.exp(-rd * t)

    a = math.sqrt(2 * math.pi) / (s * ebrt + k * ert)

    if option_type == "c":
        payoff = s * ebrt - k * ert
    else:
        payoff = k * ert - s * ebrt

    b = cp - payoff / 2
    c = (payoff ** 2) / math.pi

    v = (a * (b + math.sqrt(b ** 2 + c))) / math.sqrt(t)

    return v

def _newton_vol_implicite(val_fn, option_type, s, k, t, rf, rd, cp, precision=.00001, max_steps=100):
    _test_option_type(option_type)

    v = _vol_implicite_approx(option_type, s, k, t, rf, rd, cp)
    v = max(_GK_Limites.MIN_V, v)
    v = min(_GK_Limites.MAX_V, v)

    value, delta, gamma, theta, vega, rho_domestic, rho_foreign, omega, vanna = val_fn(option_type, s, k, t, rf, rd, v)
    min_diff = abs(cp - value)

    countr = 0
    while precision <= abs(cp - value) <= min_diff and countr < max_steps:
        v = v - (value - cp) / vega
        if (v > _GK_Limites.MAX_V) or (v < _GK_Limites.MIN_V):
            break

        value, delta, gamma, theta, vega, rho_domestic, rho_foreign, omega, vanna = val_fn(option_type, s, k, t, rf, rd, v)
        min_diff = min(abs(cp - value), min_diff)

        countr += 1

    if abs(cp - value) < precision:
        return v
    else:
        return _rech_dichotomique_vol_implicite(val_fn, option_type, s, k, t, rf, rd, cp, precision, max_steps)

def _rech_dichotomique_vol_implicite(val_fn, option_type, s, k, t, rf, rd, cp, precision=.00001, max_steps=100):
    v_mid = _vol_implicite_approx(option_type, s, k, t, rf, rd, cp)

    if (v_mid <= _GK_Limites.MIN_V) or (v_mid >= _GK_Limites.MAX_V):
        v_low = _GK_Limites.MIN_V
        v_high = _GK_Limites.MAX_V
        v_mid = (v_low + v_high) / 2
    else:
        v_low = max(_GK_Limites.MIN_V, v_mid * .5)
        v_high = min(_GK_Limites.MAX_V, v_mid * 1.5)

    cp_mid = val_fn(option_type, s, k, t, rf, rd, v_mid)[0]

    current_step = 0
    diff = abs(cp - cp_mid)

    while (diff > precision) and (current_step < max_steps):
        current_step += 1

        if cp_mid < cp:
            v_low = v_mid
        else:
            v_high = v_mid

        cp_low = val_fn(option_type, s, k, t, rf, rd, v_low)[0]
        cp_high = val_fn(option_type, s, k, t, rf, rd, v_high)[0]

        v_mid = v_low + (cp - cp_low) * (v_high - v_low) / (cp_high - cp_low)
        v_mid = max(_GK_Limites.MIN_V, v_mid)
        v_mid = min(_GK_Limites.MAX_V, v_mid)

        cp_mid = val_fn(option_type, s, k, t, rf, rd, v_mid)[0]
        diff = abs(cp - cp_mid)

    if abs(cp - cp_mid) < precision:
        return v_mid
    else:
        raise GK_CalculationError("Vol implicite n'a pas convergé. Meilleure estimation={0}, Diff prix={1}, Precision requise={2}".format(v_mid, diff, precision))

def _gk_vol_implicite(option_type, s, k, t, rf, rd, cp, precision=.00001, max_steps=100):
    return _newton_vol_implicite(_gk, option_type, s, k, t, rf, rd, cp, precision, max_steps)

def calculate_historical_volatility(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    historical_vol = log_returns.std() * np.sqrt(252)
    return historical_vol

def create_greek_plot(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y, label=ylabel)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + string.decode('utf-8')
    buf.close()
    return uri

@app.route('/tarification')
def option_pricing():
    return render_template('tarification.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        option_type = data['option_type']
        ticker = data['ticker']
        k = float(data['k'])
        t = float(data['t'])
        rf = float(data['rf'])
        rd = float(data['rd'])
        data = yf.download(ticker, period='1y')
        s = data['Close'][-1]
        hist_vol = calculate_historical_volatility(ticker)

        price, delta, gamma, theta, vega, rho_domestic, rho_foreign, omega, vanna = garman_kohlhagen(option_type, s, k, t, rf, rd, hist_vol)
        imp_vol = _gk_vol_implicite(option_type, s, k, t, rf, rd, price)
        price_imp, delta_imp, gamma_imp, theta_imp, vega_imp, rho_domestic_imp, rho_foreign_imp, omega_imp, vanna_imp = garman_kohlhagen(option_type, s, k, t, rf, rd, imp_vol)

        # Generate plots for Greeks
        x = np.linspace(s * 0.8, s * 1.2, 100)
        delta_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[1] for xi in x]
        gamma_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[2] for xi in x]
        theta_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[3] for xi in x]
        vega_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[4] for xi in x]
        rho_domestic_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[5] for xi in x]
        rho_foreign_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[6] for xi in x]
        omega_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[7] for xi in x]
        vanna_y = [garman_kohlhagen(option_type, xi, k, t, rf, rd, imp_vol)[8] for xi in x]

        delta_plot = create_greek_plot(x, delta_y, 'Delta vs Spot Price', 'Spot Price', 'Delta')
        gamma_plot = create_greek_plot(x, gamma_y, 'Gamma vs Spot Price', 'Spot Price', 'Gamma')
        theta_plot = create_greek_plot(x, theta_y, 'Theta vs Spot Price', 'Spot Price', 'Theta')
        vega_plot = create_greek_plot(x, vega_y, 'Vega vs Spot Price', 'Spot Price', 'Vega')
        rho_domestic_plot = create_greek_plot(x, rho_domestic_y, 'Rho Domestic vs Spot Price', 'Spot Price', 'Rho Domestic')
        rho_foreign_plot = create_greek_plot(x, rho_foreign_y, 'Rho Foreign vs Spot Price', 'Spot Price', 'Rho Foreign')
        omega_plot = create_greek_plot(x, omega_y, 'Omega vs Spot Price', 'Spot Price', 'Omega')
        vanna_plot = create_greek_plot(x, vanna_y, 'Vanna vs Spot Price', 'Spot Price', 'Vanna')

        results = {
            'current_price': s,
            'historical_volatility': hist_vol,
            'implied_volatility': imp_vol,
            'option_price_implied_vol': price_imp,
            'delta': delta_imp,
            'gamma': gamma_imp,
            'theta': theta_imp,
            'vega': vega_imp,
            'rho_domestic': rho_domestic_imp,
            'rho_foreign': rho_foreign_imp,
            'omega': omega_imp,
            'vanna': vanna_imp,
            'plots': {
                'delta_plot': delta_plot,
                'gamma_plot': gamma_plot,
                'theta_plot': theta_plot,
                'vega_plot': vega_plot,
                'rho_domestic_plot': rho_domestic_plot,
                'rho_foreign_plot': rho_foreign_plot,
                'omega_plot': omega_plot,
                'vanna_plot': vanna_plot
            }
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

# Définition des classes et fonctions pour la prédiction des devises
def get_economic_data(indicator, start, end):
    return pdr.get_data_fred(indicator, start, end)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs).reshape(-1)


def run_model(ticker):
    with app.app_context():
        print(f"Running model for {ticker}...")
        start_date = datetime.datetime.now() - datetime.timedelta(days=5 * 365)
        end_date = datetime.datetime.now()

        data = yf.download(ticker, period="5y", interval="1d")
        data = data.dropna()

        indicators = {
            'Interest_Rate': 'FEDFUNDS',
            'GDP': 'GDP',
            'Consumer_Confidence_Index': 'UMCSENT',
            'Inflation': 'CPIAUCSL',
            'Trade_Balance': 'BOPGSTB',
            'Unemployment_Rate': 'UNRATE',
            'Foreign_Exchange_Reserves': 'TRESEGDX',
            'Producer_Price_Index': 'PPIACO',
            'Money_Supply_M2': 'M2SL',
            'Industrial_Production_Index': 'INDPRO'
        }

        for key, value in indicators.items():
            try:
                econ_data = get_economic_data(value, start_date, end_date)
                econ_data = econ_data.resample('D').ffill().reindex(data.index).ffill()
                data[key] = econ_data
            except Exception as e:
                print(f"Error retrieving {key}: {e}")

        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']  # MACD = EMA12 - EMA26
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Ligne de signal (EMA9 du MACD)
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']  # Histogramme pour visualiser la force du signal

        data['Buy_Signal_Prediction'] = (data['MACD'] > data['Signal_Line']) & (
                data['MACD_Histogram'] > 0)  # Achat quand le MACD croise au-dessus de la ligne de signal avec histogramme positif

        data['Sell_Signal_Prediction'] = (data['MACD'] < data['Signal_Line']) & (
                data['MACD_Histogram'] < 0)

        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['Bollinger_High'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['Bollinger_Low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        data['Stochastic'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'],
                                                              window=14).stoch()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'],
                                                     window=14).average_true_range()
        data['Momentum'] = ta.momentum.ROCIndicator(data['Close'], window=14).roc()

        data = data.dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        sequence_length = 180
        x_data, y_data = [], []

        for i in range(sequence_length, len(scaled_data)):
            x_data.append(scaled_data[i - sequence_length:i])
            y_data.append(scaled_data[i, 0])

        x_data, y_data = np.array(x_data), np.array(y_data)
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        input_shape = (x_train.shape[1], x_train.shape[2])
        transformer_model = build_transformer_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        transformer_model.compile(optimizer='adam', loss='mean_squared_error')

        transformer_checkpoint = ModelCheckpoint('transformer_model.keras', save_best_only=True, monitor='val_loss',
                                                 mode='min')
        transformer_early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        transformer_model.fit(x_train, y_train, epochs=150, batch_size=32, validation_data=(x_test, y_test),
                              callbacks=[transformer_checkpoint, transformer_early_stopping])

        lstm_model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.2),
            LSTM(units=100, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')

        lstm_checkpoint = ModelCheckpoint('lstm_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        lstm_early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        lstm_model.fit(x_train, y_train, epochs=150, batch_size=32, validation_data=(x_test, y_test),
                       callbacks=[lstm_checkpoint, lstm_early_stopping])

        cnn_model = build_cnn_model((x_train.shape[1], x_train.shape[2]))

        cnn_checkpoint = ModelCheckpoint('cnn_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        cnn_early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        cnn_model.fit(x_train, y_train, epochs=150, batch_size=32, validation_data=(x_test, y_test),
                      callbacks=[cnn_checkpoint, cnn_early_stopping])

        wrapped_transformer_model = KerasRegressorWrapper(transformer_model)
        wrapped_lstm_model = KerasRegressorWrapper(lstm_model)
        wrapped_cnn_model = KerasRegressorWrapper(cnn_model)

        ensemble = VotingRegressor(
            [('transformer', wrapped_transformer_model), ('lstm', wrapped_lstm_model), ('cnn', wrapped_cnn_model)])

        # ensemble = VotingRegressor(
        #     [('cnn', wrapped_cnn_model)])

        ensemble.fit(x_train, y_train)
        predictions = ensemble.predict(x_test)

        predictions = scaler.inverse_transform(
            np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))),
                           axis=1))[:, 0]
        y_test_real = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:,
                      0]

        mse = mean_squared_error(y_test_real, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, predictions)
        r2 = r2_score(y_test_real, predictions)

        print(f'Final MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')

        future_days = 180
        future_predictions = []
        last_sequence = scaled_data[-sequence_length:].reshape((1, sequence_length, scaled_data.shape[1]))

        noise_level = 0.005

        for _ in range(future_days):
            next_pred = ensemble.predict(last_sequence)
            next_pred += noise_level * np.random.randn(*next_pred.shape)
            future_predictions.append(next_pred[0])
            last_sequence = np.append(last_sequence[:, 1:, :], np.concatenate(
                (next_pred.reshape(1, 1, 1), np.zeros((1, 1, scaled_data.shape[1] - 1))), axis=2), axis=1)

        future_predictions = np.array(future_predictions)
        future_predictions = scaler.inverse_transform(np.concatenate(
            (future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], scaled_data.shape[1] - 1))),
            axis=1))[:, 0]
        last_date = data.index[-1]
        future_dates = pd.date_range(last_date, periods=future_days + 1, inclusive='right')
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Prediction'])

        combined_df = pd.DataFrame(data={'Date': data.index, 'Actual': data['Close']})
        future_df.reset_index(inplace=True)
        future_df.rename(columns={'index': 'Date'}, inplace=True)
        combined_df = pd.concat([combined_df, future_df], ignore_index=True)

        # combined_df['SMA_20_Prediction'] = combined_df['Prediction'].rolling(window=20).mean()
        # combined_df['SMA_50_Prediction'] = combined_df['Prediction'].rolling(window=50).mean()
        #
        # combined_df['Buy_Signal_Prediction'] = (combined_df['SMA_20_Prediction'] > combined_df['SMA_50_Prediction']) & (
        #         combined_df['SMA_20_Prediction'].shift(1) <= combined_df['SMA_50_Prediction'].shift(1))
        # combined_df['Sell_Signal_Prediction'] = (combined_df['SMA_20_Prediction'] < combined_df[
        #     'SMA_50_Prediction']) & (combined_df['SMA_20_Prediction'].shift(1) >= combined_df['SMA_50_Prediction'].shift(1))

        combined_df['MACD_Prediction'] = combined_df['Prediction'].ewm(span=12, adjust=False).mean() - \
                                         combined_df['Prediction'].ewm(span=26, adjust=False).mean()
        combined_df['Signal_Line_Prediction'] = combined_df['MACD_Prediction'].ewm(span=9, adjust=False).mean()
        combined_df['MACD_Histogram_Prediction'] = combined_df['MACD_Prediction'] - combined_df[
            'Signal_Line_Prediction']

        last_signal = 0  # 1 = Buy, -1 = Sell, 0 = None

        for i in range(1, len(combined_df)):
            if combined_df['MACD_Prediction'].iloc[i] > combined_df['Signal_Line_Prediction'].iloc[i] and \
                    combined_df['MACD_Histogram_Prediction'].iloc[i] > 0:
                if last_signal != 1:  # Éviter deux signaux Buy consécutifs
                    combined_df.at[combined_df.index[i], 'Buy_Signal_Prediction'] = True
                    combined_df.at[combined_df.index[i], 'Sell_Signal_Prediction'] = False
                    last_signal = 1
                else:
                    combined_df.at[combined_df.index[i], 'Buy_Signal_Prediction'] = False
            elif combined_df['MACD_Prediction'].iloc[i] < combined_df['Signal_Line_Prediction'].iloc[i] and \
                    combined_df['MACD_Histogram_Prediction'].iloc[i] < 0:
                if last_signal != -1:  # Éviter deux signaux Sell consécutifs
                    combined_df.at[combined_df.index[i], 'Sell_Signal_Prediction'] = True
                    combined_df.at[combined_df.index[i], 'Buy_Signal_Prediction'] = False
                    last_signal = -1
                else:
                    combined_df.at[combined_df.index[i], 'Sell_Signal_Prediction'] = False
            else:
                combined_df.at[combined_df.index[i], 'Buy_Signal_Prediction'] = False
                combined_df.at[combined_df.index[i], 'Sell_Signal_Prediction'] = False

        # Remplir les NaN dans les colonnes de signaux par False pour éviter les erreurs
        combined_df['Buy_Signal_Prediction'] = combined_df['Buy_Signal_Prediction'].fillna(False).infer_objects()
        combined_df['Sell_Signal_Prediction'] = combined_df['Sell_Signal_Prediction'].fillna(False).infer_objects()

        future_df = pd.DataFrame(data={'Date': future_dates, 'Prediction': future_predictions})

        # future_df['SMA_20_Prediction'] = future_df['Prediction'].rolling(window=20).mean()
        # future_df['SMA_50_Prediction'] = future_df['Prediction'].rolling(window=50).mean()
        #
        # future_df['Buy_Signal_Prediction'] = (future_df['SMA_20_Prediction'] > future_df['SMA_50_Prediction']) & (
        #         future_df['SMA_20_Prediction'].shift(1) <= future_df['SMA_50_Prediction'].shift(1))
        # future_df['Sell_Signal_Prediction'] = (future_df['SMA_20_Prediction'] < future_df['SMA_50_Prediction']) & (
        #         future_df['SMA_20_Prediction'].shift(1) >= future_df['SMA_50_Prediction'].shift(1))

        future_df['MACD_Prediction'] = future_df['Prediction'].ewm(span=12, adjust=False).mean() - \
                                         future_df['Prediction'].ewm(span=26, adjust=False).mean()
        future_df['Signal_Line_Prediction'] = future_df['MACD_Prediction'].ewm(span=9, adjust=False).mean()
        future_df['MACD_Histogram_Prediction'] = future_df['MACD_Prediction'] - future_df[
            'Signal_Line_Prediction']

        # Éviter les signaux consécutifs et corriger l'inversion des signaux (Buy sur prix bas, Sell sur prix haut)
        last_signal = 0  # 1 = Buy, -1 = Sell, 0 = None

        for i in range(1, len(future_df)):
            if future_df['MACD_Prediction'].iloc[i] > future_df['Signal_Line_Prediction'].iloc[i] and \
                    future_df['MACD_Histogram_Prediction'].iloc[i] > 0:
                if last_signal != 1:  # Éviter deux signaux Buy consécutifs
                    future_df.at[future_df.index[i], 'Buy_Signal_Prediction'] = True
                    future_df.at[future_df.index[i], 'Sell_Signal_Prediction'] = False
                    last_signal = 1
                else:
                    future_df.at[future_df.index[i], 'Buy_Signal_Prediction'] = False
            elif future_df['MACD_Prediction'].iloc[i] < future_df['Signal_Line_Prediction'].iloc[i] and \
                    future_df['MACD_Histogram_Prediction'].iloc[i] < 0:
                if last_signal != -1:  # Éviter deux signaux Sell consécutifs
                    future_df.at[future_df.index[i], 'Sell_Signal_Prediction'] = True
                    future_df.at[future_df.index[i], 'Buy_Signal_Prediction'] = False
                    last_signal = -1
                else:
                    future_df.at[future_df.index[i], 'Sell_Signal_Prediction'] = False
            else:
                future_df.at[future_df.index[i], 'Buy_Signal_Prediction'] = False
                future_df.at[future_df.index[i], 'Sell_Signal_Prediction'] = False

        # Remplir les NaN dans les colonnes de signaux par False pour éviter les erreurs
        future_df['Buy_Signal_Prediction'] = future_df['Buy_Signal_Prediction'].fillna(False).infer_objects()
        future_df['Sell_Signal_Prediction'] = future_df['Sell_Signal_Prediction'].fillna(False).infer_objects()

        pdf_name = f"{ticker}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
        # pdf_path = os.path.join(DOWNLOAD_FOLDER, pdf_name)

        # Enregistrer les prédictions dans la base de données
        prediction_data = {
            'dates': combined_df['Date'].tolist(),
            'actual_values': combined_df['Actual'].tolist(),
            'predicted_values': combined_df['Prediction'].tolist(),
            'buy_signals': combined_df['Buy_Signal_Prediction'].tolist(),
            'sell_signals': combined_df['Sell_Signal_Prediction'].tolist()
        }
        performance_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        save_prediction_to_db(pdf_name, ticker, prediction_data, performance_metrics)

        # Use a buffer instead of saving to the file system
        pdf_buffer = io.BytesIO()

        # with PdfPages(pdf_path) as pdf:
        with PdfPages(pdf_buffer) as pdf:
            plt.figure(figsize=(14, 5))
            plt.plot(combined_df['Date'], combined_df['Actual'], color='blue', label='Prix réel')
            plt.plot(combined_df['Date'], combined_df['Prediction'], color='red', label='Prédictions futures')
            plt.scatter(combined_df[combined_df['Buy_Signal_Prediction']]['Date'],
                        combined_df[combined_df['Buy_Signal_Prediction']]['Prediction'], marker='^', color='green',
                        label='Buy Signal (Prediction)', alpha=1)
            plt.scatter(combined_df[combined_df['Sell_Signal_Prediction']]['Date'],
                        combined_df[combined_df['Sell_Signal_Prediction']]['Prediction'], marker='v', color='red',
                        label='Sell Signal (Prediction)', alpha=1)
            plt.title('Prédiction des taux de change avec signaux d\'achat et de vente')
            plt.xlabel('Date')
            plt.ylabel('Taux de change')
            plt.legend()
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(14, 5))
            plt.plot(future_df['Date'], future_df['Prediction'], color='red', label='Prédictions futures')
            plt.scatter(future_df[future_df['Buy_Signal_Prediction']]['Date'],
                        future_df[future_df['Buy_Signal_Prediction']]['Prediction'], marker='^', color='green',
                        label='Buy Signal (Prediction)', alpha=1)
            plt.scatter(future_df[future_df['Sell_Signal_Prediction']]['Date'],
                        future_df[future_df['Sell_Signal_Prediction']]['Prediction'], marker='v', color='red',
                        label='Sell Signal (Prediction)', alpha=1)
            plt.title('Prédiction des taux de change avec signaux d\'achat et de vente')
            plt.xlabel('Date')
            plt.ylabel('Taux de change')
            plt.legend()
            pdf.savefig()
            plt.close()


        # print(f"PDF saved as {pdf_path}")

        # Save PDF data to the buffer
        pdf_buffer.seek(0)
        save_pdf_to_db(pdf_name, pdf_buffer.read())
        pdf_buffer.close()

        print(f"PDF saved as: {pdf_name}")



class PDFFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    pdf_data = db.Column(db.LargeBinary, nullable=False)  # Store binary data
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __init__(self, filename, pdf_data):
        self.filename = filename
        self.pdf_data = pdf_data

class PredictionInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)  # Le nom du fichier PDF associé
    ticker = db.Column(db.String(50), nullable=False)  # Le ticker associé à la prédiction
    prediction_data = db.Column(db.JSON, nullable=False)  # Les données de prédiction (dates, valeurs, etc.)
    performance_metrics = db.Column(db.JSON, nullable=False)  # Les métriques de performance (MSE, MAE, RMSE)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)  # Date de création

def convert_timestamps_to_str(data):
    """
    Fonction récursive pour convertir les objets de type `Timestamp` en chaînes au format ISO.
    """
    if isinstance(data, list):
        return [convert_timestamps_to_str(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_timestamps_to_str(value) for key, value in data.items()}
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, datetime.datetime):
        return data.isoformat()
    else:
        return data

def replace_nan_with_none(data):
    """
    Fonction récursive pour remplacer les valeurs NaN par None dans un dictionnaire ou une liste.
    """
    if isinstance(data, list):
        return [replace_nan_with_none(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_nan_with_none(value) for key, value in data.items()}
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data

def save_prediction_to_db(pdf_name, ticker, prediction_data, performance_metrics):
    try:
        with app.app_context():
            print(f"Starting to save prediction data for {ticker}")

            # Convertir les Timestamps en chaînes de caractères
            prediction_data_clean = convert_timestamps_to_str(prediction_data)
            prediction_data_clean = replace_nan_with_none(prediction_data_clean)

            new_prediction = PredictionInfo(
                filename=pdf_name,
                ticker=ticker,
                prediction_data=prediction_data_clean,
                performance_metrics=performance_metrics
            )
            db.session.add(new_prediction)
            db.session.commit()  # Sauvegarde dans la base de données
            print(f"Prediction data for {ticker} saved in the database.")
    except exc.SQLAlchemyError as e:
        db.session.rollback()  # Gestion du rollback en cas d'erreur
        print(f"Error saving prediction data to database: {str(e)}")
    except Exception as e:
        print(f"General Error: {str(e)}")
    finally:
        db.session.remove()  # Libération du contexte de session


def save_pdf_to_db(pdf_name, pdf_data):
    try:
        # Assurez-vous que l'accès à la base de données est dans un contexte Flask
        with app.app_context():
            new_pdf = PDFFile(filename=pdf_name, pdf_data=pdf_data)
            db.session.add(new_pdf)
            db.session.commit()
            print(f"PDF saved to database: {pdf_name}")
    except exc.SQLAlchemyError as e:
        db.session.rollback()
        print(f"Error saving PDF to database: {str(e)}")
    finally:
        db.session.remove()

def get_pdf_from_db(filename):
    # Connexion à la base de données PostgreSQL
    conn = psycopg2.connect("dbname=finlogikk user=atou26ag password=qsdffdsq26")
    cur = conn.cursor()
    cur.execute("SELECT pdf_data FROM pdf_file WHERE filename  = %s", (filename,))
    pdf_data = cur.fetchone()[0]
    cur.close()
    conn.close()
    return pdf_data

@app.route('/prevision', methods=['GET'])
def list_files():
    try:
        conn = psycopg2.connect("dbname=finlogikk user=atou26ag password=qsdffdsq26")
        cur = conn.cursor()
        cur.execute("SELECT id, filename, created_at FROM pdf_file")
        pdf_files = cur.fetchall()

        if not pdf_files:
            return jsonify({'error': 'Aucun fichier trouvé'}), 404

        files = [{
            'id': file[0],
            'filename': file[1],
            'created_at': file[2]
        } for file in pdf_files]

        cur.close()
        conn.close()

        return jsonify({'files': files}), 200

    except psycopg2.OperationalError as e:
        # Gestion de l'erreur de connexion à la base de données
        return jsonify({'error': f"Erreur de connexion à la base de données: {str(e)}"}), 500
    except Exception as e:
        # Gestion de toutes les autres erreurs
        return jsonify({'error': f"Une erreur s'est produite: {str(e)}"}), 500



@app.route('/openfile/<string:filename>', methods=['GET'])
def open_pdf(filename):
    pdf_data = get_pdf_from_db(filename)
    return send_file(io.BytesIO(pdf_data), download_name=filename, mimetype='application/pdf')

@app.route('/downloadfile/<string:filename>', methods=['GET'])
def download_pdf(filename):
    pdf_data = get_pdf_from_db(filename)
    return send_file(io.BytesIO(pdf_data), download_name=filename, as_attachment=True)


# def schedule_task():
#     def job():
#         now = datetime.datetime.now()
#         if now.hour == 20 and now.minute == 45:
#             print("Running model on the first day of the month at 04:20")
#             for ticker in currency_pairs:
#                 # Créer un contexte d'application pour chaque tâche planifiée
#                 with app.app_context():
#                     run_model(ticker)
#     schedule.every().day.at("20:43").do(job)
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
#
# # Lancer la tâche planifiée dans un thread séparé
# threading.Thread(target=schedule_task).start()

@celery.task(bind=True, max_retries=5)
def run_model_task(self, ticker):
    with app.app_context():  # Assurez-vous que chaque tâche s'exécute dans le contexte Flask
        try:
            run_model(ticker)
        except Exception as exc:
            raise self.retry(exc=exc, countdown=60)  # Réessayer après 60 secondes

# Planifier la tâche pour qu'elle s'exécute chaque jour à 07:02 avec Celery Beat
@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    for pair in currency_pairs:
        sender.add_periodic_task(
            crontab(minute=0, hour=0, day_of_month=1),
            run_model_task.s(pair),  # Exécute le modèle pour la paire en cours
            name=f"run_model_for_{pair}",
            queue='currency_queue'  # Ajout de la file d'attente pour chaque tâche
        )

@app.route('/run_task', methods=['POST'])
def run_task():
    data = request.json
    ticker = data.get("ticker", "EURUSD=X")  # Récupérer la paire de devises, par défaut "EURUSD=X"
    task = run_model_task.apply_async(args=[ticker])  # Exécute la tâche Celery
    return jsonify({'task_id': task.id}), 202


@app.route('/test_db', methods=['GET'])
def test_db():
    try:
        # Utilisez la fonction text() pour exécuter une requête SQL brute
        result = db.session.execute(text('SELECT 1'))
        return jsonify({'message': 'Database connection successful!'}), 200
    except Exception as e:
        app.logger.error(f"Database connection error: {str(e)}")
        return jsonify({'error': f"Database connection error: {str(e)}"}), 500


@app.route('/test_connection')
def test_connection():
    try:
        engine = create_engine('postgresql://atou26ag:qsdffdsq26@finlogik-postgres:5432/finlogikk')
        connection = engine.connect()
        return "Connection successful!"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/test_db_psycopg2', methods=['GET'])
def test_db_psycopg2():
    connection = None  # Initialisation de la variable connection
    try:
        # Connexion à la base de données PostgreSQL
        connection = psycopg2.connect(
            database="finlogikk",
            user="atou26ag",
            password="qsdffdsq26",
            host="finlogik-postgres",  # Utilise le nom du conteneur Docker ou "localhost" si la base est locale
            port="5432"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        if result:
            return jsonify({"message": "Connexion réussie à PostgreSQL avec psycopg2!"}), 200
        else:
            return jsonify({"message": "La requête n'a pas retourné de résultat."}), 500
    except psycopg2.OperationalError as e:
        # Ajouter les détails complets de l'erreur dans la réponse
        return jsonify({"error": f"Erreur de connexion à la base de données: {str(e)}"}), 500
    except Exception as e:
        # Capturer toutes les autres exceptions pour plus de détails
        return jsonify({"error": f"Erreur inattendue: {str(e)}"}), 500
    finally:
        if connection:
            cursor.close()
            connection.close()


if __name__ == '__main__':
    app.run(debug=True)
