import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import psycopg2
import yfinance as yf

# Fonction pour se connecter à la base de données PostgreSQL
def connect_db():
    try:
        connection = psycopg2.connect(
            database="finlogikk",
            user="atou26ag",
            password="qsdffdsq26",
            host="finlogik-postgres",
            port="5432"
        )
        return connection
    except psycopg2.OperationalError as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return None

# Fonction pour récupérer les données de prédiction
def get_prediction_data(ticker):
    connection = connect_db()
    if connection is None:
        return None, None

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT prediction_data, performance_metrics, filename FROM prediction_info WHERE ticker = %s ORDER BY created_at DESC LIMIT 1", (ticker,))
        result = cursor.fetchone()
        if result:
            prediction_data = result[0]
            performance_metrics = result[1]
            filename = result[2]
            return prediction_data, performance_metrics, filename
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return None, None
    finally:
        cursor.close()
        connection.close()

# Fonction pour récupérer le fichier PDF
def get_pdf_from_db(filename):
    connection = connect_db()
    if connection is None:
        return None

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT pdf_data FROM pdf_file WHERE filename = %s", (filename,))
        pdf_data = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return bytes(pdf_data) if isinstance(pdf_data, memoryview) else pdf_data
    except Exception as e:
        st.error(f"Erreur lors de la récupération du fichier PDF : {e}")
        return None

# Fonction pour styliser les cellules des signaux d'achat et de vente
def style_signals(row):
    if row['Signal'] == 'Buy':
        return ['background-color: #d4edda; color: #155724'] * len(row)
    elif row['Signal'] == 'Sell':
        return ['background-color: #f8d7da; color: #721c24'] * len(row)
    else:
        return [''] * len(row)

def detect_head_shoulder(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_head_shoulder = ((df['high_roll_max'] > df['High'].shift(1)) &
                          (df['high_roll_max'] > df['High'].shift(-1)) &
                          (df['High'] < df['High'].shift(1)) &
                          (df['High'] < df['High'].shift(-1)))
    mask_inv_head_shoulder = ((df['low_roll_min'] < df['Low'].shift(1)) &
                              (df['low_roll_min'] < df['Low'].shift(-1)) &
                              (df['Low'] > df['Low'].shift(1)) &
                              (df['Low'] > df['Low'].shift(-1)))
    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df

def detect_multiple_tops_bottoms(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    mask_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['close_roll_max'] < df['Close'].shift(1))
    mask_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['close_roll_min'] > df['Close'].shift(1))
    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    return df

def detect_triangle_pattern(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_asc = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
    mask_desc = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    return df

def detect_wedge(df, window=3):
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    return df

def detect_channel(df, window=3):
    roll_window = window
    channel_range = 0.1
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x[-1]-x[0])>0 else -1 if (x[-1]-x[0])<0 else 0)
    mask_channel_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    mask_channel_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['high_roll_max'] - df['low_roll_min'] <= channel_range * (df['high_roll_max'] + df['low_roll_min'])/2) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    roll_window = window
    range_threshold = threshold
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    mask_double_top = (df['high_roll_max'] >= df['High'].shift(1)) & (df['high_roll_max'] >= df['High'].shift(-1)) & (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    mask_double_bottom = (df['low_roll_min'] <= df['Low'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(-1)) & (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)) & ((df['High'].shift(1) - df['Low'].shift(1)) <= range_threshold * (df['High'].shift(1) + df['Low'].shift(1))/2) & ((df['High'].shift(-1) - df['Low'].shift(-1)) <= range_threshold * (df['High'].shift(-1) + df['Low'].shift(-1))/2)
    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'
    return df

def find_pivots(df):
    high_diffs = df['High'].diff()
    low_diffs = df['Low'].diff()
    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)
    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'
    return df

# Fonction principale pour l'application Streamlit
def main():
    st.title("📈 Prédiction des Taux de Change")

    st.subheader("Sélectionnez un Ticker pour voir les prédictions et télécharger les rapports PDF")
    ticker = st.selectbox('Choisissez un ticker', ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDEUR=X'])

    # Récupération des données de prédiction depuis la base de données
    prediction_data, performance_metrics, filename = get_prediction_data(ticker)

    if prediction_data:
        st.write("### Graphique des Prédictions et des Prix Réels")

        # Convertir les données en DataFrame
        df = pd.DataFrame({
            'dates': prediction_data['dates'],
            'actual_values': prediction_data['actual_values'],
            'predicted_values': prediction_data['predicted_values'],
            'buy_signals': prediction_data['buy_signals'],
            'sell_signals': prediction_data['sell_signals']
        })

        # Affichage des courbes interactives avec un design moderne
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['dates'], y=df['actual_values'], mode='lines', name='Prix Réels'))
        fig.add_trace(go.Scatter(x=df['dates'], y=df['predicted_values'], mode='lines', name='Prédictions', line=dict(color='firebrick')))

        st.plotly_chart(fig, use_container_width=True)

        # Organisation des métriques avec des colonnes côte à côte
        st.write("### Métriques de Performance")
        col1, col2, col3, col4 = st.columns(4)

        # Affichage des métriques avec style
        col1.metric(label="MSE", value=f"{performance_metrics['mse']:.5f}")
        col2.metric(label="RMSE", value=f"{performance_metrics['rmse']:.5f}")
        col3.metric(label="MAE", value=f"{performance_metrics['mae']:.5f}")
        col4.metric(label="R²", value=f"{performance_metrics['r2']:.2f}")

        # Préparer les données pour le tableau des signaux d'achat/vente
        signals_df = pd.DataFrame({
            'Date': df['dates'],
            'Signal': ['Buy' if buy else 'Sell' if sell else None for buy, sell in
                       zip(df['buy_signals'], df['sell_signals'])]
        })

        # Filtrer les lignes vides
        signals_df = signals_df.dropna(subset=['Signal'])  # Garde seulement les lignes où il y a un signal

        # Trier les signaux par date
        signals_df = signals_df.sort_values(by='Date', ascending=True)

        st.write("### Tableau des Signaux d'Achat et de Vente")

        # Agrandir et styliser le tableau avec Streamlit et Pandas
        st.table(signals_df.style.apply(style_signals, axis=1))

        # Bouton de téléchargement du fichier PDF avec un style moderne
        st.write("### Télécharger le Rapport")
        if filename:
            pdf_data = get_pdf_from_db(filename)
            if pdf_data:
                st.download_button(
                    label="💾 Télécharger le PDF",
                    data=pdf_data,
                    file_name=filename,
                    mime='application/pdf',
                    help="Cliquez pour télécharger le rapport PDF de prédiction"
                )
            else:
                st.error("Échec du téléchargement du fichier PDF.")

        # Dictionnaire pour mapper la période choisie à la durée en heures, jours ou mois
        period_mapping = {
            '1d': '1d',  # 1 jour
            '5d': '5d',  # 5 jours
            '1mo': '1mo',  # 1 mois (30 jours)
            '3mo': '3mo',  # 3 mois
            '6mo': '6mo',  # 6 mois
            '1y': '1y'
        }

        interval_mapping = {
            '1m': '1m',  # 1 minute
            '2m': '2m',  # 2 minutes
            '5m': '5m',  # 5 minutes
            '15m': '15m',  # 15 minutes
            '30m': '30m',  # 30 minutes
            '60m': '60m',  # 1 heure
            '90m': '90m',  # 1h30
            '1h': '1h',  # 1 heure
            '1d': '1d',  # 1 jour
            '5d': '5d',  # 5 jours
            '1wk': '1wk',  # 1 semaine
            '1mo': '1mo',  # 1 mois
            '3mo': '3mo'  # 3 mois
        }

        # Sélection de la période
        time_period = st.selectbox('Choisissez une période', list(period_mapping.keys()), index=2)

        # Sélection de l'intervalle
        time_interval = st.selectbox('Choisissez un intervalle', list(interval_mapping.keys()), index=8)

        # Télécharger les données pour le ticker et la période sélectionnés
        st.write(f"### Patterns Table for {ticker} over the last {time_period}")
        data = yf.download(ticker, period=period_mapping[time_period], interval=interval_mapping[time_interval])

        # Appliquer les fonctions de détection de pattern
        data = detect_head_shoulder(data)
        data = detect_multiple_tops_bottoms(data)
        data = detect_triangle_pattern(data)
        data = detect_wedge(data)
        data = detect_channel(data)
        data = detect_double_top_bottom(data)
        data = find_pivots(data)

        # On sélectionne les colonnes des patterns sans la colonne 'signal'
        pattern_columns = ['head_shoulder_pattern', 'multiple_top_bottom_pattern', 'triangle_pattern', 'wedge_pattern',
                           'channel_pattern', 'double_pattern']

        # Supprimer les lignes où toutes les colonnes de patterns sont NaN (et garder les lignes qui ont un signal, même si les patterns sont vides)
        data_filtered = data.dropna(subset=pattern_columns, how='all')

        # Afficher le tableau avec les colonnes désirées, y compris 'signal'
        data_filtered = data_filtered[pattern_columns + ['signal']]

        # Fonction pour styliser les colonnes en fonction des types de patterns détectés
        def highlight_patterns(val):
            colors = {
                'Head and Shoulder': 'background-color: #ffcccb; color: black;',
                'Inverse Head and Shoulder': 'background-color: #ffcccb; color: black;',
                'Multiple Top': 'background-color: #b3cde0; color: black;',
                'Multiple Bottom': 'background-color: #b3cde0; color: black;',
                'Ascending Triangle': 'background-color: #dcedc1; color: black;',
                'Descending Triangle': 'background-color: #dcedc1; color: black;',
                'Wedge Up': 'background-color: #fce7b2; color: black;',
                'Wedge Down': 'background-color: #fce7b2; color: black;',
                'Channel Up': 'background-color: #fff5b1; color: black;',
                'Channel Down': 'background-color: #fff5b1; color: black;',
                'Double Top': 'background-color: #ffda79; color: black;',
                'Double Bottom': 'background-color: #ffda79; color: black;',
                'HH': 'background-color: #c3f7d4; color: black;',
                'LL': 'background-color: #f7d4d4; color: black;',
                'LH': 'background-color: #d4d4f7; color: black;',
                'HL': 'background-color: #f7f7d4; color: black;'
            }
            return colors.get(val, '')

        # Appliquer le style au DataFrame
        styled_data = data_filtered.style.applymap(highlight_patterns).set_properties(
            **{
                'border': '1px solid black',
                'font-size': '14px',
                'text-align': 'center',
                'border-color': '#ddd'
            }
        ).set_caption("Tableau des Patterns Détectés")

        # Afficher le tableau stylisé
        st.dataframe(styled_data, use_container_width=True)

    else:
        st.warning("Aucune donnée de prédiction disponible pour ce ticker.")

if __name__ == '__main__':
    main()
