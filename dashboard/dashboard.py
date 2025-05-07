import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gdown
import os
import requests


# === Conversion JSON-safe des features ===
def clean_feature(x):
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return 0.0 if (np.isnan(x) or np.isinf(x)) else x
    elif isinstance(x, (np.integer, int)):
        return int(x)
    elif isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x

st.set_page_config(page_title="Scoring Crédit Interactif", layout="wide")
st.title("💳 Dashboard de scoring client")

# === URL Google Drive direct (transformée en "uc?export=download")
data_url = "https://drive.google.com/uc?id=1DgXIYKQfbwIS3zNdVbR7nJcOWsazvS3k"

@st.cache_data
def load_data():
    df = pd.read_csv(data_url)
    return df

df = load_data()


if "TARGET" in df.columns:
    df_features = df.drop(columns=["TARGET"])
else:
    df_features = df

all_vars = df_features.select_dtypes(include="number").columns.tolist()

# === BARRE LATÉRALE ===
st.sidebar.header("🔎 Sélection du client")
client_id = st.sidebar.selectbox("Choisir un client :", df["SK_ID_CURR"])

st.sidebar.markdown("## 🎚️ Seuil de décision")
threshold = st.sidebar.slider(
    "Choisir le seuil de probabilité de défaut pour refuser le crédit :",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

# ✅ Au lieu de supprimer TARGET, on le garde et on force sa valeur à 0
client_data = df[df["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"])
client_data["TARGET"] = 0  # on force TARGET à 0

# === DONNÉES CLIENT ===
st.subheader("🧍 Informations du client sélectionné")
st.dataframe(client_data.T)

# === PRÉDICTION API ===
st.subheader("📤 Prédiction du modèle (via API Flask)")

if st.button("Obtenir la prédiction du mohttpsdèle"):
    try:
        row = client_data.iloc[0]
        input_features = [clean_feature(x) for x in row]

        api_url = "https://ocp7-25.onrender.com/predict"
        with st.spinner("⏳ Prédiction en cours..."):
            response = requests.post(api_url, json={"features": input_features})

        if response.status_code == 200:
            result = response.json()

            if "proba" in result:
                proba = result["proba"]

                st.markdown("## 🎯 Résultat de la prédiction")
                st.markdown(
                    f"<h2 style='text-align: center;'>📊 Probabilité de défaut : "
                    f"<span style='color:#e74c3c;'>{proba:.2%}</span></h2>",
                    unsafe_allow_html=True
                )

                if proba >= threshold:
                    # Refus
                    st.markdown(
                        f"<div style='background-color:#f8d7da;padding:20px;border-radius:10px;"
                        f"border:1px solid #f5c6cb;'>"
                        f"<h2 style='color:#721c24;text-align:center;'>⛔ Crédit REFUSÉ (seuil : {threshold:.2f})</h2>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # Accord
                    st.markdown(
                        f"<div style='background-color:#d4edda;padding:20px;border-radius:10px;"
                        f"border:1px solid #c3e6cb;'>"
                        f"<h2 style='color:#155724;text-align:center;'>✅ Crédit ACCORDÉ (seuil : {threshold:.2f})</h2>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.error(f"❌ Erreur dans la réponse API : {result}")
        else:
            st.error(f"❌ Erreur API : {response.text}")

    except Exception as e:
        st.error(f"❌ Erreur technique : {e}")

# === GRAPHIQUE UNIVARIÉ ===
if var_uni := st.sidebar.selectbox("Choisir une variable :", ["-- Aucune --"] + all_vars):
    if var_uni != "-- Aucune --":
        st.subheader(f"📊 Distribution de {var_uni}")
        fig_uni, ax_uni = plt.subplots()
        sns.histplot(df[var_uni], kde=True, ax=ax_uni)
        ax_uni.axvline(client_data[var_uni].values[0], color='red', linestyle='--', label="Client")
        ax_uni.legend()
        st.pyplot(fig_uni)
    else:
        st.info("ℹ️ Sélectionnez une variable univariée pour afficher un histogramme.")

# === GRAPHIQUE BIVARIÉ ===
if (var_x := st.sidebar.selectbox("Variable X :", ["-- Aucune --"] + all_vars)) and \
   (var_y := st.sidebar.selectbox("Variable Y :", ["-- Aucune --"] + all_vars)):
    if var_x != "-- Aucune --" and var_y != "-- Aucune --":
        st.subheader(f"📊 Analyse croisée : {var_x} vs {var_y}")
        fig_bi, ax_bi = plt.subplots()
        sns.scatterplot(data=df, x=var_x, y=var_y, hue="TARGET", palette="coolwarm", alpha=0.3, ax=ax_bi)
        ax_bi.scatter(client_data[var_x], client_data[var_y], color="black", s=100, label="Client")
        ax_bi.legend()
        st.pyplot(fig_bi)
    else:
        st.info("ℹ️ Sélectionnez deux variables pour afficher une analyse croisée.")




