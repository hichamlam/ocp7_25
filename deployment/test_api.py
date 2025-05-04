import requests
import pandas as pd
import numpy as np

def test_predict_api():
    # 📂 Charger une ligne de test
    df = pd.read_csv("../data/processed/test_split.csv")
    sample = df.drop(columns=["TARGET"]).iloc[0]

    # 🧼 Nettoyage et typage
    def clean_feature(x):
        if isinstance(x, (np.floating, float)):
            return 0.0 if (np.isnan(x) or np.isinf(x)) else float(x)
        elif isinstance(x, (np.integer, int)):
            return int(x)
        elif isinstance(x, (np.bool_, bool)):
            return bool(x)
        return str(x)

    features = [clean_feature(x) for x in sample]

    # 📤 Envoi de la requête
    url = "http://127.0.0.1:5000/predict"
    response = requests.post(url, json={"features": features})

    # ✅ Vérifications automatiques
    assert response.status_code == 200, f"Status code != 200 : {response.status_code}"
    json_data = response.json()
    assert "proba" in json_data, "Réponse JSON ne contient pas 'proba'"
    assert "prediction" in json_data, "Réponse JSON ne contient pas 'prediction'"
    assert isinstance(json_data["proba"], float), "La proba n'est pas un float"
    assert json_data["prediction"] in [0,1], "Prediction doit être 0 ou 1"
