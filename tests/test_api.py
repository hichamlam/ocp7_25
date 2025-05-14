import requests
import pandas as pd
import numpy as np

def test_predict_api():
    # 📥 Charger les données depuis Google Drive
    data_url = "https://drive.google.com/uc?id=1DgXIYKQfbwIS3zNdVbR7nJcOWsazvS3k"
    df = pd.read_csv(data_url)
    sample = df.drop(columns=["SK_ID_CURR"], errors="ignore").iloc[0]


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
    url = "https://ocp7-25.onrender.com/predict" 
    response = requests.post(url, json={"features": features})

    # ✅ Vérifications automatiques
    assert response.status_code == 200, f"Status code != 200 : {response.status_code}"
    json_data = response.json()
    assert "proba" in json_data, "Réponse JSON ne contient pas 'proba'"
    assert isinstance(json_data["proba"], float), "La proba n'est pas un float"


    print(f"✅ Test réussi | Proba = {json_data['proba']:.4f}")
