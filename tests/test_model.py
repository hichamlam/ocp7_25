import joblib
import pandas as pd
import numpy as np

def test_model_prediction():
    # 📥 Charger le modèle et les colonnes attendues
    model, model_columns = joblib.load("src/models/best_model.pkl")

    # 📂 Charger une ligne de données depuis Google Drive (comme dans dashboard)
    url = "https://drive.google.com/uc?id=1DgXIYKQfbwIS3zNdVbR7nJcOWsazvS3k"
    df = pd.read_csv(url)
    sample = df.drop(columns=["SK_ID_CURR"], errors="ignore").iloc[0]

    # 🧼 Nettoyage des types
    def clean_feature(x):
        if isinstance(x, (np.floating, float)):
            return 0.0 if (np.isnan(x) or np.isinf(x)) else float(x)
        elif isinstance(x, (np.integer, int)):
            return int(x)
        elif isinstance(x, (np.bool_, bool)):
            return bool(x)
        return str(x)

    features = [clean_feature(x) for x in sample]

    # 📐 Vérifie la dimension
    assert len(features) == len(model_columns), f"Colonnes attendues : {len(model_columns)}, reçues : {len(features)}"

    # 🔮 Prédiction
    df_input = pd.DataFrame([features], columns=model_columns)
    proba = model.predict_proba(df_input)[0][1]

    # ✅ Vérifications
    assert isinstance(proba, float), "La probabilité retournée n'est pas un float"
    assert 0.0 <= proba <= 1.0, "La probabilité est hors des bornes [0, 1]"

    print(f"✅ Prédiction OK | Probabilité de défaut = {proba:.4f}")
