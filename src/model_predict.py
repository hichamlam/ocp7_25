# src/model_predict.py

import joblib
import pandas as pd
from pathlib import Path

# 📥 Charger les données de test
print("📥 Chargement des données test_clean.csv...")
data_path = Path(__file__).resolve().parent.parent / "data" / "processed"
df_test = pd.read_csv(data_path / "test_clean.csv")

# 📌 Sauvegarder SK_ID_CURR pour soumission
ids = df_test["SK_ID_CURR"]

# 📤 Drop ID pour la prédiction
X_test = df_test.drop(columns=["SK_ID_CURR"], errors="ignore")

# 🔮 Charger le meilleur modèle
model_path = Path(__file__).resolve().parent / "models" / "LightGBM_Pipeline.pkl"
print(f"🧠 Chargement du modèle depuis {model_path}")
model = joblib.load(model_path)

# ✅ Faire les prédictions (proba)
print("🧪 Prédiction en cours...")
y_proba = model.predict_proba(X_test)[:, 1]

# 📄 Construire un DataFrame de prédictions
submission = pd.DataFrame({
    "SK_ID_CURR": ids,
    "TARGET": y_proba
})

# 💾 Sauvegarder
output_path = Path(__file__).resolve().parent.parent / "predictions.csv"
submission.to_csv(output_path, index=False)
print(f"✅ Prédictions sauvegardées dans: {output_path}")
