# save_best_model.py

import joblib
import pandas as pd

# Charger ton modèle final
model = joblib.load("/Users/hicham/Desktop/OCP7_25/src/models/XGBoost_smote.pkl")

# Charger les colonnes du train_clean
df_train = pd.read_csv("/Users/hicham/Desktop/OCP7_25/data/processed/train_clean.csv")
X_train = df_train.drop(columns=["TARGET", "SK_ID_CURR"])

# Sauvegarder modèle + colonnes ensemble
joblib.dump((model, list(X_train.columns)), "/Users/hicham/Desktop/OCP7_25/deployment/best_model.pkl")

print("✅ Modèle et colonnes sauvegardés proprement.")
