from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os


app = Flask(__name__)


# 📁 Chemin du modèle local
model_path = "best_model.pkl"


# ✅ Télécharger le modèle si non présent
if not os.path.exists(model_path):
    print("📥 Téléchargement du modèle depuis Google Drive...")
    url = f"https://drive.google.com/file/d/1Fu21aQVEaNMOJxLoM0yDCFZpEYZTCzXp"
    gdown.download(url, model_path, quiet=False)

# 📦 Chargement du modèle entraîné et des colonnes d'entraînement
model, model_columns = joblib.load("best_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # ✅ Vérifie que le bon nombre de features est envoyé
        if len(data['features']) != len(model_columns):
            return jsonify({
                'error': f"Longueur des features incorrecte. Attendu : {len(model_columns)}, reçu : {len(data['features'])}"
            })

        # 🧠 Création du DataFrame pour la prédiction
        input_data = pd.DataFrame([data['features']], columns=model_columns)

        # 🔮 Prédiction
        proba = model.predict_proba(input_data)[:, 1][0]

        print(f"✅ Requête reçue avec proba = {proba:.4f}")

        return jsonify({
            'proba': round(float(proba), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
