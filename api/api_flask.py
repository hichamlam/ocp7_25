from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

# 📁 Chemin du modèle local
model_path = "best_model.pkl"

# ✅ Télécharger le modèle si non présent
if not os.path.exists(model_path):
    print("📥 Téléchargement du modèle depuis Google Drive...")
    url = "https://drive.google.com/uc?id=1Fu21aQVEaNMOJxLoM0yDCFZpEYZTCzXp"
    gdown.download(url, model_path, quiet=False)

# 📦 Chargement du modèle et des colonnes d'entraînement
model, model_columns = joblib.load(model_path)

@app.route("/", methods=["GET"])
def health():
    return "✅ API scoring en ligne", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Vérifie le bon nombre de features
        if len(data['features']) != len(model_columns):
            return jsonify({
                'error': f"Longueur des features incorrecte. Attendu : {len(model_columns)}, reçu : {len(data['features'])}"
            })

        input_data = pd.DataFrame([data['features']], columns=model_columns)
        proba = model.predict_proba(input_data)[:, 1][0]

        print(f"✅ Proba calculée : {proba:.4f}")

        return jsonify({
            'probability': round(float(proba), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
