# 💳 Projet OCP7 – Modèle de scoring crédit

![Build Status](https://github.com/hichamdev/OCP7_25/actions/workflows/python-tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

Ce projet a été réalisé dans le cadre d'une mission de Data Science pour l'entreprise **Prêt à Dépenser**, spécialisée dans le crédit à la consommation. L'objectif principal est de construire un **modèle de scoring crédit** permettant de prédire la probabilité de défaut d'un client, avec une démarche complète de **mise en production** et de **suivi du modèle**.

---

## 🧠 Objectifs

- Prédire la probabilité de remboursement d’un client
- Prendre en compte l’asymétrie des erreurs (10×FN + 1×FP)
- Offrir un dashboard explicatif pour les chargés client
- Déployer une API de prédiction
- Suivre les performances et détecter le **data drift**
- Appliquer une démarche MLOps (tests, CI/CD, versioning)

---

## 🗂️ Arborescence du projet

OCP7_25/
├── api/ → API Flask pour les prédictions
├── dashboard/ → Interface Streamlit pour les chargés client
├── models/ → Modèles sauvegardés (.pkl)
├── data/ → Données brutes et clean
├── src/ → Scripts de preprocessing et entraînement
├── tests/ → Tests unitaires (API + modèle)
├── .github/workflows/ → CI GitHub Actions
├── requirements.txt
├── README.md


---

## 🚀 Déploiements

- **API Flask** : déployée sur [Render](https://ocp7-25.onrender.com)
- **Dashboard Streamlit** : [Accès ici](https://ocp7-dashboard.streamlit.app)

---

## ⚙️ Stack technique

- Python 3.11
- Scikit-learn, LightGBM, XGBoost
- SMOTE (déséquilibre)
- SHAP (interprétation)
- Streamlit (dashboard)
- Flask (API)
- MLflow (tracking)
- Evidently (data drift)
- GitHub Actions + pytest (CI)

---

## 📦 Installation locale

```bash
git clone https://github.com/hichamdev/OCP7_25.git
cd OCP7_25
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pytest -v tests/



---

## 🧠 Méthodologie

1. Prétraitement des données (`src/data_preprocessing_full.py`)
2. Modélisation et sélection (`train_models.py`) avec GridSearchCV
3. Score métier : 10×FN + FP
4. Seuil optimisé via `evaluate_models.py`
5. Interprétation locale & globale avec SHAP
6. API Flask et Dashboard Streamlit
7. Test de drift avec Evidently
8. Tests automatisés et déploiement continu (CI/CD)

---

## 📊 Résultats

- **Modèle retenu** : LightGBM + Pipeline
- **AUC** : 0.722
- **Seuil optimisé** : 0.10
- **Coût métier (optimisé)** : 35 470
- **Coût métier (seuil 0.5)** : 48 368

---

## ✅ Livrables produits

- `best_model.pkl`
- `dashboard.py` (Streamlit)
- `api_flask.py` (Flask API)
- `note_méthodologique.pdf`
- `presentation_soutenance.pdf`
- `comparaison_modeles.csv`
- `data_drift_report.html`
- `README.md`, `requirements.txt`, `tests/`



## ✅ Rapport de tests HTML

Les résultats des tests automatisés sont disponibles ici :

📄 [Voir le rapport de tests HTML](tests/report.html)

Le rapport est généré automatiquement avec `pytest-html` :
```bash
pytest -v --html=tests/report.html --self-contained-html


