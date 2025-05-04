from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# === Charger les datasets
train_data = pd.read_csv("../data/processed/train_split.csv")
test_data = pd.read_csv("../data/processed/test_clean.csv")

# === Supprimer TARGET
if "TARGET" in train_data.columns:
    train_data = train_data.drop(columns=["TARGET"])

# === Supprimer colonnes inutiles
for col in ["SOURCE_test", "SOURCE_train"]:
    if col in train_data.columns:
        train_data = train_data.drop(columns=[col])
    if col in test_data.columns:
        test_data = test_data.drop(columns=[col])

# === Vérifier colonnes manquantes
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Réordonner les colonnes
test_data = test_data[train_data.columns]

# === COLONNES CATEGORIELLES : cast en str + fillna("MISSING")
for col in train_data.select_dtypes(include=['object', 'category']).columns:
    train_data[col] = train_data[col].astype(str).fillna("MISSING")

for col in test_data.select_dtypes(include=['object', 'category']).columns:
    test_data[col] = test_data[col].astype(str).fillna("MISSING")

# === COLONNES NUMERIQUES : fillna(0)
for col in train_data.select_dtypes(include=['number']).columns:
    train_data[col] = train_data[col].fillna(0)

for col in test_data.select_dtypes(include=['number']).columns:
    test_data[col] = test_data[col].fillna(0)

# === Générer le rapport
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=test_data)
report.save_html("data_drift_report.html")

print("✅ Rapport généré dans data_drift.html")
