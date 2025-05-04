# src/data_preprocessing_full.py

import numpy as np
import pandas as pd
import gc
from pathlib import Path

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    return df

def application_train_test(path, num_rows=None, nan_as_category=False):
    train = pd.read_csv(path / 'application_train.csv', nrows=num_rows)
    test = pd.read_csv(path / 'application_test.csv', nrows=num_rows)

    train["SOURCE"] = "train"
    test["SOURCE"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df = df[df['CODE_GENDER'] != 'XNA']

    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # ⛔️ Sauvegarde temporaire de la colonne SOURCE
    source_col = df["SOURCE"]

    df = one_hot_encoder(df, nan_as_category)

    # ✅ Réinsertion manuelle
    df["SOURCE"] = source_col.values

    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    return df

def load_final_dataset(relative_data_path):
    data_path = Path(__file__).resolve().parent.parent / relative_data_path
    processed_dir = data_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = application_train_test(data_path)

    # ✅ Sauvegarde fichiers train/test
    df[df["SOURCE"] == "train"].drop(columns=["SOURCE"]).to_csv(processed_dir / "train_clean.csv", index=False)
    df[df["SOURCE"] == "test"].drop(columns=["SOURCE"]).to_csv(processed_dir / "test_clean.csv", index=False)

    print(f"✅ Fichiers sauvegardés dans {processed_dir}")
    return df

if __name__ == "__main__":
    df = load_final_dataset("data")
    print("✅ Preprocessing terminé avec succès.")



#Encodage one-hot

#Création de features utiles (INCOME_PER_PERSON, PAYMENT_RATE, etc.)

#Fusion train/test (et tu peux le couper plus tard)

#Nettoyage de base (XNA, 365243)PYTHONPATH=. python src/data_preprocessing_full.py
