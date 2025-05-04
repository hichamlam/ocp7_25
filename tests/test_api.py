import requests
import pandas as pd
import numpy as np

# Charger une ligne du fichier test
df = pd.read_csv("data/application_test.csv")

# Nettoyer les NaN / inf / -inf
row = df.iloc[0].replace({np.nan: 0, np.inf: 0, -np.inf: 0})
client_data = row.to_dict()

# Requête API
url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=client_data)

# Affichage
print("Réponse de l'API :", response.json())
