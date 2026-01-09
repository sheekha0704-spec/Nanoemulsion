import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from google.colab import files
import os

# 1. Load the data to recreate the encoders
if os.path.exists('drug_loaded_nanoemulsion_1500.csv'):
    df = pd.read_csv('drug_loaded_nanoemulsion_1500.csv')

    # 2. Re-create and fit the encoders specifically for your categories
    le_drug = LabelEncoder()
    le_oil = LabelEncoder()
    le_surf = LabelEncoder()

    le_drug.fit(df['Drug_Name'])
    le_oil.fit(df['Oil_Type'])
    le_surf.fit(df['Surfactant_Type'])

    # 3. Explicitly save them to the Colab folder
    joblib.dump(le_drug, 'le_drug.pkl')
    joblib.dump(le_oil, 'le_oil.pkl')
    joblib.dump(le_surf, 'le_surf.pkl')

    print("✅ Files created in sidebar: le_drug.pkl, le_oil.pkl, le_surf.pkl")

    # 4. Force individual downloads
    for f in ['le_drug.pkl', 'le_oil.pkl', 'le_surf.pkl']:
        print(f"Requesting download for: {f}")
        files.download(f)
else:
    print("❌ CSV file not found! Please upload 'drug_loaded_nanoemulsion_1500.csv' to Colab first.")
