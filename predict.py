import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

MODEL_FILE = "model.pkl"
DATA_DIR = "data"

# Hämta senaste CSV
files = sorted(os.listdir(DATA_DIR))
latest_file = os.path.join(DATA_DIR, files[-1])
df = pd.read_csv(latest_file)

features = ['price','ownership','points','net_transfers']

# Om modell finns, ladda den
if os.path.exists(MODEL_FILE):
    model = load(MODEL_FILE)
    print("Loaded existing ML model.")
else:
    # Skapa baseline
    df_dummy = df.copy()
    df_dummy['actual_change'] = np.random.choice(['up','down','same'], size=len(df_dummy))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df_dummy[features], df_dummy['actual_change'])
    dump(model, MODEL_FILE)
    print("No model found, baseline ML model created and saved.")

# Prediktioner
df['predicted_change'] = model.predict(df[features])
df[['first_name','last_name','predicted_change','net_transfers','ownership','price','points']].to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
