import pandas as pd
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data"
TRAIN_FILE = "training_data.csv"
MODEL_FILE = "model.pkl"

# Hämta senaste två dagar för att jämföra prisändring
files = sorted(os.listdir(DATA_DIR))
if len(files) < 2:
    print("Not enough data yet to update model.")
    exit()

today = pd.read_csv(os.path.join(DATA_DIR, files[-1]))
yesterday = pd.read_csv(os.path.join(DATA_DIR, files[-2]))

# Matcha spelare via player_id
merged = pd.merge(today, yesterday, on='player_id', suffixes=('_today','_yesterday'))
merged['actual_change'] = merged['price_today'] - merged['price_yesterday']
merged['actual_change'] = merged['actual_change'].apply(lambda x: 'up' if x>0 else ('down' if x<0 else 'same'))

# Features
features = ['price_yesterday','ownership_yesterday','points_yesterday']
X = merged[features]
y = merged['actual_change']

# Träna RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Spara modell
dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# Uppdatera träningsdata
if os.path.exists(TRAIN_FILE):
    train_df = pd.read_csv(TRAIN_FILE)
    train_df = pd.concat([train_df, merged[features + ['actual_change']]], ignore_index=True)
else:
    train_df = merged[features + ['actual_change']]
train_df.to_csv(TRAIN_FILE, index=False)
print(f"Training data updated: {TRAIN_FILE}")
