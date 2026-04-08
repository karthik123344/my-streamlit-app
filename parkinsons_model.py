import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv('parkinsons.csv')

X = data.drop(['status','name'], axis=1)  # IMPORTANT FIX
y = data['status']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open('parkinsons_model.pkl', 'wb'))

print("Parkinson model created ✅")
