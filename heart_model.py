import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1)
y = data['target']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open('heart_model.pkl', 'wb'))

print("Heart model created ✅")