import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

print("=" * 60)
print("  Training Subject Pass Probability Model")
print("=" * 60)

np.random.seed(42)
n_samples = 5000

# Synthetic Data: CIE is out of 50
cie = np.random.uniform(5, 50, n_samples)
# Difficulty noise
noise = np.random.normal(0, 10, n_samples)
# SEE is roughly proportional to CIE (max 100)
see = cie * 1.8 + noise
see = np.clip(see, 0, 100)

total = cie + see
# Target: 1 (Pass) if total >= 40 and see >= 35, else 0 (Fail)
target = ((total >= 40) & (see >= 35)).astype(int)

df = pd.DataFrame({'cie': cie, 'target': target})
X = df[['cie']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

print("\nModel Evaluation:")
print(classification_report(y_test, clf.predict(X_test)))

os.makedirs('models', exist_ok=True)
model_data = {
    'model': clf,
    'features': ['cie']
}
with open('models/subject_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("SUCCESS: Saved model to models/subject_model.pkl")
