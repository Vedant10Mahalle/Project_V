import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

# Features:
# Age (18-40)
# Gender (0 or 1)
# Admission grade (100-200) -> Wait, description says "0-200"? Usually it's 0-20 or 0-200. Let's use 0-20 scale or 0-200. I'll use 0-20 for simplicity.
# First semester grade (0-20)
# Second semester grade (0-20)
# Scholarship status (0 or 1)
# Debtor status (0 or 1)
# Tuition fee payment status (0 or 1)

data = {
    'Age': np.random.randint(18, 41, n_samples),
    'Gender': np.random.randint(0, 2, n_samples),
    'Admission_grade': np.random.uniform(10, 20, n_samples),
    'First_semester_grade': np.random.uniform(0, 20, n_samples),
    'Second_semester_grade': np.random.uniform(0, 20, n_samples),
    'Scholarship_status': np.random.randint(0, 2, n_samples),
    'Debtor_status': np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples), # 20% debtors
    'Tuition_fee_status': np.random.choice([0, 1], p=[0.1, 0.9], size=n_samples) # 90% paid
}

df = pd.DataFrame(data)

# Create logic for Target variable ("Dropout", "Enrolled", "Graduate")
def determine_outcome(row):
    score = 0
    # Higher grades -> better chance of graduate
    if row['First_semester_grade'] + row['Second_semester_grade'] > 25:
        score += 3
    elif row['First_semester_grade'] + row['Second_semester_grade'] < 15:
        score -= 2
        
    if row['Debtor_status'] == 1:
        score -= 2
    if row['Tuition_fee_status'] == 0:
        score -= 3
    if row['Scholarship_status'] == 1:
        score += 2
        
    if score >= 3:
        return 'Graduate'
    elif score >= 0:
        return 'Enrolled'
    else:
        return 'Dropout'

df['Outcome'] = df.apply(determine_outcome, axis=1)

# Save dataset
os.makedirs('training', exist_ok=True)
df.to_csv('training/student_dataset.csv', index=False)
print("Saved dataset to training/student_dataset.csv")

# Train model
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Saved model and scaler to model/")
