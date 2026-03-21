import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

try:
    print("Generating synthetic historical student dataset...")
    np.random.seed(42)

    n_samples = 2000

    # 1. Features
    attendance = np.random.uniform(40, 100, n_samples)
    sgpa = np.random.uniform(4.0, 10.0, n_samples)
    stress = np.random.uniform(1, 10, n_samples)
    missed_classes = np.random.randint(0, 10, n_samples)

    # 2. Heuristic baseline logic for dropping out (Adding noise for ML realism)
    risk_score = (
        (100 - attendance) * 0.45 + 
        (10 - sgpa) * 6 + 
        stress * 2.5 + 
        missed_classes * 3 + 
        np.random.normal(0, 6, n_samples) # Natural variance noise
    )

    # 3. Define target variable (1 = High Risk of Drop/Failure, 0 = Safe)
    target = (risk_score > 45).astype(int)

    df = pd.DataFrame({
        'attendance': attendance,
        'sgpa': sgpa,
        'stress': stress,
        'missed_classes': missed_classes,
        'target': target
    })

    X = df[['attendance', 'sgpa', 'stress', 'missed_classes']]
    y = df['target']

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y)

    os.makedirs('models', exist_ok=True)
    with open('models/risk_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("SUCCESS: Real ML Model generated and saved as models/risk_model.pkl!")
    
except Exception as e:
    print(f"Error: {e}")
