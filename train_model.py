"""
Enhanced ML Training Pipeline
Trains a RandomForestClassifier on academic + behavioral features.
Features: attendance, sgpa, avg_stress, missed_days, avg_mood, avg_energy, streak
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

try:
    print("=" * 60)
    print("  Intelligent Student Risk Prediction — Model Training")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 3000

    print(f"\n1. Generating {n_samples} synthetic student records...")

    # Academic features
    attendance = np.random.uniform(40, 100, n_samples)
    sgpa = np.random.uniform(3.0, 10.0, n_samples)

    # Behavioral features (from daily check-in aggregates)
    avg_stress = np.random.uniform(1, 5, n_samples)       # 1=chill, 5=high
    missed_days = np.random.randint(0, 8, n_samples)       # missed days per week
    avg_mood = np.random.uniform(1, 3, n_samples)          # 1=sad, 2=neutral, 3=happy
    avg_energy = np.random.uniform(1, 5, n_samples)        # 1=drained, 5=energized
    streak = np.random.randint(0, 15, n_samples)           # consecutive check-in days

    # Composite risk score with realistic weights
    base_score = (
        (100 - attendance) * 0.40 +
        (10 - sgpa) * 7.0 +
        avg_stress * 4.5 +
        missed_days * 3.5 +
        (3 - avg_mood) * 5.0 +
        (5 - avg_energy) * 2.0 +
        np.maximum(0, 7 - streak) * 1.5 +
        np.random.normal(0, 5, n_samples)
    )

    # Safety: excellent students rarely drop out
    risk_score = np.where(
        sgpa >= 8.5,
        np.minimum(base_score, np.random.uniform(5, 25, n_samples)),
        base_score
    )

    target = (risk_score > 50).astype(int)

    feature_names = [
        'attendance', 'sgpa', 'avg_stress', 'missed_days',
        'avg_mood', 'avg_energy', 'streak'
    ]

    df = pd.DataFrame({
        'attendance': attendance,
        'sgpa': sgpa,
        'avg_stress': avg_stress,
        'missed_days': missed_days,
        'avg_mood': avg_mood,
        'avg_energy': avg_energy,
        'streak': streak,
        'target': target
    })

    X = df[feature_names]
    y = df['target']

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Training set: {len(X_train)} | Test set: {len(X_test)}")
    print(f"   Risk distribution: {(target == 1).sum()} high-risk, {(target == 0).sum()} low-risk")

    print(f"\n2. Training RandomForestClassifier (100 trees, max_depth=8)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    print(f"\n3. Model Evaluation on Test Set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # Feature importances
    print("4. Feature Importances (Global):")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"   {name:20s} {imp:.4f}  {bar}")

    # Save model with metadata
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': clf,
        'feature_names': feature_names,
        'n_estimators': 100,
        'max_depth': 8,
        'training_samples': n_samples
    }
    with open('models/risk_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'=' * 60}")
    print(f"  SUCCESS: Model saved to models/risk_model.pkl")
    print(f"  Features: {len(feature_names)} ({', '.join(feature_names)})")
    print(f"{'=' * 60}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
