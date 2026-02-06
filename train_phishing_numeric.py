import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from utils import save_artifacts

print("="*60)
print("PHISHING DETECTION MODEL TRAINING")
print("(Feature-based approach)")
print("="*60)

# Load the dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv("email_phishing_data.csv")
print(f"✓ Dataset loaded: {len(df):,} emails")
print(f"✓ Features: {len(df.columns)-1}")
print(f"✓ Columns: {list(df.columns)}")

# Check label distribution
print("\n[2/6] Analyzing label distribution...")
print(df['label'].value_counts())
label_counts = df['label'].value_counts()
print(f"✓ Legitimate (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"✓ Phishing (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.2f}%)")

# Prepare features and labels
print("\n[3/6] Preparing features and labels...")
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].copy()
y = df['label'].copy()

# Handle any missing or infinite values
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Label vector shape: {y.shape}")
print("\nFeature statistics:")
print(X.describe())

# Split data
print("\n[4/6] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {len(X_train):,} samples")
print(f"✓ Test set: {len(X_test):,} samples")

# Calculate class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = {c: w for c, w in zip(classes, weights)}
print(f"✓ Class weights: {class_weight}")

# Train model
print("\n[5/6] Training Random Forest model...")
print("Configuration:")
print("  - Estimators: 200")
print("  - Class weighting: Balanced")
print("  - Random state: 42")
print("  - Parallel processing: Enabled")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight=class_weight,
    verbose=1
)
clf.fit(X_train, y_train)

# Evaluate model
print("\n[6/6] Evaluating model on test set...")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    y_test, y_pred, 
    digits=4, 
    target_names=['Legitimate', 'Phishing']
))

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(f"                  Predicted")
print(f"                  Legit  Phish")
print(f"Actual  Legit     {cm[0][0]:6,}  {cm[0][1]:6,}")
print(f"        Phish     {cm[1][0]:6,}  {cm[1][1]:6,}")

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
save_artifacts({
    'phish_model': clf,
    'phish_features': feature_columns
}, dirpath='models')
print("✓ Model saved to: models/phish_model.pkl")
print("✓ Features saved to: models/phish_features.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"✓ Total training samples: {len(X_train):,}")
print(f"✓ Total test samples: {len(X_test):,}")
print(f"✓ Model type: Random Forest with {clf.n_estimators} trees")
print(f"✓ Features used: {len(feature_columns)}")
print("✓ Model ready for deployment!")
print("="*60)
