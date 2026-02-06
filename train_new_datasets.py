import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from utils import save_artifacts, normalize_binary_labels

print("Loading datasets...")

ddos_file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_1761766968440.csv"
portscan_file = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_1761766968441.csv"

df_ddos = pd.read_csv(ddos_file)
print(f"DDoS dataset loaded: {len(df_ddos)} rows")

df_portscan = pd.read_csv(portscan_file)
print(f"PortScan dataset loaded: {len(df_portscan)} rows")

df = pd.concat([df_ddos, df_portscan], ignore_index=True)
print(f"Combined dataset: {len(df)} rows")

print("\nColumn names found:")
print(df.columns.tolist())

selected_features = [
    ' Flow Duration',
    ' Total Fwd Packets', 
    ' Total Backward Packets',
    ' Fwd Packet Length Mean',
    ' Bwd Packet Length Mean',
    'Flow Bytes/s'
]

label_col = ' Label'

print(f"\nUsing features: {selected_features}")
print(f"Label column: {label_col}")

X = df[selected_features].copy()
y = df[label_col].copy()

print(f"\nOriginal label distribution:")
print(y.value_counts())

y = normalize_binary_labels(y)

print(f"\nNormalized label distribution:")
print(y.value_counts())

X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

print(f"\nFeature statistics:")
print(X.describe())

print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = {c: w for c, w in zip(classes, weights)}

print(f"\nClass weights: {class_weight}")

print("\nTraining Random Forest model...")
clf = RandomForestClassifier(
    n_estimators=200, 
    n_jobs=-1, 
    random_state=13, 
    class_weight=class_weight
)
clf.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = clf.predict(X_test)

print("\n=== IDS Model Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\nSaving model...")
save_artifacts({
    "ids_model": clf, 
    "ids_columns": selected_features
}, dirpath="models")

print("✓ Model successfully trained and saved to models/")
print(f"✓ Model can detect: {list(np.unique(y))}")
