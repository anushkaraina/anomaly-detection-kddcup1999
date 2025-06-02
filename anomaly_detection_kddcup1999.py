# Anomaly Detection in Network Traffic using Isolation Forest
# Dataset: KDD Cup 1999 - https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# Step 1: Load Dataset
# ============================
data_path = "kddcup.data_10_percent_corrected" 

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Please download from Kaggle and place it in the project folder.")

df = pd.read_csv(data_path, names=column_names)

# ============================
# Step 2: Preprocess Data
# ============================
df['label_binary'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize features
X = df.drop(['label', 'label_binary'], axis=1)
y = df['label_binary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Step 3: Isolation Forest Model
# ============================
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled)

# Predict: 1 = normal, -1 = anomaly
y_pred = model.predict(X_scaled)
y_pred = np.where(y_pred == 1, 0, 1)  # Convert to binary: 0 = normal, 1 = anomaly

# ============================
# Step 4: Evaluation
# ============================
print("\nClassification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# Optional: Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
