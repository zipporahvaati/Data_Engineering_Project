# src/Train_Model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -------------------------------
# 1. Paths
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir,"fraud_dataset_5000.csv")  # your dataset in project root
print("Looking for dataset at:", data_path)

# -------------------------------
# 2. Verify Dataset
# -------------------------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found! Please check the path: {data_path}")

# -------------------------------
# 3. Load Data
# -------------------------------
df = pd.read_csv(data_path)
print("Dataset loaded successfully!")
print(df.head())

# -------------------------------
# 4. Preprocess Data
# -------------------------------
# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Scale numerical features (update according to your dataset)
numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig']  # example columns
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# -------------------------------
# 5. Split Features and Target
# -------------------------------
X = df.drop('isFraud', axis=1)  # target column
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Train Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# -------------------------------
# 7. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Save Model
# -------------------------------
joblib.dump(model, model_path)
print(f"Model saved successfully at {model_path}")
