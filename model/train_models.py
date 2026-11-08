"""
Train and Evaluate 5 Machine Learning Classifiers for Online Shoppers Purchasing Intention
Course: BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2
Student: 2026AJ05386

Models Trained:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVM)

Evaluation Metrics:
- Accuracy, AUC Score, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)
"""

import os
import urllib.request
import zipfile
import io
import warnings
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# 1. Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("Step 1: Downloading & Loading Dataset (UCI Online Shoppers Intention)")
print("=" * 70)

URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    df = pd.read_csv(z.open('online_shoppers_intention.csv'))

print(f"Dataset successfully loaded. Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

# 2. Preprocessing Column Definitions
# Convert Boolean targets to integer (1 = Purchase/Revenue, 0 = No Revenue)
df['Revenue'] = df['Revenue'].astype(int)
df['Weekend'] = df['Weekend'].astype(int)

numeric_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend'
]
categorical_cols = ['Month', 'VisitorType']
feature_cols = numeric_cols + categorical_cols
target_col = 'Revenue'

X = df[feature_cols]
y = df[target_col]

# 3. Train-Test Split (80% Train, 20% Test, Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# Export test_data.csv for the Streamlit application and repository
test_df = X_test.copy()
test_df['Revenue'] = y_test
test_df.to_csv(DATA_PATH, index=False)
print(f"Saved test dataset to {DATA_PATH} ({test_df.shape[0]} rows, {test_df.shape[1]} columns)")

# 4. Build Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# Fit preprocessor on X_train and save
preprocessor.fit(X_train)
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
print("Preprocessor saved successfully.")

# 5. Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Naive Bayes': GaussianNB(var_smoothing=1e-1),
    'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
}

model_filenames = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib'
}

print("\n" + "=" * 70)
print("Step 2 & 3: Training & Evaluating 5 Classification Models")
print("=" * 70)

metrics_list = []
reports = {}

for name, clf in models.items():
    print(f"--> Training {name}...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    
    # Save model pipeline
    save_path = os.path.join(MODEL_DIR, model_filenames[name])
    joblib.dump(pipe, save_path)
    
    # Predict on test data
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    reports[name] = {
        'confusion_matrix': cm,
        'classification_report': cr
    }
    
    metrics_list.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })

metrics_df = pd.DataFrame(metrics_list)
print("\n" + "=" * 70)
print("MODEL EVALUATION COMPARISON TABLE")
print("=" * 70)
print(metrics_df.to_string(index=False))

# Save metrics table to JSON and CSV for Streamlit app & documentation
metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)
with open(os.path.join(MODEL_DIR, "evaluation_details.json"), "w") as f:
    json.dump({'metrics': metrics_list, 'reports': reports}, f, indent=4)

print("\nAll models and evaluation metrics successfully saved to `model/` directory.")

