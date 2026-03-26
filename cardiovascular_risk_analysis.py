# =============================================================================
# CARDIOVASCULAR DISEASE RISK PREDICTION
# Dataset: cardio.csv
# Tools: pandas, NumPy, scikit-learn, matplotlib, seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================

df = pd.read_csv("data/cardio.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS & OUTLIER DETECTION
# =============================================================================

# Convert age from days to years for readability
df['age_years'] = df['age'] / 365.25

# --- Outlier Visualisation (Boxplots) ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['age_years'], color='skyblue')
plt.title('Boxplot of Age (Years)')
plt.ylabel('Age')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['weight'], color='lightgreen')
plt.title('Boxplot of Weight (kg)')
plt.ylabel('Weight')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['ap_hi'], color='salmon')
plt.title('Boxplot of Systolic BP (ap_hi)')
plt.ylabel('Pressure')

plt.tight_layout()
plt.savefig("outputs/outliers_boxplots.png", dpi=150)
plt.show()
print("Outlier boxplots saved to outputs/outliers_boxplots.png")

# --- Duplicate Check ---
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")

# --- Missing Value Report ---
missing_pct = (df.isnull().sum() / len(df)) * 100
print("\nMissing values (%):")
print(missing_pct[missing_pct > 0].map('{:.4f}%'.format))

# =============================================================================
# STEP 3: DATA CLEANING
# =============================================================================

# Impute 'ap_lo' (diastolic BP) with median — robust against extreme outliers
if 'ap_lo' in df.columns:
    df['ap_lo'] = df['ap_lo'].fillna(df['ap_lo'].median())

# Drop remaining rows with minimal missing values (<0.1%)
df.dropna(inplace=True)

# Remove biologically impossible values (data entry errors)
df = df[(df['ap_hi'] >= 50)  & (df['ap_hi'] <= 250)]   # Systolic BP
df = df[(df['ap_lo'] >= 30)  & (df['ap_lo'] <= 150)]   # Diastolic BP
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]  # Weight in kg
df = df[(df['height'] >= 100)& (df['height'] <= 250)]  # Height in cm
df = df[(df['age'] >= 20)    & (df['age'] <= 120)]     # Age boundaries

print(f"\nData cleaned. Valid rows remaining: {len(df)}")

# =============================================================================
# STEP 4: FEATURE ENCODING
# =============================================================================

# Gender: Binary encoding — prevents model treating labels as numeric magnitude
if 'gender' in df.columns:
    df['gender'] = df['gender'].apply(lambda x: 0 if x == 1 else 1)
    print("Gender encoded: binary (0/1)")

# Cholesterol & Glucose: Kept as ordinal (1=Normal, 2=Above Normal, 3=Well Above Normal)
# The rank order carries meaningful information for cardiovascular prediction

# =============================================================================
# STEP 5: TRAIN / TEST SPLIT (75:25)
# =============================================================================

cols_to_drop = [c for c in ['id', 'cardio', 'age_years'] if c in df.columns]
X = df.drop(cols_to_drop, axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# =============================================================================
# STEP 6: MODEL TRAINING — LOGISTIC REGRESSION vs RANDOM FOREST
# =============================================================================

# Model 1: Logistic Regression — efficient binary classifier, highly interpretable
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Model 2: Random Forest — handles non-linear relationships, reduces overfitting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nLogistic Regression Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_acc:.4f}")

# Select best performing model
if lr_acc >= rf_acc:
    best_model_name = "Logistic Regression"
    best_pred = lr_pred
else:
    best_model_name = "Random Forest"
    best_pred = rf_pred

print(f"\nBest model selected: {best_model_name}")

# =============================================================================
# STEP 7: EVALUATION METRICS
# =============================================================================

f1   = f1_score(y_test, best_pred)
prec = precision_score(y_test, best_pred)
rec  = recall_score(y_test, best_pred)

print(f"\n{'='*40}")
print(f"  {best_model_name} — Performance Report")
print(f"{'='*40}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"{'='*40}")

# NOTE: In a healthcare context, Recall is the most critical metric.
# A low Recall means the model misses patients who actually have CVD (False Negatives),
# which can lead to severe health outcomes. A False Positive (low Precision) only
# results in precautionary checks — a much safer error to make.
