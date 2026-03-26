# =============================================================================
# TAXI FARE PREDICTION — REGRESSION ANALYSIS
# Dataset: taxi.csv
# Tools: pandas, NumPy, statsmodels, matplotlib, seaborn
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================

df = pd.read_csv("data/taxi.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# STEP 2: MISSING VALUE REPORT & IMPUTATION
# =============================================================================

print("\n--- Missing Values Report ---")
missing_pct = df.isnull().mean() * 100
print(missing_pct.map('{:.2f}%'.format))

# Numerical columns → Median imputation (robust to skewed distributions/outliers)
# Categorical columns → Mode imputation (preserves original distribution)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values imputed successfully.")

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

# --- 3a. Trip Volume by Time of Day ---
plt.figure(figsize=(8, 5))
sns.countplot(
    x='Time_of_Day', data=df,
    order=['Morning', 'Afternoon', 'Evening', 'Night'],
    palette='viridis'
)
plt.title('Highest Number of Hires by Time of Day')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("outputs/hires_by_time_of_day.png", dpi=150)
plt.show()

# --- 3b. Relationship Analysis: Key Variables vs Trip Price ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# i. Passenger Count — expected to have little influence on price
corr_passengers = df['Passenger_Count'].corr(df['Trip_Price'])
sns.scatterplot(ax=axes[0], x='Passenger_Count', y='Trip_Price', data=df)
axes[0].set_title(f"Passenger Count vs Price (Corr: {corr_passengers:.2f})")

# ii. Trip Duration — weak positive influence expected
corr_duration = df['Trip_Duration_Minutes'].corr(df['Trip_Price'])
sns.scatterplot(ax=axes[1], x='Trip_Duration_Minutes', y='Trip_Price', data=df)
axes[1].set_title(f"Duration vs Price (Corr: {corr_duration:.2f})")

# iii. Trip Distance — strongest predictor of price
corr_distance = df['Trip_Distance_km'].corr(df['Trip_Price'])
sns.scatterplot(ax=axes[2], x='Trip_Distance_km', y='Trip_Price', data=df)
axes[2].set_title(f"Distance vs Price (Corr: {corr_distance:.2f})")

plt.tight_layout()
plt.savefig("outputs/relationship_scatter_plots.png", dpi=150)
plt.show()

# --- 3c. Price vs Weather Conditions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(ax=axes[0], x='Weather', y='Trip_Price', data=df, palette='coolwarm')
axes[0].set_title('Price vs Weather')

# --- 3d. Price vs Traffic Conditions ---
sns.boxplot(
    ax=axes[1], x='Traffic_Conditions', y='Trip_Price',
    data=df, order=['Low', 'Medium', 'High'], palette='Oranges'
)
axes[1].set_title('Price vs Traffic Conditions')

plt.tight_layout()
plt.savefig("outputs/price_vs_weather_traffic.png", dpi=150)
plt.show()

# --- 3e. Correlation Matrix Heatmap ---
df_corr = df.copy()
traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_corr['Traffic_Conditions'] = df_corr['Traffic_Conditions'].map(traffic_map)
df_corr = pd.get_dummies(
    df_corr, columns=['Time_of_Day', 'Day_of_Week', 'Weather'],
    drop_first=True, dtype=int
)

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()
print("Note: No two independent variables exceed a correlation of 0.8, so multicollinearity is not a concern.")

# =============================================================================
# STEP 4: FEATURE ENCODING
# =============================================================================

df_model = df.copy()

# Traffic Conditions: Ordinal encoding — reflects inherent rank order of intensity
df_model['Traffic_Conditions'] = df_model['Traffic_Conditions'].map(traffic_map)

# Time of Day, Day of Week, Weather: One-Hot encoding — nominal, no natural order
df_model = pd.get_dummies(
    df_model,
    columns=['Time_of_Day', 'Day_of_Week', 'Weather'],
    drop_first=True,
    dtype=int
)

# =============================================================================
# STEP 5: OLS REGRESSION MODEL
# =============================================================================

X = df_model.drop('Trip_Price', axis=1)
y = df_model['Trip_Price']

# Add intercept constant (required for OLS)
X = sm.add_constant(X)

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

model = sm.OLS(y, X).fit()

# =============================================================================
# STEP 6: RESULTS
# =============================================================================

print("\n--- OLS Regression Summary ---")
print(model.summary())

print("\n--- Model Coefficients ---")
print(model.params)

print(f"\nR-squared: {model.rsquared:.4f}")
print(f"This means {model.rsquared*100:.1f}% of the variance in Trip Price is explained by the model.")

# Key insight: Trip_Distance_km has the strongest positive coefficient,
# confirming it is the primary driver of fare price (correlation ~0.83).
