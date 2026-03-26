# 📊 Data Analysis Portfolio — Healthcare & Transport

Two end-to-end data science projects built using Python, covering classification and regression on real-world datasets.

---

## 🫀 Project 1: Cardiovascular Disease Risk Prediction

**File:** `cardiovascular_risk_analysis.py`  
**Dataset:** `data/cardio.csv`

### Overview
Built a machine learning classification model to predict whether a patient is at risk of cardiovascular disease, based on health indicators such as blood pressure, cholesterol, glucose levels, weight, and lifestyle factors.

### Key Steps
- **Outlier Detection** — used boxplots to identify impossible values (e.g. age of 510 years, weight of 6,662 kg, negative blood pressure)
- **Data Cleaning** — removed biologically invalid records, imputed missing diastolic BP values using median
- **Feature Encoding** — binary encoding for gender, ordinal encoding retained for cholesterol and glucose
- **Train/Test Split** — 75:25 ratio using `train_test_split`
- **Model Comparison** — Logistic Regression vs Random Forest Classifier
- **Evaluation** — F1 Score, Precision, and Recall

### Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~72.3% |
| Random Forest | ~69.4% |

**Best Model: Logistic Regression**

| Metric | Score |
|---|---|
| F1 Score | 0.6978 |
| Precision | 0.7513 |
| Recall | 0.6513 |

> **Why Recall matters most here:** In a healthcare context, missing a true positive (False Negative) means a patient with CVD goes undetected — a far more dangerous error than a False Positive, which only leads to precautionary checks.

### Libraries Used
`pandas` `numpy` `scikit-learn` `matplotlib` `seaborn`

---

## 🚕 Project 2: Taxi Fare Prediction

**File:** `taxi_fare_prediction.py`  
**Dataset:** `data/taxi.csv`

### Overview
Performed exploratory data analysis and built an OLS regression model to predict taxi trip prices based on distance, duration, passenger count, weather, and traffic conditions.

### Key Steps
- **Missing Value Imputation** — median for numerical columns, mode for categorical
- **EDA** — analysed trip volume by time of day, correlations between key variables and price, impact of weather and traffic
- **Feature Encoding** — ordinal encoding for traffic conditions, one-hot encoding for time of day, weather, and day of week
- **OLS Regression** — fitted using `statsmodels` with full coefficient analysis

### Key Findings

| Variable | Correlation with Price |
|---|---|
| Trip Distance (km) | **0.83 (strong)** |
| Trip Duration (min) | 0.21 (weak) |
| Passenger Count | -0.01 (none) |

- Afternoon sees the **highest trip volume** (~421 trips)
- Snow conditions are associated with **slightly higher prices** (surge pricing effect)
- The model explains **83.7% of variance** in trip price (R² = 0.837)

### OLS Regression Equation
```
Trip_Price = -53.79 + (1.72 × Distance) + (0.35 × Passengers) + (0.30 × Base_Fare)
           + (24.50 × Per_Km_Rate) + (53.36 × Per_Minute_Rate) + ...
```

### Libraries Used
`pandas` `numpy` `statsmodels` `matplotlib` `seaborn`

---

## 📁 Repo Structure

```
├── cardiovascular_risk_analysis.py   # CVD classification pipeline
├── taxi_fare_prediction.py           # Taxi fare regression pipeline
├── data/
│   ├── cardio.csv                    # Cardiovascular dataset
│   └── taxi.csv                      # Taxi trips dataset
├── outputs/                          # Saved charts and plots
└── README.md
```

## 🚀 How to Run

1. Clone the repo and install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

2. Place `cardio.csv` and `taxi.csv` inside the `data/` folder.

3. Run either script:
```bash
python cardiovascular_risk_analysis.py
python taxi_fare_prediction.py
```

---

## 🛠 Tech Stack
Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib · seaborn
