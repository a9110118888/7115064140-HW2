# 🚗 Dynamic Machine Learning Explorer (HW2)

An interactive Machine Learning exploration web application built with **Streamlit**, **Scikit-Learn**, and **Statsmodels**.

## 🌟 Features

- **Dynamic Dataset Input**: Load custom datasets via CSV URL (default: Car Price dataset, with auto-detection for 50 Startups and other tabular datasets).
- **Automated Target Selection**: Automatically identifies and allows users to choose the target variable $Y$.
- **Problem 1: Feature Selection & Model Evaluation**:
  - Recursive Feature Elimination (RFE) to identify optimal feature subset.
  - Interactive evaluation metric charts showing $R^2$ and RMSE trends across varying numbers of selected features ($k$).
- **Problem 2: Prediction Intervals & Live Predictor**:
  - OLS regression coefficients and p-values using statsmodels.
  - 95% Confidence Interval visualization (Actual vs. Predicted values).
  - Real-time interactive predictor with sliders and selectboxes for selected features.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/a9110118888/7115064140-HW2.git
cd 7115064140-HW2
```

### 2. Set up virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
