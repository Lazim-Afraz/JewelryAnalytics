# Jewelry Analytics System

## Overview

This project is an end-to-end analytics system designed to analyze and predict jewelry sales across multiple branches. It combines machine learning with business-oriented insights to support decision-making in inventory and sales optimization.

---

## Features

* **Sales Prediction**

  * Predicts `SALE_COUNT` based on branch and product attributes
  * Uses Random Forest Regressor

* **Branch Performance Analysis**

  * Identifies top-performing branches
  * Compares regional performance

* **Clustering**

  * Groups similar products/branches
  * Helps identify patterns in sales behavior

* **Analytics Dashboard (Streamlit)**

  * Interactive interface
  * Real-time predictions and insights

---

## Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Frontend:** Streamlit
* **ML Model:** Random Forest Regressor

---

## Project Structure

```
JewelryAnalytics/
│
├── analytics/              # ML models and analytics logic
├── services/               # Service layer (business logic)
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Clone repository

```
git clone https://github.com/Lazim-Afraz/JewelryAnalytics.git
cd JewelryAnalytics
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run application

```
streamlit run app.py
```

---

## Model Performance

* R² Score: ~0.60 – 0.70
* MAE: Moderate
* RMSE: Controlled

---

## Key Insights

* Certain branches consistently outperform others
* Sales vary significantly based on product attributes
* Demand patterns are not uniform across regions

---

## Future Improvements

* Add time-based forecasting
* Improve model interpretability
* Integrate real-time data

---

## Author

Lazim Afraz
