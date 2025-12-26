# 🌍 World Happiness Score Prediction  
### End-to-End Machine Learning Project with CI Pipeline & Flask Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview
This project predicts the **World Happiness Score** of a country using socio-economic indicators and **machine learning regression models**.  
It follows **ML engineering best practices**, including a **modular pipeline**, **automated training (CI)**, and a **Flask-based web application** for real-time predictions.

The project emphasizes **reproducibility, automation, and clean architecture**, making it suitable for real-world ML systems.

---

## 🎯 Problem Statement
Happiness is a multidimensional concept that cannot be explained by GDP alone.  
This project aims to:
- Predict happiness scores using multiple socio-economic factors
- Identify the most influential features
- Provide a deployable ML-powered web interface

---

## 📊 Dataset
**Source:** World Happiness Report  

### Target Variable
- `Happiness Score`

### Input Features
- Year  
- GDP per Capita  
- Social Support  
- Healthy Life Expectancy  
- Freedom to Make Life Choices  
- Generosity  
- Perception of Corruption  

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights:
- **GDP per Capita** and **Social Support** show strong positive correlation with happiness
- **Life Expectancy** and **Freedom** have moderate influence
- **Generosity** and **Corruption** show weak correlation
- Correlation heatmaps and statistical summaries were used

---

## 🛠 Data Preprocessing
- Missing value handling using statistical imputation
- Feature selection based on domain knowledge
- Feature scaling using `StandardScaler`
- Train-test split (80% / 20%)

---

## 🤖 Machine Learning Models
Candidate regression models:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- AdaBoost Regressor

### 🔧 Hyperparameter Tuning
- `GridSearchCV` / `RandomizedSearchCV`
- Optimized parameters:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`

### 📈 Evaluation Metrics
- **R² Score (Train/Test)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

✅ **Best Model:** Random Forest Regressor  
- Highest test R² score  
- Low error metrics  
- Strong generalization

---

## 🧠 Feature Importance
Tree-based models show:
1. GDP per Capita
2. Social Support
3. Healthy Life Expectancy
4. Freedom
5. Generosity
6. Corruption

---

## 🌐 Web Application Deployment (Flask)
The trained model is deployed as a **Flask web application** that allows users to input socio-economic values and receive a predicted happiness score in real time.

### 🔹 Features
- User-friendly web interface
- Real-time predictions
- Input validation based on historical ranges
- Backend powered by trained ML model

### 🔹 Technologies
- Flask
- HTML, CSS
- Python
- Scikit-learn

### ▶️ Run Locally
```bash
python app.py
