# 🫀 Heart Stroke Prediction using Machine Learning

## 📌 Project Overview
This project predicts the likelihood of a person having a **stroke** based on health and demographic data using machine learning algorithms. The goal is to support early diagnosis and potentially reduce mortality rates through proactive medical care.

It also includes a feature to recommend **nearby hospitals** for recovery assistance based on the user’s city and locality.

## 🎯 Problem Statement
Cardiovascular diseases, particularly strokes, are among the leading causes of death globally. Early detection using predictive modeling can help in timely intervention.

## 🧰 Tools & Technologies
- **Python**
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn** – ML algorithms (Logistic Regression, Decision Tree,Random Forest etc.)
- **Matplotlib, Seaborn** – Data visualization
- **Flask** – Web framework
- **HTML, CSS** – Frontend UI
- **CSV File** – Used for hospital location data, health data for prediction

## 🧪 Dataset1
- Source: kaggle dataset (health data)
- Features: age, hypertension, heart_disease, avg_glucose_level, BMI, smoking_status, etc.
- Label: `stroke` (0: No, 1: Yes)

## 🧪 Dataset2
- Source: kaggle dataset (hospital location data)
- Features: hospital name , address,pincode.

## 🧠 ML Workflow
1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Model Training & Evaluation
4. Accuracy Comparison of Models
5. Model Deployment using Flask

## 📈 Models Used
- Random Forest

## 🖥️ User Interface
- Home page with input form for user details
- Prediction result display (Stroke risk: Yes/No)
- Recommended nearby hospitals based on user city/locality (Tagify auto-suggestions)
