# csit360project

# Obesity & Overweight Prevalence Predictor

## **Project Overview**

This project predicts obesity and overweight prevalence rates using demographic data from the CDC's Behavioral Risk Factor Surveillance
System. The pipeline includes data cleaning, exploratory analysis, and comparative machine learning modeling.

## **Data**

* **Source**: [https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)
* **Variables Used**:
* Target: Obese (%) and Overweight (%) prevalence
* Features: StateID, Age, Education, Sex, Income, Race/Ethnicity
* **Preprocessing**:
* Filtered to obesity/overweight-related questions (Q036, Q037)
* Handled missing values with "Unknown"
* Pivoted to wide format (one row per demographic group)

## **Modeling Approach**

* **Algorithms**:
* Linear Regression (baseline)
* Ridge Regression (regularized linear)
* Random Forest (non-linear relationships)
* XGBoost (gradient boosting)

## **Performance Metrics**

### **Obesity Prediction (Target: Obese)**

| Model             | MAE             | RMSE            | R²             | Important Features            |
| ----------------- | --------------- | --------------- | --------------- | ----------------------------- |
| Linear            | 3.225           | 4.361           | 0.677           |                               |
| Ridge             | 3.225           | 4.361           | 0.677           |                               |
| Random Forest     | 3.121           | 4.209           | 0.699           | Race_Asian (23.4% importance) |
| **XGBoost** | **3.089** | **4.164** | **0.705** | Age_18-24 (10.7% importance)  |

### **Overweight Prediction (Target: Overweight)**

| Model           | MAE             | RMSE            | R²             | Important Features           |
| --------------- | --------------- | --------------- | --------------- | ---------------------------- |
| Linear          | 2.324           | 3.431           | 0.472           |                              |
| **Ridge** | **2.324** | **3.431** | **0.472** |                              |
| Random Forest   | 2.346           | 3.631           | 0.408           | Age_18-24 (22.4% importance) |
| XGBoost         | 2.310           | 3.550           | 0.435           | Sex_Male (21.9% importance)  |

## **Key Findings**

1. **Best Models** :

* Obesity: XGBoost (MAE: 3.09%, R²: 0.705)
* Overweight: Ridge (MAE: 2.324%, R²: 0.472)

1. **Top Predictors** :

* Obesity: Asian ethnicity, young adults (18-24), college education
* Overweight: Male sex, low income, young adults
