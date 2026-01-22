# AI Model Reliability Audit — Credit Risk

This project audits the reliability of a machine learning model for credit risk prediction.  
Rather than focusing only on accuracy, the analysis emphasizes **calibration, probabilistic reliability, and subgroup-level failure modes**, which are critical in high-stakes domains like lending.

---

## Problem Statement
Traditional ML evaluation metrics (e.g., accuracy, ROC-AUC) can hide systematic risks.  
In credit risk modeling, miscalibrated probabilities or subgroup-specific errors can lead to unsafe or unfair decisions.

This project answers:
- Is the model **well-calibrated**?
- Does reliability differ across **age** and **income** groups?
- What **failure modes** are not visible in global metrics?

---

## Dataset
- Public credit risk dataset
- Binary target: loan default (`loan_status`)
- Mixed numerical and categorical features
- Realistic class imbalance

---

## Methodology
1. **Data Quality & EDA**
   - Missing value analysis
   - Target distribution
   - Feature slicing by age and income
## Notebook
The full analysis is available as a Kaggle notebook:

👉 [Kaggle Notebook – Credit Risk Reliability Audit](PASTE_KAGGLE_LINK_HERE)

2. **Baseline Model**
   - Logistic Regression
   - Full preprocessing pipeline:
     - Numerical & categorical imputation
     - One-hot encoding
   - Train / test split with stratification

3. **Reliability Evaluation**
   - ROC-AUC
   - Brier Score
   - Calibration curve

4. **Subgroup Analysis**
   - Calibration by age group
   - Calibration by income group
   - Identification of subgroup-specific reliability gaps

5. **Failure Mode Analysis**
   - False positives vs false negatives
   - Business risk interpretation of errors

---

## Key Results
- ROC-AUC ≈ **0.85**
- Brier Score ≈ **0.11**
- Calibration deviations observed in younger and older age groups
- Reliability issues not visible from aggregate performance alone

---

## Project Structure
```text
ai-model-reliability-audit/
├── data/
├── notebooks/
├── src/
├── reports/
├── figures/
├── results/
├── requirements.txt
└── README.md
