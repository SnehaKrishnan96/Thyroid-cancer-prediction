# Predicting Whether a Thyroid Nodule is Cancerous using Machine Learning Algorithms

## Overview

Early diagnosis of thyroid nodules as cancerous or benign is crucial for effective treatment. Traditionally, this involves a series of medical tests. This project leverages machine learning algorithms to predict the likelihood of a thyroid nodule being cancerous based on patient data such as nodule size, diabetes status, smoking habits, age, and more, with a primary focus on maximizing **recall** to reduce the risk of false negatives—crucial in medical diagnostics. The workflow includes extensive data preprocessing, statistical analysis, model training, and depiction of prediction via an interactive **Gradio** interface.

---

## Data Preprocessing & Statistical Analysis

- Performed exploratory data analysis and summary statistics to understand feature distributions.
- One-hot encoded categorical variables.
- Balanced the dataset using SMOTE after comparing it with Random Oversampling (ROS) and Random Undersampling (RUS).
- Visualized data distributions and relationships.
- Conducted normality checks using:
  - Shapiro-Wilk Test
  - Kolmogorov-Smirnov Test
  - Anderson-Darling Test
- Normalized skewed features using **log** and **Box-Cox** transformations, then revalidated using Shapiro-Wilk.
- Used the **Mann-Whitney U Test** to compare independent groups due to non-normal distributions.
- Assessed multicollinearity using **Variance Inflation Factor (VIF)**.
- Selected top features using:
  - Spearman Correlation
  - Chi-Squared Test

---

## Modeling & Evaluation

- Split dataset into training and testing sets.
- Trained the following models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - AdaBoost
  - Stochastic Gradient Boosting
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **XGBoost** achieved the best performance with strong recall.

---

## User Interface

- Developed an interactive UI using **Gradio**.
- Allows real-time prediction, feature input, and visualization of feature impacts.
- Enhances model interpretability and accessibility for clinical use.

---

## Conclusion

This project highlights the effectiveness of machine learning—particularly **XGBoost**—in predicting thyroid cancer with high recall. By minimizing false negatives, the system aims to aid early diagnosis and improve clinical outcomes. The Gradio interface ensures the tool is user-friendly and interpretable.

---

## Future Work

- Train on larger and more diverse datasets to improve generalization across demographics.
- Fine-tune the trade-off between recall and precision to reduce false positives.
- Enhance model transparency and robustness for deployment in real-world medical settings.

---

## Technologies & Skills Used

- **Python**, **Pandas**, **Scikit-learn**, **XGBoost**, **LightGBM**, **Matplotlib**, **Seaborn**
- **SMOTE**, **Chi-Squared Test**, **Statistical Tests for Normality**
- **Gradio**, **Feature Selection**, **Model Evaluation Metrics**

---

