# Heart Disease Predictor 

This project applies machine learning models to predict whether a patient has heart disease based on clinical attributes like age, cholesterol, chest pain type, and more. The notebook includes the entire process of the project, from cleaning the data to training and evaluating multiple models with hyperparameter tuning. 

---

## Dataset
- **Source**: [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) 
- **Rows**: 918 observations
- **Target Variable**: 'HeartDisease' (1 = presence, 0 = absence)

**Included Attributes**
- Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, MaxHR, ExerciseAngina, OldPeak, ST_Slope, and HeartDisease.

---

## Tools

- Jupyter Notebook
- Libraries:
    - 'pandas', 'numpy', matplotlib', 'seaborn'
    - 'scikit-learn' for ML models and evaluation

---

## Models Trained

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

All models were trained with:
- Feature scaling ('StandardScaler')
- GridSearchCV for hyperparameter tuning
- Confusion matrix evaluation

---

## Workflow Summary

1. **Data Exploration**
   Used '.info()', '.describe()', and boxplots to understand the dataset.

2. **Preprocessing**
   - Handled outliers using the IQR method.
   - Factorized binary columns.
   - One hote encoded multi-class columns.

3. **Feature Scaling**
   Used 'StandardScaler' on all numeric features before training.

4. **Train/Test Split**
   80/20 split to preserve class balance.

5. **Hyperparameter Tuning**
   Used 'GridSearchCV' to optimize hyperparameters for each model.

6. **Model Evaluation**
   Compared accuracy, precision, recall, and F1 scores across all models.

---

## Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | ~0.86 | ~0.86   | ~0.86 | ~0.86   |
| KNN                | ~0.87  | ~0.87   | ~0.87 | ~0.87   |
| SVM                | ~0.86  | ~0.86   | ~0.86 | ~0.86   |
| Decision Tree      | ~0.82  | ~0.82   | ~0.82 | ~0.82   |
| Random Forest      | ~0.85  | ~0.85   | ~0.85 | ~0.85   |

> KNN showed the best overall performance, balancing speed and predictive accuracy. 
