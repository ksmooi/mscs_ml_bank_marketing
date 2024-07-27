# Bank Marketing Predictive Model

This project involves analyzing and predicting whether a client will subscribe to a term deposit based on various attributes collected from a bank marketing dataset. The dataset includes information such as age, job, marital status, education, balance, and previous marketing campaign outcomes. By applying machine learning techniques, the goal is to build a predictive model that can accurately forecast a client's decision.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Feature Selection](#4-feature-selection)
5. [Dimensionality Reduction](#5-dimensionality-reduction)
6. [Model Training and Evaluation](#6-model-training-and-evaluation)
7. [Conclusion](#7-conclusion)
8. [Future Work](#8-future-work)
9. [References](#9-references)

## 1. [Introduction](notebooks/bank_marketing.ipynb)

This project involves analyzing and predicting whether a client will subscribe to a term deposit based on various attributes collected from a bank marketing dataset. The dataset includes information such as age, job, marital status, education, balance, and previous marketing campaign outcomes. By applying machine learning techniques, the goal is to build a predictive model that can accurately forecast a client's decision.

**Objective:**
The primary objective of this project is to develop a predictive model that can determine if a client will subscribe to a term deposit. This involves:
- **Data Preprocessing:** Cleaning and preparing the dataset for analysis.
- **Feature Selection and Dimensionality Reduction:** Identifying the most relevant features and reducing the dimensionality to improve model performance.
- **Model Training:** Training various supervised learning models to predict the target variable.
- **Model Evaluation:** Evaluating the models based on their accuracy and other performance metrics to select the best model.

By achieving this objective, the project aims to provide insights into the key factors influencing a client's decision and enhance the bank's marketing strategies.

## 2. [Data Preprocessing](notebooks/bank_marketing.ipynb)

The data preprocessing involves the following steps:

1. **Handling Missing Values:** Identify and handle missing values in the dataset.
2. **Converting Categorical to Numerical:** Convert categorical variables to numerical values using one-hot encoding.
3. **Normalizing Numerical Features:** Normalize numerical features to ensure they are on a similar scale.
4. **Removing Outliers:** Identify and remove outliers from the dataset to improve model performance.
5. **Handling Imbalanced Data:** Use techniques like SMOTE to handle imbalanced data.
6. **Dropping Unnecessary Features:** Drop features that are not relevant for the analysis.

## 3. [Exploratory Data Analysis (EDA)](notebooks/bank_marketing.ipynb)

The EDA process involves examining the dataset to uncover patterns, relationships, and insights that can inform the modeling process. This includes:

- **Summary Statistics:** Generate summary statistics for numerical and categorical features.
- **Distribution Plots:** Plot distributions of features to understand their spread and central tendency.
- **Correlation Analysis:** Assess the correlation between numerical features to identify potential relationships.

## 4. [Feature Selection](notebooks/bank_marketing.ipynb)

Feature selection involves selecting a subset of relevant features for model training. The following methods are used:

- **Mutual Information:** Measure the mutual dependence between two variables to select relevant features.
- **Random Forest Feature Importance:** Use the feature importance scores from a trained Random Forest model to select features.

## 5. [Dimensionality Reduction](notebooks/bank_marketing.ipynb)

Dimensionality reduction techniques are used to reduce the number of features while retaining most of the information. The following methods are applied:

- **Principal Component Analysis (PCA):** Reduce the dimensionality by transforming the original features into a new set of orthogonal components.
- **Singular Value Decomposition (SVD):** Factorize the data matrix into three matrices to reduce dimensionality.
- **Non-negative Matrix Factorization (NMF):** Decompose the data matrix into two lower-dimensional matrices with non-negative elements.

## 6. [Model Training and Evaluation](notebooks/bank_marketing.ipynb)

Various supervised learning models are trained and evaluated to predict whether a client will subscribe to a term deposit. The models include:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (KNN)**

The models are evaluated based on their accuracy, precision, recall, F1-score, and ROC-AUC.

## 7. [Conclusion](notebooks/bank_marketing.ipynb)

**Best Performing Model:**
- The Random Forest model with SVD reduction achieved the highest test accuracy (0.78256), suggesting that SVD is a suitable dimensionality reduction technique for this dataset.

**Effective Techniques:**
- PCA and SVD are effective in preserving data variance and improving model performance.

**Feature Selection Alone Insufficient:**
- Relying solely on feature selection (common features method) may not provide sufficient information for accurate predictions.

## 8. [Future Work](notebooks/bank_marketing.ipynb)

1. **Hyperparameter Tuning:** Further tuning of hyperparameters for each model, particularly for dimensionality reduction methods, could potentially improve performance.
2. **Combination Methods:** Explore combining feature selection with dimensionality reduction to see if it provides a more robust feature set.
3. **Regularization Techniques:** Apply regularization techniques to mitigate overfitting observed in models like Decision Trees and Random Forests.
4. **Advanced Models:** Evaluate the performance of more advanced models, such as XGBoost or deep learning techniques, on the reduced datasets.
5. **Cross-Validation:** Implement cross-validation to ensure the robustness of model performance across different data splits.

## 9. [References](notebooks/bank_marketing.ipynb)

The references below provide access to the datasets and additional information on the techniques and models used in the analysis:

- **Bank Marketing Dataset:**
  - Kaggle: [https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)
  - UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- **Principal Component Analysis (PCA):**
  - Wikipedia: [https://en.wikipedia.org/wiki/Principal_component_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/decomposition.html#pca](https://scikit-learn.org/stable/modules/decomposition.html#pca)

- **Singular Value Decomposition (SVD):**
  - Wikipedia: [https://en.wikipedia.org/wiki/Singular_value_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

- **Non-negative Matrix Factorization (NMF):**
  - Wikipedia: [https://en.wikipedia.org/wiki/Non-negative_matrix_factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

- **Mutual Information:**
  - Wikipedia: [https://en.wikipedia.org/wiki/Mutual_information](https://en.wikipedia.org/wiki/Mutual_information)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)

- **Random Forest:**
  - Wikipedia: [https://en.wikipedia.org/wiki/Random_forest](https://en.wikipedia.org/wiki/Random_forest)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- **Support Vector Machine (SVM):**
  - Wikipedia: [https://en.wikipedia.org/wiki/Support_vector_machine](https://en.wikipedia.org/wiki/Support_vector_machine)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)

- **Gradient Boosting:**
  - Wikipedia: [https://en.wikipedia.org/wiki/Gradient_boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

- **K-Nearest Neighbors (KNN):**
  - Wikipedia: [https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  - Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
