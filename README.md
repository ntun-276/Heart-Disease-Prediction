# Artificial Intelligence Project 🔬💡
Heart Disease Prediction (Dự đoán bệnh tim) ❤️‍🩹


## 🔍 Overview:
This repository contains a project focused on heart disease prediction. The data, derived from heart patients, includes various health metrics such as age, blood pressure, heart rate, and more. The primary objective is to create a predictive model that accurately identifies individuals at risk of heart disease. The emphasis is on achieving a high recall to ensure no potential heart disease case is missed.

## 🚀 Problem:
In this project, they delve into a dataset encapsulating various health metrics from heart patients, including age, blood pressure, heart rate, and more. The goal is to develop a predictive model capable of accurately identifying individuals with heart disease. Given the grave implications of missing a positive diagnosis, our primary emphasis is on ensuring that the model identifies all potential patients, making recall for the positive class a crucial metric.


## 🎯 Objectives:
<b> 1. Data Understanding: </b> Familiarize ourselves with the dataset and its features.

<b> 2. Exploratory Data Analysis (EDA): </b> Unveil patterns, trends, and relationships between different variables.
- Univariate Analysis
- Bivariate Analysis

<b> 3. Data Preprocessing: </b> Prepare the data for future machine learning tasks.
- Remove irrelevant features
- Address missing values
- Treat outliers
- Encode categorical variables
- Transform skewed features to achieve normal-like distributions

<b> 4. Model Building: </b> Develop and refine the prediction models.
- Establish pipelines for models that require scaling
- Implement and tune classification models including Decision Tree, Random Forest, KNN & SVM
- Emphasize achieving high recall for class 1, ensuring comprehensive identification of heart patients

<b> 5. Evaluate and Compare Model Performance: </b> Utilize precision, recall, and F1-score to gauge models' effectiveness.


## 📝 Key Features:
- `age` : Age of the patient in years.
- `sex` : Gender of the patient (0 = male, 1 = female).
- `cp` : Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic).
- `trestbps` : Resting blood pressure in mm Hg.
- `chol` : Serum cholesterol in mg/dl.
- `fbs` : Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false).
- `restecg` : Resting electrocardiographic results (0: Normal, 1: Having ST-T wave abnormality, 2: Showing probable or definite left ventricular hypertrophy).
- `thalach` : Maximum heart rate achieved during a stress test.
- `exang` : Exercise-induced angina (1 = yes, 0 = no).
- `oldpeak` : ST depression induced by exercise relative to rest.
- `slope` : Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping).
- `ca` : Number of major vessels (0-4) colored by fluoroscopy.
- `thal` : Thalium stress test result (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described).
- `target` : Heart disease status (0 = no disease, 1 = presence of disease).

Data Source: https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction


## 📈 Evaluations:
|       |  precision_0  |  precision_1  |  recall_0  |  recall_1  |  f1_0  |  f1_1  |  macro_avg_precision  |  macro_avg_recall  |  macro_avg_f1  |  Accuracy  |
|-------|---------------|---------------|------------|------------|--------|--------|-----------------------|--------------------|----------------|------------|
|  DT   |     0.80      |     0.78      |    0.71    |    0.85    |  0.75  |  0.81  |         0.79          |        0.78        |      0.78      |    0.79    |
|  RF   |     0.85      |     0.83      |    0.79    |    0.88    |  0.81  |  0.85  |         0.84          |        0.83        |      0.83      |    0.84    |
|  KNN  |     0.82      |     0.85      |    0.82    |    0.85    |  0.82  |  0.85  |         0.83          |        0.83        |      0.83      |    0.84    |
|  SVM  |     0.94      |     0.73      |    0.57    |    0.97    |  0.71  |  0.83  |         0.83          |        0.77        |      0.77      |    0.79    |


## 📑 Project Organization:

```text
├── LICENSE
├── README.md
├── data/
│   └── heart.csv
│
├── models
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── reports
│   ├── best_model.json
│   ├── model_results.csv
│   └── figures
│
├── src/
│    ├── __init__.py
│    │
│    ├── data_processing/
│    │    ├── load_data.py
│    │    └── preprocess.py
│    │
│    ├── feature_engineering/
│    │    └── build_features.py
│    │
│    ├── models/
│    │    ├── train.py
│    │    └── predict.py
│    │
│    ├── evaluation/
│    │    └── evaluate.py
│    │
│    └── utils/
│         └── helpers.py
│
├── main.py
└── requirements.txt
```

## Quick Start

```powershell
python -m pip install -r requirements.txt
python main.py
```

## Output Artifacts

- Trained models are saved to `models/`.
- Evaluation results are saved to `reports/model_results.csv`.
- Best model metadata is saved to `reports/best_model.json`.

## 📫 Contact me:
📧 <a href="#"> ntruynhi276@gmail.com </a>
