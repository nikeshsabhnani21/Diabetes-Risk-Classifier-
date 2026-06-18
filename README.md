# Diabetes Risk Classifier 

## Project Description 

A supervised machine learning approach is used to predict the likelihood of future diabetes mellitus occurrence using standardized medical record data. Specifically, the focus of this project will be to determine whether the machine learning algorithms produce medically interpretable relationships between the different variables and how accurately classification methods perform given real-world constraints (i.e., instances where there is not enough representation of diabetics). To evaluate the various models during the development and evaluation phase, the training/testing data was pre-processed, separated into features/results, trained, and compared across models. In addition to classifying diabetic patients, developing better metrics (other than accuracy) for all classifiers who were evaluated was a top priority. Due to the low occurrence of diabetics in this data set (a class imbalance problem), it emphasised the importance of precision/recall trade-offs for the classifiers evaluated.

## Project Key Features

- Used train-test split for strong evaluation
- Implementation of ML models (Logistic Regression, Random Forest, Gradient Boosting)
- Handling class imbalance using class weights
- Evaluation using classification (precision, recall, F1-score) report and heatmaps

## How does it work?

Health indicator data was used to classify individuals as diabetic or non-diabetic. Only a small proportion of the dataset was actually diabetic. Therefore, the models were evaluated on how well they identified true diabetic cases, rather than on overall accuracy using the three different model classification methods. Logistic Regression with class weighting performed substantially better than the other two models at identifying true positives in the minority class, even though it wasn't the most accurate model overall. In addition, this project highlights the importance of model comparison, interpretation, and metric choice when applying machine learning to ethical decisions in healthcare.

## Installation
1. Clone the repository
2. Make sure you have Python 3.10 or greater
3. Install scikit-learn, pandas and matplotlib
4. Run: python main.py









