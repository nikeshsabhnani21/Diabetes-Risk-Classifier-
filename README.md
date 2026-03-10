# Project Name 

Diabetes Risk Classifier

## Project Description 

This project employs a supervised algorithmic framework to ascertain possible future occurrence(s) of diabetes mellitus, using only standardized medical data points. The specific goal is to investigate whether or not models would produce medically understandable relationships among variables and how various classification techniques would be able to function while adhering to the real-time boundaries present (e.g., lack of class representation).
Pre-processing data, separating or splitting features and results, training the model, and conducting a comparative analysis across multiple classifiers were all part of the model development and evaluation process. Developing more precise model metrics than just accuracy for all the classifiers measured was highly prioritised. The low frequencies of diabetic patients in this specific data set emphasised the precision-recall trade-offs for these classifiers.

## Project Key Features

- Used train-test split for strong evaluation
- Implementation of ML models (Logistic Regression, Random Forest, Gradient Boosting)
- Handling class imbalance using class weights
- Evaluation suing classification (precision, recall, F1-score) report and heatmaps
- Input validation for deadlines and difficulty levels

## How does it work?

Health indicator data was used for Classifying the binary outcome for diabetes. The data showed a small proportion of the population classified as diabetic actually had the true definition of diabetes. As a result, the classification models were evaluated based on the performance with regard to identifying diabetic patients by looking at the proportion of true asymptomatic diabetic cases identified by each of the three classification models. It was shown that the performance of the Logistic Regression with a class weight was substantially better than the other two classification models with regard to identifying true positives among the minority class compared to the overall accuracy of each model.
This project also emphasized the importance of model comparison and interpretation, along with the evaluation methods with regard to the performance of the models for ethical decisions in the field of health care.

## Installation
1. Clone the repository
2. Make sure you have Python 3.10 or greater
3. Install scikit-learn, pandas and matplotlib
4. Run: python main.py









