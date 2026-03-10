import numpy as np
from data_loader import data_load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(path):
    # Loading data from the data_loader
    df = data_load(path)

    # Setting all the features as X-axis and Diabetes as Y-axis/Target
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # Splitting the data into train and test sets
    # 25% of the dataset used for testing, and 75% for training
    # Random state = 104, ensures training and testing sets remain consistent across different executions
    # Randomises dataset before division between test and training sets
    # stratify = y, ensures both sets have the same ratio of people with/without diabetes
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=104,shuffle=True,stratify=y)

    scaler = StandardScaler()

    # Scaling BMI in both training and testing sets
    X_train["BMI"] = scaler.fit_transform(X_train[["BMI"]])
    X_test["BMI"] = scaler.transform(X_test[["BMI"]])

    return X, y, X_train, X_test, y_train, y_test