import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from evaluation import evaluation, heatmaps
from Models import logistic_regression_model, random_forest_model, gradient_boosting_model

X, y, X_train, X_test, y_train, y_test = preprocess_data("Dataset.csv")

# LOGISITC REGRESSION MODEL
y_pred_lr = logistic_regression_model(X_train, X_test, y_train)
print("Logistic Regression Model")
print(evaluation(y_test, y_pred_lr))
print(heatmaps(y_test, y_pred_lr))

# RANDOM FOREST MODEL
y_pred_rf = random_forest_model(X_train, X_test, y_train)
print("Random Forest Model")
print(evaluation(y_test, y_pred_rf))
print(heatmaps(y_test, y_pred_rf))

# GRADIENT BOOSTING MODEL
gb_model, y_pred_gb = gradient_boosting_model(X_train, X_test, y_train)
print("Gradient Boosting Model")
print(evaluation(y_test, y_pred_gb))
print(heatmaps(y_test, y_pred_gb))

# Helps visualize the importance of each feature towards diabetes result
feature_importance = gb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature importance:\n", feature_importance_df)

feature_importance_df = feature_importance_df.iloc[::-1]
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

    