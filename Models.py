from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def logistic_regression_model(X_train, X_test, y_train):
    # Initiating the model
    # class_weight = "balanced", adjusts class imbalance
    # random_state = 16, ensures results are consistent across different executions
    # solver="lbfgs", Algorithm used for optimization, also is a default solver for Scikit-learn version.
    # Number of iterations for convergence 
    logreg = LogisticRegression(class_weight = "balanced", random_state=16, solver="lbfgs", max_iter=400)
    # Fitting the modle with the data
    logreg.fit(X_train, y_train)
    # Predictions for the test dataset using Logistic Regression 
    y_pred_lr = logreg.predict(X_test)

    return y_pred_lr


def random_forest_model(X_train, X_test, y_train):
    # n_estimators=100, Creates 100 trees
    # random_state=42, Ensures consistent outcomes
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fitting the modle with the data
    rf.fit(X_train, y_train)
    # Predictions for the test dataset using Random Forest
    y_pred_rf = rf.predict(X_test)

    return y_pred_rf


def gradient_boosting_model(X_train, X_test, y_train):
    # n_estimators=100, Creates 100 trees
    # learning_rate=0.1, Dafault and Learning rate shrinks the contribution of each tree.
    # random_state=42, Ensures consistent outcomes
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # Fitting the modle with the data
    gbm.fit(X_train, y_train)
    # Predictions for the test dataset using Gradient Boosting  
    y_pred_gb = gbm.predict(X_test)

    return gbm, y_pred_gb


