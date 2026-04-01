from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import RegressorMixin
import numpy as np

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=5, max_features=None, min_samples_leaf=5, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate regression model and return metrics"""
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)*100

    print(f"\n{model_name} Performance:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2 score : {r2:.2f}%")

    return {
        "model_name" : model_name,
        "MAE" : mae,
        "RMSE" : rmse,
        "R2(%)" : r2
    }
