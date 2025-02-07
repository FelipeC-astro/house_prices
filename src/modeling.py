from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_evaluate(X_train, X_valid, y_train, y_valid):
    """
    Trains and evaluates different regression models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }
    
    best_model = None
    best_rmse = float("inf")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        print(f"RMSE - {name}: {rmse:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
    
    return best_model