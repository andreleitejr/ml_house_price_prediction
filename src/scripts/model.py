from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(train_X, train_y):
    """Train a machine learning models."""

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)

    return model

def get_prediction(model, X_valid):
    """Use the trained model to make predictions on the given dataset."""

    return model.predict(X_valid)

def evaluate_model(model, X_valid, y_valid):
    """Evaluate the trained models using Mean Absolute Error."""

    predictions = get_prediction(model, X_valid)

    return mean_absolute_error(y_valid, predictions)

def save_model(model, model_path: str):
    """Save the trained models to a file."""

    joblib.dump(model, model_path)

def load_model(model_path: str):
    """Load a pre-trained models."""

    return joblib.load(model_path)
