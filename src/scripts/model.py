from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(train_X, train_y):
    """Train a machine learning models."""

    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)

    return model

def get_prediction(model, val_X):
    """Use the trained model to make predictions on the given dataset."""

    return model.predict(val_X)

def evaluate_model(model, val_X, val_y):
    """Evaluate the trained models using Mean Absolute Error."""

    predictions = get_prediction(model, val_X)

    return mean_absolute_error(val_y, predictions)

def save_model(model, model_path: str):
    """Save the trained models to a file."""

    joblib.dump(model, model_path)

def load_model(model_path: str):
    """Load a pre-trained models."""

    return joblib.load(model_path)
