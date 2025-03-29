from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model(x_train, y_train):
    """Train a machine learning models."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluate the trained models using Mean Squared Error."""
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def save_model(model, model_path: str):
    """Save the trained models to a file."""
    joblib.dump(model, model_path)

def load_model(model_path: str):
    """Load a pre-trained models."""
    return joblib.load(model_path)
