from sklearn.metrics import mean_absolute_error
import joblib
from xgboost import XGBRegressor


def load_model(model_path: str):
    """Load a pre-trained model."""
    return joblib.load(model_path)


def train_model(X_train, X_valid, y_train, y_valid) -> XGBRegressor:
    """Train an XGBoost model with early stopping."""
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def evaluate_model(model, X_valid, y_valid) -> float:
    """Evaluate the trained model using Mean Absolute Error."""
    return mean_absolute_error(y_valid, model.predict(X_valid))


def save_model(model, model_path: str) -> None:
    """Save the trained model to a file."""
    joblib.dump(model, model_path)
