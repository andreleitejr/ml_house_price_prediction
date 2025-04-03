import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


def train_model(X_train, X_valid, y_train, y_valid) -> XGBRegressor:
    """Trains an XGBoost regressor with early stopping and predefined hyperparameters."""
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
    }
    model = XGBRegressor(**params, n_jobs=4)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return model


def evaluate_model(model, X_valid, y_valid) -> None:
    """Calculates and prints the Mean Absolute Error (MAE) between predictions and actual values."""
    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)

    print(f"Mean Absolute Error: {mae:.2f}")


def cross_validate_model(model, X_valid, y_valid) -> None:
    """Calculates and prints the cross-validated Mean Absolute Error (MAE) of the model."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds

    scores = cross_val_score(model, X_valid, y_valid, cv=cv, scoring='neg_mean_absolute_error')
    mae_scores = -scores

    print(f"Mean Absolute Error (Cross Validation): {np.mean(mae_scores):.2f} (± {np.std(mae_scores):.2f})")


def save_model(models, model_path: str) -> None:
    """Saves a trained model to a file using joblib."""
    joblib.dump(models, model_path)


def load_model(model_path: str):
    """Loads a trained model from a file using joblib."""
    return joblib.load(model_path)
