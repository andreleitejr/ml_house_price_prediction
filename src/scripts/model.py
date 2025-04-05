import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer, mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score
from src.utils.logger import logger
from xgboost import XGBRegressor


def load_model(model_path: str):
    """Loads a trained model from a file using joblib."""
    model = joblib.load(model_path)

    return model


def train_model(X_train, X_valid, y_train, y_valid) -> XGBRegressor:
    """Trains an XGBoost regressor with early stopping and predefined hyperparameters."""
    params = {
        'n_estimators': 800,
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.55,
        'colsample_bytree': 0.4,
    }

    model = XGBRegressor(**params, n_jobs=4)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return model



def evaluate_model(model, X_valid, y_valid) -> None:
    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)
    rmsle = np.sqrt(mean_squared_log_error(y_valid, predictions))

    logger.result(
        f'✅ Model evaluation completed.\n'
        f'🎯 MAE: {mae:.2f}\n'
        f'📈 RMSLE: {rmsle:.4f}\n'
        f'🔍 Accuracy (R² Score): {r2:.4f} ({r2:.0%})\n'
    )


def cross_validate_model(model, X_valid, y_valid) -> None:
    """Performs cross-validation and calculates the Mean Absolute Error (MAE) and Accuracy (R² Score)."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    mae_scores = -cross_val_score(model, X_valid, y_valid, cv=cv, scoring='neg_mean_absolute_error')
    mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)

    r2_scores = cross_val_score(model, X_valid, y_valid, cv=cv, scoring=make_scorer(r2_score))
    mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)

    logger.result(
        f'✅ Cross-validation completed.\n'
        f'📊 MAE (Cross-Validation): {mean_mae:.2f} (± {std_mae:.2f})\n'
        f'🔍 Accuracy (R² Score - Cross Validation): {mean_r2:.4f} ({mean_r2:.0%}) (± {std_r2:.2f})\n'
    )


def save_model(model, model_path: str) -> None:
    """Saves a trained model to a file using joblib."""
    joblib.dump(model, model_path)
