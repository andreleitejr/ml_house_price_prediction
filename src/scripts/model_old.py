from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def train_model(X_train, y_train):
    """Train a machine learning models."""

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    return model

def xgboost_train_model(X_train, X_valid, y_train, y_valid):
    """Train a machine learning models."""
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=10)

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return model


def preprocessor_train_model(X_train, y_train, categorical_cols, numerical_cols):
    """Train a machine learning model using preprocessor (ColumnTransformer)."""

    cols = categorical_cols + numerical_cols
    X_train = X_train[cols].copy()

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    preprocessor = get_preprocessor(categorical_cols, numerical_cols)
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    clf.fit(X_train, y_train)

    return clf


# TODO: Remove this
def get_prediction(model, X_valid):
    """Use the trained model to make predictions on the given dataset."""

    return model.predict(X_valid)


def evaluate_model(model, X_valid, y_valid):
    """Evaluate the trained models using Mean Absolute Error."""

    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)


def cross_validate_model(X, y, n_estimators):
    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0)),
    ])

    scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()


def get_preprocessor(categorical_cols, numerical_cols):
    numerical_transformer = SimpleImputer(strategy='constant')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor


def save_model(model, model_path: str):
    """Save the trained models to a file."""

    joblib.dump(model, model_path)


def load_model(model_path: str):
    """Load a pre-trained models."""

    return joblib.load(model_path)
