from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import OneHotEncoder


def train_model(X_train, y_train, preprocessor):
    """Train a machine learning models."""

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

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


def pipeline_model(categorical_cols, numerical_cols):

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
