import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost.testing.data import joblib


def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from a CSV file using 'Id' as the index."""
    return pd.read_csv(file_path, index_col="Id")


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering transformations to enhance the dataset with additional relevant features."""
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]

    return data


def columns_to_drop() -> list:
    """Returns a list of columns to remove."""
    target = 'SalePrice'
    return [target, 'PoolQC', 'Utilities']


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into 80% training and 20% validation while separating the target variable."""
    X = data.drop(columns=columns_to_drop()).copy()
    y = data["SalePrice"].copy()

    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)


def preprocess_data(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies numerical imputation, categorical encoding, and saves the preprocessing pipeline."""
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    joblib.dump(preprocessor, "src/models/preprocessor.pkl")

    return X_train, X_valid


def export_data(data: dict, filepath: str) -> None:
    """Exports a dictionary to a CSV file without an index."""
    pd.DataFrame(data).to_csv(filepath, index=False)
