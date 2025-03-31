import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Load the train dataset from a CSV file."""

    return pd.read_csv(file_path, index_col='Id')


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing numerical columns by filling with the median."""

    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and validation sets after preprocessing."""

    data = data.dropna(subset=['SalePrice'])

    y = data['SalePrice']
    X = data.drop(columns=['SalePrice'])

    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X = X.drop(columns=cols_with_missing)

    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


def reduce_data(X_train, X_valid):
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
    return X_train.drop(cols_with_missing, axis=1), X_valid.drop(cols_with_missing, axis=1)


def impute_data(X_train, X_valid):
    imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    imputed_X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
    return imputed_X_train, imputed_X_valid


def impute_extension_data(X_train, X_valid):
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    return impute_data(X_train_plus, X_valid_plus)


def remove_categorical_data(X_train, X_valid):
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    return drop_X_train, drop_X_valid


def ordinal_encode_data(X_train, X_valid):
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    ordinal_encoder = OrdinalEncoder()
    label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
    label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

    return label_X_train, label_X_valid


def get_target(data: pd.DataFrame) -> pd.Series:
    """Extract the target variable (SalePrice) from the dataset."""

    return data.SalePrice


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select the relevant features for model training."""

    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    return data[features]


def get_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retrieve both the feature set and target variable from the dataset."""

    return get_features(data), get_target(data)


def export_data(data: dict, filepath: str) -> None:
    """Save a dictionary as a CSV file."""
    pd.DataFrame(data).to_csv(filepath, index=False)
