import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils.logger import logger


def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from a CSV file using 'Id' as the index."""
    logger.info(f'Loading data from {file_path}...')

    data = pd.read_csv(file_path, index_col='Id')

    logger.info(f'Data loaded successfully. Shape: {data.shape}')
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering transformations to enhance the dataset with additional relevant features."""
    logger.info('Applying feature engineering...')

    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    logger.info('Feature engineering applied successfully.')
    return data


def columns_to_drop() -> list:
    """Returns a list of columns to remove."""
    return ['SalePrice', 'PoolQC', 'Utilities', 'Alley']


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into 80% training and 20% validation while separating the target variable."""
    logger.info('Splitting data into training and validation sets...')

    X = data.drop(columns=columns_to_drop()).copy()
    y = data['SalePrice'].copy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    logger.info(f'Data split completed. Training size: {X_train.shape}, Validation size: {X_valid.shape}')
    return X_train, X_valid, y_train, y_valid


def preprocess_data(X_train: pd.DataFrame, X_valid: pd.DataFrame, preprocessor_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies numerical imputation, categorical encoding, and saves the preprocessing pipeline."""
    logger.info('Preprocessing data...')

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f'Data preprocessing completed. Preprocessor saved to {preprocessor_path}.')
    return X_train, X_valid


def export_data(data: dict, filepath: str) -> None:
    """Exports a dictionary to a CSV file without an index."""
    logger.info(f'Exporting data to {filepath}...')

    pd.DataFrame(data).to_csv(filepath, index=False)

    logger.info('Data export completed.')
