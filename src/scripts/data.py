import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


target = 'SalePrice'

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a dataset from a CSV file using 'Id' as the index."""
    data = pd.read_csv(file_path, index_col='Id')

    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering transformations to enhance the dataset with additional relevant features."""
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['TotalBathrooms'] = data['FullBath'] + (data['HalfBath'] * 0.5) + data['BsmtFullBath'] + (data['BsmtHalfBath'] * 0.5)
    data['PorchArea'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    data['AboveGroundLivingArea'] = data['1stFlrSF'] + data['2ndFlrSF']
    data['BsmtFinishedRatio'] = data['BsmtFinSF1'] / data['TotalBsmtSF']
    data['BsmtFinishedRatio'] = data['BsmtFinishedRatio'].replace([np.inf, -np.inf], 0).fillna(0)
    data['LotToLivingRatio'] = data['LotArea'] / data['AboveGroundLivingArea']
    data['LotToLivingRatio'] = data['LotToLivingRatio'].replace([np.inf, -np.inf], 0).fillna(0)
    garage_qc_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
    data['GarageQualityScore'] = data['GarageQual'].map(garage_qc_mapping).fillna(0)
    data['GarageConditionScore'] = data['GarageCond'].map(garage_qc_mapping).fillna(0)
    data['GarageQualityConditionScore'] = data['GarageQualityScore'] * data['GarageConditionScore']
    fireplace_qu_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}
    data['FireplaceQualityScore'] = data['FireplaceQu'].map(fireplace_qu_mapping).fillna(0)
    data['HasCentralAir'] = data['CentralAir'].eq('Y').astype(int)
    data['RoomsToAreaRatio'] = data['TotRmsAbvGrd'] / data['AboveGroundLivingArea']
    data['RoomsToAreaRatio'] = data['RoomsToAreaRatio'].replace([np.inf, -np.inf], 0).fillna(0)
    data['GarageAge'] = np.where(data['HasGarage'] == 1, data['YrSold'] - data['GarageYrBlt'], 0)
    data['GarageAge'] = data['GarageAge'].replace([np.inf, -np.inf], 0).fillna(0)
    paved_drive_mapping = {'Y': 2, 'P': 1, 'N': 0}
    data['PavedDriveScore'] = data['PavedDrive'].map(paved_drive_mapping).fillna(0)
    data['OverallQual_TotalSF'] = data['OverallQual'] * data['TotalSF']

    return data


def columns_to_drop() -> list:
    """Returns a list of columns to remove."""
    return [target, 'PoolQC', 'Utilities', 'Alley']


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into 80% training and 20% validation while separating the target variable."""
    X = data.drop(columns=columns_to_drop()).copy()
    y = data[target].copy()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid


def preprocess_data(X_train: pd.DataFrame, X_valid: pd.DataFrame, preprocessor_path: str) -> tuple[  pd.DataFrame, pd.DataFrame]:
    """Applies numerical imputation, categorical encoding, and saves the preprocessing pipeline."""

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    low_cardinality_cols = [col for col in cat_cols if X_train[col].nunique() < 10]
    high_cardinality_cols = list(set(cat_cols) - set(low_cardinality_cols))

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    low_cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    high_cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('low_cat', low_cat_transformer, low_cardinality_cols),
        ('high_cat', high_cat_transformer, high_cardinality_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    joblib.dump(preprocessor, preprocessor_path)

    return X_train, X_valid


def export_data(data: dict, filepath: str) -> None:
    """Exports a dictionary to a CSV file without an index."""
    pd.DataFrame(data).to_csv(filepath, index=False)
