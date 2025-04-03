import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col='Id')


def split_data(data: pd.DataFrame):
    data = data.dropna(subset=['SalePrice'])
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


def get_numerical_and_categorical_columns(data: pd.DataFrame):
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in data.select_dtypes(include=['object']).columns if data[col].nunique() < 10]
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


def train_random_forest(X_train, y_train, numerical_cols, categorical_cols):
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, X_valid, y_train, y_valid):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def evaluate_model(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictions)


def cross_validate(X, y, n_estimators):
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('regressor', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()


def save_model(model, file_path: str):
    joblib.dump(model, file_path)


def main():
    data = load_data('src/datasets/train/train_house_prices.csv')
    X_train, X_valid, y_train, y_valid = split_data(data)

    numerical_cols, categorical_cols = get_numerical_and_categorical_columns(X_train)

    model_rf = train_random_forest(X_train, y_train, numerical_cols, categorical_cols)
    mae_rf = evaluate_model(model_rf, X_valid, y_valid)
    print(f'Random Forest MAE: {mae_rf}')

    model_xgb = train_xgboost(X_train, X_valid, y_train, y_valid)
    mae_xgb = evaluate_model(model_xgb, X_valid, y_valid)
    print(f'XGBoost MAE: {mae_xgb}')

    save_model(model_xgb, 'src/models/house_prices_xgb.pkl')


if __name__ == '__main__':
    main()
