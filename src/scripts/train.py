from src.scripts.data import load_data, preprocess_data, split_data, reduce_data, impute_data, impute_extension_data, \
    remove_categorical_data, ordinal_encode_data, one_hot_encode_data, get_features
from src.scripts.model import train_model, evaluate_model, save_model, cross_validate_model, preprocessor_train_model, \
    xgboost_train_model


def main():
    training_data = load_data('src/datasets/train/train_house_prices.csv')
    extreme_gradient_boosting(training_data)


def deal_with_values(training_data):
    X_train, X_valid, y_train, y_valid = split_data(training_data)

    # Use reduce_data(), impute_data() or impute_extension_data() methods to numerical numbers
    X_train_preprocessed, X_valid_preprocessed = impute_data(X_train, X_valid)

    # Use remove_categorical_data() or ordinal_encode_data() methods to categorical numbers
    X_train_final, X_valid_final = one_hot_encode_data(X_train_preprocessed, X_valid_preprocessed)

    model = train_model(X_train_final, y_train)
    mae = evaluate_model(model, X_valid_final, y_valid)
    print(f'Mean Absolute Error: {mae}')


def pipeline(training_data):
    X_train, X_valid, y_train, y_valid = split_data(training_data)

    model = preprocessor_train_model(X_train, y_train)

    mae = evaluate_model(model, X_valid, y_valid)
    print(f'Mean Absolute Error: {mae}')

    save_model(model, 'src/models/house_prices_model.pkl')


def cross_validate(training_data):
    training_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = training_data.SalePrice
    training_data.drop(['SalePrice'], axis=1, inplace=True)

    numeric_cols = [cname for cname in training_data.columns if training_data[cname].dtype in ['int64', 'float64']]
    X = training_data[numeric_cols].copy()
    results = {}

    for i in range(1, 9):
        j = 50 * i
        results[j] = cross_validate_model(X, y, j)

    key = min(results, key=results.get)
    value = results[key]
    print(f'Best estimator: "{key}"\nMAE: {value}')


def extreme_gradient_boosting(training_data):
    X_train, X_valid, y_train, y_valid = split_data(training_data)

    model = xgboost_train_model(X_train, X_valid, y_train, y_valid)
    mae = evaluate_model(model, X_valid, y_valid)
    print(f'Mean Absolute Error: {mae}')


if __name__ == '__main__':
    main()
