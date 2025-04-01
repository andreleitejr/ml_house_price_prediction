import pandas as pd
from src.scripts.data import load_data, preprocess_data, split_data, reduce_data, impute_data, impute_extension_data, \
    remove_categorical_data, ordinal_encode_data, one_hot_encode_data
from src.scripts.model import train_model, evaluate_model, save_model, pipeline_model


def main():
    home_data = load_data('src/datasets/train/train_house_prices.csv')
    # home_data = preprocess_data(home_data)

    X_train, X_valid, y_train, y_valid = split_data(home_data)

    # Use the following methods reduce_data, impute_data, impute_extension_data to numerical numbers
    # X_train_preprocessed, X_valid_preprocessed = impute_data(X_train, X_valid)
    #
    # # Use the following methods remove_categorical_data or ordinal_encode_data to categorical numbers
    # X_train_final, X_valid_final = one_hot_encode_data(X_train_preprocessed, X_valid_preprocessed)

    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"]

    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

    cols = categorical_cols + numerical_cols
    X_train = X_train[cols].copy()
    X_valid = X_valid[cols].copy()

    preprocessor = pipeline_model(categorical_cols, numerical_cols)
    model = train_model(X_train, y_train, preprocessor)
    mae = evaluate_model(model, X_valid, y_valid)
    print(f'Mean Absolute Error: {mae}')

    save_model(model, 'src/models/house_prices_model.pkl')


if __name__ == '__main__':
    main()
