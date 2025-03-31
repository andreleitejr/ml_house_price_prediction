from src.scripts.data import load_data, preprocess_data, split_data, reduce_data, impute_data, impute_extension_data, remove_categorical_data, ordinal_encode_data
from src.scripts.model import train_model, evaluate_model, save_model


def main():
    home_data = load_data('src/datasets/train/train_house_prices.csv')
    home_data = preprocess_data(home_data)

    X_train, X_valid, y_train, y_valid = split_data(home_data)

    # Use the following methods reduce_data, impute_data, impute_extension_data, remove_categorical_data or ordinal_encode_data
    X_train_final, X_valid_final = ordinal_encode_data(X_train, X_valid)

    model = train_model(X_train_final, y_train)

    mae = evaluate_model(model, X_valid_final, y_valid)
    print(f'Mean Absolute Error: {mae}')

    save_model(model, 'src/models/house_prices_model.pkl')


if __name__ == '__main__':
    main()
