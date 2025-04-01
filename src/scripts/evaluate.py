from src.scripts.data import load_data, preprocess_data, get_features_and_target, get_features, export_data
from src.scripts.model import get_prediction, train_model, evaluate_model


def main():
    home_data = load_data('src/datasets/train/train_house_prices.csv')
    home_data = preprocess_data(home_data)
    X, y = get_features_and_target(home_data)

    full_model = train_model(X, y)

    test_data = load_data('src/datasets/test/test_house_prices.csv')
    test_data = preprocess_data(test_data)
    X_test = get_features(test_data)

    # TODO: Change this
    predictions = get_prediction(full_model, X_test)
    mae = evaluate_model(full_model, X, y)
    print(f'Mean Absolute Error: {mae}')

    data = {'Id': test_data.index, 'SalePrice': predictions}

    export_data(data, filepath='src/datasets/result/predictions_house_prices.csv')

if __name__ == '__main__':
    main()
