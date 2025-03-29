from src.model import load_model, evaluate_model
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data

def main():
    model = load_model('models/house_price_model.pkl')

    df = load_data('data/raw/house_prices.csv')
    df = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(df)
    x_train_scaled, x_test_scaled, _ = scale_data(x_train, x_test)

    mse = evaluate_model(model, x_test_scaled, y_test)
    print(f'Mean Squared Error on Test Set: {mse}')

if __name__ == '__main__':
    main()
