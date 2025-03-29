from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from src.model import train_model, evaluate_model, save_model

def main():
    df = load_data('data/raw/house_prices.csv')

    df = preprocess_data(df)

    x_train, x_test, y_train, y_test = split_data(df)

    x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

    model = train_model(x_train_scaled, y_train)

    mse = evaluate_model(model, x_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')

    save_model(model, 'models/house_price_model.pkl')

if __name__ == '__main__':
    main()
