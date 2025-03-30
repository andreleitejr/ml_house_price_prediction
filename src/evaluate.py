from src.model import get_prediction, train_model
from src.data_preprocessing import load_data, preprocess_data, get_X_y, get_X
from src.output import generate

def main():
    home_data = load_data('data/raw/iowa_house_prices.csv')
    home_data = preprocess_data(home_data)
    X, y = get_X_y(home_data)

    full_model = train_model(X, y)

    test_data = load_data('data/raw/test.csv')
    test_data = preprocess_data(test_data)
    test_X =  get_X(test_data)

    predictions = get_prediction(full_model, test_X)

    generate(test_data, predictions)

if __name__ == '__main__':
    main()
