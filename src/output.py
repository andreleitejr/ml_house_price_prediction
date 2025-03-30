import pandas as pd

def generate(data, predictions):
    """Generate a predictions.csv file with the predictions results."""
    output = pd.DataFrame({'Id': data.Id, 'SalePrice': predictions})
    output.to_csv('predictions.csv', index=False)