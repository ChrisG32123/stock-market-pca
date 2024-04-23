import pandas as pd

def load_stock_data(filepath):
    # Load the stock data from CSV
    data = pd.read_csv(filepath)
    return data

if __name__ == "__main__":
    filepath = 'stocks.csv'
    stock_data = load_stock_data(filepath)
    print(stock_data)
