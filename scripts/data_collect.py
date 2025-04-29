import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker="AAPL", period="2y"):
    stock_data = yf.download(ticker, period=period)
    stock_data = stock_data[['Close']]  # Keep only the closing prices
    return stock_data
if __name__ == "__main__":
    data = fetch_stock_data()
    data.to_csv("stock_data.csv")