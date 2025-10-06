# src/data_ingestion/scraper.py

import yfinance as yf
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def get_stock_data(
    yfinance_link: str,
    period: str = "1y",
    save_path: None|str = f"{PROJECT_ROOT}/data/processed/",
    delete_after_use: bool = False
) -> pd.DataFrame:
    """
    Scrapes the most recent stock data from a Yahoo Finance link, processes it,
    saves it to a file, and optionally deletes the file after returning the data.

    Args:
        yfinance_link (str): The full URL to the stock's page on Yahoo Finance.
        period (str, optional): The period of historical data to fetch (e.g., "1y", "6mo", "30d"). Defaults to "1y".
        save_path (str, optional): The directory path to save the processed data file. Defaults to "data/processed/".
        delete_after_use (bool, optional): If True, deletes the saved file after returning the data. Defaults to False.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the processed stock data.
    """
    print(f"Starting data ingestion for link: {yfinance_link}")

    # --- 1. Extract Ticker from URL ---
    try:
        # A more robust way to get the ticker from various URL formats
        parsed_url = urlparse(yfinance_link)
        ticker = [path for path in parsed_url.path.split('/') if path][-1]
        print(f"Extracted ticker: {ticker}")
    except Exception as e:
        print(f"Error: Could not extract ticker from URL. {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

    # --- 2. Scrape Data using yfinance ---
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period=period)

    if hist_data.empty:
        print(f"Error: No data found for ticker {ticker} for the period {period}.")
        return pd.DataFrame()

    print(f"Successfully scraped {len(hist_data)} data points.")

    # --- 3. Minor Preprocessing ---
    # Reset index to make 'Date' a column
    hist_data.reset_index(inplace=True)
    # Ensure 'Date' is in the correct format (YYYY-MM-DD)
    hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.strftime('%Y-%m-%d')
    # Simple moving average as an example feature
    hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
    # Dropping initial rows where moving average is NaN
    hist_data.dropna(inplace=True)
    
    print("Minor preprocessing complete. (Date formatting, 20-day MA calculated)")

    # --- 4. Save Data to a Temporary File ---
    # Create the directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{ticker}_processed_data.csv")
        hist_data.to_csv(file_path, index=False)
        print(f"Data temporarily saved to: {file_path}")

    # --- 5. Conditional Deletion Logic ---
    if delete_after_use:
        try:
            os.remove(file_path)
            print(f"Ephemeral mode: Deleted temporary file at {file_path}")
        except OSError as e:
            print(f"Error deleting file: {e}")

    return hist_data

# This block allows you to test the script directly
if __name__ == '__main__':
    print("--- Running in local test mode (deletion disabled) ---")
    test_link = "https://finance.yahoo.com/quote/BTC-USD/" 
    
    # In a local run, we call with delete_after_use=False (the default)
    # This is for development, like preparing data for fine-tuning.
    data = get_stock_data(yfinance_link=test_link, period="6mo")
    
    if not data.empty:
        print("\n--- Test Run Successful ---")
        print("Data for fine-tuning has been processed and saved.")
        print("First 5 rows of data:")
        print(data.head())