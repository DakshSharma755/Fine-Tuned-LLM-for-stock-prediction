# src/model_training/datacftimeser.py

import pandas as pd
import json
import random
from pathlib import Path
from datasets import Dataset
import sys
import time
import re
import os

# Add project root to path to allow importing 'src'
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_ingestion.scraper import get_stock_data

# --- Configuration ---
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "timeseries_batch"
TICKER_CONFIG_PATH = PROJECT_ROOT / "src" / "model_training" / "weblinks.json"
PROCESSED_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "timeseries_training_data.jsonl"
DATA_PERIOD = "5y" # Scrape a long history for diverse contexts
BATCH_SIZE = 10
REQUEST_DELAY = 2

TIME_WINDOW_LOGIC = [
    {"pred_days": 1, "context_range_days": 30},
    {"pred_days": 5, "context_range_days": 60},
    {"pred_days": 7, "context_range_days": 90},
    {"pred_days": 10, "context_range_days": 180},
    {"pred_days": 30, "context_range_days": 365},
    {"pred_days": 90, "context_range_days": 1000},
    {"pred_days": 365, "context_range_days": 1825},
]

def format_training_prompt(df, ticker_symbol):
    """Creates a rich text prompt with statistical context and a one-shot example."""
    prompts = []
    # Loop through the dataframe to create sliding window prompts
    for i in range(len(df)):
        context_correlation = random.choice(TIME_WINDOW_LOGIC)
        prediction_days = context_correlation["pred_days"]
        context_days = context_correlation["context_range_days"]

        if i < context_days:
            continue
        
        prediction_end_index = i + prediction_days
        if prediction_end_index > len(df):
            continue

        context_df = df.iloc[i - context_days : i]
        context_prices = context_df['Close'].tolist()
        prediction_prices = df.iloc[i : prediction_end_index]['Close'].tolist()

        mean = context_df['Close'].mean()
        std_dev = context_df['Close'].std()
        trend = "upward" if context_prices[-1] > context_prices[0] else "downward"

        context_str = ", ".join([f"{p:.2f}" for p in context_prices])
        prediction_str = ", ".join([f"{p:.2f}" for p in prediction_prices])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a financial analyst specializing in time series forecasting.
Now, perform the following task.
TASK: Time Series Forecast.
STOCK: {ticker_symbol}
STATISTICS: mean={mean:.2f}, std_dev={std_dev:.2f}, trend={trend}
CONTEXT_DAYS: {context_days}
DATA: [{context_str}]
Predict the next {prediction_days} closing prices.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
PREDICTION: [{prediction_str}]<|eot_id|>"""
        
        prompts.append({"text": prompt})
    return prompts

def main():
    """Main function to generate and save the training dataset."""
    print("--- Starting Data Preparation for Generalist Time Series Model ---")
    
    with open(TICKER_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    tickers = config['tickers']
    # Add this entire block after 'tickers = config['tickers']'

    # --- Resume Logic ---
    if PROCESSED_JSONL_PATH.exists():
        print(f"Found existing dataset file: {PROCESSED_JSONL_PATH}")
        with open(PROCESSED_JSONL_PATH, 'rb') as f:
            try:  # Read the last line of the file
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()

        if last_line:
            last_prompt = json.loads(last_line)
            # Use regex to find the STOCK ticker in the last prompt
            match = re.search(r"STOCK: (\S+)", last_prompt['text'])
            if match:
                last_ticker_processed = match.group(1)
                print(f"The last ticker successfully processed was: {last_ticker_processed}")
                try:
                    last_ticker_index = tickers.index(last_ticker_processed)
                    tickers_to_process = tickers[last_ticker_index + 1:]
                    
                    if not tickers_to_process:
                        print("Dataset already complete. Exiting.")
                        return # Exit the main function

                    print(f"There are {len(tickers_to_process)} tickers left to process.")
                    resume_choice = input("Do you want to resume from the next ticker? (y/n): ").lower()
                    
                    if resume_choice == 'y':
                        tickers = tickers_to_process # Trim the list to only remaining tickers
                    else:
                        overwrite_choice = input("Do you want to overwrite the existing file and start from scratch? (y/n): ").lower()
                        if overwrite_choice == 'y':
                            PROCESSED_JSONL_PATH.unlink()
                            print("Existing file removed. Starting from the beginning.")
                        else:
                            print("Aborting.")
                            return # Exit the main function
                except ValueError:
                    print(f"Could not find last ticker '{last_ticker_processed}' in the config. Starting fresh.")
    #all_prompts = []
    # Ensure the target directories exist
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_prompts_generated = 0

    # Process tickers in batches
    for i in range(0, len(tickers), BATCH_SIZE):
        batch_tickers = tickers[i:i+BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1}/{-(-len(tickers)//BATCH_SIZE)} ---")

        for ticker in batch_tickers:
            print(f"\nProcessing data for {ticker}...")
            link = f"https://finance.yahoo.com/quote/{ticker}"
            df = get_stock_data(yfinance_link=link, period=DATA_PERIOD, save_path=str(RAW_DATA_PATH))
            if df.empty:
                print(f"No data found for {ticker}, skipping.")
                continue
                
            ticker_prompts = format_training_prompt(df, ticker)
            # Append the prompts for this ticker to the final .jsonl file
            if ticker_prompts:
                with open(PROCESSED_JSONL_PATH, 'a') as f:
                    for prompt in ticker_prompts:
                        f.write(json.dumps(prompt) + '\n')
                total_prompts_generated += len(ticker_prompts)
            # Be polite to the server
            print(f"Generated {len(ticker_prompts)} prompts for {ticker}.")
            time.sleep(REQUEST_DELAY)   
        # --- Cleanup raw files for the current batch ---
        print("Cleaning up raw data files for the batch...")
        for file in RAW_DATA_PATH.glob('*.csv'):
            file.unlink()

    """print(f"\nTotal prompts generated from all stocks: {len(all_prompts)}")
    
    # Convert to Hugging Face Dataset object
    dataset = Dataset.from_list(all_prompts)
    
    # Save the processed dataset to disk
    print(f"Saving processed dataset to: {PROCESSED_DATASET_PATH}")
    dataset.save_to_disk(PROCESSED_DATASET_PATH)
    print("--- Data Preparation Complete ---")
    """
    print(f"\n--- Data Preparation Complete ---")
    print(f"Total prompts generated and saved to {PROCESSED_JSONL_PATH}: {total_prompts_generated}")



if __name__ == "__main__":
    main()