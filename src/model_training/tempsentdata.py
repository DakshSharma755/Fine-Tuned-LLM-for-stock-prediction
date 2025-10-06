# src/model_training/tempsentdata.py

import pandas as pd
import json
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import logging

# --- Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# This is the directory where all the old CSV files are
DATA_SOURCE_DIR = PROJECT_ROOT / "data" / "processed" 
# This is the file where the final list of 15 tickers will be saved
OUTPUT_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "raw" / "selected_sentiment_portfolio.json"

# Portfolio Selection Config (matches your other script)
NUM_VOLATILE = 8
NUM_STABLE = 7

def main():
    """
    Analyzes existing CSV files to select a portfolio, saves the portfolio,
    and then deletes the original CSV files.
    """
    logger.info("--- Starting Temporary Analysis of Existing Data ---")
    
    # Find all the old processed_data.csv files
    csv_files = list(DATA_SOURCE_DIR.glob("*_processed_data.csv"))
    if not csv_files:
        logger.error("No processed CSV files found in 'data/processed/'. Cannot continue.")
        return

    logger.info(f"Found {len(csv_files)} CSV files to analyze...")
    
    volatilities = {}
    # Loop through all found files
    for file_path in tqdm(csv_files, desc="Calculating Volatility"):
        try:
            # Extract ticker from filename
            ticker = file_path.stem.replace('_processed_data', '')
            
            df = pd.read_csv(file_path)
            
            if len(df) < 50: # Ensure we have enough data
                continue
            
            df['returns'] = df['Close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(252) # Annualized volatility
            
            if pd.notna(volatility):
                volatilities[ticker] = volatility
                
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Could not process file {file_path.name}: {e}")

    # --- Select the Portfolio ---
    logger.info("Selecting final portfolio from calculated volatilities...")
    sorted_tickers = sorted(volatilities.items(), key=lambda item: item[1], reverse=True)
    
    most_volatile = [t[0] for t in sorted_tickers if t[1] > 0][:NUM_VOLATILE]
    least_volatile = [t[0] for t in sorted_tickers if t[1] > 0][-NUM_STABLE:]
    
    portfolio = most_volatile + least_volatile
    
    # --- Save the Selected Portfolio to a JSON file ---
    logger.info(f"Saving selected portfolio of {len(portfolio)} stocks to {OUTPUT_PORTFOLIO_PATH}")
    with open(OUTPUT_PORTFOLIO_PATH, 'w') as f:
        json.dump(portfolio, f, indent=2)
    logger.info(f"Most volatile: {most_volatile}")
    logger.info(f"Most stable: {least_volatile}")

    # --- Cleanup Phase ---
    logger.info(f"Cleaning up {len(csv_files)} original CSV files...")
    for file_path in tqdm(csv_files, desc="Cleaning Up Files"):
        if file_path:
            try:
                file_path.unlink() # Deletes the file
            except OSError as e:
                logger.error(f"Error deleting file {file_path.name}: {e}")
            
    logger.info("--- Temporary Analysis and Cleanup Complete ---")

if __name__ == "__main__":
    main()