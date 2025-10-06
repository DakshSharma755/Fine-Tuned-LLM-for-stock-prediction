# src/model_training/dataftsentiment.py

import pandas as pd
import json
from pathlib import Path
import sys
import time
import re
from tqdm import tqdm
import numpy as np
from newsapi import NewsApiClient
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
# --- Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.data_ingestion.scraper import get_stock_data

# (Assuming you have a log_config.yaml, otherwise logging will be basic)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
TICKER_CONFIG_PATH = PROJECT_ROOT / "src" / "model_training" / "weblinks.json"
OUTPUT_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_training_data.jsonl"
COMPANIES_SELECTED = PROJECT_ROOT / "data" / "raw" / "selected_sentiment_portfolio.json"
BASE_MODEL_ID = "FreedomIntelligence/TinyDeepSeek-0.5B-base" # The model to use for labeling

# --- Portfolio Selection Config ---
NUM_VOLATILE = 8
NUM_STABLE = 7
VOLATILITY_PERIOD = "2y" # Use 2 years of data to judge volatility

# --- News Scraping Config ---
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY") # IMPORTANT: Add your key here or load from .env
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY not found. Make sure it is set in your .env file.")
NEWS_PERIOD_DAYS = 30 # Free NewsAPI is limited to 30 days
VOLATILITY_TO_ARTICLE_COUNT = {"low": 10, "medium": 25, "high": 50}

"""def calculate_volatility(ticker: str) -> float:
    #Calculates the volatility of a stock as the std dev of its daily returns.
    link = f"https://finance.yahoo.com/quote/{ticker}"
    df = get_stock_data(yfinance_link=link, period=VOLATILITY_PERIOD, save_path=None)
    if len(df) < 50: # Ensure we have enough data
        return 0.0
    
    df['returns'] = df['Close'].pct_change()
    volatility = df['returns'].std() * np.sqrt(252) # Annualized volatility
    return volatility if pd.notna(volatility) else 0.0"""

"""def select_portfolio(tickers: list) -> list:
    #Selects a portfolio of the most and least volatile stocks.
    logger.info("Analyzing volatility for all tickers to select portfolio...")
    volatilities = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"Analyzing ticker {i+1}/{len(tickers)}: {ticker}")
        volatilities[ticker] = calculate_volatility(ticker)
        time.sleep(1) # Be polite to the API

    sorted_tickers = sorted(volatilities.items(), key=lambda item: item[1], reverse=True)
    
    most_volatile = [t[0] for t in sorted_tickers[:NUM_VOLATILE]]
    least_volatile = [t[0] for t in sorted_tickers[-NUM_STABLE:]]
    
    portfolio = most_volatile + least_volatile
    logger.info(f"Selected portfolio of {len(portfolio)} stocks.")
    logger.info(f"Most volatile: {most_volatile}")
    logger.info(f"Most stable: {least_volatile}")
    return portfolio"""

def label_headline(model, tokenizer, headline: str) -> tuple[float, float]:
    """Uses the base LLM to perform zero-shot labeling for sentiment and price impact."""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Analyze the financial headline. Provide its sentiment score from -1.0 (very negative) to 1.0 (very positive) and predict the stock's percentage change for the next trading day.
Respond in the format: SENTIMENT_SCORE: [score], NEXT_DAY_PRICE_CHANGE: [change]%
HEADLINE: {headline}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Use regex to robustly parse the output
    sentiment_match = re.search(r"SENTIMENT_SCORE:\s*(-?\d+\.\d+)", response_text)
    price_change_match = re.search(r"NEXT_DAY_PRICE_CHANGE:\s*(-?\d+\.\d+)%", response_text)
    
    sentiment_score = float(sentiment_match.group(1)) if sentiment_match else 0.0
    price_change_pred = float(price_change_match.group(1)) if price_change_match else 0.0
    
    return sentiment_score, price_change_pred

# Replace your existing main() function with this one

def main():
    """Main function to select portfolio, scrape, label, and save the dataset."""
    logger.info("--- Starting Data Preparation for Generalist Sentiment Model ---")
      
    # --- 1. Load the Selected Portfolio ---
    logger.info(f"Loading selected portfolio from {COMPANIES_SELECTED}")
    with open(COMPANIES_SELECTED, 'r') as f:
        portfolio_tickers = json.load(f)
    """# --- 1. Select Portfolio (with caching) ---
    if PORTFOLIO_CACHE_PATH.exists():
        logger.info(f"Loading cached portfolio from {PORTFOLIO_CACHE_PATH}")
        with open(PORTFOLIO_CACHE_PATH, 'r') as f:
            portfolio_tickers = json.load(f)
    else:
        logger.info("No cached portfolio found. Selecting a new one (this will take a long time)...")
        portfolio_tickers = select_portfolio(config['tickers'])
        with open(PORTFOLIO_CACHE_PATH, 'w') as f:
            json.dump(portfolio_tickers, f, indent=2)
        logger.info(f"Portfolio saved to {PORTFOLIO_CACHE_PATH} for future runs.")"""
    # --- Resume Logic ---
    processed_tickers = set()
    if OUTPUT_JSONL_PATH.exists() and OUTPUT_JSONL_PATH.stat().st_size > 0:
        logger.info(f"Found existing dataset file. Checking for completed tickers...")
        with open(OUTPUT_JSONL_PATH, 'r') as f:
            for line in f:
                # Added a try-except for safety against corrupted lines
                try:
                    prompt_text = json.loads(line)['text']
                    match = re.search(r"STOCK: (\S+)", prompt_text)
                    if match:
                        processed_tickers.add(match.group(1))
                except (json.JSONDecodeError, KeyError):
                    continue # Skip corrupted lines

    if processed_tickers:
        logger.info(f"Found {len(processed_tickers)} already processed tickers in the output file.")
        
        # Filter the portfolio to only include tickers that have not been processed
        remaining_tickers = [t for t in portfolio_tickers if t not in processed_tickers]
        
        if not remaining_tickers:
            logger.info("All tickers from the selected portfolio have already been processed. Exiting.")
            return # Exit the main function

        logger.info(f"Skipping {len(portfolio_tickers) - len(remaining_tickers)} tickers. {len(remaining_tickers)} remain.")
        resume_choice = input("Do you want to resume and process only the remaining tickers? (y/n): ").lower()

        if resume_choice == 'y':
            portfolio_tickers = remaining_tickers # Trim the list to only remaining tickers
        else:
            overwrite_choice = input("Do you want to overwrite the existing file and start from scratch? (y/n): ").lower()
            if overwrite_choice == 'y':
                OUTPUT_JSONL_PATH.unlink()
                logger.info("Existing file removed. Starting from the beginning.")
            else:
                logger.info("Aborting.")
                return # Exit the main function
    # --- End of Resume Logic ---
    
    # --- 2. Load Labeling Model ---
    logger.info(f"Loading labeling model '{BASE_MODEL_ID}'...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    labeling_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    labeling_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    
    # --- 3. Scrape, Label, and Save Data (with Alternating Strategy) ---
    if portfolio_tickers:
        most_volatile = portfolio_tickers[:NUM_VOLATILE]
        least_volatile = portfolio_tickers[NUM_VOLATILE:]
        least_volatile.reverse() # Puts the most stable stocks first

        processing_order = []
        # Interleave the two lists to ensure diversity
        for i in range(NUM_VOLATILE):
            processing_order.append(most_volatile[i])
            if i < len(least_volatile):
                processing_order.append(least_volatile[i])
        
        logger.info(f"Processing stocks in a diversity-focused order: {processing_order}")

        for ticker in processing_order:
            logger.info(f"\nProcessing news for selected ticker: {ticker}")
            link = f"https://finance.yahoo.com/quote/{ticker}"
            df_stock = get_stock_data(yfinance_link=link, period="3mo")
            
            if df_stock.empty:
                logger.warning(f"No stock data for {ticker}, skipping.")
                continue

            df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            
            from_date = (datetime.now() - timedelta(days=NEWS_PERIOD_DAYS)).strftime('%Y-%m-%d')
            all_articles_for_ticker = []
            try:
                articles_response = newsapi.get_everything(
                    q=ticker, 
                    from_param=from_date, 
                    language='en', 
                    sort_by='relevancy', 
                    page_size=100
                )
                all_articles_for_ticker = articles_response.get('articles', [])
            except Exception as e:
                logger.warning(f"Could not fetch news for {ticker}: {e}")
                # If we hit a rate limit, stop processing this run
                if 'rateLimited' in str(e):
                    logger.error("NewsAPI rate limit hit. Please wait before running the script again.")
                    break # Exit the main loop

            articles = {'articles': all_articles_for_ticker}
            
            logger.info(f"Found {len(articles['articles'])} articles for {ticker}. Now labeling...")
            for article in tqdm(articles['articles'], desc=f"Labeling {ticker}"):
                headline_date = pd.to_datetime(article['publishedAt'].split('T')[0])
                
                current_day_data = df_stock[df_stock['Date'] == headline_date]
                
                if not current_day_data.empty:
                    next_trading_day_rows = df_stock[df_stock['Date'] > headline_date]
                    
                    if not next_trading_day_rows.empty:
                        current_day_close = current_day_data['Close'].iloc[0]
                        next_trading_day_close = next_trading_day_rows.iloc[0]['Close']
                        next_day_change = (next_trading_day_close / current_day_close - 1) * 100
                        
                        sentiment_score, _ = label_headline(labeling_model, labeling_tokenizer, article['title'])
                        
                        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Analyze the financial headline. Provide its sentiment score from -1.0 to 1.0 and predict the stock's percentage change for the next trading day.
    HEADLINE: {article['title']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    SENTIMENT_SCORE: {sentiment_score:.2f}, NEXT_DAY_PRICE_CHANGE: {next_day_change:.2f}%<|eot_id|>"""
                                        
                        with open(OUTPUT_JSONL_PATH, 'a') as f:
                            f.write(json.dumps({"text": prompt}) + '\n')
    else:
        logger.info("All selected portfolio tickers have already been processed.")

    logger.info(f"\n--- Data Preparation Complete ---")
    logger.info(f"Final labeled dataset saved to: {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()