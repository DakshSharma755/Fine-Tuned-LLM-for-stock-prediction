# src/pipeline/main.py

import torch
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import logging
import sys
from dotenv import load_dotenv
import os
import yfinance as yf

load_dotenv()
# --- Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.data_ingestion.scraper import get_stock_data

# (Setup basic logging if the main app doesn't)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_MODEL_ID = "FreedomIntelligence/TinyDeepSeek-0.5B-base"
REPORTING_MODEL_ID = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

TIME_WINDOW_LOGIC = [
    {"pred_days": 1, "context_range_days": 30},
    {"pred_days": 5, "context_range_days": 60},
    {"pred_days": 7, "context_range_days": 90},
    {"pred_days": 10, "context_range_days": 180},
    {"pred_days": 30, "context_range_days": 365},
    {"pred_days": 90, "context_range_days": 1000},
]

# --- Load Models (Global Scope) ---
# We load the models once when the script starts to avoid reloading on every request.
#logger.info(f"Loading base model '{BASE_MODEL_ID}' with quantization...")
"""BNB_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=BNB_CONFIG, device_map="auto")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

# Define paths to your fine-tuned adapters
TIMESERIES_ADAPTER_PATH = PROJECT_ROOT / "models" / "processed" / f"{BASE_MODEL_ID.replace('/', '_')}-timeseries-generalist-v1" / "final_model"

logger.info("Loading FinBERT sentiment analysis pipeline...")
SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="ProsusAI/finbert")

logger.info(f"Loading reporting model '{REPORTING_MODEL_ID}' with quantization...")
REPORTING_MODEL = AutoModelForCausalLM.from_pretrained(
    REPORTING_MODEL_ID, quantization_config=BNB_CONFIG, device_map="auto"
)
REPORTING_TOKENIZER = AutoTokenizer.from_pretrained(REPORTING_MODEL_ID, use_fast=True)

# Attach both adapters
logger.info("Attaching time-series adapter...")
MODEL = PeftModel.from_pretrained(MODEL, str(TIMESERIES_ADAPTER_PATH), adapter_name="timeseries")
"""
#logger.info("All models loaded and ready.")


# In src/pipeline/main.py, replace the entire run_analysis_pipeline function with this one

def run_analysis_pipeline(yfinance_link: str, prediction_days: int) -> dict:
    """
    Runs the full 3-instance pipeline with a merged sentiment approach,
    loading and releasing models sequentially to manage memory.
    """
    results = {"status": "Failure", "error_message": None}
    # Initialize all model variables to None for robust cleanup in the 'finally' block
    model, tokenizer, sentiment_pipeline, reporting_model, reporting_tokenizer = None, None, None, None, None
    
    if not (1 <= prediction_days <= 90):
        results["error_message"] = "Prediction days must be between 1 and 90."
        return results

    try:
        ticker_symbol = yfinance_link.split("/")[-1].strip()
        if not ticker_symbol: ticker_symbol = yfinance_link.split("/")[-2].strip()

        # === INSTANCE 1: TIME SERIES FORECASTING ===
        logger.info("--- Loading and Running Instance 1: Time Series Model ---")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        ts_adapter_path = PROJECT_ROOT / "models" / "processed" / f"{BASE_MODEL_ID.replace('/', '_')}-timeseries-generalist-v1" / "final_model"
        model = PeftModel.from_pretrained(model, str(ts_adapter_path), adapter_name="timeseries")
        context_days_to_fetch = TIME_WINDOW_LOGIC[-1]['context_range_days'] 
        for window in sorted(TIME_WINDOW_LOGIC, key=lambda x: x['pred_days']):
            if prediction_days <= window['pred_days']:
                context_days_to_fetch = window['context_range_days']
                break

        if context_days_to_fetch <= 30: period = "1mo"
        elif context_days_to_fetch <= 60: period = "2mo"
        elif context_days_to_fetch <= 90: period = "3mo"
        elif context_days_to_fetch <= 180: period = "6mo"
        elif context_days_to_fetch <= 365: period = "1y"
        elif context_days_to_fetch <= 1000: period = "3y"
        else: period = "5y"

        logger.info(f"Prediction horizon: {prediction_days} days. Fetching {period} of context data.")
        
        df_stock = get_stock_data(yfinance_link=yfinance_link, period=period, save_path=None)
        if df_stock.empty:
            results["error_message"] = f"Could not fetch historical data for {ticker_symbol}."
            return results
            
        context_days, context_prices = len(df_stock), df_stock['Close'].tolist()
        context_str = ", ".join([f"{p:.2f}" for p in context_prices])
        mean, std_dev = df_stock['Close'].mean(), df_stock['Close'].std()
        trend = "upward" if context_prices[-1] > context_prices[0] else "downward"
        
        ts_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a financial analyst specializing in time series forecasting.
Now, perform the following task.
TASK: Time Series Forecast.
STOCK: {ticker_symbol}
STATISTICS: mean={mean:.2f}, std_dev={std_dev:.2f}, trend={trend}
CONTEXT_DAYS: {context_days}
DATA: [{context_str}]
Predict the next {prediction_days} closing prices.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
PREDICTION:"""
        
        model.set_adapter("timeseries")
        inputs = tokenizer(ts_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
            )
        forecast_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        price_forecast = [float(p.strip()) for p in re.findall(r'[\d\.]+', forecast_text.split("PREDICTION:")[1])]
        results["forecast_prices"] = price_forecast

        logger.info("--- Releasing Time Series model ---")
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        model, tokenizer = None, None

        # === INSTANCE 2 (Part A): FinBERT Labeling ===
        logger.info("--- Loading and Running Instance 2a: FinBERT Labeling ---")
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0)
        
        all_headlines = set()
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        yf_ticker = yf.Ticker(ticker_symbol)
        from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        try:
            articles = newsapi.get_everything(q=ticker_symbol, from_param=from_date, language='en', sort_by='relevancy', page_size=20)['articles']
            for article in articles:
                if article.get('title'): all_headlines.add(article['title'])
        except Exception as e: logger.warning(f"Could not fetch news from NewsAPI: {e}")
        
        try:
            for article in yf_ticker.news:
                if article.get('title'): all_headlines.add(article['title'])
        except Exception as e: logger.warning(f"Could not fetch news from yfinance: {e}")

        labeled_headlines = []
        if all_headlines:
            analyses = sentiment_pipeline(list(all_headlines))
            for headline, analysis in zip(all_headlines, analyses):
                labeled_headlines.append(f"'{headline}' (Sentiment: {analysis['label'].capitalize()}, Score: {analysis['score']:.2f})")
        
        logger.info("--- Releasing FinBERT model ---")
        del sentiment_pipeline, analyses
        torch.cuda.empty_cache()
        sentiment_pipeline = None
        
        # === INSTANCE 2 (Part B) & INSTANCE 3: Llama Synthesis ===
        logger.info("--- Loading and Running Instances 2b & 3: Llama Synthesis ---")
        reporting_model = AutoModelForCausalLM.from_pretrained(REPORTING_MODEL_ID, quantization_config=bnb_config, device_map="auto",torch_dtype=torch.bfloat16)
        reporting_tokenizer = AutoTokenizer.from_pretrained(REPORTING_MODEL_ID, use_fast=True)

        # Instance 2b: Create a daily sentiment summary
        sentiment_summary = "No recent news found to analyze."
        if labeled_headlines:
            headlines_str = "\n".join([f"- {h}" for h in labeled_headlines])
            sentiment_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Analyze the following list of recent, pre-analyzed headlines for the stock {ticker_symbol}.
Synthesize them into a one-sentence summary of the overall daily sentiment.

PRE-ANALYZED HEADLINES:
{headlines_str}

OVERALL SENTIMENT SUMMARY:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            inputs = reporting_tokenizer(sentiment_prompt, return_tensors="pt").to(reporting_model.device)
            outputs = reporting_model.generate(**inputs, max_new_tokens=100, pad_token_id=reporting_tokenizer.eos_token_id)
            sentiment_summary = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Instance 3: Create the final report
        info_for_report = f"Price Forecast for {ticker_symbol} (next {prediction_days} days): {price_forecast}\n\nRecent News Sentiment Summary: {sentiment_summary}"
        reporting_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a senior financial analyst. Synthesize the following quantitative forecast and qualitative news analysis into a concise, professional report for an investor.
Do not just list the data. Provide a concluding summary of the likely outlook for the stock.
--- DATA ---
{info_for_report}
--- END DATA ---
ANALYST REPORT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs = reporting_tokenizer(reporting_prompt, return_tensors="pt").to(reporting_model.device)
        outputs = reporting_model.generate(**inputs, max_new_tokens=512, pad_token_id=reporting_tokenizer.eos_token_id)
        final_report_text = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        results["analyst_report"] = final_report_text.strip()
        results["historical_data"] = df_stock.to_dict('records')
        last_date = pd.to_datetime(df_stock['Date'].iloc[-1])
        results["forecast_dates"] = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, len(price_forecast) + 1)]
        results["status"] = "Success"

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {e}", exc_info=True)
        results["error_message"] = str(e)

    finally:
        # This block will run NO MATTER WHAT, ensuring memory is always freed.
        logger.info("--- Releasing all models from memory ---")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if sentiment_pipeline is not None: del sentiment_pipeline
        if reporting_model is not None: del reporting_model
        if reporting_tokenizer is not None: del reporting_tokenizer
        torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    # This block allows you to test the script directly
    logger.info("--- Running Rudimentary Pipeline Test ---")
    
    test_link = "https://finance.yahoo.com/quote/ONDS"
    test_days = 7
    
    pipeline_results = run_analysis_pipeline(yfinance_link=test_link, prediction_days=test_days)
    
    if pipeline_results["status"] == "Success":
        print("\n--- PIPELINE COMPLETED SUCCESSFULLY ---")
        print("\n--- FINAL ANALYST REPORT ---")
        print(pipeline_results["analyst_report"])
        print("\n--- FORECAST DATA ---")
        print(f"Dates: {pipeline_results['forecast_dates']}")
        print(f"Prices: {pipeline_results['forecast_prices']}")
    else:
        print(f"\n--- PIPELINE FAILED ---")
        print(f"Error: {pipeline_results['error_message']}")