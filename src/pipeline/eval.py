#src/pipeline/eval.py
import torch
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import logging
import sys
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import List
import time
import numpy as np
from newsapi import NewsApiClient
import gc

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.data_ingestion.scraper import get_stock_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03dZ - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime
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

# --- Helper Functions (Copied from main.py) ---
def create_overlapping_chunks(df: pd.DataFrame, chunk_size: int, overlap: int) -> List[pd.DataFrame]:
    chunks = []
    start = 0
    while start < len(df):
        end = start + chunk_size
        chunks.append(df.iloc[start:end])
        start += (chunk_size - overlap)
        if end >= len(df):
            break
    return chunks

def create_summarizer_prompt(prior_summary: str, chunk_data_str: str) -> str:
    """Creates a prompt with a strict rulebook to enforce a textual summary."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    You are a financial analyst. Your task is to update a rolling summary of a stock's history with new data. You will be given the data in small chunks to make sure each piece of context is analysed correctly then recursively fed your own output till you end up with the final chunk output which will then be saved as natural language context for time series analysis.
    --- OUTPUT RULES ---
    1. The output MUST be a concise, natural language sentence or short paragraph.
    2. DO NOT output a list of numbers or raw prices.
    3. Describe the key behavior in the NEW_DATA_CHUNK, such as the trend (upward, downward, stable), volatility (high, low), and any significant price levels tested or broken.
    --- END RULES ---
    Now, perform the task described.
    PRIOR_SUMMARY: {prior_summary}
    NEW_DATA_CHUNK: [{chunk_data_str}]
    UPDATED_SUMMARY:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

# --- Main Test Pipeline Function ---

def run_test_pipeline(yfinance_link: str, prediction_days: int) -> dict:
    """
    Runs the full pipeline in backtesting mode, generating a forecast,
    an "as-if" report, accuracy metrics, and a final performance review.
    """
    results = {"status": "Failure", "error_message": None}
    model, tokenizer, sentiment_pipeline, reporting_model, reporting_tokenizer = None, None, None, None, None
    
    try:
        # === SETUP: DATA FETCHING AND SPLITTING ===
        ticker_symbol = yfinance_link.split("/")[-1].strip()
        if not ticker_symbol: ticker_symbol = yfinance_link.split("/")[-2].strip()
        model_request_days = prediction_days
        for window in sorted(TIME_WINDOW_LOGIC, key=lambda x: x['pred_days']):
            if prediction_days <= window['pred_days']:
                model_request_days = window['pred_days']
                break
        
        context_days_to_use = 1000
        for window in TIME_WINDOW_LOGIC:
            if window['pred_days'] == model_request_days:
                context_days_to_use = window['context_range_days']
                break
        
        logger.info(f"TEST MODE: Predicting {prediction_days} days. Model will use {context_days_to_use} days of context to predict {model_request_days} days.")
        
        df_full = get_stock_data(yfinance_link=yfinance_link, period="5y", save_path=None)
        if df_full.empty or len(df_full) < context_days_to_use + prediction_days:
            raise ValueError(f"Not enough historical data to perform backtest (need {context_days_to_use + prediction_days} days).")

        holdout_df = df_full.tail(prediction_days)
        context_df = df_full.iloc[:-(prediction_days)]
        actual_prices = holdout_df['Close'].tolist()
        last_price_in_context = context_df['Close'].iloc[-1]

        # === STEP 1: RUN INSTANCE 1 (FORECAST) ===
       # === INSTANCE 1a: CONTEXT SUMMARIZATION (with Llama-3.2) ===
        logger.info("--- Loading and Running Instance 1a: Context Summarization (Llama) ---")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        # Load the powerful REPORTING model for the summarization task
        reporting_model = AutoModelForCausalLM.from_pretrained(REPORTING_MODEL_ID, quantization_config=bnb_config, device_map="auto")
        reporting_tokenizer = AutoTokenizer.from_pretrained(REPORTING_MODEL_ID)
        rolling_summary = "No significant trend observed yet."
        
        data_chunks = create_overlapping_chunks(context_df, chunk_size=180, overlap=30)
        for i, chunk in enumerate(data_chunks):
            chunk_data_str = ", ".join([f"{p:.2f}" for p in chunk['Close'].tolist()])
            prompt = create_summarizer_prompt(rolling_summary, chunk_data_str)
            
            inputs = reporting_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(reporting_model.device)
            with torch.no_grad():
                outputs = reporting_model.generate(**inputs, max_new_tokens=150, pad_token_id=reporting_tokenizer.eos_token_id, do_sample=True, temperature=0.7)
            
            # The entire output IS the new summary. No fragile parsing needed.
            new_summary = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            
            if new_summary: # As long as the model says something, update the summary
                rolling_summary = new_summary

        comma_index = rolling_summary.find(',')
        if comma_index != -1:
            rolling_summary = rolling_summary[comma_index + 1:].strip()
        logger.info(f"Final Historical Summary: {rolling_summary}")
        logger.info("--- Releasing Summarization model ---")
        del reporting_model, reporting_tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        reporting_model, reporting_tokenizer = None, None
        # === INSTANCE 1b: TIME SERIES FORECASTING (with TinyDeepSeek) ===
        logger.info("--- Loading and Running Instance 1b: Time Series Forecasting (Fine-tuned) ---")
        # Load the specialized, fine-tuned model ONLY for the final forecast
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        ts_adapter_path = PROJECT_ROOT / "models" / "processed" / f"{BASE_MODEL_ID.replace('/', '_')}-timeseries-generalist-v1" / "final_model"
        model = PeftModel.from_pretrained(model, str(ts_adapter_path))
        
        final_context_df = context_df.tail(180)
        final_context_str = ", ".join([f"{p:.2f}" for p in final_context_df['Close'].tolist()])
        mean, std_dev = final_context_df['Close'].mean(), final_context_df['Close'].std()
        stability_threshold = 5.0 
        first_price = final_context_df['Close'].iloc[0]
        last_price_from_context = final_context_df['Close'].iloc[-1] # Use a different name to avoid conflict
        if first_price > 0:
            percent_change = ((last_price_from_context - first_price) / first_price) * 100
        else:
            percent_change = 0
        if abs(percent_change) < stability_threshold:
            trend = "stable"
        elif last_price_from_context > first_price:
            trend = "upward"
        else:
            trend = "downward"
        last_price = last_price_in_context 
        final_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        You are a financial analyst... Based on your comprehensive analysis summarized as: "{rolling_summary}", perform the following task.

        TASK: Time Series Forecast.
        STOCK: {ticker_symbol}
        LAST_PRICE: {last_price:.2f}
        STATISTICS: mean={mean:.2f}, std_dev={std_dev:.2f}, trend={trend}
        CONTEXT_DAYS: {len(final_context_df)}
        DATA: [{final_context_str}]
        Predict the next {model_request_days} closing prices.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        PREDICTION:"""

        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=model_request_days * 5, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.4, top_p=0.9)
        
        final_assistant_response = tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
        price_strings = re.findall(r'(\d+\.?\d*)', final_assistant_response)
        price_forecast = [float(p) for p in price_strings if p and p != '.']
        price_forecast = price_forecast[:prediction_days] # Truncate to user's desired length

        if not price_forecast: raise ValueError("Model failed to generate a parsable forecast.")
        
        # Added the sanity check for statistically impossible jumps
        for i in range(1, len(price_forecast)):
            if price_forecast[i] < price_forecast[i-1] * 0.5 or price_forecast[i] > price_forecast[i-1] * 2.0:
                logger.warning(f"Statistically unlikely jump detected: {price_forecast}. Flagging as unreliable.")
                results["is_unreliable"] = True
                break
        
        results["forecast_prices"] = price_forecast
        results["historical_data"] = df_full.to_dict('records')

        logger.info("--- Releasing Time Series model ---")
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        model, tokenizer = None, None
        
        # === STEP 2: EVALUATE INSTANCE 1 (METRICS) ===
        logger.info("--- [STEP 2] Evaluating forecast accuracy ---")

        # Ensure arrays are numpy arrays for calculations
        forecast_np = np.array(price_forecast)
        actual_np = np.array(actual_prices)

        # Basic Error Metrics
        errors = forecast_np - actual_np
        results['mae'] = np.mean(np.abs(errors))
        results['mape'] = np.mean(np.abs(errors) / actual_np) * 100
        results['mse'] = np.mean(errors**2)
        results['rmse'] = np.sqrt(results['mse'])

        # Directional Accuracy (crucial for finance)
        actual_diff = np.diff(actual_np)
        forecast_diff = np.diff(forecast_np)
        # Check if the sign of the change is the same
        correct_direction = (np.sign(actual_diff) == np.sign(forecast_diff)).sum()
        results['directional_accuracy'] = (correct_direction / len(actual_diff)) * 100 if len(actual_diff) > 0 else 0

        logger.info(f"""Forecast Accuracy --> 
            MAE: ${results['mae']:.2f}
            MAPE: {results['mape']:.2f}%
            RMSE: ${results['rmse']:.2f}
            Directional Accuracy: {results['directional_accuracy']:.2f}%
        """)

        # === STEP 3 & 4: RUN INSTANCE 2 (SENTIMENT) ===
        # === INSTANCE 2 (Part A): FinBERT Labeling ===
        logger.info("--- Loading and Running Instance 2a: FinBERT Labeling ---")
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0)
        
        all_headlines = set()
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        yf_ticker = yf.Ticker(ticker_symbol)
        company_name = yf_ticker.info.get('longName', ticker_symbol)
        search_query = f'"{company_name}" OR {ticker_symbol}'
        logger.info(f"Searching for news with query: {search_query}")
        last_context_date = pd.to_datetime(context_df['Date'].iloc[-1])
        from_date = (last_context_date - timedelta(days=5)).strftime('%Y-%m-%d')
        
        try:
            articles = newsapi.get_everything(q=search_query, from_param=from_date, language='en', sort_by='relevancy', page_size=20)['articles']
            for article in articles:
                if article.get('title'): all_headlines.add(article['title'])
        except Exception as e:
            logger.error(f"Could not fetch news from NewsAPI: {e}", exc_info=True)
        
        try:
            for article in yf_ticker.news:
                if article.get('title'): all_headlines.add(article['title'])
        except Exception as e:
            logger.error(f"Could not fetch news from yfinance: {e}", exc_info=True)
        r=0
        labeled_headlines = []
        if all_headlines:
            with torch.no_grad():
                analyses = sentiment_pipeline(list(all_headlines))
                r=69
            for headline, analysis in zip(all_headlines, analyses):
                labeled_headlines.append(f"'{headline}' (Sentiment: {analysis['label'].capitalize()}, Score: {analysis['score']:.2f})")
        
        logger.info("--- Releasing FinBERT model ---")
        # Explicitly delete the model from the pipeline object first
        if hasattr(sentiment_pipeline, 'model'):
            del sentiment_pipeline.model
        # Now delete the rest
        del sentiment_pipeline
        if r==69:
            del analyses
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
            Synthesize the key themes and overall sentiment from these headlines into a concise 2-3 sentence paragraph.

            PRE-ANALYZED HEADLINES:
            {headlines_str}

            CONCISE SENTIMENT SUMMARY:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            inputs = reporting_tokenizer(sentiment_prompt, return_tensors="pt").to(reporting_model.device)
            with torch.no_grad():
                outputs = reporting_model.generate(**inputs, max_new_tokens=100, pad_token_id=reporting_tokenizer.eos_token_id)
            sentiment_summary = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # === STEP 5: GENERATE "AS-IF" ANALYST REPORT ===
        logger.info("--- [STEP 5] Generating 'as-if' analyst report ---")
        original_report_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        You are a senior financial analyst. Your task is to write a professional investor report for the stock **{ticker_symbol}**.
        Critically analyze the data provided.

        **CONTEXT:**
        - The most recent closing price for {ticker_symbol} is **${last_price_in_context:.2f}**.

        **DATA TO SYNTHESIZE:**
        1.  **Quantitative Model Forecast:** The price forecast for the next {prediction_days} days is {price_forecast}.
        2.  **Qualitative News Sentiment:** The recent news sentiment is summarized as: "{sentiment_summary}".

        Based on all of this information, write a concise, professional report.
        - **Crucially, evaluate the model's forecast. Is it bullish, bearish, or neutral compared to the most recent actual price?**
        - Provide a concluding summary of the likely outlook for **{ticker_symbol}**.
        - Do not use generic placeholders.

        ANALYST REPORT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        inputs = reporting_tokenizer(original_report_prompt, return_tensors="pt").to(reporting_model.device)
        with torch.no_grad():
            outputs = reporting_model.generate(**inputs, max_new_tokens=512, pad_token_id=reporting_tokenizer.eos_token_id)
        original_analyst_report = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
        results['original_analyst_report'] = original_analyst_report.strip()

        # === STEP 6: GENERATE FINAL TEST PERFORMANCE REVIEW ===
        logger.info("--- [STEP 6] Generating final test performance review ---")
        test_report_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        You are a lead data scientist reviewing the performance of a multi-agent AI forecasting pipeline. Your task is to write a **Test Performance Review**.
        You have the following information:
        - **Stock:** {ticker_symbol}
        - **Original AI Analyst Report:** "{original_analyst_report}"
        - **AI's Price Forecast:** {price_forecast}
        - **Actual Outcome (Ground Truth):** {actual_prices}
        - **Forecast Error Metrics:**   - Mean Absolute Error (MAE): ${results['mae']:.2f} (Average dollar error)
        - Root Mean Squared Error (RMSE): ${results['rmse']:.2f} (Penalizes large errors more)
        - Mean Absolute Percentage Error (MAPE): {results['mape']:.2f}%
        - Directional Accuracy: {results['directional_accuracy']:.2f}% (Correctly predicted up/down trend)
        - **News Sentiment Summary Used:** "{sentiment_summary}"

        **Instructions for your review:**
        1. Start with a clear verdict: Did the pipeline perform well, poorly, or moderately?
        2. Critique the AI's price forecast. How did it compare to the actual outcome? Did it predict the trend correctly?
        3. Critique the original AI analyst report. Was its assessment (bullish/bearish/neutral) correct in hindsight? Did it correctly identify the risks?
        4. Provide a concluding summary on the pipeline's overall performance for this specific test case.
        TEST PERFORMANCE REVIEW:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs = reporting_tokenizer(test_report_prompt, return_tensors="pt").to(reporting_model.device)
        with torch.no_grad():
            outputs = reporting_model.generate(**inputs, max_new_tokens=512, pad_token_id=reporting_tokenizer.eos_token_id)
        final_test_report = reporting_tokenizer.decode(outputs[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
        results['final_test_report'] = final_test_report.strip()

        results['status'] = "Success"

    except Exception as e:
        logger.error(f"An error occurred during the test pipeline: {e}", exc_info=True)
        results["error_message"] = str(e)
    finally:
        logger.info("--- Releasing all models from memory ---")
        # Create a list of all variables that might hold GPU memory
        vars_to_del = [
            'model', 'tokenizer', 'reporting_model', 'reporting_tokenizer',
            'sentiment_pipeline', 'inputs', 'outputs'
        ]
        for var_name in vars_to_del:
            if var_name in locals() and locals()[var_name] is not None:
                if var_name == 'sentiment_pipeline' and hasattr(locals()[var_name], 'model'):
                    del locals()[var_name].model
                del locals()[var_name]
        torch.cuda.empty_cache()
        gc.collect()
        
    summary_full = torch.cuda.memory.memory_summary()
    print(summary_full) 
    return results

if __name__ == '__main__':
    logger.info("--- Running Pipeline in Test Mode ---")
    
    test_link = "https://finance.yahoo.com/quote/AAPL"
    test_days = 30 
    test_results = run_test_pipeline(yfinance_link=test_link, prediction_days=test_days)
    
    if test_results["status"] == "Success":
        print("\n" + "="*50)
        print("--- TEST MODE COMPLETED SUCCESSFULLY ---")
        print("="*50 + "\n")
        print("--- ORIGINAL 'AS-IF' REPORT ---")
        print(test_results['original_analyst_report'])
        print("\n" + "="*50 + "\n")
        print("--- FINAL TEST PERFORMANCE REVIEW ---")
        print(test_results['final_test_report'])
        print("\n" + "="*50 + "\n")
    else:
        print(f"\n--- TEST MODE FAILED ---\nError: {test_results.get('error_message', 'Unknown error')}")