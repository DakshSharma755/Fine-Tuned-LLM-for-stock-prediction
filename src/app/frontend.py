# src/app/frontend.py

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from pathlib import Path

# --- Page Configuration ---
# Set the page title, icon, and layout. This must be the first Streamlit command.
st.set_page_config(
    page_title="FT Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Configuration ---
# URL of your running FastAPI backend
# Make sure the port number matches the one in your backend.py
# src/app/frontend.py

# ... (existing code) ...
# URLs of your running FastAPI backend
PREDICTION_ENDPOINT = "http://127.0.0.1:8069/analyze"
TEST_ENDPOINT = "http://127.0.0.1:8069/test"

# --- UI Layout ---
st.title("📈 FT Stock Analyzer")

# ADD THIS MODE TOGGLE
mode = st.radio(
    "Select Mode",
    ["Prediction Mode", "Test Mode"],
    horizontal=True,
    label_visibility="collapsed"
)

if mode == "Prediction Mode":
    st.markdown("An AI-powered tool that analyzes historical stock data and real-time market sentiment to generate financial forecasts and reports.")
else:
    st.markdown("A backtesting tool to evaluate the AI's performance by comparing its historical forecasts against actual market outcomes.")

st.divider()

# --- Input Section ---
# --- Input Section ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Yahoo Finance Link")
    st.markdown("Paste the full URL of the stock you want to analyze from Yahoo Finance.")

    # Descriptions change based on mode
    if mode == "Prediction Mode":
        st.subheader("Prediction Horizon")
        st.markdown("Select the number of days you want to forecast into the future (max 90).")
    else:
        st.subheader("Backtesting Horizon")
        st.markdown("Select the number of past days to use for the performance evaluation (max 90).")

with col2:
    yfinance_link = st.text_input("Enter URL", "https://finance.yahoo.com/quote/AAPL", label_visibility="collapsed")
    st.markdown("""
        <style>
        .custom-space {
            margin-top: 90px; /* Adds 20px of top margin */
            margin-bottom: 90px; /* Adds 20px of bottom margin */
        }
        </style>
        <div class="custom-space"></div>
    """, unsafe_allow_html=True)
    prediction_days = st.number_input("Enter days", min_value=1, max_value=90, value=7, label_visibility="collapsed")
    st.markdown("""
        <style>
        .custom-space {
            margin-top: 90px; /* Adds 20px of top margin */
            margin-bottom: 90px; /* Adds 20px of bottom margin */
        }
        </style>
        <div class="custom-space"></div>
    """, unsafe_allow_html=True)

# --- Analysis Trigger ---
button_label = "Analyze Stock" if mode == "Prediction Mode" else "Run Performance Test"
if st.button(button_label, use_container_width=True, type="primary"):
    if not yfinance_link:
        st.error("Please enter a Yahoo Finance link.")
    else:
        # Determine which endpoint and session state key to use
        if mode == "Prediction Mode":
            endpoint = PREDICTION_ENDPOINT
            results_key = "prediction_results"
        else: # Test Mode
            endpoint = TEST_ENDPOINT
            results_key = "test_results"

        with st.spinner("Processing... This may take a few minutes as the AI models are running."):
            try:
                # --- API Call ---
                payload = {"yfinance_link": yfinance_link, "prediction_days": prediction_days}
                response = requests.post(endpoint, json=payload, timeout=600) 

                if response.status_code == 200:
                    st.session_state[results_key] = response.json()
                else:
                    st.session_state[results_key] = None
                    st.error(f"Failed to get a response from the backend. Status code: {response.status_code}")
                    st.error(response.text)

            except requests.exceptions.RequestException as e:
                st.session_state[results_key] = None
                st.error(f"Connection to backend failed: {e}")

st.divider()
# --- Output Display ---

# --- Prediction Mode Output ---
if mode == "Prediction Mode" and 'prediction_results' in st.session_state and st.session_state.prediction_results:
    results = st.session_state.prediction_results

    if results['status'] == "Success":
        st.subheader("Analyst Report")
        st.markdown(results['analyst_report'])

        st.subheader("Forecast Visualization")
        df_hist = pd.DataFrame(results['historical_data'])
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])

        fig = go.Figure()
        last_hist_date = df_hist['Date'].iloc[-1]
        last_hist_price = df_hist['Close'].iloc[-1]
        forecast_avg = pd.Series(results['forecast_prices']).mean()
        if forecast_avg > last_hist_price:
            zone_color = "rgba(0, 255, 0, 0.2)" # Light green
        else:
            zone_color = "rgba(255, 0, 0, 0.2)" # Light red
        fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Close'], mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=results['forecast_dates'], y=results['forecast_prices'], mode='lines', name='Forecasted Prices', line=dict(color='red')))
        fig.add_vline(x=last_hist_date, line_width=2, line_dash="dash", line_color="white")
        fig.add_vrect(x0=last_hist_date, x1=results['forecast_dates'][-1],
              fillcolor=zone_color, opacity=0.5, layer="below", line_width=0)
        fig.update_layout(title_text=f"{yfinance_link.split('/')[-1]} Price Forecast", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"Analysis Failed: {results['error_message']}")

# --- Test Mode Output ---
elif mode == "Test Mode" and 'test_results' in st.session_state and st.session_state.test_results:
    results = st.session_state.test_results

    if results['status'] == "Success":
        st.subheader("Test Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"${results['mae']:.2f}", help="Mean Absolute Error: The average dollar amount the forecast was off by.")
        col2.metric("RMSE", f"${results['rmse']:.2f}", help="Root Mean Squared Error: Similar to MAE, but penalizes large errors more heavily.")
        col3.metric("MAPE", f"{results['mape']:.2f}%", help="Mean Absolute Percentage Error: The average percentage the forecast was off by.")
        col4.metric("Directional Accuracy", f"{results['directional_accuracy']:.2f}%", help="How often the model correctly predicted if the price would go up or down.")

        st.subheader("Original 'As-If' Analyst Report")
        st.info("This is the report the AI would have generated at the time, using only data available before the test period.")
        st.markdown(results['original_analyst_report'])

        st.subheader("Final Test Performance Review")
        st.success("This is the AI's review of its own performance after comparing its forecast to the actual prices.")
        st.markdown(results['final_test_report'])

    else:
        st.error(f"Test Failed: {results['error_message']}")