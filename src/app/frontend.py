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
BACKEND_URL = "http://127.0.0.1:8069/analyze"

# --- UI Layout ---
st.title("📈 FT Stock Analyzer")
st.markdown("An AI-powered tool that analyzes historical stock data and real-time market sentiment to generate financial forecasts and reports.")

st.divider()

# --- Input Section ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Yahoo Finance Link")
    st.markdown("Paste the full URL of the stock you want to analyze from Yahoo Finance.")
    
    st.subheader("Prediction Horizon")
    st.markdown("Select the number of days you want to forecast into the future (max 90).")

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
if st.button("Analyze Stock", use_container_width=True, type="primary"):
    if not yfinance_link:
        st.error("Please enter a Yahoo Finance link.")
    else:
        # Show a spinner while the backend is processing
        with st.spinner("Processing... This may take a few minutes as the AI models are running."):
            try:
                # --- API Call ---
                payload = {"yfinance_link": yfinance_link, "prediction_days": prediction_days}
                # A long timeout is needed as the pipeline is slow
                response = requests.post(BACKEND_URL, json=payload, timeout=600) 
                
                if response.status_code == 200:
                    st.session_state.results = response.json() # Save results to session state
                else:
                    st.session_state.results = None
                    st.error(f"Failed to get a response from the backend. Status code: {response.status_code}")
                    st.error(response.text)

            except requests.exceptions.RequestException as e:
                st.session_state.results = None
                st.error(f"Connection to backend failed: {e}")

st.divider()

# --- Output Display ---
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    if results['status'] == "Success":
        st.subheader("Analyst Report")
        st.markdown(results['analyst_report'])

        st.subheader("Forecast Visualization")
        
        # --- Create Plot ---
        df_hist = pd.DataFrame(results['historical_data'])
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])

        fig = go.Figure()

        # Add historical data trace
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], 
            y=df_hist['Close'],
            mode='lines',
            name='Historical Prices',
            line=dict(color='royalblue')
        ))

        # Add forecast data trace
        fig.add_trace(go.Scatter(
            x=results['forecast_dates'],
            y=results['forecast_prices'],
            mode='lines',
            name='Forecasted Prices',
            line=dict(color='red')
        ))

        # Update layout for a professional look
        fig.update_layout(
            title_text=f"{yfinance_link.split('/')[-1]} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Legend",
            template="plotly_dark" # Use a dark theme
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("The analysis failed. Please see the error message below:")
        st.error(results['error_message'])
        
