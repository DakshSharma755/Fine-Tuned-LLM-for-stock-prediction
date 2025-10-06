# src/app/backend.py

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# --- Setup Paths and Imports ---
# This ensures the script can find your 'src' directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import your main pipeline function
from src.pipeline.main import run_analysis_pipeline

# Load environment variables (like NEWS_API_KEY) from a .env file
load_dotenv()

# --- API Application ---
app = FastAPI(
    title="FT Stock Analyzer API",
    description="An API that uses a multi-stage LLM pipeline to forecast stock prices and generate analyst reports.",
    version="1.0.0"
)

# --- Pydantic Models for Input and Output Validation ---
class AnalysisRequest(BaseModel):
    yfinance_link: str = Field(..., example="https://finance.yahoo.com/quote/AAPL")
    prediction_days: int = Field(..., example=7, gt=0, le=90)

class AnalysisResponse(BaseModel):
    status: str
    error_message: str | None = None
    analyst_report: str | None = None
    historical_data: list | None = None
    forecast_dates: list[str] | None = None
    forecast_prices: list[float] | None = None

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisResponse)
async def create_analysis(request: AnalysisRequest):
    """
    Accepts a stock link and prediction days, and returns a full analysis.
    This runs the synchronous, CPU/GPU-bound ML pipeline in a separate
    thread to avoid blocking the main server process.
    """
    # Use run_in_threadpool to run our synchronous ML function in a non-blocking way
    results = await run_in_threadpool(
        run_analysis_pipeline, 
        yfinance_link=request.yfinance_link, 
        prediction_days=request.prediction_days
    )
    return results


if __name__=='__main__':
    import uvicorn
    uvicorn.run("backend:app", host = "0.0.0.0", port = 8069, reload = True)