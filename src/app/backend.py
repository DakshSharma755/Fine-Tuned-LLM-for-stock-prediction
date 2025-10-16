# src/app/backend.py

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.pipeline.worker import run_pipeline_in_worker


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

class TestResponse(BaseModel):
    status: str
    error_message: str | None = None
    original_analyst_report: str | None = None
    final_test_report: str | None = None
    mae: float | None = None
    mape: float | None = None
    rmse: float | None = None
    directional_accuracy: float | None = None

def run_process_and_get_result(task: str, payload: dict):
    """
    Manages the lifecycle of the worker process.
    """
    # A Manager allows processes to share Python objects, like a queue.
    with mp.Manager() as manager:
        queue = manager.Queue() # Create a queue to get the result back

        process = mp.Process(
            target=run_pipeline_in_worker,
            args=(task, payload, queue)
        )
        process.start()
        process.join() # Wait for the process to finish

        results = queue.get() # Retrieve the result from the queue
        return results

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisResponse)
async def create_analysis(request: AnalysisRequest):
    """
    Accepts a stock link and prediction days, and returns a full analysis.
    This runs the synchronous, CPU/GPU-bound ML pipeline in a separate
    thread to avoid blocking the main server process.
    """
    # Use run_in_threadpool to run our synchronous ML function in a non-blocking way
    payload = request.dict()
    results = await run_in_threadpool(run_process_and_get_result, task='analyze', payload=payload)
    return results

@app.post("/test", response_model=TestResponse)
async def create_test_run(request: AnalysisRequest):
    """
    Accepts a stock link and prediction days, and runs the backtesting
    pipeline to evaluate the model's performance against historical data.
    """
    payload = request.dict()
    results = await run_in_threadpool(run_process_and_get_result, task='test', payload=payload)
    return results


if __name__=='__main__':
    mp.set_start_method("spawn", force=True)
    import uvicorn
    uvicorn.run("backend:app", host = "0.0.0.0", port = 8069)