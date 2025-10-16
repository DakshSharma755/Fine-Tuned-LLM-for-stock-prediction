# src/pipeline/worker.py

# This worker needs all the same imports as your main.py pipeline
import torch
import gc
from src.pipeline.main import run_analysis_pipeline
from src.pipeline.eval import run_test_pipeline

def run_pipeline_in_worker(task: str, payload: dict, queue):
    """
    This function is the entry point for the isolated process.
    It runs the requested pipeline and puts the result in a shared queue.
    """
    try:
        if task == 'analyze':
            result = run_analysis_pipeline(
                yfinance_link=payload['yfinance_link'],
                prediction_days=payload['prediction_days']
            )
        elif task == 'test':
            result = run_test_pipeline(
                yfinance_link=payload['yfinance_link'],
                prediction_days=payload['prediction_days']
            )
        else:
            result = {"status": "Failure", "error_message": "Invalid task specified."}

        queue.put(result)
    except Exception as e:
        # Ensure any exception is caught and returned
        error_result = {"status": "Failure", "error_message": str(e)}
        queue.put(error_result)
    finally:
        # This process will be destroyed, so cleanup is less critical,
        # but it's good practice.
        torch.cuda.empty_cache()
        gc.collect()