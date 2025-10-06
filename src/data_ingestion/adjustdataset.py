# src/data_ingestion/adjustdataset.py

import json
import re
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Configuration ---
INPUT_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "timeseries_training_data.jsonl"
OUTPUT_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "timeseries_training_data_filtered.jsonl"
MAX_CONTEXT_DAYS = 1000

def main():
    """
    Reads the raw dataset, filters out samples with a context longer than
    MAX_CONTEXT_DAYS, and saves a new, training-ready dataset.
    """
    print("--- Starting Dataset Filtering Process ---")
    
    if not INPUT_JSONL_PATH.exists():
        print(f"Error: Input file not found at {INPUT_JSONL_PATH}. Please run datacftimeser.py first.")
        return

    print(f"Processing {INPUT_JSONL_PATH}...")
    original_count = 0
    kept_count = 0

    with open(INPUT_JSONL_PATH, 'r') as infile, open(OUTPUT_JSONL_PATH, 'w') as outfile:
        for line in infile:
            original_count += 1
            prompt_text = json.loads(line)['text']
            
            # Use a regular expression to find the CONTEXT_DAYS value
            match = re.search(r"CONTEXT_DAYS: (\d+)", prompt_text)
            
            if match:
                context_days = int(match.group(1))
                # Keep the sample only if it meets the condition
                if context_days <= MAX_CONTEXT_DAYS:
                    outfile.write(line)
                    kept_count += 1
            else:
                # Keep samples where the tag might be missing, just in case
                outfile.write(line)
                kept_count += 1
    
    dropped_count = original_count - kept_count
    print("\n--- Filtering Complete ---")
    print(f"Total prompts processed: {original_count}")
    print(f"Prompts kept (<= {MAX_CONTEXT_DAYS} context days): {kept_count}")
    print(f"Prompts dropped (> {MAX_CONTEXT_DAYS} context days): {dropped_count}")
    print(f"Filtered dataset saved to: {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()