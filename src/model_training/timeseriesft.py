# src/model_training/timeseriesft.py

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging.config
import yaml
from transformers import EarlyStoppingCallback

"""PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_CONFIG_PATH = PROJECT_ROOT / "logconfig.yaml"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(exist_ok=True) 
LOG_FILENAME_BASE = "timeseries"

# Load the YAML config file and apply its settings
with open(LOG_CONFIG_PATH, 'r') as f:
    logconfig = yaml.safe_load(f)

log_filename = f"train_log_{LOG_FILENAME_BASE}.txt"
logconfig['handlers']['file_handler']['filename'] = str(LOG_DIR / log_filename)
logging.config.dictConfig(logconfig)

# Get a logger for this specific script
logger = logging.getLogger(__name__)

logger.info(f"Logging configured. Output will be saved to {log_filename}")"""

def main():
    """Main function to load data and train the model."""
    
    # --- 1. Configuration ---
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    """#LOG_CONFIG_PATH = PROJECT_ROOT / "logconfig.yaml"
    LOG_DIR = PROJECT_ROOT / "logs"

    LOG_DIR.mkdir(exist_ok=True) 
    LOG_FILENAME_BASE = "timeseries"

    # Load the YAML config file and apply its settings
    with open(LOG_CONFIG_PATH, 'r') as f:
        logconfig = yaml.safe_load(f)

    log_filename = f"train_log_{LOG_FILENAME_BASE}.txt"
    logconfig['handlers']['file_handler']['filename'] = str(LOG_DIR / log_filename)
    logging.config.dictConfig(logconfig)

    # Get a logger for this specific script
    logger = logging.getLogger(__name__)

    logger.info(f"Logging configured. Output will be saved to {log_filename}")
    """
    #BASE_MODEL_ID = "chuanli11/Llama-3.2-3B-Instruct-uncensored"  
    BASE_MODEL_ID = "FreedomIntelligence/TinyDeepSeek-0.5B-base"
    
    PROCESSED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "timeseries_training_data_filtered.jsonl"
    PROCESSED_MODEL_PATH = PROJECT_ROOT / "models" / "processed" / f"{BASE_MODEL_ID.replace('/', '_')}-timeseries-generalist-v1"

    # --- 2. Load the Pre-processed Dataset ---
    print("--- Starting Model Fine-Tuning ---")
    print(f"Loading dataset from: {PROCESSED_DATASET_PATH}")
    dataset = load_dataset('json', data_files=str(PROCESSED_DATASET_PATH), split='train')
    print(f"Dataset loaded with {len(dataset)} samples.")
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    EVAL_SUBSET_SIZE = 10000
    if len(eval_dataset) > EVAL_SUBSET_SIZE:
        eval_dataset = eval_dataset.select(range(EVAL_SUBSET_SIZE))
        
    print(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples.")
    
    # --- 3. Load Model and Tokenizer ---
    print(f"Loading base model '{BASE_MODEL_ID}' with quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")
    print(model)
    
    # --- 4. Configure PEFT (LoRA) and Trainer ---
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "q_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=PROCESSED_MODEL_PATH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        #num_train_epochs=1, 
        max_steps = 8000,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    def collator(data):
        tokenized_batch = tokenizer(
            [sample['text'] for sample in data],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        # For Causal LM, the labels are the same as the input_ids.
        # The model internally handles shifting them for next-token prediction.
        tokenized_batch["labels"] = tokenized_batch["input_ids"].clone()
        return tokenized_batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # --- 5. Start Training ---
    last_checkpoint = get_last_checkpoint(PROCESSED_MODEL_PATH)
    print("Starting training...")
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("--- Training Complete ---")
    final_model_path = PROCESSED_MODEL_PATH / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"Final fine-tuned model adapters saved to: {final_model_path}")

if __name__ == "__main__":
    main()