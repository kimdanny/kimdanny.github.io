#!/usr/bin/env python
# Fine-tuning script for Pythia-2.8B model with PANDA dataset using LoRA and GPU optimization

import os
import json
import torch
import wandb
import logging
import numpy as np
from datasets import Dataset
from typing import Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_seed(42)

# Configuration
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = "EleutherAI/pythia-2.8b"
PANDA_DATASET_PATH = os.path.join(
    CUR_DIR_PATH, "dataset", "PANDA-annotated-100k-shard0.jsonl"
)
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "pythia-panda-finetuned")
WANDB_PROJECT = "pythia-panda-finetuning"
WANDB_NAME = "pythia-2.8b-lora-panda-debiasing"
LOGGING_DIR = os.path.join(CUR_DIR_PATH, "logs")

# GPU and training parameters
USE_4BIT = False  # Set to True for 4-bit quantization if you have limited GPU memory
USE_8BIT = True  # Set to True for 8-bit quantization (ignored if USE_4BIT is True)
BATCH_SIZE = 8  # Adjust based on your GPU memory
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_LENGTH = 512

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_panda_dataset(dataset_path: str) -> List[Dict]:
    """Load the PANDA dataset from a JSONL file"""
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_dataset_from_panda(data: List[Dict]) -> Dataset:
    """Convert PANDA data to a HuggingFace Dataset"""
    dataset_dict = {
        "original": [],
        "rewrite": [],
        "selected_word": [],
        "perturbed_category": [],
        "source_data_type": [],
    }

    for item in data:
        dataset_dict["original"].append(item["original"])
        dataset_dict["rewrite"].append(item["rewrite"])
        dataset_dict["selected_word"].append(item["selected_word"])
        dataset_dict["perturbed_category"].append(item["perturbed_category"])
        dataset_dict["source_data_type"].append(item["source_data_type"])

    return Dataset.from_dict(dataset_dict)


def format_for_instruction_tuning(examples, tokenizer):
    """Format PANDA examples for instruction tuning with counterfactual data augmentation"""
    prompts = []

    for original, rewrite, word, category, source in zip(
        examples["original"],
        examples["rewrite"],
        examples["selected_word"],
        examples["perturbed_category"],
        examples["source_data_type"],
    ):
        # Create instruction format
        instruction = f"Rewrite the following text to change '{word}' to a {category} reference while preserving meaning:"
        formatted_prompt = f"<|prompt|>{instruction}\nOriginal: {original}\n<|response|>{rewrite}<|endoftext|>"
        prompts.append(formatted_prompt)

    # Tokenize inputs
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Create labels that are the same as inputs for causal LM training
    tokenized["labels"] = tokenized["input_ids"].clone()

    # Mask the prompt part for loss calculation (we only want to compute loss on the response)
    response_token_id = tokenizer.convert_tokens_to_ids("<|response|>")

    for i, input_ids in enumerate(tokenized["input_ids"]):
        # Find position of response token
        response_pos = (input_ids == response_token_id).nonzero(as_tuple=True)[0]
        if len(response_pos) > 0:
            # Mask everything up to and including the response token
            tokenized["labels"][i, : response_pos[0] + 1] = -100

    return tokenized


def prepare_dataset_for_training(dataset_path: str, tokenizer):
    """Load, prepare, and split PANDA dataset for training"""
    # Load PANDA data
    logger.info(f"Loading PANDA dataset from {dataset_path}")
    panda_data = load_panda_dataset(dataset_path)

    # Convert to HuggingFace Dataset
    panda_dataset = create_dataset_from_panda(panda_data)

    # Split dataset into train and eval (90/10 split)
    panda_dataset = panda_dataset.train_test_split(test_size=0.1, seed=42)

    # Format dataset for instruction tuning
    tokenized_dataset = panda_dataset.map(
        lambda examples: format_for_instruction_tuning(examples, tokenizer),
        batched=True,
        remove_columns=panda_dataset["train"].column_names,
        num_proc=4,  # Use multiple processes for faster preprocessing
    )

    return tokenized_dataset


def compute_metrics(eval_preds):
    """Calculate and return metrics for evaluation"""
    logits, labels = eval_preds

    # Get predictions
    predictions = logits.argmax(axis=-1)

    # Create a mask for tokens we care about (not -100)
    mask = labels != -100

    # Calculate accuracy only on non-masked tokens
    valid_preds = predictions[mask]
    valid_labels = labels[mask]
    accuracy = (valid_preds == valid_labels).mean()

    return {"accuracy": float(accuracy)}


def print_gpu_utilization():
    """Print GPU memory usage for monitoring"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            free = total_memory - reserved
            logger.info(
                f"GPU {i}: Total: {total_memory:.2f}GB | Reserved: {reserved:.2f}GB | Allocated: {allocated:.2f}GB | Free: {free:.2f}GB"
            )
    else:
        logger.info("No GPU available")


def main():
    """Main training function"""
    # Initialize Weights & Biases
    wandb.init(project=WANDB_PROJECT, name=WANDB_NAME)

    # Check available GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print_gpu_utilization()
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix padding token issue
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens if they don't exist
    special_tokens = {
        "additional_special_tokens": ["<|prompt|>", "<|response|>", "<|endoftext|>"]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added_tokens} special tokens to the tokenizer")

    # Load and prepare dataset
    tokenized_dataset = prepare_dataset_for_training(PANDA_DATASET_PATH, tokenizer)
    logger.info(f"Training dataset size: {len(tokenized_dataset['train'])}")
    logger.info(f"Evaluation dataset size: {len(tokenized_dataset['test'])}")

    # Load the model with appropriate quantization
    logger.info(f"Loading model from {MODEL_NAME}")
    load_in_4bit = USE_4BIT
    load_in_8bit = USE_8BIT and not USE_4BIT  # Only use 8-bit if 4-bit is disabled

    # Model loading options
    if load_in_4bit:
        logger.info("Loading model in 4-bit quantization")
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    elif load_in_8bit:
        logger.info("Loading model in 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model in 16-bit (float16)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Print memory usage after model loading
    print_gpu_utilization()

    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Configure LoRA
    logger.info("Configuring LoRA for parameter-efficient fine-tuning")

    # For GPT-NeoX models like Pythia, these are the standard target modules
    target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )

    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb",
        fp16=True,
        save_total_limit=3,
        logging_dir=LOGGING_DIR,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        # Memory optimization
        gradient_checkpointing=True,
        # Avoid memory leaks
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    logger.info("Starting training with LoRA")
    trainer.train()

    # Save the model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
