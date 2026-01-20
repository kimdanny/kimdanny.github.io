#!/usr/bin/env python
# Debiasing fine-tuning script for Pythia-2.8B model with PANDA dataset

import os
import json
import torch
import wandb
import logging
import numpy as np
import random
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
    CUR_DIR_PATH, "dataset"
)
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "tuned_models", "pythia-panda-finetuned")
WANDB_PROJECT = "pythia-debiasing"
WANDB_NAME = "pythia-2.8b-panda-general-debiasing"
LOGGING_DIR = os.path.join(CUR_DIR_PATH, "logs")

# GPU and training parameters
USE_8BIT = True
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6  # Lower learning rate for more subtle changes
NUM_EPOCHS = 1
MAX_LENGTH = 512

# LoRA parameters - smaller values for less aggressive fine-tuning
LORA_RANK = 4  # Smaller rank
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training formats
FORMATS = [
    "diversity_exposure",
    "demographic_examples",
    "knowledge_exposure",
    "side_by_side_comparison",
    "demographic_pairs"
]

def load_panda_dataset(dataset_dir: str) -> List[Dict]:
    """
    Load the PANDA dataset from multiple JSONL files (shards)
    Randomly selects 20% of data from each shard to maintain total dataset size
    
    Args:
        dataset_dir: Directory containing the PANDA dataset shards
        
    Returns:
        List of combined examples from all shards
    """
    data = []
    
    # Define shard files
    shard_files = [
        os.path.join(dataset_dir, f"PANDA-annotated-100k-shard{i}.jsonl") 
        for i in range(5)
    ]
    
    logger.info(f"Loading PANDA dataset from {len(shard_files)} shards")
    
    # Process each shard
    for shard_file in shard_files:
        shard_data = []
        
        # Load all data from this shard
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                shard_data.append(json.loads(line))
        
        # Randomly select 20% of examples from this shard
        num_to_select = max(1, int(len(shard_data) * 0.2))
        selected_examples = random.sample(shard_data, num_to_select)
        
        # Add selected examples to the combined dataset
        data.extend(selected_examples)
        
        logger.info(f"Selected {len(selected_examples)} examples from {shard_file}")
    
    # Shuffle the combined dataset
    random.shuffle(data)
    logger.info(f"Total dataset size after sampling: {len(data)} examples")
    
    return data

def create_diverse_training_examples(data: List[Dict]) -> List[Dict]:
    """Convert PANDA data into diverse training examples for debiasing"""
    training_examples = []
    
    # Group examples by original text for side-by-side examples
    grouped_examples = {}
    for item in data:
        original = item["original"]
        if original not in grouped_examples:
            grouped_examples[original] = []
        grouped_examples[original].append(item)
    
    # Create diverse training examples
    for item in data:
        original = item["original"]
        rewrite = item["rewrite"]
        word = item["selected_word"]
        category = item["perturbed_category"]
        
        # Format 1: Simple demographic exposure - expose the model to diverse content
        diversity_example = {
            "format": "diversity_exposure",
            "text": f"{rewrite}"
        }
        training_examples.append(diversity_example)
        
        # Format 2: Side-by-side demographic examples - show multiple variations
        if len(grouped_examples[original]) > 1:
            # Get all rewrites for this original text
            rewrites = [ex["rewrite"] for ex in grouped_examples[original]]
            categories = [ex["perturbed_category"] for ex in grouped_examples[original]]
            
            # Create side-by-side examples with 2-3 variations
            sample_size = min(3, len(rewrites))
            sample_indices = random.sample(range(len(rewrites)), sample_size)
            
            comparison_text = f"Original: {original}\n\n"
            for i in sample_indices:
                comparison_text += f"{categories[i]} version: {rewrites[i]}\n\n"
            
            side_by_side = {
                "format": "side_by_side_comparison",
                "text": comparison_text
            }
            training_examples.append(side_by_side)
        
        # Format 3: Demographic knowledge exposure - teach about demographics without explicit rewriting
        knowledge_text = f"The statement '{original}' includes a reference to {word}. "\
                         f"This can be represented in different ways to be inclusive of various demographics, "\
                         f"including {category} perspectives."
        
        knowledge_example = {
            "format": "knowledge_exposure",
            "text": knowledge_text
        }
        training_examples.append(knowledge_example)
        
        # Format 4: Demographic examples without instructions
        example_text = f"Here's a statement that uses inclusive language for {category} demographics: {rewrite}"
        
        demographic_example = {
            "format": "demographic_examples",
            "text": example_text
        }
        training_examples.append(demographic_example)
        
        # Format 5: Simple pairs of original and rewritten text
        pair_text = f"Text: {original}\nMore inclusive version: {rewrite}"
        
        pair_example = {
            "format": "demographic_pairs",
            "text": pair_text
        }
        training_examples.append(pair_example)
    
    # Shuffle and return examples
    random.shuffle(training_examples)
    return training_examples

def prepare_dataset_for_training(dataset_path: str, tokenizer):
    """Load, prepare, and split PANDA dataset for training"""
    # Load PANDA data
    logger.info(f"Loading PANDA dataset from {dataset_path}")
    panda_data = load_panda_dataset(dataset_path)
    
    # Create diverse training examples
    logger.info("Creating diverse training examples")
    training_examples = create_diverse_training_examples(panda_data)
    logger.info(f"Created {len(training_examples)} training examples")
    
    # Create a HuggingFace dataset
    dataset = Dataset.from_dict({"text": [example["text"] for example in training_examples],
                               "format": [example["format"] for example in training_examples]})
    
    # Split dataset into train and eval (90/10 split)
    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Function to tokenize examples
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    # Tokenize datasets
    tokenized_dataset = {}
    for split in ["train", "test"]:
        # Tokenize
        tokenized = tokenize_function(dataset_split[split])
        
        # Set labels for causal language modeling (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Convert to Dataset
        tokenized_dataset[split] = Dataset.from_dict(tokenized)
    
    # Log stats on formats
    format_counts = {}
    for fmt in dataset_split["train"]["format"]:
        if fmt not in format_counts:
            format_counts[fmt] = 0
        format_counts[fmt] += 1
    
    logger.info("Training example format distribution:")
    for fmt, count in format_counts.items():
        logger.info(f"  {fmt}: {count} examples")
    
    return tokenized_dataset

def main():
    """Main training function"""
    # Initialize Weights & Biases
    wandb.init(project=WANDB_PROJECT, name=WANDB_NAME)

    # Check available GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix padding token issue
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset - no special tokens this time
    tokenized_dataset = prepare_dataset_for_training(PANDA_DATASET_PATH, tokenizer)
    logger.info(f"Training dataset size: {len(tokenized_dataset['train'])}")
    logger.info(f"Evaluation dataset size: {len(tokenized_dataset['test'])}")

    # Load the model with appropriate quantization
    logger.info(f"Loading model from {MODEL_NAME}")
    
    if USE_8BIT:
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,  # Longer warmup
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb",
        fp16=True,
        save_total_limit=3,
        logging_dir=LOGGING_DIR,
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
    )

    # Start training
    logger.info("Starting training for general debiasing")
    trainer.train()

    # Save the model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()