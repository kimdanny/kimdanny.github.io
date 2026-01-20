#!/usr/bin/env python
# Inference script for fine-tuned Pythia-2.8B model with LoRA merging

import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Path to your fine-tuned model
MODEL_PATH = "tuned_models/pythia-panda-instruction-tuned"
BASE_MODEL = "EleutherAI/pythia-2.8b"  # Original base model
MERGED_MODEL_PATH = "tuned_models/pythia-panda-instruction-tuned-merged-model"  # Where to save the merged model
MERGE_ALPHA = 0.001  # Lower alpha means less influence from the fine-tuned weights

def manual_weight_merging(base_model, peft_model_path, alpha=MERGE_ALPHA):
    """
    Manually merge LoRA weights with base model at a specified alpha
    
    Args:
        base_model: The base model to merge into
        peft_model_path: Path to the LoRA adapter
        alpha: Weight to apply to the adapter (0-1), 0 = all base model, 1 = all adapter
        
    Returns:
        The merged model
    """
    print(f"Performing manual weight merging with alpha={alpha}")
    
    # Load adapter config to get target modules and LoRA parameters
    adapter_config = PeftConfig.from_pretrained(peft_model_path)
    
    # Get target modules from the config
    target_modules = adapter_config.target_modules
    r = adapter_config.r  # LoRA rank
    lora_alpha = adapter_config.lora_alpha  # Original scaling factor
    
    print(f"LoRA config: target_modules={target_modules}, r={r}, lora_alpha={lora_alpha}")
    
    # This is the scaling used in the LoRA paper (alpha/r)
    lora_scaling = lora_alpha / r
    
    # Create a deepcopy of the base model's state dict as the starting point
    base_state_dict = base_model.state_dict()
    merged_state_dict = base_state_dict.copy()
    
    # Load LoRA weights from adapter file
    # We need to handle safetensors file format
    from safetensors.torch import load_file
    
    adapter_model_path = os.path.join(peft_model_path, "adapter_model.safetensors")
    print(f"Loading adapter weights from: {adapter_model_path}")
    adapter_state_dict = load_file(adapter_model_path)
    
    # Pattern to identify LoRA A and B matrices in adapter state dict
    lora_a_pattern = re.compile(r'.*\.lora_A\.\w+$')
    lora_b_pattern = re.compile(r'.*\.lora_B\.\w+$')
    
    # Find all A matrices and their corresponding B matrices
    lora_a_keys = [k for k in adapter_state_dict.keys() if lora_a_pattern.match(k)]
    merged_weights = {}
    
    # Get the parameter names in the base model that need to be updated
    for a_key in lora_a_keys:
        # Extract the prefix and the matrix name (weight/bias)
        prefix = a_key.split('.lora_A.')[0]
        matrix_name = a_key.split('.')[-1]  # should be 'weight' or 'bias'
        
        # Construct the corresponding B key
        b_key = f"{prefix}.lora_B.{matrix_name}"
        
        if b_key in adapter_state_dict:
            # The parameter name in the base model
            param_name = f"{prefix}.{matrix_name}"
            
            if param_name in base_state_dict:
                # Get weights
                a_weight = adapter_state_dict[a_key]
                b_weight = adapter_state_dict[b_key]
                
                # Compute LoRA weight adjustment (B×A)
                lora_adjustment = torch.matmul(b_weight, a_weight) * lora_scaling
                
                # Adjust the weight by the specified alpha
                base_weight = base_state_dict[param_name]
                merged_weights[param_name] = base_weight + alpha * lora_adjustment
                
                print(f"Merged {param_name} with LoRA weights from {a_key} and {b_key}")
    
    # Update the state dict with the merged weights
    for param_name, merged_weight in merged_weights.items():
        merged_state_dict[param_name] = merged_weight
    
    # Load the merged state dict into the model
    base_model.load_state_dict(merged_state_dict)
    
    print("Manual weight merging completed")
    return base_model

def load_and_merge_model(save_merged=True):
    """Load, merge, and optionally save the model"""
    # First load the tokenizer with special tokens
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Ensure the tokenizer has special tokens
    special_tokens = {"additional_special_tokens": ["<|prompt|>", "<|response|>", "<|endoftext|>"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_tokens} special tokens to tokenizer")
    
    # Fix padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model with the SAME tokenizer vocabulary size
    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Resize the token embeddings of the model to match the tokenizer
    print("Resizing token embeddings to match tokenizer")
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Try to use the PEFT merger with alpha parameter
    try:
        print("Attempting to use PEFT's merge_and_unload with alpha parameter")
        peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        merged_model = peft_model.merge_and_unload(alpha=MERGE_ALPHA)
        print("Successfully used PEFT's built-in merging")
    except (TypeError, AttributeError):
        print("PEFT merge_and_unload with alpha parameter not supported")
        print("Falling back to manual weight merging")
        # Use our manual merging function
        merged_model = manual_weight_merging(base_model, MODEL_PATH, alpha=MERGE_ALPHA)
    
    # Save the merged model if requested
    if save_merged and MERGED_MODEL_PATH:
        print(f"Saving merged model to: {MERGED_MODEL_PATH}")
        merged_model.save_pretrained(MERGED_MODEL_PATH)
        tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    return merged_model, tokenizer

def load_merged_model(merged_model_path=MERGED_MODEL_PATH):
    """Load an already merged model if it exists"""
    if os.path.exists(merged_model_path):
        print(f"Loading pre-merged model from: {merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        
        # Ensure the tokenizer has the needed special tokens
        special_tokens = {"additional_special_tokens": ["<|prompt|>", "<|response|>", "<|endoftext|>"]}
        tokenizer.add_special_tokens(special_tokens)
        
        # Fix padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Resize model embeddings if needed
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    else:
        print("Pre-merged model not found. Creating merged model...")
        return load_and_merge_model()

def generate_text(model, tokenizer, prompt, use_special_tokens=False, max_length=200):
    """Generate text based on a prompt with optional special token formatting"""
    if use_special_tokens and "<|prompt|>" not in prompt:
        # Format with special tokens
        formatted_prompt = f"<|prompt|>{prompt}\n<|response|>"
    else:
        # Use as-is if special tokens are already included or not requested
        formatted_prompt = prompt
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Set up appropriate generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Add eos token for special token formatted prompts
    if use_special_tokens:
        gen_kwargs["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, **gen_kwargs)
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=not use_special_tokens)
    
    # Extract only the response part if using special tokens
    if use_special_tokens:
        response_start = generated_text.find("<|response|>") + len("<|response|>")
        response_end = generated_text.find("<|endoftext|>", response_start)
        if response_end == -1:
            response = generated_text[response_start:]
        else:
            response = generated_text[response_start:response_end]
        return response.strip()
    else:
        return generated_text

def run_inference_interactive():
    """Run an interactive inference session"""
    # Load model and tokenizer
    model, tokenizer = load_merged_model()
    model.eval()
    
    print("\n=== Pythia Merged Model Inference ===")
    print(f"Using merge alpha: {MERGE_ALPHA}")
    print("Type 'quit' to exit")
    print("\nOptions:")
    print("1: General Q&A (without special tokens)")
    print("2: Demographic rewriting (with special tokens)")
    
    while True:
        print("\n--- New Query ---")
        mode = input("Enter mode (1 or 2): ")
        if mode.lower() == 'quit':
            break
            
        use_special_tokens = (mode == '2')
        
        if mode == '1':
            prompt = input("Enter your question: ")
        elif mode == '2':
            instruction = input("Enter instruction (e.g., 'Rewrite the following text to change 'John' to a Woman reference'): ")
            original = input("Enter original text: ")
            prompt = f"Rewrite the following text to change '{instruction}' while preserving meaning:\nOriginal: {original}"
        else:
            print("Invalid mode. Please enter 1 or 2.")
            continue
            
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating response...")
        response = generate_text(model, tokenizer, prompt, use_special_tokens=use_special_tokens)
        print(f"\nResponse: {response}")

def test_examples():
    """Run standard examples to test general capabilities"""
    model, tokenizer = load_merged_model()
    model.eval()
    
    # General knowledge questions (without special tokens)
    general_questions = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Who wrote Romeo and Juliet?",
        "What is the theory of relativity?",
        "What are the major organs in the human body?"
    ]
    
    # Demographic rewriting tasks (with special tokens)
    demographic_tasks = [
        "Rewrite the following text to change 'businessman' to a Woman reference while preserving meaning:\nOriginal: The businessman closed the deal after months of negotiations.",
        "Rewrite the following text to change 'actress' to a Non-Binary reference while preserving meaning:\nOriginal: The actress delivered a stunning performance in the latest movie.",
        "Rewrite the following text to change 'fireman' to a Woman reference while preserving meaning:\nOriginal: The fireman rescued the cat from the tree.",
    ]
    
    # General demographic understanding (without special tokens)
    demographic_questions = [
        "How can we ensure diverse representation in media?",
        "What are some examples of gender-neutral language?",
        "Why is diversity important in the workplace?",
    ]
    
    # Test general knowledge questions (without special tokens)
    print("\n=== GENERAL KNOWLEDGE QUESTIONS ===")
    for question in general_questions:
        print(f"\nQ: {question}")
        response = generate_text(model, tokenizer, question, use_special_tokens=False)
        print(f"A: {response}")
        print("-" * 50)
    
    # Test demographic rewriting (with special tokens)
    print("\n=== DEMOGRAPHIC REWRITING TASKS ===")
    for task in demographic_tasks:
        print(f"\nTask: {task}")
        response = generate_text(model, tokenizer, task, use_special_tokens=True)
        print(f"Result: {response}")
        print("-" * 50)
    
    # Test demographic understanding (without special tokens)
    print("\n=== DEMOGRAPHIC UNDERSTANDING QUESTIONS ===")
    for question in demographic_questions:
        print(f"\nQ: {question}")
        response = generate_text(model, tokenizer, question, use_special_tokens=False)
        print(f"A: {response}")
        print("-" * 50)

if __name__ == "__main__":
    # Choose one:
    test_examples()      # Run predefined examples
    # run_inference_interactive()  # Interactive mode