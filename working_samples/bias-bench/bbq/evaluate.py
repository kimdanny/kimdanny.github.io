import json
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mamba models on the BBQ dataset")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-2.8b",
                        help="Model name or path")
    parser.add_argument("--bbq_dir", type=str, default="bbq/data",
                        help="Directory containing BBQ JSONL files")
    parser.add_argument("--results_path", type=str, default="bbq/results.txt",
                        help="Path to save evaluation results")
    parser.add_argument("--predictions_dir", type=str, default="bbq/results",
                        help="Directory to save prediction JSONL files")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    is_mamba = model_name.startswith("state-spaces/mamba") or model_name.startswith("state-spaces/transformerpp")
    
    if is_mamba:
        # If <ModuleNotFoundError: No module named 'mamba_ssm'> cannot be solved, add the following line.
        # from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        # And remove it from the top.
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device}, torch_dtype=dtype)
    
    model.eval()
    return model, tokenizer, device


def load_bbq_category_data(file_path):
    """Load BBQ data from a category-specific JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def get_choice_scores(model, tokenizer, prompt, choices, device):
    """Get scores for each answer choice."""
    scores = []
    
    for choice in choices:
        # Create input with the choice appended
        input_text = f"{prompt} {choice}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get outputs from model
            # If <ModuleNotFoundError: No module named 'mamba_ssm'> cannot be solved, modify the following line to
            # if False:
            if isinstance(model, MambaLMHeadModel):
                outputs = model(input_ids=inputs['input_ids'])
            else:
                outputs = model(**inputs)
            
            # Calculate choice score (mean logit as a simple metric)
            score = outputs.logits.mean().item()
            scores.append(score)
    
    return scores


def evaluate_example(model, tokenizer, example, device):
    """Evaluate a single BBQ example."""
    context = example.get("context", "")
    question = example["question"]
    
    # Format the prompt based on whether context exists
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"
    
    # Get answer choices
    choices = [example["ans0"], example["ans1"], example["ans2"]]
    
    # Get scores for each choice
    scores = get_choice_scores(model, tokenizer, prompt, choices, device)
    
    # Find predicted answer (highest score)
    predicted_index = np.argmax(scores)
    predicted_answer = choices[predicted_index]
    
    # Get ground truth
    correct_index = example["label"]
    
    # Create a copy of the original example and add predictions
    prediction = example.copy()
    prediction["predicted_index"] = int(predicted_index)
    prediction["predicted_answer"] = predicted_answer
    prediction["is_correct"] = bool(predicted_index == correct_index)
    prediction["choice_scores"] = [float(score) for score in scores]
    
    # Create a simplified result for metrics calculation
    result = {
        "example_id": example["example_id"],
        "category": example["category"],
        "subcategory": example["additional_metadata"]["subcategory"],
        "context_condition": example["context_condition"],
        "question_polarity": example["question_polarity"],
        "predicted_index": int(predicted_index),
        "correct_index": correct_index,
        "is_correct": bool(predicted_index == correct_index),
        "choice_scores": [float(score) for score in scores],
        "stereotyped_groups": example["additional_metadata"]["stereotyped_groups"]
    }
    
    return result, prediction


def evaluate_category(model, tokenizer, category_examples, device):
    """Evaluate all examples in a category."""
    results = []
    predictions = []
    
    for example in tqdm(category_examples, desc=f"Evaluating {category_examples[0]['category']}"):
        result, prediction = evaluate_example(model, tokenizer, example, device)
        results.append(result)
        predictions.append(prediction)
    
    return results, predictions


def calculate_category_metrics(results):
    """Calculate metrics for a category."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0
    
    # Calculate metrics by context condition
    context_conditions = {}
    for result in results:
        condition = result["context_condition"]
        if condition not in context_conditions:
            context_conditions[condition] = {"total": 0, "correct": 0}
        
        context_conditions[condition]["total"] += 1
        if result["is_correct"]:
            context_conditions[condition]["correct"] += 1
    
    condition_accuracies = {
        cond: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for cond, stats in context_conditions.items()
    }
    
    # Calculate metrics by question polarity
    polarities = {}
    for result in results:
        polarity = result["question_polarity"]
        if polarity not in polarities:
            polarities[polarity] = {"total": 0, "correct": 0}
        
        polarities[polarity]["total"] += 1
        if result["is_correct"]:
            polarities[polarity]["correct"] += 1
    
    polarity_accuracies = {
        pol: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for pol, stats in polarities.items()
    }
    
    return {
        "category": results[0]["category"] if results else "Unknown",
        "overall_accuracy": accuracy,
        "total_examples": total,
        "total_correct": correct,
        "condition_accuracies": condition_accuracies,
        "polarity_accuracies": polarity_accuracies
    }


def write_results_to_file(all_metrics, model_name, output_path):
    """Write evaluation results to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"BBQ Evaluation Results for Model: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        # Write overall metrics across all categories
        total_examples = sum(m["total_examples"] for m in all_metrics)
        total_correct = sum(m["total_correct"] for m in all_metrics)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_examples})\n\n")
        
        # Write detailed metrics for each category
        f.write("Category-wise Results:\n")
        f.write("-"*80 + "\n")
        
        for metrics in all_metrics:
            category = metrics["category"]
            acc = metrics["overall_accuracy"]
            correct = metrics["total_correct"]
            total = metrics["total_examples"]
            
            f.write(f"Category: {category}\n")
            f.write(f"  Accuracy: {acc:.4f} ({correct}/{total})\n")
            
            # Write condition-based accuracies
            f.write("  Context Condition Accuracies:\n")
            for cond, cond_acc in metrics["condition_accuracies"].items():
                f.write(f"    {cond}: {cond_acc:.4f}\n")
            
            # Write polarity-based accuracies
            f.write("  Question Polarity Accuracies:\n")
            for pol, pol_acc in metrics["polarity_accuracies"].items():
                f.write(f"    {pol}: {pol_acc:.4f}\n")
            
            f.write("-"*80 + "\n")
    
    print(f"Results saved to {output_path}")


def save_predictions_to_jsonl(predictions, model_name, category_name, predictions_dir):
    """Save category predictions to a JSONL file."""
    # Create the directory structure
    output_dir = os.path.join(predictions_dir, model_name.replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output path
    output_path = os.path.join(output_dir, f"preds_{category_name}.jsonl")
    
    # Define a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # Write predictions to JSONL file
    with open(output_path, 'w') as f:
        for prediction in predictions:
            # Convert any NumPy types to Python native types
            f.write(json.dumps(prediction, cls=NumpyEncoder) + '\n')
    
    print(f"Predictions for {category_name} saved to {output_path}")
import random

def main():
    args = parse_args()
    
    # Extract model name for directory naming
    #model_name_dir = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    
    # Load model and tokenizer
    #model, tokenizer, device = load_model_and_tokenizer(args.model_name)
    
    # Get all category files
    category_files = [f for f in os.listdir(args.bbq_dir) if f.endswith('.jsonl')]
    print(f"Found {len(category_files)} category files in {args.bbq_dir}")
    
    all_metrics = []
    #category_files = ['Race_x_SES.jsonl', 'Gender_identity.jsonl', 'Nationality.jsonl', 'Religion.jsonl', 'Physical_appearance.jsonl']
    # Evaluate each category
    """  for category_file in category_files:
        
        file_path = os.path.join(args.bbq_dir, category_file)
        category_name = os.path.splitext(category_file)[0]
        
        print(f"Processing category: {category_name}")
        
        # Load category data
        category_examples = load_bbq_category_data(file_path)
        random.shuffle(category_examples)
        category_examples = category_examples[:len(category_examples)//4]  # Limit to 1000 examples for testing
        print(f"Loaded {len(category_examples)} examples")
        
        # Evaluate examples in this category
        category_results, category_predictions = evaluate_category(model, tokenizer, category_examples, device)
        
        # Calculate metrics for this category
        category_metrics = calculate_category_metrics(category_results)
        all_metrics.append(category_metrics)
        
        # Save predictions for this category
        save_predictions_to_jsonl(category_predictions, model_name_dir, category_name, args.predictions_dir)
        
        # Print summary for this category
        print(f"Category: {category_name}, Accuracy: {category_metrics['overall_accuracy']:.4f}")
    """
    # Write all results to file
    all_metrics = []
    for category_name in category_files:
        with open(f"/ocean/projects/cis250019p/sfragara/bias-bench/bbq/results/mamba-2.8b/preds_{category_name}", "r") as f:
            category_results = []
            for line in f:
                x = json.loads(line.strip())
                category_results.append(x)
        metrics = calculate_category_metrics(category_results)
        all_metrics.append(metrics)
    #print(metrics)
    write_results_to_file(all_metrics, args.model_name, args.results_path)


if __name__ == "__main__":
    main()