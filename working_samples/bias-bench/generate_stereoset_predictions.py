import json
import torch
import torch.nn.functional as F
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from peft import PeftModel, PeftConfig
import os

# Model name
# MODEL_NAME = "state-spaces/mamba2-2.7b"
MODEL_NAME = sys.argv[1]
print(f"Model: {MODEL_NAME}")



# Load model and tokenizer
print("Loading model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda"
dtype = torch.bfloat16
is_mamba = MODEL_NAME.startswith("state-spaces/mamba") or MODEL_NAME.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(MODEL_NAME, device=device, dtype=dtype)

else: 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, torch_dtype=dtype)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)


model.eval()

# Load StereoSet data
DATA_PATH = "data/stereoset/test.json"
# DATA_PATH = "data/stereoset/test_small.json"
PREDICTIONS_PATH = "predictionsStereoSet_PythiaFT.json"
#PREDICTIONS_PATH = sys.argv[2]

print(f"Loading StereoSet data from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    stereoset_data = json.load(f)

# Extract intrasentence examples
sentences = []
for example in stereoset_data["data"]["intrasentence"]:
    for sentence in example["sentences"]:
        sentences.append({
            "id": sentence["id"],
            "text": sentence["sentence"]
        })

# Generate predictions
print("Running inference on StereoSet sentences...")
predictions = []

for sentence in tqdm(sentences, total=len(sentences)):
    # print(sentence["text"])
    
    inputs = tokenizer("<|endoftext|>" + sentence["text"], return_tensors="pt").to(device)
    # print(inputs)
    
    # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    
    with torch.no_grad():
        if isinstance(model, MambaLMHeadModel): 
            outputs = model(input_ids=inputs['input_ids'])
        else:
            outputs = model(**inputs)

    # Using the mean logit value as a proxy for a score
    # print(outputs.logits.shape)
    # score = outputs.logits.mean().item()
    
    # Shift logits and labels so we're predicting token t given tokens < t
    shift_logits = outputs.logits[:, :-1, :]  # ignore last token's prediction
    shift_labels = inputs['input_ids'][:, 1:]   # ignore first token (e.g., BOS)
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum log-probs for total sentence log-probability
    total_log_prob = token_log_probs.sum().item()
    avg_log_prob = token_log_probs.mean().item()  # normalized per-token

    predictions.append({
        "id": sentence["id"],
        "score": avg_log_prob,
        "total_log_prob": total_log_prob,
        "length": shift_labels.shape[1]
    })
    # break

# Save predictions to JSON file
print(f"Saving predictions to {PREDICTIONS_PATH}...")
with open(PREDICTIONS_PATH, "w") as f:
    json.dump({"intrasentence": predictions}, f, indent=4)

print("Prediction generation completed!")
