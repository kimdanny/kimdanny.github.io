# 11830 Ethics, Safety, and Social Impact in NLP and LLMs
# Project: Bias Evaluation on BBQ / StereoSet

This repository evaluates the bias of language models using the **StereoSet** benchmark.

---

### Install Dependencies
Create a **Conda environment** and install dependencies:

```bash
conda create -n bias-bench python=3.8 -y
conda activate bias-bench
pip install -e .
pip install transformers torch tqdm numpy
```

---
## StereoSet

### Run Evaluation
Generate Predictions
```bash
python generate_stereoset_predictions.py
```

The current implementation uses `data/stereoset/test_small.json`. Switch the code to `data/stereoset/test.json` for full StereoSet evaluation.

Evaluate Predictions
```bash
python experiments/stereoset_evaluation.py --persistent_dir $(pwd) --predictions_dir $(pwd) --predictions_file predictions.json --output_file results.json
```

### Understanding the Results
After running the evaluation, check `results.json`:
```json
{
  "intrasentence": {
    "race": {
      "Count": 10,
      "LM Score": 78.4,
      "SS Score": 65.2,
      "ICAT Score": 35.6
    },
    "overall": {
      "Count": 10,
      "LM Score": 76.2,
      "SS Score": 60.8,
      "ICAT Score": 40.1
    }
  }
}
```

---
## BBQ

### Run Evaluation

Format:
```bash
python bbq/evaluate.py --model_name {model name} --bbq_dir bbq/data --results_path bbq/results_{model name}.txt --predictions_dir bbq/results
```
Example:
```bash
python bbq/evaluate.py --model_name gpt2 --bbq_dir bbq/data --results_path bbq/results_gpt2.txt --predictions_dir bbq/results
python bbq/evaluate.py --model_name state-spaces/mamba2-2.7b --bbq_dir bbq/data --results_path bbq/results_mamba2-2.7b.txt --predictions_dir bbq/results
```

### Understanding the Results
After running the evaluation, check `bbq/results_{model name}.txt`:
```txt
BBQ Evaluation Results for Model: gpt2
================================================================================

Overall Accuracy: 0.4000 (4/10)

Category-wise Results:
--------------------------------------------------------------------------------
Category: Age
  Accuracy: 0.4000 (4/10)
  Context Condition Accuracies:
    ambig: 0.6000
    disambig: 0.2000
  Question Polarity Accuracies:
    neg: 0.3333
    nonneg: 0.5000
--------------------------------------------------------------------------------
...
```