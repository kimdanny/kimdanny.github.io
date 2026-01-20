from collections import Counter
import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return " ".join(
            [word for word in text.split() if word not in ["a", "an", "the"]]
        )

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punct(text):
        return "".join([char for char in text if char.isalnum() or char == " "])

    def extract_binary(text):
        if text.startswith("yes") or text.startswith("no"):
            return text.split()[0]
        return text

    return white_space_fix(extract_binary(remove_articles(remove_punct(s.lower()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate(gold_answers, predictions):
    exact_match = 0
    total = len(gold_answers)
    f1_scores = []
    recalls = []

    for ground_truth, prediction in zip(gold_answers, predictions):
        exact_match += int(exact_match_score(prediction, ground_truth))
        f1, recall = f1_score(prediction, ground_truth)
        f1_scores.append(f1)
        recalls.append(recall)

    exact_match = 100.0 * exact_match / total
    macro_f1 = np.mean(f1_scores) * 100.0
    answer_recall = np.mean(recalls) * 100.0

    return {
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "answer_recall": answer_recall,
    }


# # Example usage:
# gold_answers = ["Paris", "42", "Tom Hanks"] # Ground truth answers
# predictions = ["paris", "42", "Tom Hanks"] # Model's predictions

# # Evaluate performance
# results = evaluate(gold_answers, predictions)
# print("Exact Match: {:.2f}".format(results['exact_match']))
# print("Macro F1: {:.2f}".format(results['macro_f1']))
# print("Answer Recall: {:.2f}".format(results['answer_recall']))
