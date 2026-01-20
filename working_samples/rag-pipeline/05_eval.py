"""
Evaluate the model's predictions against the gold answers
Utilizes functions from eval_utils.py
"""

from eval_utils import *
from typing import List
import csv
import argparse
import re
import os


def load_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        data = [line.strip() for line in file.readlines()]

    return data


def main(args):
    gold_answers = load_file(args.gold_answers)
    predictions = load_file(args.predictions)

    pattern = r"system_out_original_(?P<k>\d+)_(?P<document_format>[^.]+)\.txt"

    match = re.search(pattern, args.predictions)

    k = match.group("k")
    document_format = match.group("document_format")

    results = evaluate(gold_answers, predictions)

    results_file_path = "results.csv"
    file_exists = (
        os.path.isfile(results_file_path) and os.path.getsize(results_file_path) > 0
    )

    with open(results_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["k", "doc_prompt_format", "Exact Match", "Macro F1", "Answer Recall"]
            )
        writer.writerow(
            [
                str(k),
                document_format,
                results["exact_match"],
                results["macro_f1"],
                results["answer_recall"],
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_answers",
        type=str,
        help="Path to the file containing the gold answers",
    )

    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to the file containing the model's predictions",
    )

    args = parser.parse_args()

    main(args)
