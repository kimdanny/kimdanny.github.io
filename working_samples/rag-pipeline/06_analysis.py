import pandas as pd
from pathlib import Path
from tqdm import tqdm
from word2number import w2n
from dateutil.parser import parse, ParserError
from eval_utils import evaluate, normalize_answer
from nltk.tag import StanfordNERTagger
from itertools import product


def _filename(prompt_restrict, numbered, retriever_k):
    if prompt_restrict == "none":
        folder = "no_instruct"
    elif prompt_restrict == "basic":
        folder = "no_repeat"
    elif prompt_restrict == "emphasis":
        folder = "original"

    if retriever_k == 0:
        fname = "system_out_False_0_None.txt"
    else:
        if numbered:
            fname = f"system_out_{folder.replace('no_', '')}_{retriever_k}_numbered.txt"
        else:
            fname = f"system_out_True_{retriever_k}_simple.txt"
    return Path("data/experiments") / folder / fname


def _is_numeric(x):
    for word in normalize_answer(x["Prediction"]).split():
        try:
            w2n.word_to_num(word)
            return True
        except ValueError:
            continue
    return x["Prediction"] == normalize_answer(x["Answer"])


def _is_date(x):
    try:
        parse(normalize_answer(x["Prediction"]))
        return True
    except ParserError:
        return x["Prediction"] == normalize_answer(x["Answer"])


def _is_yesno(x):
    return normalize_answer(x["Prediction"]) in ("yes", "no")


def _is_person(x):
    return ("PERSON" in [item[1] for item in x["ner"]]) or normalize_answer(x["Prediction"]) == normalize_answer(x["Answer"])


def _is_org(x):
    return ("ORGANIZATION" in [item[1] for item in x["ner"]]) or normalize_answer(x["Prediction"]) == normalize_answer(x["Answer"])


def _is_loc(x):
    return ("LOCATION" in [item[1] for item in x["ner"]]) or normalize_answer(x["Prediction"]) == normalize_answer(x["Answer"])


if __name__ == "__main__":
    df = pd.read_csv("data/datasheet.csv")
    df["Prediction"] = _filename(prompt_restrict="emphasis", numbered=False, retriever_k=2).read_text().split("\n")
    st = StanfordNERTagger(
        "data/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz",
        "data/stanford-ner-2020-11-17/stanford-ner.jar", "UTF-8"
    )

    result_list = []
    for i, (p, n, k) in enumerate(tqdm(product(("none", "basic", "emphasis"), (True, False), range(6)))):
        results = {"prompt_restrict": p, "numbered": n, "retriever_k": k}
        df["Prediction"] = _filename(**results).read_text().split("\n")
        df["ner"] = df["Prediction"].map(lambda x: st.tag(normalize_answer(x).split()))

        for name, func in (
            ("YesNo", _is_yesno), ("Date", _is_date), ("Numeric", _is_numeric),
            ("Person", _is_person), ("Organization", _is_org), ("Location", _is_loc),
        ):
            _df = df[df["Answer Type"] == name]
            for key, val in evaluate(_df["Answer"].tolist(), _df["Prediction"].tolist()).items():
                results[f"class/{name}/{key}"] = val

            results[f"classtest/{name}/exact_match"] = _df.apply(func, axis=1).mean()

        for hop in df["Hop"].unique():
            for key, val in evaluate(df["Answer"].tolist(), df["Prediction"].tolist()).items():
                results[f"hop/{hop}/{key}"] = val

        for hop in df["Hop"].unique():
            for key, val in evaluate(df[df["Hop"] == hop]["Answer"].tolist(), df[df["Hop"] == hop]["Prediction"].tolist()).items():
                results[f"hop/{hop}/{key}"] = val

        _df = df[df["Topic"] == 1]
        for key, val in evaluate(_df["Answer"].tolist(), _df["Prediction"].tolist()).items():
            results[f"topic/single/{key}"] = val
        _df = df[df["Topic"] > 1]
        for key, val in evaluate(_df["Answer"].tolist(), _df["Prediction"].tolist()).items():
            results[f"topic/multi/{key}"] = val

        for key, val in evaluate(df["Answer"].tolist(), df["Prediction"].tolist()).items():
            results[f"overall/{key}"] = val
        results["overall/word_length"] = df["Prediction"].map(lambda x: len(normalize_answer(x).split())).mean()
        results["overall/char_length"] = df["Prediction"].map(lambda x: len(normalize_answer(x))).mean()

        print(results)
        result_list.append(results)

    pd.DataFrame(result_list).to_csv("data/metrics.csv", index=False)
