import pandas as pd
from nltk import word_tokenize
from nltk.translate import meteor
from rouge_score import rouge_scorer


def calc_meteor(reference: str, candidate: str):
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    meteor_score = meteor([candidate], reference)
    return meteor_score


def rogue(reference: str, candidate: str) -> dict[str : rouge_scorer.RougeScorer]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    return scorer.score(reference, candidate)


def exact_match(reference: str, candidate: str) -> bool:
    return reference == candidate


def f1(reference: str, candidate: str) -> float:
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    common_tokens = set(reference_tokens) & set(candidate_tokens)
    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(candidate_tokens)
    recall = len(common_tokens) / len(reference_tokens)

    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    df = pd.read_csv(
        "test_llama.csv", dtype={"answer": "string", "predicted_answer": "string"}
    ).dropna(subset=["answer", "predicted_answer"])
    df.loc["f1"] = df.apply(lambda row: f1(row["answer"], row["predicted_answer"]), axis=1)
    df.loc[:, "exact_match"] = df.apply(
        lambda row: exact_match(row["answer"], row["predicted_answer"]), axis=1
    )
    df.loc[:, "rouge_1"] = df.apply(
        lambda row: rogue(row["answer"], row["predicted_answer"])["rouge1"].fmeasure,
        axis=1,
    )
    df.loc[:, "rouge_2"] = df.apply(
        lambda row: rogue(row["answer"], row["predicted_answer"])["rouge2"].fmeasure,
        axis=1,
    )
    df.loc[:, "meteor"] = df.apply(
        lambda row: calc_meteor(row["answer"], row["predicted_answer"]), axis=1
    )
    df.to_csv("test_eval.csv", index=False)
