import re

import numpy as np
import pandas as pd

CARDINAL_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_pipe_between_words = "|".join(CARDINAL_TO_DIGIT.keys())
CARDINAL_TO_DIGIT_REGEX = re.compile(r"\b(" + _pipe_between_words + r")\b", re.IGNORECASE)


def cardinal_to_digit(s: str) -> str:
    return CARDINAL_TO_DIGIT_REGEX.sub(_replace_match, s)


# Ignore case and lower because some digits are writing with capital letters
def _replace_match(match: re.Match) -> str:
    word = match.group(1)  # The matched word (e.g. "one")
    return CARDINAL_TO_DIGIT[word.lower()]  # The corresponding digit (e.g. "1")


def flatten_list(xss):
    return [x for xs in xss for x in xs]


def add_temp_errors(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        answer_digits=lambda x: x["answer_digits"].apply(lambda y: y[0]).astype(int),
        predicted_answer_digits=lambda x: x["predicted_answer_digits"]
        .apply(lambda y: y[0])
        .astype(int),
        err=lambda x: x["predicted_answer_digits"] - x["answer_digits"],
        rel_err=lambda x: (x["predicted_answer_digits"] - x["answer_digits"]) / x["answer_digits"],
        abs_err=lambda x: (x["predicted_answer_digits"] - x["answer_digits"]).abs(),
        abs_per_err=lambda x: (
            (x["predicted_answer_digits"] - x["answer_digits"]).abs() / x["answer_digits"]
        )
        * 100,
        log_err=lambda x: np.log(x["predicted_answer_digits"] + 1) - np.log(x["answer_digits"] + 1),
    )


def merge_with_indomain_year_only(df: pd.DataFrame) -> pd.DataFrame:
    indomain_year_only = pd.read_csv(
        "./../temp_error_analyis/label_temp_answers_head_dataset/data/notebook_output/03_label_answers_with_year_timeunit_head_data.csv"
    ).query("answer_timeunit == 'year'")

    return (
        indomain_year_only.merge(
            df,
            left_on=[
                "question",
                "answer_old",
            ],  # answer_old is unmodified, i.e., contains digits still as words ("three" instead of "3")
            right_on=["question", "actual_answer"],
            how="inner",
        )
        .drop(
            columns=[
                "table",
                "answer_timeunit",
                "is_temporal",
                "answer_old",
                "actual_answer",
                "answer_type",
            ]
        )
        .dropna()
    )


def extract_digits_from_answers(df: pd.DataFrame, split_str_reas: str = "&&") -> pd.DataFrame:
    return df.assign(
        predicted_answer_old=lambda x: x["predicted_answer"],
        predicted_answer_wo_reas=lambda x: x["predicted_answer"]
        .str.split(split_str_reas)  # Seperate answer from reasoning
        .apply(lambda y: y[0]),
        predicted_answer=lambda x: x["predicted_answer_wo_reas"].apply(
            lambda y: cardinal_to_digit(y)
        ),
        answer_digits=lambda x: x["answer"].str.findall("\d+"),
        predicted_answer_digits=lambda x: x["predicted_answer"].str.findall("\d+"),
    )
