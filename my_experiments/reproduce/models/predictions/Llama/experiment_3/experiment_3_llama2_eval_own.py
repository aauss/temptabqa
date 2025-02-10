import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Because HF_HOME is saved in dotenv, .env needs to be loaded before torch to take effect.
# https://github.com/huggingface/transformers/issues/25305
CWD = Path().resolve()
load_dotenv(CWD / "../../..//../../.env")

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)


def construct_conversation(context: str, question: str, method: str) -> list[dict[str, str]]:
    # Use messages to do few shot prompting: https://github.com/philschmid/easyllm/blob/main/notebooks/chat-completion-api.ipynb
    # Intersting insight how messages are translated to prompt: https://github.com/philschmid/easyllm/blob/main/easyllm/prompt_utils/llama2.py#L8
    # Some other reference how to prompt LlaMa2: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    # Replace prior ### for role changes. Seems to be Alpaca style and may work better for bigger LlaMa model: https://www.reddit.com/r/LocalLLaMA/comments/15uu6lk/how_do_you_really_really_prompt_llama_2/
    match method:
        case "zero_shot":
            user = f"""question : {question}
            context : ***{context}***
            answer :"""
            message = [
                {
                    "role": "system",
                    "content": """
                    Answer the question based on the context provided (take the current year as 2022)
                    generate small answer within 1 to 10 words
                    do not include verb in the answer
                    do not add the phrase Sure! Here's a small answer within 1 to 10 words""",
                },
                {"role": "user", "content": user},
            ]
            return message

        case "few_shot":
            user = f"""question : {question}
            context : ***{context}***
            answer :"""
            message = [
                {
                    "role": "system",
                    "content": """
                    Answer the question based on the context provided (take the current year as 2022)
                    generate only the relevant answer do not include the question in it thus generate small answer within 1 to 10 words.""",
                },
                {
                    "role": "user",
                    "content": """
                    questions : How old was Charles Brenton Huggins when he won the Nobel Prize in 1966?

                    context : ***category
                    nobel

                    Charles Brenton Huggins
                    Born	( | 1901-09-22 | ) | September 22, 1901 | Halifax, Nova Scotia
                    Citizenship	Canadian / American
                    Awards	Nobel Prize for Physiology or Medicine | (1966) | Gairdner Foundation International Award | (1966)
                    ['Charles Brenton Huggins nobel.jpg | Charles Brenton Huggins']***
                    answers :""",
                },
                {"role": "assistant", "content": "65 years old"},
                {
                    "role": "user",
                    "content": """
                    question : When was the last time that Frederick John Perry won the Davis Cup?
                    context : ***category
                    tennis

                    Full name
                    Frederick John Perry

                    Died
                    2 February 1995 | (1995-02-02) | (aged 85) | Melbourne, Victoria, Australia

                    Team competitions
                    Davis Cup	W | (1933, 1934, 1935, 1936)***
                    answer :""",
                },
                {"role": "assistant", "content": "1936"},
                {"role": "user", "content": user},
            ]
            return message
        case "CoT":
            user = f"""question : {question}
            context : ***{context}***
            answer and reasoning:"""
            message = [
                {
                    "role": "system",
                    "content": """
                    Answer the question based on the context provided (take the current year as 2022)
                    generate only the relevant answer do not include the question in it thus generate small answer within 1 to 10 words.
                    Also give the proper reasoning with the answer, first answer then the reason for the generated answer.""",
                },
                {
                    "role": "user",
                    "content": """
                    questions : How old was Charles Brenton Huggins when he won the Nobel Prize in 1966?

                    context : ***category
                    nobel

                    Charles Brenton Huggins
                    Born	( | 1901-09-22 | ) | September 22, 1901 | Halifax, Nova Scotia
                    Citizenship	Canadian / American
                    Awards	Nobel Prize for Physiology or Medicine | (1966) | Gairdner Foundation International Award | (1966)
                    ['Charles Brenton Huggins nobel.jpg | Charles Brenton Huggins']***
                    answers and reasoning:""",
                },
                {
                    "role": "assistant",
                    "content": "65 years old && Charles was born in 1901 and he won the price in 1966, thus he was 65 years old. (1966 - 1901)",
                },
                {
                    "role": "user",
                    "content": """
                    question : When was the last time that Frederick John Perry won the Davis Cup?
                    context : ***category
                    tennis

                    Full name
                    Frederick John Perry

                    Died
                    2 February 1995 | (1995-02-02) | (aged 85) | Melbourne, Victoria, Australia

                    Team competitions
                    Davis Cup	W | (1933, 1934, 1935, 1936)***
                    answer and reasoning:""",
                },
                {
                    "role": "assistant",
                    "content": "1936 && He won the davis cup four times among them 1936 is the lastest year in which he won.",
                },
                {"role": "user", "content": user},
            ]
            # .strip() because of: https://huggingface.co/meta-llama/Llama-2-7b-hf
            # but did not work well for a few examples so leave commented out for the moment
            # return [{k: v.strip() for k, v in _dict.items()} for _dict in message]
            return message


def init_model_and_tokenizer() -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    # Reference: https://www.reddit.com/r/huggingface/comments/1bv1kfk/what_is_the_difference_between/
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    access_token = os.environ["HF_TOKEN"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=access_token,
        resume_download=None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=access_token,
        resume_download=None,
    )
    return model, tokenizer


def generate(conversation, model: LlamaPreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> str:
    # Ref: https://huggingface.co/docs/transformers/main/en/chat_templating
    tokenized_chat = tokenizer.apply_chat_template(
        conversation, tokenize=True, return_tensors="pt"
    ).to("cuda:0")
    outputs = model.generate(tokenized_chat, max_new_tokens=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1]


def _load_tables_as_json() -> dict[int, dict]:
    """Load wiki tables as json into dict with table ID as key and table as value"""
    id_to_json = {}

    for filename in glob.glob(str(CWD / "../../../../../data/maindata/tables/json/*")):
        filename: Path = Path(filename)
        with open(filename, "r") as f:
            wiki_table_json = json.load(f)
            id_of_table = int(filename.stem)
            id_to_json[id_of_table] = wiki_table_json
    return id_to_json


def _load_qa_data_set(data_set: str) -> pd.DataFrame:
    """Load dev set containing questions, answers, and categories that can be linked to tables via their table ID"""
    return pd.read_csv(
        CWD / f"../../../../../data/maindata/qapairs/{data_set}-set/{data_set}-set.csv", index_col=0
    )


def _linearize_json_tables(id_to_json: dict[int, dict]) -> dict[int, str]:
    """Linearizes json Wiki tables to strings

    Note: I checked if my adopted code would produce the same as the original code and it does.

    """

    # Init dict and table IDs
    tables_as_str = {}
    table_ids = id_to_json.keys()

    for table_id in tqdm(table_ids, "Linearise tables"):
        table_as_str = ""
        table = id_to_json[table_id]
        for key in table.keys():
            # JSON Tables are transformed to key + "\n" + value and "\n\n" for new line.
            if type(table[key]) is type(dict()):
                # Deals with nested tables.
                # Several values under same key are added with "\n"
                # Different keys under key are seperated with "\t"
                table_as_str = table_as_str + key + "\n"
                for sub_key in table[key].keys():
                    if sub_key == key:
                        table_as_str = table_as_str + str(table[key][sub_key]) + "\n"
                    else:
                        table_as_str = table_as_str + sub_key + "\t" + table[key][sub_key] + "\n"
                table_as_str = table_as_str + "\n"
            else:
                table_as_str = table_as_str + key + "\n" + table[key] + "\n\n"

        tables_as_str[table_id] = table_as_str

    return tables_as_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LlaMa2 experiments in various prompting modes."
    )
    parser.add_argument(
        "-d",
        "--data_set",
        required=True,
        type=str,
        choices=["dev", "head", "tail"],
        help="Select the dataset that to evaluate on: 'dev', 'head', or 'tail'",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        type=str,
        choices=["zero_shot", "few_shot", "CoT"],
        help="Select the prompting mode: 'zero_shot', 'few_shot', or 'CoT' (Chain-of-Thought).",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        help="Path to the output CSV file where the results will be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    id_to_json = _load_tables_as_json()
    tables_as_str = _linearize_json_tables(id_to_json)
    df_eval = _load_qa_data_set(args.data_set)

    # Example of nested values
    id_to_json[998]
    # Contains nested subkey "Nigerian Civil War": ['Part of Cold War and Decolonisation of Africa', 'Biafra independent state map-en.svg | The de facto independent Republic of Biafra in June 1967']
    # Value is just appended to the above key without re-mentioning the sub-key
    tables_as_str[998]

    model, tokenizer = init_model_and_tokenizer()
    all_output = []
    for i in tqdm(range(len(df_eval)), "Eval table QA"):
        table_id = df_eval["table_id"][i]
        question = df_eval["question"][i]

        context = tables_as_str[table_id]
        conversation = construct_conversation(context, question, args.mode)
        result = generate(conversation, model, tokenizer)
        if args.mode == "CoT":
            result = result.split("&&")[
                0
            ]  # Prompting encourages to respond with answer && reasoning
            result = result.split("Reasoning:")[
                0
            ]  # Remove reasoning if above format is not followed
        all_output.append(result.strip().replace("\n", ""))  # Remove newlines as they break CSV
        if i % 10 == 0:
            df_checkpoint = df_eval.copy().iloc[: len(all_output)]
            df_checkpoint["predicted_answer"] = all_output
            df_checkpoint.to_csv(f"checkpoint_{args.output_path}", index=False, quoting=1)

    df_eval["predicted_answer"] = all_output
    df_eval.to_csv(args.output_path, index=False, quoting=1)
