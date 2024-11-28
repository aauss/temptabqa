import glob
import json
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)

load_dotenv()
CWD = Path().resolve()


def prepare_prompt(context: str, question: str, method: str) -> str:
    match method:
        case "zero_shot":
            return f"""
            Answer the question based on the context provided (take the current year as 2022)
            generate small answer within 1 to 10 words
            do not include verb in the answer
            do not add the phrase Sure! Here's a small answer within 1 to 10 words:

            question : {question}
            context : ***{context}***
            answer :"""
        case "few_shpt":
            return f"""
            Answer the question based on the context provided (take the current year as 2022)
            generate only the relevant answer do not include the question in it thus generate small answer within 1 to 10 words.
            question :
            How old was Charles Brenton Huggins when he won the Nobel Prize in 1966?

            context : ***category
            nobel

            Charles Brenton Huggins
            Born	( | 1901-09-22 | ) | September 22, 1901 | Halifax, Nova Scotia
            Citizenship	Canadian / American
            Awards	Nobel Prize for Physiology or Medicine | (1966) | Gairdner Foundation International Award | (1966)
            ['Charles Brenton Huggins nobel.jpg | Charles Brenton Huggins']***
            answer : 65 years old
            ###
            question : When was the last time that Frederick John Perry won the Davis Cup?
            context : ***category
            tennis

            Full name
            Frederick John Perry

            Died
            2 February 1995 | (1995-02-02) | (aged 85) | Melbourne, Victoria, Australia

            Team competitions
            Davis Cup	W | (1933, 1934, 1935, 1936)***
            answer : 1936
            ###
            question : {question}
            context : ***{context}***
            answer :"""


def init_model_and_tokenizer() -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    # Reference: https://www.reddit.com/r/huggingface/comments/1bv1kfk/what_is_the_difference_between/
    model_name = "meta-llama/Llama-2-7b-chat-hf"
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
        model_name, use_fast=True, token=access_token, resume_download=None
    )
    return model, tokenizer


def generate(
    prompt, model: LlamaPreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> str:
    # Strip because of: https://huggingface.co/meta-llama/Llama-2-7b-hf
    message = [
        {"role": "user", "content": prompt.strip()},
    ]
    # Ref: https://huggingface.co/docs/transformers/main/en/chat_templating
    tokenized_chat = tokenizer.apply_chat_template(
        message, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to("cuda:0")
    outputs = model.generate(tokenized_chat, max_new_tokens=200)
    return tokenizer.decode(
        outputs[0][len(tokenized_chat[0]) :], skip_special_tokens=True
    )


# def generate(
#     prompt, model: LlamaPreTrainedModel, tokenizer: PreTrainedTokenizerFast
# ) -> str:
#     model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
#     print(model_inputs)
#     output = model.generate(**model_inputs, max_new_tokens=10)
#     print(output)
#     return tokenizer.decode(
#         output[0][len(model_inputs["input_ids"][0]) :], skip_special_tokens=True
#     ).strip()


def _load_tables_as_json() -> dict[int, dict]:
    """Load wiki tables as json into dict with table ID as key and table as value"""
    id_to_json = {}

    for filename in glob.glob(str(CWD / "../../../data/maindata/tables/json/*")):
        filename: Path = Path(filename)
        with open(filename, "r") as f:
            wiki_table_json = json.load(f)
            id_of_table = int(filename.stem)
            id_to_json[id_of_table] = wiki_table_json
    return id_to_json


def _load_qa_dev_set() -> pd.DataFrame:
    """Load dev set containing questions, answers, and categories that can be linked to tables via their table ID"""
    return pd.read_csv(
        CWD / "../../../data/maindata/qapairs/dev-set/dev-set.csv", index_col=0
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
                        table_as_str = (
                            table_as_str + sub_key + "\t" + table[key][sub_key] + "\n"
                        )
                table_as_str = table_as_str + "\n"
            else:
                table_as_str = table_as_str + key + "\n" + table[key] + "\n\n"

        tables_as_str[table_id] = table_as_str

    return tables_as_str


if __name__ == "__main__":
    id_to_json = _load_tables_as_json()
    tables_as_str = _linearize_json_tables(id_to_json)

    # Example of nested values
    id_to_json[998]
    # Contains nested subkey "Nigerian Civil War": ['Part of Cold War and Decolonisation of Africa', 'Biafra independent state map-en.svg | The de facto independent Republic of Biafra in June 1967']
    # Value is just appended to the above key without re-mentioning the sub-key
    tables_as_str[998]

    df = _load_qa_dev_set()

    df_eval = df.copy().loc[:100]

    model, tokenizer = init_model_and_tokenizer()
    all_output = []
    for i in tqdm(range(len(df_eval)), "Eval table QA"):
        table_id = df_eval["table_id"][i]
        question = df_eval["question"][i]

        context = tables_as_str[table_id]
        input_text = prepare_prompt(context, question, "zero_shot")
        result = generate(input_text, model, tokenizer)
        all_output.append(result)
    df_eval["predicted_answer"] = all_output
    df_eval.to_csv("test_llama.csv", index=False, quoting=1)
