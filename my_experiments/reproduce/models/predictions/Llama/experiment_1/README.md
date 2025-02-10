# Short protocol of 1st experiment

## What did I do?
In experiment 1, I copied the prompt from the author's Llama2.py script as found in the models/evaluation/LLama folder. Since the few shot and few shot with chain of thought (CoT) experiment were missing in this script, I copied these experiments from the gpt3.5.py code, which is also in the evaluation folder. The few shot experiment included two examples and CoT added reasoning to the answers and added a phrase asking the LLM to reason over its answer.

## Modification of results
In the few shot settings, llama2 would reiterate the whole conversation with the examples given in the prompt. With a simple regex, I extracted the final answer searching for the last phrase starting with "answer:" and its variations and saved the result in FILENAME_modified.csv.

## Code to produce this result
A python script is added in this folder to describe the way I did prompting to produce these results. Note, you will need to adopt paths to re-run experiments

## Open questions
- Why is the answer so long for the few shot experiments?
- Why does performance drop so bad with few shot and more so with CoT?
- Why are some responses truncated?