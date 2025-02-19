# Short protocol of 2nd experiment

## What did I do?
In experiment 2, I translated the prompt from the author's Llama2.py script as found in the models/evaluation/LLama folder to follow the message format. The message format is a list of dicts where the first key-value-pair assigns a role (user, assistant or system) and the second dict assigns the content, i.e., the prompt. I believe the author where encoding a switch between roles with "###" in a single prompt. Doing this seems to follow the Alpaca prompting syntax. Lllama2-7b does not seem to understand this very well. Therefore, I explicitly break-up the prompt and assigns its parts to the adequate role using the message format. This is more relevant for the few-shot and CoT experiments. In the zero-shot setting, the performance was not hurt by not assigning a system prompt originally. For completeness, I still translated the prompt to follow the message format in the zero-shot experiment too.

Specifically, I use the first part of the prompt as the system prompt. The most changes are seen in the  few shot and few shot with chain of thought (CoT) experiment. Examples are translated to interaction between the user and the assistant. Missing CoT instructions where copied from the gpt3.5.py code, which is also in the evaluation folder.

I stopped stripping the prompt because in a small sample, it seemed to hurt performance although Meta's model card encourages its use.

## Modification of results
In the CoT experiment, the prompts as found in this repo encouraged the model to separate the answer and the reasoning with "&&". I split the string to keep just the answer. I also explicitly search for "Reasoning:" in the string for the few cases where it did not work and keep the part before the mention of the "Reasoning:" string.

## Code to produce this result
A python script is added in this folder to describe the way I did prompting to produce these results.
