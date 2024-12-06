import os

import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Using pipelines
load_dotenv()

login(os.environ["HF_TOKEN"])
model_id = "meta-llama/Llama-2-7B-chat-hf"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# response = pipeline("What is the weather today? Give one single answers.")
# print(response)
# print(response[0]["generated_text"].split("\n\n"))

# Using Automodel
model_name = "meta-llama/Llama-2-7b-chat-hf"
prompt = "Tell me about gravity"
access_token = os.environ["HF_TOKEN"]
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, resume_download=None, device_map="cuda"
)


messages = [{"role": "user", "content": prompt}]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(tokenized_chat)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1])
