import os
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name_or_path = "wxjiao/alpaca-7b"
it_src = "Always trust a glue salesman. They tend to stick to their word."
humour = "Topic: Trust, Glue, Salesman. Punchline: The punchline is that glue salesmen are reliable and trustworthy, as they tend to stick to their promises."

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
gen_config = GenerationConfig(
                        temperature=0.2,
                        do_sample=True,
                        num_beams=1,
                        max_new_tokens=256,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token=tokenizer.pad_token_id,
            )
s = f"Given the humour theory analysis, translate the following input text into Chinese.\n" + \
                    f"Input text: {it_src}\n"+ "Analysis: {humour}\n" + "Translation:"

tokenized = tokenizer(s, padding=True, return_tensors="pt")
print(tokenized)
input_ids = tokenized.input_ids.cuda()
attn_mask = tokenized.attention_mask.cuda()
input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
print(generated_ids)
original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
print(new_text, flush=True)