import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

input_file = '/data/home/ysu132/Github/DUAL/Dual-Reflect/output/baseline_dual_reflect_qwen_4lp.jsonl'
final = {"data":[]}
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


filtered_joke = read_jsonl(input_file)

model_name_or_path = "Qwen/Qwen2.5-14B-Instruct"
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

def humour_decomposition(paths,joke):
    messages=[
        {"role": "system", "content": "A joke can be thought of as being composed based on three components. Under a particular theory of joke information, those components are: \
        1. The topic, which is the news item that the joke is based on. \
        2. The angle, which is the particular direction that the joke takes.\
        3. The punch line, which is the surprise at the end of the joke.\
        Please analyze the following joke and provide the best estimate of what the topic is, what the angle is, and what the punch line is. "},
        {"role": "user", "content": f"Joke: {joke}"}
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    tokenized = tokenizer(text, return_tensors="pt")

    input_ids = tokenized.input_ids.cuda()
    attn_mask = tokenized.attention_mask.cuda()
    input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
    attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
    original_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
    
    return new_text


def translation_module(paths,explain):
    messages=[
        {"role": "system", "content": "Please translate the analysis from Englsih into Chinese:"},
        {"role": "user", "content": f"Analysis: {explain}"}
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    tokenized = tokenizer(text, return_tensors="pt")

    input_ids = tokenized.input_ids.cuda()
    attn_mask = tokenized.attention_mask.cuda()
    input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
    attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
    original_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
    
    return new_text

def humour_composition(paths,translate):
    
    messages=[
        {"role": "system", "content": "Please generate a Chinese joke based on the analysis (only output the joke): "},
        {"role": "user", "content": f"Analysis: {translate}"}
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    tokenized = tokenizer(text, return_tensors="pt")

    input_ids = tokenized.input_ids.cuda()
    attn_mask = tokenized.attention_mask.cuda()
    input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
    attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
    original_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
    
    return new_text


if __name__ == "__main__":

    for each in tqdm(range(len(filtered_joke[:200]))):

        temp = {}
        temp["original_text"] = filtered_joke[each]
        
        #################################################Humour Decomposition
        explain = humour_decomposition([],filtered_joke[each])

        temp["English_explain"] = explain
        ################################################Translation module
        translate = translation_module([],explain)

        temp["Chinese_explain"] = translate
        #################################################Humour Composition
        joke = humour_composition([],translate)
 
        temp["Chinese_joke"] = joke

        final["data"].append(temp)


with open('/data/home/ysu132/HTDM/test.json', 'w',encoding='utf-8') as f1:
    f1.write(json.dumps(final, ensure_ascii=False, indent=4)) 
