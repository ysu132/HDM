import os
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import csv
import json
from consistency import cos_sim_text

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name_or_path = "wxjiao/alpaca-7b"
model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
# input_file = '/data/home/ysu132/HTDM/data/filtered_jokes.json'
input_file = '/data/home/ysu132/Github/DUAL/Dual-Reflect/output/baseline_dual_reflect_qwen_4lp.jsonl'
output_file = '/data/home/ysu132/HTDM/qwen/Data_mining/Mining_data_consistency/punchline.json'

num_samples = 5

# with open(input_file, 'r', encoding='utf-8') as csvfile:
#     filtered_joke = json.load(csvfile)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


# def filter(paths,joke):
#     messages = [
#                 {"role": "system", "content": "Given the joke and it candidate punchlines, please identify the most suitable description punchlines for this joke and only output the description punchline."},
#                 {"role": "user", "content": f"Joke: {joke}\n" + f"Candidate: {paths}\n"+ "Description:"}
#             ]
#     text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
    
#     tokenized = tokenizer(text, return_tensors="pt")

#     input_ids = tokenized.input_ids.cuda()
#     attn_mask = tokenized.attention_mask.cuda()
#     input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
#     attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask

#     generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
#     original_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
#     gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
#     new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
    
#     return new_text


filtered_joke = read_jsonl(input_file)
    

# it_src = "What do you call a green cow in a field? Invisibull."

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

res = []
for i in tqdm(range(len(filtered_joke))):

    # s = f"The punchline is the surprise at the end of the joke. Please provide a brief description of the punchline for the input sentence.\n" + \
    #                     f"Input: {jokes[i]}\n" + "Description:"
    paths = []
    final_text = ""
    joke = filtered_joke[i]["src_text"]
    for _ in range(num_samples):
        
        messages = [
                {"role": "system", "content": "The punchline is the surprise at the end of the joke. Please provide a brief description of the punchline for the input sentence."},
                {"role": "user", "content": f"Input: {joke}\n" + "Description:"}
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
        paths.append(new_text)
        
    final_text = cos_sim_text(paths)
    # final_text = filter(paths,joke)
    temp = {"ID":i,"joke": joke, "punchline": final_text}
    res.append(temp)
# 将过滤后的数据写入JSON文件
with open(output_file, 'w', encoding='utf-8') as jsonfile:
    json.dump(res, jsonfile, indent=4, ensure_ascii=False)
# for original_input, gen_id in zip(input_ids, generated_ids):
#     original_text = tokenizer.decode(original_input, skip_special_tokens=True)

#     gen_text = tokenizer.decode(gen_id, skip_special_tokens=True)

#     new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
#     print(new_text, flush=True)