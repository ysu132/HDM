import os
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from Data_mining.consistency import cos_sim_text
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_samples = 5

model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
input_topic_file = '/data/home/ysu132/HTDM/qwen/Data_mining/Mining_data_consistency/background.json'
input_angle_file = '/data/home/ysu132/HTDM/qwen/Data_mining/Mining_data_consistency/angle.json'
input_punchline_file = '/data/home/ysu132/HTDM/qwen/Data_mining/Mining_data_consistency/punchline.json'
output_file = '/data/home/ysu132/HTDM/qwen/humour_consistency.json'
with open(input_topic_file, 'r', encoding='utf-8') as csvtopic:
    filtered_topic = json.load(csvtopic)
with open(input_angle_file, 'r', encoding='utf-8') as csvangle:
    filtered_angle = json.load(csvangle)
with open(input_punchline_file, 'r', encoding='utf-8') as csvpunchline:
    filtered_punchline = json.load(csvpunchline)

humour_array = []
for i in range(len(filtered_topic)):
    humour_temp = {}
    humour_temp["joke"] = filtered_topic[i]["joke"]
    humour_temp["topic"] = filtered_topic[i]["topics"]
    humour_temp["angle"] = filtered_angle[i]["angle"]
    humour_temp["punchline"] = filtered_punchline[i]["punchline"]
    humour_array.append(humour_temp)

# it_src = "Always trust a glue salesman. They tend to stick to their word."
# humour = "Topic: Based on the given sentence, the topics are: glue salesman, trustworthiness, and adhesive products. If you have a specific news item in mind, please provide more details so I can accurately describe the topic. For example, if the news item was about a glue salesman who always delivered on his promises, the topics would be: glue salesman, trustworthiness, and business ethics. If the news item was about a company that produces a particularly strong adhesive, the topics would be: adhesive products, business, and technology; Salesman. Punchline: The punchline is a play on words, where \"stick to their word\" has a double meaning - it means to adhere to their promise, but also to be sticky, like glue. The unexpected twist is that the glue salesman is reliable in a literal sense. The sentence implies that the glue salesman is trustworthy because they are sticky, like the product they sell. This unexpected connection between the salesman's profession and their reliability creates the humor. The punchline is the final part of the sentence that delivers this unexpected twist. In this case, the punchline is \"They tend to stick to their word.\" It is the part that makes the joke funny by connecting the idea of a glue salesman being reliable to the physical property of being sticky. The humor comes from the clever wordplay and the surprising connection between the two ideas. The punchline is the final statement that delivers the joke's surprise and humor. In this case, it is \"They tend to stick to their word.\" The punchline is the unexpected connection between the glue salesman's profession and their reliability, which is the source of the joke's humor. The punchline is the final part of the sentence that delivers the joke's surprise and humor. In this case, it is \"They tend to stick to their word"


def filter(paths,joke):
    messages = [
                {"role": "system", "content": "Given the joke and it candidate translations, please identify the most suitable translations for this joke and only output the translation."},
                {"role": "user", "content": f"Joke: {joke}\n" + f"Candidate: {paths}\n"+ "Translation:"}
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
for i in tqdm(range(len(humour_array))):
    topic = humour_array[i]["topic"]
    angle = humour_array[i]["angle"]
    punchline = humour_array[i]["punchline"]
    joke = humour_array[i]["joke"]
    paths = []
    final_text = ""
    for _ in range(num_samples):
        messages = [
                    {"role": "system", "content": "Given the following knowledge, translate the following joke from English into Chinese."},
                    {"role": "user", "content": f"Topic: {topic}\n"+ f"Angle: {angle}\n"+ f"Punchline: {punchline}\n"+ f"Joke: {joke}\n" + "Translation:"}
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

        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
        paths.append(new_text)
        # print(new_text, flush=True)

    # final_text = cos_sim_text(paths)
    final_text = filter(paths,joke)
    temp = {"ID":i,"joke": joke,"translation": new_text}
    res.append(temp)

    
with open(output_file, 'w', encoding='utf-8') as jsonfile:
    json.dump(res, jsonfile, indent=4, ensure_ascii=False)