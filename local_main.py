import os
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
it_src = "Always trust a glue salesman. They tend to stick to their word."
humour = "Topic: Based on the given sentence, the topics are: glue salesman, trustworthiness, and adhesive products. If you have a specific news item in mind, please provide more details so I can accurately describe the topic. For example, if the news item was about a glue salesman who always delivered on his promises, the topics would be: glue salesman, trustworthiness, and business ethics. If the news item was about a company that produces a particularly strong adhesive, the topics would be: adhesive products, business, and technology; Salesman. Punchline: The punchline is a play on words, where \"stick to their word\" has a double meaning - it means to adhere to their promise, but also to be sticky, like glue. The unexpected twist is that the glue salesman is reliable in a literal sense. The sentence implies that the glue salesman is trustworthy because they are sticky, like the product they sell. This unexpected connection between the salesman's profession and their reliability creates the humor. The punchline is the final part of the sentence that delivers this unexpected twist. In this case, the punchline is \"They tend to stick to their word.\" It is the part that makes the joke funny by connecting the idea of a glue salesman being reliable to the physical property of being sticky. The humor comes from the clever wordplay and the surprising connection between the two ideas. The punchline is the final statement that delivers the joke's surprise and humor. In this case, it is \"They tend to stick to their word.\" The punchline is the unexpected connection between the glue salesman's profession and their reliability, which is the source of the joke's humor. The punchline is the final part of the sentence that delivers the joke's surprise and humor. In this case, it is \"They tend to stick to their word"

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
s = f"Translate the following input text into Chinese.\n" + \
                    f"Input text: {it_src}\n" + "Translation:"

tokenized = tokenizer(s, padding=True, return_tensors="pt")
input_ids = tokenized.input_ids.cuda()
attn_mask = tokenized.attention_mask.cuda()
print(input_ids, attn_mask, flush=True)
input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
print(input_ids, attn_mask, flush=True)
generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)

original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
new_text = gen_text.replace(original_text, "").replace("\n", "").strip()
print(new_text, flush=True)