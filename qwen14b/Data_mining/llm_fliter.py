from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

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

def filter(paths,joke):
    messages = [
                {"role": "system", "content": "Given the joke and it candidate angles, please identify the most suitable description angle for this joke."},
                {"role": "user", "content": f"Joke: {joke}\n" + f"Candidate: {paths}\n"+ "Result:"}
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