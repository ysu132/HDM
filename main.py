import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import json


with open('Path', 'r') as f:
    data = json.load(f)
    Chinese_data = data['data']
final = {"data":[]}
err=0
client = OpenAI(
        api_key='Your Openai Key',
        )
for each in Chinese_data[:]:

    # try:
    print(each)
    temp = {}
    temp["original_text"] = each
    

    #################################################Humour Decomposition

    completion = client.chat.completions.create(
    model="gpt-4o",
    seed=40,
    messages=[
        {"role": "system", "content": "A joke can be thought of as being composed based on three components. Under a particular theory of joke information, those components are: \
        1. The topic, which is the news item that the joke is based on. \
        2. The angle, which is the particular direction that the joke takes.\
        3. The punch line, which is the surprise at the end of the joke.\
        Please analyze the following joke and provide the best estimate of what the topic is, what the angle is, and what the punch line is. "},
        {"role": "user", "content": each}
    ]
    )    
    explain = completion.choices[0].message.content
    print(completion.choices[0].message.content)
    temp["English_explain"] = explain
    ################################################Translation module
    completion1 = client.chat.completions.create(
    model="gpt-4o",
    seed=40,
    messages=[
        {"role": "system", "content": "Please translate the analysis from Englsih into Spanish:"},
        {"role": "user", "content": explain}
    ]
    )    
    translate = completion1.choices[0].message.content
    print(translate)
    temp["Chinese_explain"] = translate
    #################################################Humour Composition
    completion2 = client.chat.completions.create(
    model="gpt-4o",
    seed=40,
    messages=[
        {"role": "system", "content": "Please generate a Spanish joke based on the analysis (only output the joke): "},
        {"role": "user", "content": translate}
    ]
    )    
    joke = completion2.choices[0].message.content
    print(joke)
    temp["Chinese_joke"] = joke


with open('Path', 'w',encoding='utf-8') as f1:
    f1.write(json.dumps(final, ensure_ascii=False, indent=4)) 
