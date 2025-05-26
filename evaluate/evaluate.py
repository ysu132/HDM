from prompt import prompts, language_codes
import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import json

baseline = 0
theory_final = 0
Chinese_joke_only = 0
final_final = 0
client = OpenAI(
        api_key='',
        )
count=0
time=0
with open("/data/home/ysu132/HTDM/baseline/qwen2.5/base.json", "r") as f:
    data_json = json.load(f)
    # data = data_json["data"]
    score = []
    for i in data_json[100:200]:
        try:
            print(count)
            
            #############################################################
            data1 = {
                    "sentence": i["translation"],
                    "lang": "Chinese"
                }
            # data1 = {
            #     "target_seg": i["translation"],
            #     "lang": "Chinese",
            #     "target_lang": "Chinese",
            #     "source_lang": "English",
            #     "source_seg": i["joke"],
            # }

            com = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            seed=50,
            messages=[
                # {"role": "system", "content": "You are a translation assistant, skilled in translating the text /to Chinese."},
                {"role": "user", "content": prompts["GEMBA-SQM-in"]["prompt"].format(**data1)}
            ]
            )    
            result1 = com.choices[0].message.content
            print("Chinese_joke:",result1)
            result1 = int(result1)
            score.append(result1)
            # theory_final+=theory
            Chinese_joke_only+=result1
            count+=1
            print("###############################")
        except:
            print("error")

print(count)
#print("Chinese_joke:",theory_final)
# print("explanation :",explanation_final)
print("Chinese_joke_only:",Chinese_joke_only)
#print("ground_truth:",baseline)
print("score:",score)