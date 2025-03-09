import bert_score
from bert_score import score
import json
from tqdm import tqdm

input_humour = '/home/ysu132/HTDM/qwen/humour_llmfilter.json'
with open(input_humour, 'r', encoding='utf-8') as csvhumour:
    filtered_humour = json.load(csvhumour)

source = []
translation = []
for i in tqdm(range(len(filtered_humour))):
    source.append(filtered_humour[i]["joke"])
    translation.append(filtered_humour[i]["translation"])



# 计算 BERTScore
P, R, F1 = score(translation, source, lang="en", model_type="bert-base-multilingual-uncased")

print(sum(P).item()/len(P))
print(sum(R).item()/len(R))
print(sum(F1).item()/len(F1))
# 输出 BERTScore 评分
# print(f"Precision: {P.item():.4f}")
# print(f"Recall: {R.item():.4f}")
# print(f"F1 Score: {F1.item():.4f}")
