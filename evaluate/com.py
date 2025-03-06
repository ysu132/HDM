from comet import download_model, load_from_checkpoint
import json
from tqdm import tqdm

input_humour = '/data/home/ysu132/HTDM/qwen2.5/humour2.json'
with open(input_humour, 'r', encoding='utf-8') as csvhumour:
    filtered_humour = json.load(csvhumour)


# 下载并加载 COMET-QE 预训练模型
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
model = load_from_checkpoint(model_path)

# score = 0
data = []
for i in tqdm(range(len(filtered_humour))):
    humour_temp = {
            "src": filtered_humour[i]["joke"],
            "mt": filtered_humour[i]["translation"]
        }
    data.append(humour_temp)

# 计算 COMET-QE 评分
scores = model.predict(data)
# score += scores[0]
final = sum(scores[0])/len(filtered_humour)
print("COMET-QE Score:", final)
    
# print("Average COMET-QE Score:", score / 300)
