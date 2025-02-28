import csv
import json

# 文件路径
input_file = '/data/home/ysu132/HTDM/data/shortjokes.csv'
output_file = '/data/home/ysu132/HTDM/data/filtered_jokes.json'

count = 0
# 读取CSV文件并过滤数据
filtered_jokes = []
with open(input_file, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        joke = row['Joke']
        count += 1
        if count>10000:
            break
        
        if len(joke.split()) >= 15:
            filtered_jokes.append(row)

# 将过滤后的数据写入JSON文件
with open(output_file, 'w', encoding='utf-8') as jsonfile:
    json.dump(filtered_jokes, jsonfile, indent=4, ensure_ascii=False)

print(f"Filtered jokes have been saved to {output_file}")