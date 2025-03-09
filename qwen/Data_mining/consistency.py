from sentence_transformers import SentenceTransformer, util
import numpy as np

# 加载 Sentence-BERT 预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def cos_sim_text(texts):
    # 示例文本
    # texts = [   
    #     "The capital of France is Paris.",
    #     "Paris is the capital of France.",
    #     "France's capital is Paris.",
    #     "The biggest city in France is Paris.",
    #     "Berlin is the capital of France."  # 一个错误回答
    # ]

    # 计算文本的向量表示，并转换为 NumPy 数组
    embeddings = model.encode(texts)  # 默认返回 NumPy 数组

    # 计算余弦相似度矩阵（修正 util.pytorch_cos_sim -> util.cos_sim）
    cosine_scores = util.cos_sim(embeddings, embeddings).numpy()


    # 计算每个文本的相似度总和（排除自身）
    total_similarities = np.sum(cosine_scores, axis=1) - np.diag(cosine_scores)

    # 找到相似度最高的文本索引
    best_idx = np.argmax(total_similarities)
    final_answer = texts[best_idx]

    # print(f"最终选择的答案: {final_answer}")
    
    return final_answer
