import bert_score

generated_texts = ["The cat is sitting on the mat."]
P, R, F1 = bert_score.score(generated_texts, generated_texts, lang="en", model_type="bert-base-uncased")

print(f"Self-BERTScore F1 Score: {F1.item():.4f}")