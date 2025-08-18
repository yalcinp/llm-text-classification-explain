from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset
import numpy as np

def main():
    model_dir = "outputs/best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

    test = load_dataset("imdb")["test"]
    texts = test["text"][:10]
    labels = test["label"][:10]

    preds = pipe(texts)
    y_pred = [np.argmax([s["score"] for s in p]) for p in preds]
    acc = np.mean([p == y for p, y in zip(y_pred, labels)])
    print("Quick ACC@10:", acc)

if __name__ == "__main__":
    main()

