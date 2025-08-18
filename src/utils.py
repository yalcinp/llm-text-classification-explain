from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

def load_imdb(tokenizer_name="distilbert-base-uncased", max_length=256, split_small=False, seed=42):
    """
    split_small=True: eğitimi hızlandırmak için eğitim setinden ~20k örnek kullanır.
    """
    raw = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    if split_small:
        # 25k train -> 20k train, 5k valid
        raw_train = raw["train"].train_test_split(test_size=0.2, seed=seed)
        train_ds = raw_train["train"]
        valid_ds = raw_train["test"]
    else:
        # orijinal: 25k train, 25k test (valid oluştur)
        raw_train = raw["train"].train_test_split(test_size=0.1, seed=seed)
        train_ds = raw_train["train"]
        valid_ds = raw_train["test"]

    test_ds = raw["test"]

    train_ds = train_ds.map(tok_fn, batched=True)
    valid_ds = valid_ds.map(tok_fn, batched=True)
    test_ds  = test_ds.map(tok_fn, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return train_ds, valid_ds, test_ds, tokenizer

def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

