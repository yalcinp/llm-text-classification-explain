# src/train.py
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import load_imdb, compute_metrics

def main():
    model_name = "distilbert-base-uncased"

    # IMDB veri seti + tokenizer
    train_ds, valid_ds, test_ds, tokenizer = load_imdb(
        tokenizer_name=model_name,
        max_length=256,
        split_small=True,
        seed=42
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Cihaz seçimi (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        device = torch.device("cpu")

    model.to(device)

    # Eğitim parametreleri
    args = TrainingArguments(
        output_dir="outputs",
        eval_strategy="epoch",            # validasyonu her epoch'ta yap
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,

        per_device_train_batch_size=32,   # bellek yetersizse 16 yap
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        fp16=False,                       # CUDA yoksa kapalı
        bf16=False,                       # MPS/CPU için kapalı
        dataloader_pin_memory=False,      # MPS uyarısını kapatır
        dataloader_num_workers=0,

        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Eğitim
    trainer.train()

    # Test seti üzerinde değerlendirme
    metrics = trainer.evaluate(test_ds)
    print("Test metrics:", metrics)

    # En iyi modeli kaydet
    out_dir = "outputs/best_model"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()

