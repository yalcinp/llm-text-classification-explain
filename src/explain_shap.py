import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Not: SHAP, pipeline üzerinden token attributions çıkarabilir.
# Büyük veri seçmeyin; örnek sayısını düşük tutun (örn. 50-100) yoksa yavaşlar.

def main():
    model_dir = "outputs/best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True)

    # Arka plan/özet örnekleri (SHAP için) – kısa bir liste
    background_texts = [
        "This movie was okay, not the best but not the worst.",
        "I absolutely loved this film, it was fantastic!",
        "I hated this movie. It was boring and too long."
    ]

    explainer = shap.Explainer(pipe, shap.maskers.Text(tokenizer))
    # Açıklamak istediğin örnekler
    samples = [
        "The plot was engaging and the acting was brilliant.",
        "Terrible script and I fell asleep halfway through.",
        "It has flaws, but overall I enjoyed it."
    ]

    shap_values = explainer(samples)
    # Görselleştirme – jupyter'de çalıştırırsan:
    # shap.plots.text(shap_values[0]); shap.plots.text(shap_values[1]); ...
    for i, s in enumerate(samples):
        print(f"\nText {i}: {s}")
        # En etkili kelimeleri yazdır (pozitif/negatif katkılar)
        # (Jupyter’de plots.text çok daha anlaşılır)
        print("Token importances (abs sum):",
              float(torch.tensor(shap_values.values[i]).abs().sum()))

if __name__ == "__main__":
    main()

