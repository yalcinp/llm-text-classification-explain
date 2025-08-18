from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import numpy as np

class_names = ["neg", "pos"]

def main():
    model_dir = "outputs/best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True)

    def predictor(texts):
        # LIME, probas ister
        outs = []
        for t in texts:
            scores = pipe(t)[0]
            outs.append([scores[0]["score"], scores[1]["score"]])
        return np.array(outs)

    explainer = LimeTextExplainer(class_names=class_names)
    sample = "The movie was surprisingly good; performances were strong and the pacing kept me hooked."
    exp = explainer.explain_instance(sample, predictor, num_features=10, top_labels=1)
    print(exp.as_list(label=1))  # pozitif sınıf için en önemli kelimeler

    # Jupyter’de görsel:
    # exp.show_in_notebook(text=sample)

if __name__ == "__main__":
    main()

