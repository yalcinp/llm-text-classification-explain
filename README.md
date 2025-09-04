# IMDb Sentiment Classification with DistilBERT + Explainability (SHAP/LIME)

## 🎯 Project Goal
This project classifies IMDb movie reviews as **positive** or **negative** using DistilBERT.  
Additionally, it provides **explainability** by applying SHAP and LIME to highlight which words influence the model’s decision.

## 🛠 Tech Stack
- Python 3.9+
- HuggingFace Transformers (DistilBERT)
- scikit-learn
- SHAP
- LIME
- PyTorch

## 🎯 Explainability Demo

We provide a Jupyter notebook to demonstrate explainability:

```bash
jupyter notebook notebooks/explainability_demo.ipynb
```

## ⚙️ Installation
```bash
git clone https://github.com/yalcinp/llm-text-classification-explain.git
cd llm-text-classification-explain
pip install -r requirements.txt
```
