# Multimodal Image–Text Retrieval (CLIP)

This is a minimal demo of **multimodal retrieval** using [OpenAI's CLIP](https://github.com/openai/CLIP).
It allows you to:
- Search images given a text query (Text → Image).
- Search captions given an image (Image → Text).

Built with:
- HuggingFace `transformers` (CLIPModel + CLIPProcessor)
- `scikit-learn` Nearest Neighbors
- `streamlit` for a simple UI

---

## 🚀 Quickstart
1. Clone repo and install requirements:
```bash
pip install -r requirements.txt
