import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
data = pd.read_csv("gita_full.csv")

print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

langs = {
    "en": data["translation_en"].astype(str).tolist(),
    "hi": data["translation_hi"].astype(str).tolist(),
    "mr": data["translation_mr"].astype(str).tolist()
}

for lang, texts in langs.items():
    print(f"Encoding {lang}...")
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.array(emb).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)

    faiss.write_index(index, f"gita_{lang}.index")

print("Indexes saved to disk.")
