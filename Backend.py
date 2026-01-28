from flask import Flask, request, jsonify
import pip
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
app = Flask(__name__)
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

print("Loading dataset...")

data = pd.read_csv(r"C:\Users\Admin\Backend\backend\gita_full.csv")
info = data.info()
data.head(5)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Language-specific corpora
langs = {
    "en": data['translation_en'].astype(str).tolist(),
    "hi": data['translation_hi'].astype(str).tolist(),
    "mr": data['translation_mr'].astype(str).tolist()
}

# Build FAISS indexes
indexes = {}

for lang, texts in langs.items():
    print(f"Building index for {lang}...")
    emb = model.encode(texts, show_progress_bar=True)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(emb))
    indexes[lang] = idx

print("Gita AI ready!")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]

    try:
        detected_lang = detect(user_msg)
    except:
        detected_lang = "en"

    # Map langdetect output
    if detected_lang.startswith("hi"):
        lang = "hi"
    elif detected_lang.startswith("mr"):
        lang = "mr"
    else:
        lang = "en"

    query_vec = model.encode([user_msg])
    D, I = indexes[lang].search(query_vec, k=3)

    results = []
    for i in I[0]:
        row = data.iloc[i]
        results.append({
            "language": lang,
            "chapter": int(row["chapter"]),
            "verse": int(row["verse"]),
            "sanskrit": row["shloka_sanskrit"],
            "answer": row[f"translation_{lang}"]
        })

    return jsonify(results)

if __name__ == "__main__":

    app.run()
