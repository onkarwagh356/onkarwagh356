from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
from flask_cors import CORS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)
CORS(app)

print("Loading dataset...")

# Load CSV
data = pd.read_csv("gita_full.csv")

# Load multilingual embedding model
print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Prepare language corpora
langs = {
    "en": data["translation_en"].astype(str).tolist(),
    "hi": data["translation_hi"].astype(str).tolist(),
    "mr": data["translation_mr"].astype(str).tolist()
}

# Build FAISS indexes
indexes = {}
for lang, texts in langs.items():
    print(f"Building index for {lang}...")
    emb = model.encode(texts, show_progress_bar=True)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(emb).astype("float32"))
    indexes[lang] = idx

print("Gita AI ready!")

# ------------------ ROUTES ------------------

@app.route("/", methods=["GET"])
def home():
    return "Gita AI API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    try:
        detected_lang = detect(user_msg)
    except:
        detected_lang = "en"

    if detected_lang.startswith("hi"):
        lang = "hi"
    elif detected_lang.startswith("mr"):
        lang = "mr"
    else:
        lang = "en"

    query_vec = model.encode([user_msg]).astype("float32")
    D, I = indexes[lang].search(query_vec, k=3)

    results = []
    for i in I[0]:
        row = data.iloc[int(i)]
        results.append({
            "language": lang,
            "chapter": int(row["chapter"]),
            "verse": int(row["verse"]),
            "sanskrit": row["shloka_sanskrit"],
            "answer": row[f"translation_{lang}"]
        })

    return jsonify(results)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

