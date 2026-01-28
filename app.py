from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

print("Loading dataset...")
data = pd.read_csv("gita_full.csv")

print("Loading model...")
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

print("Loading FAISS indexes...")
indexes = {
    "en": faiss.read_index("gita_en.index"),
    "hi": faiss.read_index("gita_hi.index"),
    "mr": faiss.read_index("gita_mr.index")
}

@app.route("/")
def home():
    return "Gita AI (low RAM) is running!"

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

