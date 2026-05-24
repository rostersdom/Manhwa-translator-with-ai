import json, os, re, gc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "training_data", "manga_translation_pairs.jsonl")

class RAGFewShot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)
        self.tfidf_matrix = None
        self.pairs = []
        self._load()

    def _load(self):
        if not os.path.exists(DATASET_PATH):
            print(f"RAG: dataset not found at {DATASET_PATH}")
            return
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line.strip())
                self.pairs.append(pair)

        if not self.pairs:
            print("RAG: empty dataset")
            return

        texts = [p["en"] for p in self.pairs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"RAG: ready ({len(self.pairs)} examples)")

    def get_fewshot(self, query: str, top_k: int = 3) -> str:
        if self.tfidf_matrix is None or len(self.pairs) == 0:
            return ""
        q_vec = self.vectorizer.transform([query])
        scores = np.asarray((self.tfidf_matrix @ q_vec.T).toarray().flatten())
        top_indices = np.argsort(scores)[-top_k:][::-1]

        examples = []
        seen = set()
        for idx in top_indices:
            pair = self.pairs[idx]
            key = pair["en"].lower().strip()
            if key not in seen:
                examples.append(pair)
                seen.add(key)

        lines = []
        for ex in examples:
            lines.append(f"EN: {ex['en']}")
            lines.append(f"RU: {ex['ru']}")
        return "\n".join(lines)

    def add_pair(self, en: str, ru: str):
        pair = {"en": en, "ru": ru}
        self.pairs.append(pair)

        new_vec = self.vectorizer.transform([en])
        if self.tfidf_matrix is not None:
            from scipy.sparse import vstack
            self.tfidf_matrix = vstack([self.tfidf_matrix, new_vec])

        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def cleanup(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.pairs = []
        gc.collect()

rag = RAGFewShot()
