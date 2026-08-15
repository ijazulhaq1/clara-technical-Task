from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CatalogueRetriever:
    """Retrieve only educator-authored strategies; avoid_when is a hard filter."""
    def __init__(self, catalogue):
        self.docs = catalogue["strategies"]
        corpus = [
            " ".join([d["function"], d["name"], d["goal"], d["appropriate_when"], d["keywords"]])
            for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, function, phase, severity=0.5, avoid_ids=()):
        allowed = [
            (i, d) for i, d in enumerate(self.docs)
            if (d["function"] == function or d["function"] == "any")
            and phase in d["phases"]
            and d["severity_range"][0] <= severity <= d["severity_range"][1]
            and d["id"] not in set(avoid_ids)
        ]
        if not allowed:
            return None
        query = f"{function} {phase} severity {severity:.2f}"
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        i, d = max(allowed, key=lambda x: scores[x[0]])
        result = dict(d)
        result["retrieval_score"] = float(scores[i])
        return result
