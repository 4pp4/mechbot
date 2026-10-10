"""
Very small Retrieval-Augmented Generation (RAG) helper for uploaded notes.

Pipeline:  uploaded PDF/TXT  ->  plain text  ->  overlapping chunks
           ->  TF-IDF vectors  ->  cosine similarity with the question  ->  top-k chunks

No vector database or embeddings API is needed; scikit-learn does everything locally.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(file_name: str, data: bytes) -> str:
    """Return the plain text of a PDF or TXT file given its raw bytes."""
    if file_name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_words: int = 150, overlap: int = 30) -> list[str]:
    """Split text into chunks of ~chunk_words words that overlap a little,
    so a sentence cut at a boundary still appears whole in one chunk."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


class NotesIndex:
    """TF-IDF index over the chunks of one uploaded document."""

    def __init__(self, text: str, source_name: str):
        self.source_name = source_name
        self.chunks = chunk_text(text)
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[str, float]]:
        """Return up to k (chunk, score) pairs most similar to the query."""
        if self.vectorizer is None or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] >= min_score]
