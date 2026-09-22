from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser import Segment


@dataclass
class Match:
    segment: Segment
    score: float


class Retriever:
    # TF-IDF is enough here because the transcripts are small
    # search(query, top_k) is used by the rest of the app
    # can be replaced with embeddings later if needed

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._vectorizer = TfidfVectorizer(stop_words="english")

        # Build the TF-IDF matrix once when the retriever is created
        texts = [s.text for s in segments if s.text.strip()]
        if not texts:
            self._matrix = None
            return

        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            # If everything is a stop word, try without the stop word filter
            self._vectorizer = TfidfVectorizer()
            try:
                self._matrix = self._vectorizer.fit_transform(texts)
            except ValueError:
                self._matrix = None

    def search(self, query: str, top_k: int = 1) -> list[Match]:
        if not self.segments or self._matrix is None:
            return []

        try:
            # Convert the question into the same TF-IDF space as the transcript
            query_vec = self._vectorizer.transform([query])

            # Compare the question with each transcript segment
            scores = cosine_similarity(query_vec, self._matrix)[0]

            # Return the most relevant segments first
            ranked = sorted(
                zip(self.segments, scores),
                key=lambda pair: pair[1],
                reverse=True
            )

            return [
                Match(segment=seg, score=float(score))
                for seg, score in ranked[:top_k]
                if score > 0
            ]

        except Exception:
            return []