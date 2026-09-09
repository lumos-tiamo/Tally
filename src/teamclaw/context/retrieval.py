"""Retrieval used for tool selection and memory recall.

Two backends, chosen at runtime:

* ``BM25Retriever`` — pure Python, zero dependencies, no model download. This is
  the default and it is a deliberate choice, not a placeholder: tool selection
  matches a step description against function names and docstrings, which is
  largely a lexical problem, and BM25 needs no warm-up on a 16GB laptop.
* ``EmbeddingRetriever`` — BGE-m3 via sentence-transformers when the optional
  ``local`` extra is installed. Better for memory recall, where paraphrase
  matters.

Both expose the same ``search(query, k)`` returning ``(doc_id, score)`` pairs,
so the ledger's tool slot does not care which is active.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, Sequence

_WORD = re.compile(r"[a-z0-9_]+|[一-鿿]")

# Function words carry no retrieval signal. They are removed explicitly rather
# than left to IDF because IDF is a *corpus* statistic and our corpora are tiny:
# a tool registry has ~40 entries and a skill library ~5. At n=3, a stopword
# appearing in one document scores idf≈0.98 — indistinguishable from a genuinely
# rare domain term. Only at corpus sizes in the thousands does IDF learn to
# discount them on its own, and none of this project's retrieval targets are
# that large. Both the index and the query are filtered, which is standard.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those and or but not no nor so if then than as
    of in on at to for from by with without into onto over under about
    is are was were be been being am do does did doing done have has had having
    i you he she it we they me him her us them my your his its our their
    what which who whom whose when where why how all any both each few more
    most other some such only own same too very can will just should now
    上 下 中 的 了 和 与 或 是 在 有 也 都 就 不 为 对 到 从 把 被 而 及 等 之 其 此 该
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens plus individual CJK characters.

    CJK is split per character rather than per word because we have no segmenter
    available offline; character unigrams are a weak but workable proxy for
    Chinese filings, and BM25's IDF term keeps common characters from dominating.
    """
    toks = _WORD.findall(text.lower())
    if drop_stopwords:
        return [t for t in toks if t not in STOPWORDS]
    return toks


@dataclass(frozen=True)
class Hit:
    """A retrieval result carrying enough detail to gate on match *quality*.

    Why three numbers instead of one
    --------------------------------
    ``score`` cannot be thresholded across queries: BM25 is unbounded and grows
    with query length and IDF, so a fixed floor admits junk on long queries and
    rejects good matches on short ones.

    Counting matched *terms* fails differently — stopwords dominate. "compute the
    gross margin ratio for FY2024" has four low-information terms out of seven, so
    a correct match scores 2/7 coverage and gets rejected, while "the of and a"
    scores 3/4 and gets accepted. Both outcomes are backwards.

    So coverage is weighted by IDF mass:

    ``coverage = matched_idf / query_idf``
        Of the query's *informative* content that this corpus could match at all,
        how much did this document match? Terms absent from the whole corpus are
        excluded from the denominator — they carry no discriminative information
        here, and leaving them in would make coverage unreachable.
    ``matched_idf``
        The absolute information mass matched. A pure-stopword query can reach
        coverage 1.0, so callers also need a floor on how *informative* the match
        was. This is that number.
    """

    doc_id: str
    score: float
    matched_terms: int = 0
    query_terms: int = 0
    matched_idf: float = 0.0
    query_idf: float = 0.0

    @property
    def coverage(self) -> float:
        return self.matched_idf / self.query_idf if self.query_idf else 0.0

    @property
    def term_coverage(self) -> float:
        """Unweighted term coverage. Diagnostics only — do not gate on this."""
        return self.matched_terms / self.query_terms if self.query_terms else 0.0

    def as_tuple(self) -> tuple[str, float]:
        return self.doc_id, self.score


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: dict[str, object] = field(default_factory=dict)


class Retriever(Protocol):
    def index(self, docs: Sequence[Doc]) -> None: ...
    def search(self, query: str, k: int = 8) -> list[tuple[str, float]]: ...
    def search_detailed(self, query: str, k: int = 8) -> list[Hit]: ...


class BM25Retriever:
    """Okapi BM25 over an in-memory corpus."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Doc] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._len: list[int] = []
        self._avg_len = 0.0

    def index(self, docs: Sequence[Doc]) -> None:
        self._docs = list(docs)
        self._tf = []
        self._df = Counter()
        self._len = []
        for doc in self._docs:
            toks = tokenize(doc.text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

    def search(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        return [h.as_tuple() for h in self.search_detailed(query, k=k)]

    def _idf(self, term: str) -> float:
        """Okapi IDF with +0.5 smoothing, so terms in every doc stay positive."""
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search_detailed(self, query: str, k: int = 8) -> list[Hit]:
        if not self._docs:
            return []
        distinct = set(tokenize(query))
        # Denominator counts only terms this corpus could match; a term with df=0
        # is not evidence about any document here.
        matchable = {t: self._idf(t) for t in distinct if self._df.get(t, 0) > 0}
        query_idf = sum(matchable.values())

        hits: list[Hit] = []
        for i, doc in enumerate(self._docs):
            tf, dl = self._tf[i], self._len[i]
            score = 0.0
            matched = 0
            matched_idf = 0.0
            for term, idf in matchable.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                matched += 1
                matched_idf += idf
                denom = f + self.k1 * (
                    1 - self.b + self.b * (dl / self._avg_len if self._avg_len else 1.0)
                )
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                hits.append(
                    Hit(
                        doc.doc_id,
                        score,
                        matched_terms=matched,
                        query_terms=len(matchable),
                        matched_idf=matched_idf,
                        query_idf=query_idf,
                    )
                )
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]


class EmbeddingRetriever:
    """Dense retrieval via sentence-transformers. Requires the `local` extra."""

    name = "bge-m3"

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name, device=device)
        self._docs: list[Doc] = []
        self._emb = None

    def index(self, docs: Sequence[Doc]) -> None:
        import numpy as np  # noqa: PLC0415

        self._docs = list(docs)
        if not self._docs:
            self._emb = None
            return
        vecs = self._model.encode(
            [d.text for d in self._docs], normalize_embeddings=True, show_progress_bar=False
        )
        self._emb = np.asarray(vecs, dtype="float32")

    def search(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        return [h.as_tuple() for h in self.search_detailed(query, k=k)]

    def search_detailed(self, query: str, k: int = 8) -> list[Hit]:
        import numpy as np  # noqa: PLC0415

        if self._emb is None or not self._docs:
            return []
        q = np.asarray(
            self._model.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )[0]
        sims = self._emb @ q
        order = np.argsort(-sims)[:k]
        # Dense retrieval has no term notion. Cosine similarity is already
        # scale-free in [0,1], so it stands in for both coverage and matched
        # information mass; callers gating on ``matched_idf`` should therefore
        # use a cosine-scaled floor when running dense-only.
        return [
            Hit(
                self._docs[i].doc_id,
                float(sims[i]),
                matched_terms=1,
                query_terms=1,
                matched_idf=float(sims[i]),
                query_idf=1.0,
            )
            for i in order
            if sims[i] > 0
        ]


class HybridRetriever:
    """Reciprocal-rank fusion of lexical and dense results.

    RRF rather than score addition, because BM25 scores and cosine similarities
    live on incomparable scales and normalising them per query is fragile.
    """

    name = "hybrid-rrf"

    def __init__(self, lexical: Retriever, dense: Retriever, rrf_k: int = 60) -> None:
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k

    def index(self, docs: Sequence[Doc]) -> None:
        self.lexical.index(docs)
        self.dense.index(docs)

    def search(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        return [h.as_tuple() for h in self.search_detailed(query, k=k)]

    def search_detailed(self, query: str, k: int = 8) -> list[Hit]:
        fused: dict[str, float] = {}
        coverage: dict[str, Hit] = {}
        for retriever in (self.lexical, self.dense):
            for rank, hit in enumerate(retriever.search_detailed(query, k=k * 2)):
                fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                # Keep the lexical arm's term coverage; dense has none to give.
                if hit.doc_id not in coverage or hit.matched_idf > coverage[hit.doc_id].matched_idf:
                    coverage[hit.doc_id] = hit
        ranked = sorted(fused.items(), key=lambda t: (-t[1], t[0]))[:k]
        out: list[Hit] = []
        for doc_id, score in ranked:
            lex = coverage.get(doc_id)
            out.append(
                Hit(
                    doc_id,
                    score,
                    matched_terms=lex.matched_terms if lex else 0,
                    query_terms=lex.query_terms if lex else 0,
                    matched_idf=lex.matched_idf if lex else 0.0,
                    query_idf=lex.query_idf if lex else 0.0,
                )
            )
        return out


def build_retriever(prefer_dense: bool = False) -> Retriever:
    """Dense when the optional extra is installed, lexical otherwise."""
    if prefer_dense:
        try:
            return HybridRetriever(BM25Retriever(), EmbeddingRetriever())
        except Exception:  # noqa: BLE001 - missing torch/ST or no model cache
            pass
    return BM25Retriever()

