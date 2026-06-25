# TwelveLabs Marengo-backed retriever.
#
# Drop-in replacement for `retrieve_documents_with_dynamic` in
# `rag_retriever_dynamic.py`. Instead of the local Contriever model, it embeds
# the auxiliary texts (OCR / ASR) and the query with TwelveLabs Marengo
# (512-dim multimodal embeddings) and keeps the same FAISS inner-product
# range-search retrieval, so the rest of the pipeline is unchanged.
#
# Opt-in only: nothing imports this unless `USE_TWELVELABS_RETRIEVER=1` is set
# in `vidrag_pipeline.py`. Requires `pip install twelvelabs` and the
# environment variable `TWELVELABS_API_KEY` (grab a free key at
# https://twelvelabs.io).

import os
import numpy as np
import faiss
from twelvelabs import TwelveLabs

# Marengo text embeddings are 512-dim; embedding the OCR/ASR docs and the query
# in the same space lets us reuse the existing FAISS retrieval unchanged.
MARENGO_MODEL = os.getenv("TWELVELABS_EMBED_MODEL", "marengo3.0")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TWELVELABS_API_KEY is not set. Get a free key at https://twelvelabs.io"
            )
        _client = TwelveLabs(api_key=api_key)
    return _client


def text_to_vector(text):
    """Embed a single string with Marengo, returning a 512-dim numpy vector."""
    resp = _get_client().embed.create(model_name=MARENGO_MODEL, text=text)
    return np.array(resp.text_embedding.segments[0].float_, dtype=np.float64)


def retrieve_documents_with_dynamic(documents, queries, threshold=0.4):
    """Marengo-backed twin of `rag_retriever_dynamic.retrieve_documents_with_dynamic`.

    Same signature, same return value `(top_documents, idx)`, so it can be
    swapped in without touching the calling code.
    """
    if isinstance(queries, list):
        query_vectors = np.array([text_to_vector(query) for query in queries])
        average_query_vector = np.mean(query_vectors, axis=0)
        query_vector = average_query_vector / np.linalg.norm(average_query_vector)
        query_vector = query_vector.reshape(1, -1)
    else:
        query_vector = text_to_vector(queries)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1)

    document_vectors = np.array([text_to_vector(doc) for doc in documents])
    document_vectors = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
    dimension = document_vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(document_vectors)
    lims, D, I = index.range_search(query_vector, threshold)
    start = lims[0]
    end = lims[1]
    I = I[start:end]

    if len(I) == 0:
        top_documents = []
        idx = []
    else:
        idx = I.tolist()
        top_documents = [documents[i] for i in idx]

    return top_documents, idx
