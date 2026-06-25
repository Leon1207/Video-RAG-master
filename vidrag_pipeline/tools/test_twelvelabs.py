# Focused tests for the TwelveLabs integration. No test framework needed:
#   python tools/test_twelvelabs.py
#
# - test_retriever_logic: no network. Stubs Marengo embeddings so the FAISS
#   range-search retrieval is exercised deterministically.
# - test_marengo_live: skipped unless TWELVELABS_API_KEY is set; asserts a real
#   512-dim Marengo text embedding comes back.

import os
import sys

import numpy as np

import rag_retriever_twelvelabs as r


def test_retriever_logic():
    # Three docs in a tiny embedding space; the query points at doc 0/1.
    space = {
        "cat on a mat": [1.0, 0.0, 0.0],
        "kitten on a rug": [0.9, 0.1, 0.0],
        "stock market crash": [0.0, 0.0, 1.0],
        "house cats and rugs": [0.95, 0.05, 0.0],  # the query text
    }
    r.text_to_vector = lambda t: np.array(space[t], dtype=np.float64)

    docs = ["cat on a mat", "kitten on a rug", "stock market crash"]
    top, idx = r.retrieve_documents_with_dynamic(docs, ["house cats and rugs"], threshold=0.8)

    assert "stock market crash" not in top, top
    assert "cat on a mat" in top and "kitten on a rug" in top, top
    assert set(idx) == {0, 1}, idx
    print("test_retriever_logic: ok")


def test_marengo_live():
    if not os.getenv("TWELVELABS_API_KEY"):
        print("test_marengo_live: skipped (no TWELVELABS_API_KEY)")
        return
    # Force a fresh client (the module caches a stub-friendly one).
    r._client = None
    vec = r.text_to_vector("a person walking a dog in a park")
    assert vec.shape == (512,), vec.shape
    print("test_marengo_live: ok (512-dim)")


if __name__ == "__main__":
    # Live test first: test_retriever_logic monkeypatches text_to_vector.
    test_marengo_live()
    test_retriever_logic()
    print("all tests passed")
    sys.exit(0)
