# TwelveLabs Pegasus-backed video understanding.
#
# Drop-in replacement for the local `llava_inference(qs, video)` in
# `vidrag_pipeline.py`. Pegasus is a video-language model served by TwelveLabs,
# so the heavy LLaVA-Video weights are not needed locally: the prompt (already
# augmented with the retrieved OCR / ASR / DET auxiliary texts) is sent to
# Pegasus together with the source video, and the generated answer is returned.
#
# Opt-in only: nothing imports this unless `USE_PEGASUS=1` is set in
# `vidrag_pipeline.py`. Requires `pip install twelvelabs` and the environment
# variable `TWELVELABS_API_KEY` (grab a free key at https://twelvelabs.io).

import os
from twelvelabs import TwelveLabs

PEGASUS_MODEL = os.getenv("TWELVELABS_ANALYZE_MODEL", "pegasus1.5")

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


def pegasus_inference(qs, video):
    """Pegasus-backed twin of `vidrag_pipeline.llava_inference`.

    `qs` is the (RAG-augmented) prompt. `video` selects the source clip and is
    one of:
      - a `{"url": "https://..."}` dict for a publicly reachable video URL,
      - a `{"video_id": "<indexed-id>"}` dict for a video already indexed in
        TwelveLabs,
      - a `{"asset_id": "<asset-id>"}` dict for an uploaded asset,
      - or `None`, which sends a text-only prompt (mirrors `llava_inference`'s
        text-only branch used for the chain-of-thought retrieval step).
    """
    client = _get_client()
    max_tokens = int(os.getenv("TWELVELABS_MAX_TOKENS", "2048"))

    if video is None:
        # Text-only step (e.g. the retrieve-request JSON generation). Pegasus
        # always needs a video, so callers should keep using their text LLM for
        # this branch; we surface a clear error rather than guess.
        raise ValueError(
            "pegasus_inference requires a video context; use a text model for "
            "the video-less chain-of-thought step."
        )

    if "url" in video:
        kwargs = {"video": {"type": "url", "url": video["url"]}}
    elif "asset_id" in video:
        kwargs = {"video": {"type": "asset_id", "asset_id": video["asset_id"]}}
    elif "video_id" in video:
        kwargs = {"video_id": video["video_id"]}
    else:
        raise ValueError(
            "video must contain one of 'url', 'asset_id', or 'video_id'."
        )

    resp = client.analyze(
        model_name=PEGASUS_MODEL,
        prompt=qs,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.data.strip() if isinstance(resp.data, str) else resp.data
