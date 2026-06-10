"""
Tests for llama-index-embeddings-voxell.

Shape tests run with no network. Live tests run only when FORGE_API_KEY is set
(against api.voxell.ai). Run: pytest tests
"""

import os

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voxell import VoxellEmbedding

LIVE = os.environ.get("FORGE_API_KEY")
DIMS = {"turbo": 1024, "pro": 2560, "ultra": 4096}


def test_class_is_base_embedding():
    names = [b.__name__ for b in VoxellEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names
    assert VoxellEmbedding.class_name() == "VoxellEmbedding"


def test_missing_key_raises():
    saved = os.environ.pop("FORGE_API_KEY", None)
    try:
        with pytest.raises(ValueError):
            VoxellEmbedding(model="turbo")
    finally:
        if saved is not None:
            os.environ["FORGE_API_KEY"] = saved


def test_body_building():
    emb = VoxellEmbedding(model="pro", api_key="test-key", dimensions=256)
    body = emb._body(["a", "b"], "document")
    assert body["texts"] == ["a", "b"]
    assert body["model"] == "pro"
    assert body["input_type"] == "document"
    assert body["dim"] == 256
    assert emb._url == "https://api.voxell.ai/v1/embed"
    assert emb._headers["Authorization"] == "Bearer test-key"


def test_no_dim_omits_field():
    emb = VoxellEmbedding(model="turbo", api_key="test-key")
    assert "dim" not in emb._body(["x"], "query")


@pytest.mark.skipif(not LIVE, reason="FORGE_API_KEY not set, skipping live tests")
@pytest.mark.parametrize("tier", list(DIMS))
def test_live_text_embedding_dim(tier):
    emb = VoxellEmbedding(model=tier)
    vector = emb.get_text_embedding("hello world")
    assert len(vector) == DIMS[tier]


@pytest.mark.skipif(not LIVE, reason="FORGE_API_KEY not set, skipping live tests")
def test_live_query_and_batch():
    emb = VoxellEmbedding(model="turbo")
    assert len(emb.get_query_embedding("a search query")) == DIMS["turbo"]
    vectors = emb.get_text_embedding_batch(["alpha", "beta"])
    assert len(vectors) == 2
    assert len(vectors[0]) == DIMS["turbo"]


@pytest.mark.skipif(not LIVE, reason="FORGE_API_KEY not set, skipping live tests")
def test_live_matryoshka():
    emb = VoxellEmbedding(model="turbo", dimensions=256)
    assert len(emb.get_query_embedding("truncate me")) == 256


@pytest.mark.skipif(not LIVE, reason="FORGE_API_KEY not set, skipping live tests")
@pytest.mark.asyncio
async def test_live_async():
    emb = VoxellEmbedding(model="turbo")
    vector = await emb.aget_text_embedding("async text")
    assert len(vector) == DIMS["turbo"]
