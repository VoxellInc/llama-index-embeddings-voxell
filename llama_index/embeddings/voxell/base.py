"""Voxell Forge embeddings for LlamaIndex."""

from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager

DEFAULT_BASE_URL = "https://api.voxell.ai"


class VoxellEmbedding(BaseEmbedding):
    """
    Voxell Forge embeddings.

    Forge is Voxell's hosted text-embedding API with turbo, pro, and ultra tiers.
    Set the FORGE_API_KEY environment variable or pass ``api_key``. Create a free
    key at https://dash.voxell.ai.

    Examples:
        >>> from llama_index.embeddings.voxell import VoxellEmbedding
        >>> emb = VoxellEmbedding(model="turbo")  # FORGE_API_KEY from env
        >>> vector = emb.get_text_embedding("hello world")

    """

    base_url: str = Field(default=DEFAULT_BASE_URL, description="Forge API base URL.")
    dimensions: Optional[int] = Field(
        default=None,
        description="Optional Matryoshka truncation dimension; vectors are re-normalized.",
    )
    timeout: float = Field(default=30.0, description="Per-request timeout in seconds.")

    _api_key: str = PrivateAttr()
    _headers: Dict[str, str] = PrivateAttr()

    def __init__(
        self,
        model: str = "turbo",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model,
            base_url=base_url.rstrip("/"),
            dimensions=dimensions,
            timeout=timeout,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )
        api_key = get_from_param_or_env("api_key", api_key, "FORGE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Forge API key missing: pass api_key=... or set FORGE_API_KEY. "
                "Create one at https://dash.voxell.ai."
            )
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llama-index-embeddings-voxell",
        }

    @classmethod
    def class_name(cls) -> str:
        return "VoxellEmbedding"

    @property
    def _url(self) -> str:
        return f"{self.base_url}/v1/embed"

    def _body(self, texts: List[str], input_type: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "texts": texts,
            "model": self.model_name,
            "input_type": input_type,
        }
        if self.dimensions is not None:
            body["dim"] = self.dimensions
        return body

    @staticmethod
    def _parse(resp: httpx.Response) -> List[List[float]]:
        if resp.status_code != 200:
            raise RuntimeError(f"Forge API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Unexpected Forge response: {data}")
        return embeddings

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._url, headers=self._headers, json=self._body(texts, input_type)
            )
        return self._parse(resp)

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._url, headers=self._headers, json=self._body(texts, input_type)
            )
        return self._parse(resp)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], "query")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], "document")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), "document")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed([query], "query"))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed([text], "document"))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(list(texts), "document")
