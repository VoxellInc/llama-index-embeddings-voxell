# LlamaIndex Embeddings Integration: Voxell

[Voxell Forge](https://voxell.ai/forge) is a hosted text-embedding API with three tiers (turbo, pro, and ultra) on an OpenAI-compatible endpoint.

Voxell's Ingot-8B-R3 ranks #1 for English on the public MTEB leaderboard (English v2), with a 75.98 mean task score across 41 tasks. It is the top usable English embedding model. See the [model card](https://huggingface.co/JCorners/Ingot-8B-R3), or try Forge with no signup on the [playground](https://playground.voxell.ai).

## Installation

```bash
pip install llama-index-embeddings-voxell
```

## Setup

Create a free API key at [dash.voxell.ai](https://dash.voxell.ai) (new accounts include 10M free tokens), then set it in your environment:

```bash
export FORGE_API_KEY="your_api_key_here"
```

## Usage

```python
from llama_index.embeddings.voxell import VoxellEmbedding

emb = VoxellEmbedding(model="turbo")  # reads FORGE_API_KEY from the environment

vector = emb.get_text_embedding("Voxell Forge turns text into vectors.")
print(len(vector))  # 1024

query_vector = emb.get_query_embedding("How do I turn text into vectors?")
```

## Tiers

Pick your point on the quality and cost curve:

| Tier | Dimensions |
| ------- | ---------- |
| `turbo` | 1024 |
| `pro` | 2560 |
| `ultra` | 4096 |

## Shorter vectors with Matryoshka

Pass `dimensions` to get a shorter, re-normalized vector, which means a smaller index with minimal quality loss.

```python
emb = VoxellEmbedding(model="turbo", dimensions=256)
short_vector = emb.get_query_embedding("a compact embedding")
```

## Async

```python
vector = await emb.aget_text_embedding("async text")
```
