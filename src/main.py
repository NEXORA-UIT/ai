"""HTTP process entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import environ

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from src.api.answers import router as answers_router
from src.api.documents import router as documents_router
from src.api.search import router as search_router
from src.embedding.runtime import get_default_embedder
from src.reranking.config import get_rerank_candidate_limit, get_rerank_max_length
from src.reranking.cross_encoder import get_default_reranker
from src.retrieval.config import DEFAULT_SEARCH_TOP_K


def _warmup_models() -> None:
    get_default_embedder().embed_texts(["Nexora model warmup"])
    get_default_reranker().score_pairs("Nexora", ["Nexora model warmup"])


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_rerank_candidate_limit(top_k=DEFAULT_SEARCH_TOP_K)
    get_rerank_max_length()
    if environ.get("AI_WARMUP_MODELS") == "1":
        await run_in_threadpool(_warmup_models)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Nexora AI Service", version="0.1.0", lifespan=_lifespan)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(answers_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
