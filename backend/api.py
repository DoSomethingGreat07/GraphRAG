import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.service import (
    CUSTOM_BACKEND_EXACT,
    discover_artifact_profiles,
    execute_custom_query,
    execute_dataset_query,
    get_runtime_config,
    list_examples,
)


app = FastAPI(title="GraphRAG API", version="0.1.0")

allowed_origins_env = os.getenv("GRAPH_RAG_CORS_ORIGINS", "*")
allowed_origins = [
    origin.strip()
    for origin in allowed_origins_env.split(",")
    if origin.strip()
]
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    profile_id: str
    retrieval_mode: str = Field(default="FAISS-only retrieval")
    top_k: int = Field(default=5, ge=1, le=10)
    lambda_dense: float = Field(default=0.5, ge=0.0, le=1.0)
    llm_enabled: bool = False
    compare_all_modes: bool = False
    question: str | None = None
    example_id: str | None = None
    custom_backend: str = CUSTOM_BACKEND_EXACT


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/profiles")
def profiles():
    return {"profiles": discover_artifact_profiles()}


@app.get("/api/config")
def config(profile_id: str = Query(...)):
    try:
        return get_runtime_config(profile_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/examples")
def examples(profile_id: str = Query(...), question_type: str = Query(default="all")):
    try:
        return {"examples": list_examples(profile_id, question_type=question_type)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/query/dataset")
def query_dataset(payload: QueryRequest):
    if not payload.example_id:
        raise HTTPException(status_code=400, detail="example_id is required for dataset queries")

    try:
        return execute_dataset_query(
            profile_id=payload.profile_id,
            example_id=payload.example_id,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
            lambda_dense=payload.lambda_dense,
            llm_enabled=payload.llm_enabled,
            compare_all_modes=payload.compare_all_modes,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/query/custom")
def query_custom(payload: QueryRequest):
    if not payload.question:
        raise HTTPException(status_code=400, detail="question is required for custom queries")

    try:
        return execute_custom_query(
            profile_id=payload.profile_id,
            question=payload.question,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
            lambda_dense=payload.lambda_dense,
            llm_enabled=payload.llm_enabled,
            compare_all_modes=payload.compare_all_modes,
            custom_backend=payload.custom_backend,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
