# GraphRAG Deployment Guide

This project can be deployed as a FastAPI backend plus React frontend. The backend expects prebuilt artifacts in `artifacts/`; artifact generation and GNN training are offline steps and should not run inside the production web process.

## 1. Prerequisites

- Docker and Docker Compose
- A populated `artifacts/` directory
- Optional `.env` with `OPENAI_API_KEY` if LLM answer generation should be enabled

Required artifact files for the default profile include:

```text
artifacts/hotpotqa_train_n10000_cs300_co50_manifest.json
artifacts/hotpotqa_train_n10000_cs300_co50_chunked_examples.pkl
artifacts/hotpotqa_train_n10000_cs300_co50_graph_examples.pkl
artifacts/hotpotqa_train_n10000_cs300_co50_example_lookup.pkl
artifacts/hotpotqa_train_n10000_cs300_co50_global_example.pkl
artifacts/hotpotqa_train_n10000_cs300_co50_sample_questions.json
artifacts/hotpotqa_train_n10000_cs300_co50_query_aware_graphsage_best.pt
```

## 2. Local Docker Deployment

From the repository root:

```sh
cp .env.example .env
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- API health: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`

The frontend container proxies `/api/*` to the backend container, so production frontend builds can use `VITE_API_BASE_URL=/api`.

## 3. Backend-Only Deployment

Build and run the API image:

```sh
docker build -f Dockerfile.api -t graphrag-api .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e GRAPH_RAG_CORS_ORIGINS="https://your-frontend.example.com" \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  graphrag-api
```

## 4. Frontend-Only Deployment

If the backend is hosted separately, build the frontend with the deployed API URL:

```sh
cd frontend
npm ci
VITE_API_BASE_URL=https://your-api.example.com/api npm run build
```

Deploy `frontend/dist/` to a static hosting provider such as Netlify, Vercel static output, Cloudflare Pages, or an object-store/CDN.

## 5. Environment Variables

Backend:

- `OPENAI_API_KEY`: optional; required only for LLM-generated answers.
- `GRAPH_RAG_CORS_ORIGINS`: comma-separated frontend origins. Use the deployed frontend URL in production.
- `PORT`: backend port; defaults to `8000` in the Docker image.

Frontend:

- `VITE_API_BASE_URL`: API base URL, for example `/api` behind the included Nginx proxy or `https://your-api.example.com/api`.

## 6. Production Notes

- Keep `artifacts/` outside the image when possible and mount it read-only.
- The first request may be slow because the API loads artifacts, the embedding model, ANN indexes, and the GNN checkpoint lazily.
- Use a machine with enough memory for the artifact bundle and embedding model.
- Keep `OPENAI_API_KEY` secret; do not expose it to the frontend.
- For a public API, replace wildcard CORS with the exact deployed frontend origin.
- For faster cold starts, use persistent Hugging Face cache storage or bake the model cache into a private image.

## 7. Smoke Tests

After deployment:

```sh
curl http://localhost:8000/api/health
curl http://localhost:8000/api/profiles
```

Then open the frontend and run a dataset query with `FAISS-only retrieval` before trying GNN or PCST modes.
