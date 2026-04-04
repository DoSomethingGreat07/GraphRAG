# GraphRAG Multi-Hop Question Answering

GraphRAG is a Streamlit application for multi-hop question answering over a HotpotQA-style Wikipedia corpus. It combines dense semantic retrieval with graph-based reasoning, optional GNN reranking, and optional LLM answer generation.

This project is built around a simple idea:

multi-hop questions are not just about finding the most similar passage, they are about finding the chain of evidence that connects the answer.

Instead of treating retrieval as a flat top-k ranking problem, this system builds a graph over chunked Wikipedia-style documents and uses that graph to surface bridge evidence that a standard RAG pipeline may miss.

## Why This Repo Is Useful

- It shows a full GraphRAG workflow from data preparation to interactive QA.
- It separates offline indexing from online inference, which makes the runtime app easier to understand.
- It supports both evaluation on labeled dataset questions and open-ended custom questions.
- It exposes multiple retrieval strategies so you can compare dense-only retrieval against graph-aware approaches.
- It is practical to demo through Streamlit while still being structured enough to study as a research-style system.

This repository is structured around two stages:

1. Offline preparation
   Build chunked documents, embeddings, graphs, and lookup artifacts.
2. Online inference
   Run the Streamlit app and answer either predefined dataset questions or custom user questions from the indexed corpus.

The goal of this README is to explain the system end to end in a way that is easy to follow, from raw data to the final answer shown in the UI.

## Quick Start

If you want the fastest path from clone to a working demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 prepare_artifacts.py --max-samples 1000
streamlit run app.py
```

Then open the app and:

1. start with `Dataset Question Mode`
2. keep retrieval mode on `Dense`
3. verify artifacts load correctly
4. train the GNN later if you want `Fusion` and `PCST`

If you only want retrieval and do not need GPT-generated answers, you can still run the app without enabling answer generation in the UI.

## Demo Walkthrough

Once the app is running, a typical flow looks like this:

1. Select an artifact profile from the sidebar.
2. Choose a retrieval mode: `Dense`, `Fusion`, or `PCST`.
3. Open `Dataset Question Mode` to inspect known examples with gold supporting titles.
4. Open `Custom Question Mode` to test free-form queries against the indexed corpus.
5. Review retrieved titles and evidence chunks.
6. Optionally enable GPT answer generation to synthesize a final answer from retrieved evidence.

If you want to add screenshots later, this is a good place to include:

- the main app landing page
- dataset question mode with gold-title comparison
- custom question mode with retrieved evidence
- retrieval mode comparison table

## What This Project Does

Standard dense retrieval is often good at finding passages that look semantically similar to the question, but multi-hop questions usually require connecting facts across multiple documents.

Example:

`Where was the director of Titanic born?`

To answer that, the system may need to retrieve:

- a chunk about `Titanic`
- a chunk about `James Cameron`
- a chunk containing the birthplace fact

This project is designed to improve retrieval for those kinds of questions by building a graph over the corpus and then reasoning over that graph.

## High-Level Architecture

The full pipeline looks like this:

```text
HotpotQA / Wikipedia-style data
  -> document chunking
  -> embedding generation
  -> hybrid graph construction
  -> cached artifacts saved to disk
  -> Streamlit app loads artifacts
  -> retrieval over indexed corpus
  -> optional graph/GNN reranking
  -> optional GPT answer generation
  -> answer + evidence shown in UI
```

There are two main usage paths in the app:

- `Dataset Question Mode`
  Uses a known question from the indexed dataset. Ground-truth supporting titles are available, so the app can compare retrieved evidence against gold evidence.
- `Custom Question Mode`
  Uses any user-typed question. The system searches the indexed corpus, but there is no ground-truth label for that question, so only retrieved evidence can be shown.

## Repository Layout

```text
app.py                          Streamlit application
prepare_artifacts.py            Offline artifact creation script
requirements.txt                Python dependencies
.env.example                    Example environment file
artifacts/                      Generated artifact cache (not committed)
graphrag_env/src/
  artifact_runtime.py           Artifact loading and checkpoint loading
  artifact_utils.py             Artifact pathing and save/load helpers
  loading.py                    HotpotQA data loading
  chunking.py                   Document chunking
  embeddings.py                 Embedding generation
  hybrid_graph_builder.py       Graph construction
  retrieval.py                  Dense retrieval
  gnn_fusion_retreival.py       Dense + GNN fusion retrieval
  pcst.py                       PCST retrieval
  gnn_train.py                  GNN training
  llm_eval.py                   LLM answer generation
```

## End-to-End Flow

### 1. Data Loading

The pipeline begins by loading HotpotQA-style examples. Each example contains:

- a question
- an answer
- multiple context documents
- supporting facts for dataset-labeled questions

In the distractor setting, a question may come with multiple context rows. Typically only a small subset are the true supporting documents, while the rest are distractors.

### 2. Chunking

Each context document is split into smaller text chunks. This helps retrieval because:

- long documents are hard to embed and rank as a whole
- smaller units give more precise matching
- graph reasoning can operate at chunk level

Chunk configuration is controlled by artifact parameters such as:

- `chunk_size`
- `chunk_overlap`

### 3. Embedding Generation

Each chunk is converted into a dense vector using a sentence-transformer model. The default artifact preparation currently uses:

- `BAAI/bge-base-en-v1.5`

At query time, the question is embedded with the same model so chunk similarity can be computed.

### 4. Hybrid Graph Construction

After chunking and embedding, the project builds a graph over the chunks. Nodes are chunks. Edges connect chunks that appear related based on corpus structure or content.

The graph can include signals such as:

- chunks from the same document
- adjacent chunks
- title or hyperlink-like mentions
- keyword overlap
- semantic relationships when enabled

This graph is what allows graph-aware retrieval modes to expand beyond purely lexical or dense similarity.

### 5. Artifact Creation

The offline script [`prepare_artifacts.py`](/Users/nikhiljuluri/Desktop/GraphRAG/prepare_artifacts.py) saves reusable files into [`artifacts/`](/Users/nikhiljuluri/Desktop/GraphRAG/artifacts), including:

- chunked examples
- graph examples
- example lookup table
- one global merged example used for custom search
- sample questions
- a manifest describing the artifact profile

These artifacts let the app start quickly without rebuilding the full pipeline every time.

### 6. Optional GNN Training

If you want to use `Fusion` and `PCST` retrieval modes fully, train the GNN checkpoint with:

```bash
python3 graphrag_env/src/gnn_train.py
```

Without a trained checkpoint, the app can still run in `Dense` mode.

### 7. Runtime App Loading

When the app starts, [`app.py`](/Users/nikhiljuluri/Desktop/GraphRAG/app.py) does the following:

- discovers available artifact profiles from the manifest files
- loads the selected artifact bundle
- loads the embedding model named in the manifest
- tries to load the GNN checkpoint for that profile
- prepares global chunk and graph structures for custom retrieval
- optionally prepares ANN indexes for custom question retrieval if `hnswlib` or `faiss` is installed

### 8. Query-Time Retrieval

At runtime, the app supports three retrieval modes:

- `Dense`
- `Fusion`
- `PCST`

These modes behave slightly differently depending on whether the question comes from the dataset or from custom input.

## Dataset Question Mode

Dataset mode is used for predefined indexed questions.

Flow:

1. Select a dataset question from the UI.
2. Load the corresponding indexed example.
3. Run retrieval with the selected mode.
4. Show retrieved titles and evidence chunks.
5. Compare retrieved titles to gold supporting titles.
6. Optionally generate a final answer with GPT.

Important property:

- because these questions come from the labeled dataset, the app knows the gold supporting titles
- this is why the UI can show both `Retrieved titles` and `Gold supporting titles`

This makes dataset mode useful for inspection and qualitative evaluation.

## Custom Question Mode

Custom mode is used when a user types any free-form question.

Flow:

1. User enters a question.
2. The question is embedded.
3. The system searches the indexed corpus.
4. A candidate pool is selected.
5. Retrieval mode runs on those candidates.
6. Retrieved evidence is shown.
7. Optionally GPT generates the final answer from the retrieved chunks.

Important property:

- custom questions do not have dataset ground truth
- the app cannot know the gold supporting titles unless the question exactly maps to a labeled dataset item and you explicitly build that lookup behavior
- therefore custom mode shows retrieved evidence only

## Retrieval Modes Explained

### Dense

Dense retrieval ranks chunks by embedding similarity to the question.

This is the simplest mode and works even when there is no trained GNN checkpoint.

Strengths:

- simple
- reliable baseline
- fast compared with graph-heavy reasoning

Limitations:

- may miss bridge documents needed for multi-hop reasoning

### Fusion

Fusion combines dense retrieval signals with GNN-based scores.

This mode is designed to preserve direct semantic relevance while also surfacing graph-supported bridge evidence.

Strengths:

- better at multi-hop evidence than dense alone
- balances direct query similarity and graph-aware relevance

Requirements:

- trained GNN checkpoint

### PCST

PCST uses Prize-Collecting Steiner Tree style selection to choose a connected, high-value subgraph.

This mode can be useful when the best answer depends on a connected chain of evidence rather than just the individually highest-scoring chunks.

Strengths:

- encourages coherent evidence selection
- useful for multi-hop reasoning paths

Requirements:

- trained GNN checkpoint

## Custom Retrieval Backends

Custom questions support different first-pass retrieval backends:

- `Exact`
- `HNSW`
- `IVF`

These affect only `Custom Question Mode`.

### Exact

The app embeds the question and scores it against every indexed chunk embedding. This is exact but its latency grows with corpus size.

Best for:

- smaller corpora
- debugging
- maximum exactness

### HNSW

If `hnswlib` is installed, the app can use an HNSW approximate nearest-neighbor index for custom questions.

Best for:

- lower latency on larger corpora
- interactive search

### IVF

If `faiss` is installed, the app can use an IVF index for custom questions.

Best for:

- larger-scale approximate retrieval with FAISS

If ANN libraries are not installed, the app falls back to `Exact`.

## Why Dataset Flow and Custom Flow Differ

This distinction is important:

- `Dataset Question Mode` works with a single labeled example that already has known supporting facts.
- `Custom Question Mode` searches across the indexed global corpus and has no gold labels.

So:

- dataset mode can display gold supporting titles
- custom mode cannot

Also:

- dataset mode preserves the original retrieval path for each indexed example
- custom mode may use exact or ANN candidate search across the full indexed corpus before running graph-based retrieval on the smaller candidate pool

## Artifact Profiles

The app supports multiple artifact profiles. A profile is defined by settings like:

- `split`
- `max_samples`
- `chunk_size`
- `chunk_overlap`

When you change the selected artifact profile in the sidebar, the app loads the corresponding:

- manifest
- example lookup
- global example
- graph examples
- checkpoint, if available

This makes it possible to compare different corpus sizes or chunking choices without changing application code.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/DoSomethingGreat07/graphrag-hotpotqa
cd graphrag-hotpotqa
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional ANN dependencies for custom question acceleration:

```bash
pip install hnswlib
```

or

```bash
pip install faiss-cpu
```

### 4. Add Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then set:

```env
OPENAI_API_KEY=your_key_here
```

If you do not enable GPT answer generation in the UI, the app can still be used in retrieval-only mode.

### 5. Build Artifacts

Run:

```bash
python3 prepare_artifacts.py
```

This creates the cached files needed by the app under [`artifacts/`](/Users/nikhiljuluri/Desktop/GraphRAG/artifacts).

Example with explicit parameters:

```bash
python3 prepare_artifacts.py \
  --split train \
  --max-samples 10000 \
  --chunk-size 300 \
  --chunk-overlap 50
```

### 6. Train the GNN Checkpoint

If you want `Fusion` and `PCST`:

```bash
python3 graphrag_env/src/gnn_train.py
```

If you only need `Dense`, you can skip this step.

### 7. Launch the App

```bash
streamlit run app.py
```

## Typical Usage

### Evaluate Known Dataset Questions

Use this when you want to inspect whether the system retrieves the known gold supporting evidence.

Good for:

- debugging retrieval quality
- comparing Dense vs Fusion vs PCST
- checking gold-title overlap

### Ask Custom Questions

Use this when you want to treat the system like a real QA application over the indexed corpus.

Good for:

- demos
- qualitative testing
- interactive exploration

Best results usually come from:

- multi-hop factoid questions
- HotpotQA-style phrasing
- questions whose evidence exists in the indexed corpus

## What Gets Stored in `artifacts/`

The `artifacts/` directory can become large, so it is intentionally excluded from git.

It typically contains:

- `*_manifest.json`
- `*_chunked_examples.pkl`
- `*_graph_examples.pkl`
- `*_example_lookup.pkl`
- `*_global_example.pkl`
- `*_sample_questions.json`
- GNN checkpoints for a matching artifact profile

This separation keeps the repository lightweight while still allowing reproducible local builds.

## Deployment Notes

For GitHub:

- commit source code
- commit `README.md`, `requirements.txt`, `.env.example`, and `.gitignore`
- do not commit `artifacts/`
- do not commit local `.env`
- do not commit caches, checkpoints, or editor metadata unless intentionally versioned

For a fresh deployment environment:

1. install dependencies
2. create `.env`
3. rebuild artifacts
4. optionally train or copy the GNN checkpoint
5. run `streamlit run app.py`

If you plan to deploy to a hosted environment, make sure that:

- the host has enough disk space for artifacts
- model downloads are allowed
- large artifact generation happens ahead of time if startup time matters

## Known Practical Tradeoffs

### Exact custom retrieval vs ANN custom retrieval

Exact search:

- scores against all indexed chunks
- highest exactness
- slower as corpus grows

ANN search:

- faster for larger corpora
- approximate candidate selection
- requires optional libraries like `hnswlib` or `faiss`

### Dense vs graph-aware retrieval

Dense:

- simpler
- usually faster
- good baseline

Fusion and PCST:

- better for bridge-document style reasoning
- more complex
- require checkpoint support

## Recommended First Run

If you want the fastest path to seeing the app work:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 prepare_artifacts.py --max-samples 1000
streamlit run app.py
```

Then:

- start in `Dataset Question Mode`
- use `Dense`
- verify artifacts loaded correctly
- train the GNN later if you want `Fusion` and `PCST`

## Troubleshooting

### `Artifacts not found`

Run:

```bash
python3 prepare_artifacts.py
```

### `GNN checkpoint not found`

Run:

```bash
python3 graphrag_env/src/gnn_train.py
```

Or switch the app to `Dense` mode.

### Custom ANN backend is not available

Install one of:

```bash
pip install hnswlib
```

or

```bash
pip install faiss-cpu
```

Then restart the app.

### GPT answers are unavailable

Check:

- `.env` exists
- `OPENAI_API_KEY` is set
- GPT answer generation is enabled in the sidebar

## Summary

This project is best understood as a two-stage GraphRAG system:

1. build a searchable graph-structured corpus offline
2. run retrieval and answer generation online through a Streamlit app

If you are evaluating retrieval quality, use dataset mode.
If you are demoing open-ended QA over the indexed corpus, use custom mode.

Both flows share the same indexed knowledge base, but they differ in one critical way: dataset questions have gold evidence labels, while custom questions do not.
