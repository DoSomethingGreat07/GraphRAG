# Project Status Update

## GraphRAG for Multi-Hop Question Answering

Status date: April 6, 2026  
Submission target: April 7, 2026 at 23:59

### 1. Introduction and Motivation

This project explores a graph-based retrieval-augmented generation system for multi-hop question answering. The task is based on HotpotQA-style questions, where answering correctly often requires combining evidence from multiple documents rather than extracting a fact from one passage. A standard dense retriever can work for direct fact lookup, but multi-hop reasoning is harder because the correct answer may depend on a chain of related evidence spread across different Wikipedia articles.

For example, a question such as “Where was the director of *Titanic* born?” cannot be answered reliably by matching only the surface form of the question to one passage. The system may first need to retrieve a passage about *Titanic*, then identify that James Cameron is the director, and then retrieve a passage containing James Cameron’s birthplace. This type of reasoning motivates the use of graph structure on top of dense retrieval.

The main idea of this project is to represent the corpus as a graph of text chunks and use graph-aware retrieval to surface bridge evidence that a flat top-k retriever may miss. Instead of ranking chunks independently, the system links related chunks using document structure, semantic similarity, and keyword overlap. This allows retrieval to move from an initial relevant node to neighboring evidence that may complete the reasoning chain.

In addition to improved retrieval, the project also aims to provide an end-to-end interactive demo. The current system includes an offline artifact-building stage, multiple retrieval modes, an optional Graph Neural Network (GNN) reranker, and a Streamlit interface for both benchmark questions and user-entered custom questions. The broader motivation is to study whether graph-aware retrieval improves evidence quality and final answer quality for multi-hop QA, while still keeping the system practical enough to demo and extend.

### 2. Problem Formulation

The problem can be framed as multi-hop question answering over a large set of Wikipedia-style documents. Given a question, a collection of chunked passages, and a graph whose nodes correspond to chunks, the system must retrieve a small evidence set that collectively supports the answer and then optionally generate a final answer from those chunks.

There are two linked subproblems:

1. **Evidence retrieval:** identify the top-k chunks most useful for answering the question.
2. **Answer generation:** produce a short final answer using only the retrieved evidence.

The difficulty is that relevant evidence may not be concentrated in one chunk. Multi-hop questions often require retrieving at least two supporting documents, and dense similarity alone may favor locally similar but incomplete passages. Therefore, the project treats retrieval as a graph-aware reasoning problem rather than a purely independent ranking problem.

The current system uses HotpotQA-style examples with known supporting titles for dataset-mode evaluation. This makes it possible to compare retrieved evidence against gold supporting documents and assess downstream answer quality. For custom questions entered in the app, there is no gold label, so evaluation is qualitative.

### 3. Methods and Current Approach

The system currently consists of two stages: offline artifact preparation and online inference.

#### 3.1 Offline data preparation

The offline pipeline starts from HotpotQA distractor-format data. Each example contains a question, answer, context documents, and supporting facts. The documents are chunked into smaller units so retrieval can operate at a finer granularity than full articles. The current artifact profile uses:

- training split
- 10,000 examples
- chunk size 300
- chunk overlap 50
- embedding model `BAAI/bge-base-en-v1.5`

The prepared artifact manifest shows that this pipeline currently indexes **10,000 examples** and produces **263,113 global chunks**. These artifacts are cached to disk so the app does not need to rebuild the dataset each time.

#### 3.2 Dense retrieval baseline

Each chunk is embedded using a sentence-transformer model, and a question is embedded with the same model at query time. Dense retrieval ranks chunks by similarity to the query embedding. This serves as the baseline and is also used as the first-stage candidate generator for custom questions.

For custom search, the system also supports approximate nearest-neighbor backends when available, including HNSW and FAISS IVF. This is useful because the full global chunk pool is large enough that efficient retrieval becomes important for interactive use.

#### 3.3 Hybrid graph construction

After chunking and embedding, the system constructs a hybrid graph over the chunks. Nodes represent chunks, and edges capture structural proximity, semantic similarity, and keyword overlap. The current artifact-building script exposes graph parameters including:

- `semantic_k = 2`
- `semantic_min_sim = 0.40`
- `keyword_overlap_threshold = 3`

This graph is the key component that makes the project a GraphRAG system rather than a standard dense retriever. The goal is to allow evidence expansion from an initially relevant node to neighboring nodes that may complete a reasoning path.

#### 3.4 Query-aware GraphSAGE model

To further improve retrieval, the project includes a query-aware GraphSAGE model. For each node, the feature vector concatenates:

- the chunk embedding
- the repeated query embedding
- the chunk-query cosine similarity

The GNN is trained as a node classification model to score whether a chunk is likely to be supporting evidence for the current question. The implementation includes two graph convolution layers, batch normalization, dropout, a residual connection, and a final linear classifier. The repository already contains trained checkpoint files, so the graph-aware modes can be run directly in the app.

#### 3.5 Retrieval modes

The current app supports three retrieval modes:

- **Dense:** direct embedding similarity baseline
- **Fusion:** combines dense retrieval and GNN-based relevance scores
- **PCST:** uses graph-based selection over fused scores, intended to encourage more coherent evidence sets

For dataset questions, these modes can be compared against gold supporting titles. For custom questions, the app displays retrieved chunks, titles, and optional final answers.

#### 3.6 Answer generation

The system optionally generates a final answer from retrieved evidence using an LLM. The current evaluation script uses `gpt-4o-mini` with a constrained prompt that asks for a short JSON-formatted answer based only on the provided context. This stage is designed to test whether better retrieval quality leads to better answer quality.

### 4. Preliminary Results

The project has moved beyond setup and now has a working end-to-end pipeline: artifact preparation, graph construction, GNN training/checkpoint loading, retrieval, optional answer generation, and an interactive Streamlit interface.

The infrastructure side is functioning: the repository contains a prepared artifact bundle for the main profile (`train`, 10,000 examples, chunk size 300, overlap 50), a merged global retrieval index, and a saved best GNN checkpoint. In addition, the repository includes answer-level evaluation results on **300 questions** using top-5 retrieved evidence and LLM answer generation. These numbers let us compare the effect of different retrieval strategies on downstream question answering.

The current saved results are:

- **Dense retrieval:** EM = **0.6567**, F1 = **0.7305**
- **Fusion retrieval:** EM = **0.6733**, F1 = **0.7535**
- **PCST retrieval:** EM = **0.6667**, F1 = **0.7468**

These results suggest that graph-aware retrieval is helping. In particular, the **Fusion** method currently gives the best overall answer quality among the three tested modes, improving over the dense baseline by about **1.7 EM points** and **2.3 F1 points** on this 300-question evaluation subset. This is a promising sign that combining dense similarity with graph-based node scoring improves retrieved evidence and, in turn, final answers.

The saved breakdown by question type is also informative. For **bridge questions**, Fusion achieves:

- bridge EM = **0.6624**
- bridge F1 = **0.7508**

compared with Dense:

- bridge EM = **0.6414**
- bridge F1 = **0.7216**

This is encouraging because bridge questions are exactly the setting where graph-aware evidence chaining should matter most. For **comparison questions**, performance is closer across methods, and PCST shows the best comparison F1 at **0.7745**, although the sample size is smaller (**63 questions**). This suggests that different retrieval strategies may have different strengths depending on question type.

At the same time, the system is not solved yet. Some failures happen because the correct evidence is not fully retrieved, while others happen because the answer generator outputs an incomplete or overly literal response even when relevant titles are present. There are also examples where retrieval returns near-miss passages from related entities or duplicate titles instead of a cleaner multi-document evidence chain. The current system should therefore be viewed as a working and promising prototype rather than a finished high-accuracy QA solution.

### 5. Future Plans

The next phase of the project will focus on making the evaluation stronger and the graph-aware retrieval more robust. One immediate priority is to run more systematic retrieval evaluation, not only answer-level EM and F1. Because this is fundamentally a retrieval project, it is important to measure support recall, partial support recall, and evidence coverage across the full dataset or a larger held-out subset. This will help separate retrieval failures from answer-generation failures.

Another important direction is improving graph construction. The current graph already uses semantic and keyword-based connections, but it can likely be strengthened with better edge weighting, title-link heuristics, or entity-based edges. Since multi-hop reasoning depends heavily on the quality of the graph, improvements here may have a large effect on bridge-question performance.

On the modeling side, we plan to tune the GNN and fusion strategy further. Potential improvements include:

- tuning the dense/GNN fusion weight
- experimenting with deeper or alternative GNN architectures
- refining positive labels at the chunk level
- adjusting candidate-pool sizes before graph reranking

We also plan to study PCST behavior more carefully. It appears competitive, especially on some comparison-style questions, but its tradeoff between coherence and recall needs more analysis.

From a system perspective, the Streamlit app can be expanded to show more diagnostics, such as graph statistics, retrieval overlap with gold titles, and side-by-side comparisons of retrieval modes on the same question. This would make the project stronger both as a research prototype and as a presentation/demo artifact.

Finally, a likely near-term milestone is to prepare a cleaner experimental section with tables summarizing:

- retrieval metrics by mode
- answer metrics by mode
- bridge vs. comparison performance
- qualitative case studies of successful and failed multi-hop retrieval

### 6. Summary

Overall, the project is in a solid intermediate stage. The main GraphRAG pipeline is implemented, artifacts have been built at nontrivial scale, graph-aware retrieval modes are working, and initial answer-level results suggest that graph-based reranking improves over a dense-only baseline. The current best preliminary setting is the Fusion approach, which gives the strongest overall EM and F1 on the saved 300-question evaluation run.

The most important remaining work is to deepen the experimental analysis and continue improving retrieval quality, especially for challenging bridge questions. Even at this stage, however, the project already demonstrates the central hypothesis: for multi-hop QA, it is useful to move beyond flat retrieval and explicitly model relationships between evidence chunks.
