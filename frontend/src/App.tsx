import { FormEvent, useEffect, useMemo, useState } from "react";

type Profile = {
  id: string;
  label: string;
};

type RuntimeConfig = {
  manifest: {
    model_name: string;
    num_examples: number;
    num_global_chunks: number;
  };
  sample_questions: { question: string }[];
  custom_backends: string[];
  has_gnn: boolean;
};

type Example = {
  id: string;
  question: string;
  type: string;
  answer: string;
};

type RetrievedChunk = {
  rank: number;
  title: string;
  text: string;
  is_supporting: boolean;
  dense_scores?: number;
  gnn_scores?: number;
  fusion_scores?: number;
};

type QueryResult = {
  final_answer: string;
  retrieved_titles: string[];
  retrieved_chunks: RetrievedChunk[];
  title_overlap?: {
    match_count: number;
    gold_count: number;
    all_matched: boolean;
    overlap_titles: string[];
  };
};

type QueryResponse = {
  question: string;
  question_type?: string;
  gold_answer?: string;
  graph_stats?: { num_nodes: number; num_edges: number; supporting_nodes: number };
  result: QueryResult;
  comparison: Array<{
    mode: string;
    retrieved_titles: string[];
    final_answer: string;
    gold_title_matches?: number;
    all_gold_titles_matched?: boolean;
  }>;
  best_mode?: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";

const retrievalModes = [
  "FAISS-only retrieval",
  "FAISS + heuristic PCST",
  "GNN retrieval",
  "Dense retrieval + Query-Aware GraphSAGE",
  "Dense retrieval + Query-Aware GraphSAGE + PCST (Main Method)"
] as const;

function App() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState("");
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [questionType, setQuestionType] = useState("all");
  const [selectedExample, setSelectedExample] = useState("");
  const [activeTab, setActiveTab] = useState<"dataset" | "custom">("dataset");
  const [retrievalMode, setRetrievalMode] = useState<(typeof retrievalModes)[number]>("FAISS-only retrieval");
  const [topK, setTopK] = useState(5);
  const [lambdaDense, setLambdaDense] = useState(0.5);
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [compareModes, setCompareModes] = useState(false);
  const [customBackend, setCustomBackend] = useState("Exact");
  const [customQuestion, setCustomQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void fetch(`${API_BASE}/profiles`)
      .then((response) => response.json())
      .then((data) => {
        setProfiles(data.profiles);
        if (data.profiles.length > 0) {
          setSelectedProfile(data.profiles[0].id);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedProfile) {
      return;
    }

    void Promise.all([
      fetch(`${API_BASE}/config?profile_id=${selectedProfile}`).then((response) => response.json()),
      fetch(`${API_BASE}/examples?profile_id=${selectedProfile}&question_type=${questionType}`).then((response) =>
        response.json()
      )
    ])
      .then(([cfg, dataset]) => {
        setConfig(cfg);
        setExamples(dataset.examples);
        setCustomBackend(cfg.custom_backends[0] ?? "Exact");
        if (dataset.examples.length > 0) {
          setSelectedExample(dataset.examples[0].id);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, [selectedProfile, questionType]);

  const selectedExampleDetails = useMemo(
    () => examples.find((example) => example.id === selectedExample),
    [examples, selectedExample]
  );

  async function runDatasetQuery(event: FormEvent) {
    event.preventDefault();
    if (!selectedExample) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/query/dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile_id: selectedProfile,
          example_id: selectedExample,
          retrieval_mode: retrievalMode,
          top_k: topK,
          lambda_dense: lambdaDense,
          llm_enabled: llmEnabled,
          compare_all_modes: compareModes
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Failed to run dataset query");
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run dataset query");
    } finally {
      setLoading(false);
    }
  }

  async function runCustomQuery(event: FormEvent) {
    event.preventDefault();
    if (!customQuestion.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/query/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile_id: selectedProfile,
          question: customQuestion,
          retrieval_mode: retrievalMode,
          top_k: topK,
          lambda_dense: lambdaDense,
          llm_enabled: llmEnabled,
          compare_all_modes: compareModes,
          custom_backend: customBackend
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Failed to run custom query");
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run custom query");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-mist text-ink">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="relative overflow-hidden rounded-[2rem] bg-grain p-8 text-white shadow-panel">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.12),transparent_40%)]" />
          <div className="relative grid gap-8 lg:grid-cols-[1.3fr_0.7fr]">
            <div className="space-y-5">
              <span className="inline-flex rounded-full border border-white/20 px-4 py-1 text-xs uppercase tracking-[0.3em] text-white/70">
                GraphRAG Explorer
              </span>
              <h1 className="max-w-3xl font-display text-4xl leading-tight sm:text-5xl">
                Multi-hop retrieval that feels like a product, not a notebook.
              </h1>
              <p className="max-w-2xl text-base text-white/78 sm:text-lg">
                Compare dense retrieval baselines against query-aware graph scoring and the final dense +
                Query-Aware GraphSAGE + PCST retrieval pipeline over the indexed HotpotQA-style corpus.
              </p>
              <div className="flex flex-wrap gap-3 text-sm text-white/75">
                <span className="rounded-full border border-white/15 bg-white/5 px-4 py-2">Bridge-aware evidence recovery</span>
                <span className="rounded-full border border-white/15 bg-white/5 px-4 py-2">Query-Aware GraphSAGE + PCST main method</span>
                <span className="rounded-full border border-white/15 bg-white/5 px-4 py-2">Corpus-scale artifact reuse</span>
              </div>
            </div>
            <div className="grid gap-4 rounded-[1.5rem] border border-white/10 bg-white/10 p-5 backdrop-blur">
              <div>
                <p className="text-xs uppercase tracking-[0.25em] text-white/55">Artifact Stats</p>
                <p className="mt-2 text-3xl font-bold">{config?.manifest?.num_examples ?? "..."}</p>
                <p className="text-sm text-white/70">indexed examples</p>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-2xl bg-white/10 p-4">
                  <p className="text-white/60">Global chunks</p>
                  <p className="mt-2 text-xl font-semibold">{config?.manifest?.num_global_chunks ?? "..."}</p>
                </div>
                <div className="rounded-2xl bg-white/10 p-4">
                  <p className="text-white/60">Model</p>
                  <p className="mt-2 text-base font-semibold">{config?.manifest?.model_name ?? "Loading..."}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="mt-8 grid gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="space-y-6 rounded-[1.75rem] bg-white p-6 shadow-panel">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-slate/50">Controls</p>
              <h2 className="mt-2 font-display text-2xl">Configure the run</h2>
            </div>

            <label className="block text-sm font-medium text-slate">
              Artifact profile
              <select
                className="mt-2 w-full rounded-2xl border border-slate/10 bg-mist px-4 py-3"
                value={selectedProfile}
                onChange={(event) => setSelectedProfile(event.target.value)}
              >
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block text-sm font-medium text-slate">
              Retrieval mode
              <select
                className="mt-2 w-full rounded-2xl border border-slate/10 bg-mist px-4 py-3"
                value={retrievalMode}
                onChange={(event) => setRetrievalMode(event.target.value as (typeof retrievalModes)[number])}
              >
                {retrievalModes.map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>

            <label className="block text-sm font-medium text-slate">
              Top-k: <span className="text-ember">{topK}</span>
              <input
                className="mt-3 w-full accent-ember"
                type="range"
                min={1}
                max={10}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>

            <label className="block text-sm font-medium text-slate">
              Fusion weight: <span className="text-teal">{lambdaDense.toFixed(2)}</span>
              <input
                className="mt-3 w-full accent-teal"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={lambdaDense}
                onChange={(event) => setLambdaDense(Number(event.target.value))}
              />
            </label>

            <div className="space-y-3 rounded-2xl bg-mist p-4 text-sm text-slate">
              <label className="flex items-center justify-between gap-4">
                <span>Enable GPT answer generation</span>
                <input type="checkbox" checked={llmEnabled} onChange={() => setLlmEnabled((value) => !value)} />
              </label>
              <label className="flex items-center justify-between gap-4">
                <span>Compare all retrieval modes</span>
                <input type="checkbox" checked={compareModes} onChange={() => setCompareModes((value) => !value)} />
              </label>
            </div>

            <div className="rounded-2xl border border-ember/20 bg-ember/5 p-4 text-sm text-slate">
              <p className="font-semibold text-ink">Sample prompts</p>
              <ul className="mt-3 space-y-2">
                {config?.sample_questions?.slice(0, 3).map((item) => (
                  <li key={item.question} className="leading-relaxed">
                    {item.question}
                  </li>
                ))}
              </ul>
            </div>
          </aside>

          <main className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4 rounded-[1.75rem] bg-white p-4 shadow-panel">
              <div className="inline-flex rounded-full bg-mist p-1">
                {(["dataset", "custom"] as const).map((tab) => (
                  <button
                    key={tab}
                    className={`rounded-full px-5 py-2 text-sm font-medium transition ${
                      activeTab === tab ? "bg-ink text-white" : "text-slate"
                    }`}
                    onClick={() => {
                      setActiveTab(tab);
                      setResult(null);
                    }}
                  >
                    {tab === "dataset" ? "Dataset Mode" : "Custom Mode"}
                  </button>
                ))}
              </div>
              <div className="rounded-full border border-slate/10 px-4 py-2 text-sm text-slate">
                {config?.has_gnn ? "GNN checkpoint ready" : "FAISS/heuristic-only profile"}
              </div>
            </div>

            {activeTab === "dataset" ? (
              <form className="rounded-[1.75rem] bg-white p-6 shadow-panel" onSubmit={runDatasetQuery}>
                <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
                  <div className="space-y-5">
                    <div>
                      <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Benchmark mode</p>
                      <h2 className="mt-2 font-display text-3xl">Inspect labeled retrieval behavior</h2>
                    </div>

                    <label className="block text-sm font-medium text-slate">
                      Question type
                      <select
                        className="mt-2 w-full rounded-2xl border border-slate/10 bg-mist px-4 py-3"
                        value={questionType}
                        onChange={(event) => setQuestionType(event.target.value)}
                      >
                        <option value="all">All</option>
                        <option value="bridge">Bridge</option>
                        <option value="comparison">Comparison</option>
                      </select>
                    </label>

                    <label className="block text-sm font-medium text-slate">
                      Dataset question
                      <select
                        className="mt-2 w-full rounded-2xl border border-slate/10 bg-mist px-4 py-3"
                        value={selectedExample}
                        onChange={(event) => setSelectedExample(event.target.value)}
                      >
                        {examples.map((example) => (
                          <option key={example.id} value={example.id}>
                            {example.question}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="rounded-[1.5rem] bg-mist p-5">
                    <p className="text-sm uppercase tracking-[0.25em] text-slate/45">Question snapshot</p>
                    <p className="mt-4 text-sm text-slate">{selectedExampleDetails?.question}</p>
                    <div className="mt-5 grid gap-3 text-sm text-slate">
                      <div className="rounded-2xl bg-white px-4 py-3">
                        <span className="text-slate/50">Type</span>
                        <p className="mt-1 font-semibold">{selectedExampleDetails?.type ?? "Unknown"}</p>
                      </div>
                      <div className="rounded-2xl bg-white px-4 py-3">
                        <span className="text-slate/50">Gold answer</span>
                        <p className="mt-1 font-semibold">{selectedExampleDetails?.answer ?? "N/A"}</p>
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  className="mt-6 rounded-full bg-ink px-6 py-3 text-sm font-semibold text-white transition hover:bg-slate"
                  disabled={loading}
                >
                  {loading ? "Running..." : "Run dataset query"}
                </button>
              </form>
            ) : (
              <form className="rounded-[1.75rem] bg-white p-6 shadow-panel" onSubmit={runCustomQuery}>
                <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
                  <div className="space-y-5">
                    <div>
                      <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Open-ended mode</p>
                      <h2 className="mt-2 font-display text-3xl">Search the indexed corpus like a research tool</h2>
                    </div>
                    <label className="block text-sm font-medium text-slate">
                      Ask a question
                      <textarea
                        className="mt-2 min-h-32 w-full rounded-[1.25rem] border border-slate/10 bg-mist px-4 py-3"
                        value={customQuestion}
                        onChange={(event) => setCustomQuestion(event.target.value)}
                        placeholder="Where was the director of Titanic born?"
                      />
                    </label>
                  </div>
                  <div className="rounded-[1.5rem] bg-mist p-5">
                    <label className="block text-sm font-medium text-slate">
                      Candidate backend
                      <select
                        className="mt-2 w-full rounded-2xl border border-slate/10 bg-white px-4 py-3"
                        value={customBackend}
                        onChange={(event) => setCustomBackend(event.target.value)}
                      >
                        {(config?.custom_backends ?? ["Exact"]).map((backend) => (
                          <option key={backend} value={backend}>
                            {backend}
                          </option>
                        ))}
                      </select>
                    </label>
                    <p className="mt-4 text-sm leading-relaxed text-slate/80">
                      Exact scores against the full corpus. ANN backends trade a little recall for faster candidate
                      narrowing on large artifact bundles.
                    </p>
                  </div>
                </div>

                <button
                  type="submit"
                  className="mt-6 rounded-full bg-ember px-6 py-3 text-sm font-semibold text-white transition hover:bg-[#cf5f20]"
                  disabled={loading}
                >
                  {loading ? "Running..." : "Run custom query"}
                </button>
              </form>
            )}

            {error ? <div className="rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">{error}</div> : null}

            {result ? (
              <section className="space-y-6">
                <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
                  <div className="rounded-[1.75rem] bg-white p-6 shadow-panel">
                    <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Final answer</p>
                    <h3 className="mt-3 font-display text-3xl text-ink">{result.result.final_answer}</h3>
                    <p className="mt-4 text-sm leading-relaxed text-slate/80">{result.question}</p>

                    <div className="mt-6 grid gap-3 sm:grid-cols-3">
                      <Metric label="Retrieved titles" value={String(result.result.retrieved_titles.length)} />
                      <Metric label="Evidence chunks" value={String(result.result.retrieved_chunks.length)} />
                      <Metric label="Question type" value={result.question_type ?? "Custom"} />
                    </div>
                  </div>

                  <div className="rounded-[1.75rem] bg-white p-6 shadow-panel">
                    <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Coverage snapshot</p>
                    {result.result.title_overlap ? (
                      <div className="mt-4 space-y-4 text-sm text-slate">
                        <div className="rounded-2xl bg-mist p-4">
                          <p className="text-slate/55">Gold-title coverage</p>
                          <p className="mt-2 text-2xl font-semibold">
                            {result.result.title_overlap.match_count}/{result.result.title_overlap.gold_count}
                          </p>
                        </div>
                        <div className="rounded-2xl bg-mist p-4">
                          <p className="text-slate/55">Overlap titles</p>
                          <p className="mt-2 leading-relaxed">
                            {result.result.title_overlap.overlap_titles.join(", ") || "No overlap yet"}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="mt-4 rounded-2xl bg-mist p-4 text-sm text-slate/80">
                        Custom questions do not have gold titles, so the UI shows the retrieved evidence only.
                      </div>
                    )}
                  </div>
                </div>

                {result.comparison.length > 0 ? (
                  <div className="rounded-[1.75rem] bg-white p-6 shadow-panel">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Mode comparison</p>
                        <h3 className="mt-2 font-display text-2xl">Side-by-side retrieval outcome</h3>
                      </div>
                      {result.best_mode ? (
                        <div className="rounded-full bg-teal px-4 py-2 text-sm font-medium text-white">
                          Best coverage: {result.best_mode}
                        </div>
                      ) : null}
                    </div>
                    <div className="mt-6 overflow-x-auto">
                      <table className="min-w-full text-left text-sm">
                        <thead className="text-slate/55">
                          <tr>
                            <th className="pb-3 pr-4">Mode</th>
                            <th className="pb-3 pr-4">Final answer</th>
                            <th className="pb-3 pr-4">Retrieved titles</th>
                            <th className="pb-3">Gold matches</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.comparison.map((row) => (
                            <tr key={row.mode} className="border-t border-slate/8 align-top">
                              <td className="py-4 pr-4 font-semibold">{row.mode}</td>
                              <td className="py-4 pr-4">{row.final_answer}</td>
                              <td className="py-4 pr-4">{row.retrieved_titles.join(", ")}</td>
                              <td className="py-4">{row.gold_title_matches ?? "N/A"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div className="rounded-[1.75rem] bg-white p-6 shadow-panel">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Evidence trail</p>
                      <h3 className="mt-2 font-display text-2xl">Retrieved chunks</h3>
                    </div>
                    {result.graph_stats ? (
                      <div className="grid grid-cols-3 gap-2 text-xs text-slate sm:text-sm">
                        <Metric label="Nodes" value={String(result.graph_stats.num_nodes)} compact />
                        <Metric label="Edges" value={String(result.graph_stats.num_edges)} compact />
                        <Metric label="Supporting" value={String(result.graph_stats.supporting_nodes)} compact />
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-6 grid gap-4">
                    {result.result.retrieved_chunks.map((chunk) => (
                      <article key={`${chunk.rank}-${chunk.title}`} className="rounded-[1.5rem] border border-slate/10 bg-mist p-5">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-xs uppercase tracking-[0.25em] text-slate/45">Rank {chunk.rank}</p>
                            <h4 className="mt-1 text-lg font-semibold text-ink">{chunk.title}</h4>
                          </div>
                          <span
                            className={`rounded-full px-3 py-1 text-xs font-medium ${
                              chunk.is_supporting ? "bg-teal text-white" : "bg-white text-slate"
                            }`}
                          >
                            {chunk.is_supporting ? "Supporting" : "Retrieved"}
                          </span>
                        </div>
                        <p className="mt-4 text-sm leading-7 text-slate">{chunk.text}</p>
                      </article>
                    ))}
                  </div>
                </div>
              </section>
            ) : null}
          </main>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, compact = false }: { label: string; value: string; compact?: boolean }) {
  return (
    <div className={`rounded-2xl ${compact ? "bg-mist px-3 py-3" : "bg-mist p-4"}`}>
      <p className="text-xs uppercase tracking-[0.2em] text-slate/45">{label}</p>
      <p className="mt-2 text-xl font-semibold text-ink">{value}</p>
    </div>
  );
}

export default App;
