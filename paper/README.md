# GraphRAG NeurIPS Report

Compile from this directory with:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The generated report is `main.pdf`. The paper uses `neurips_2022.sty`, `references.bib`, and figures copied into `figures/`.

The inference-time table was generated from the repository root with:

```sh
graphrag_env/bin/python graphrag_env/src/benchmark_inference_time.py \
  --num-queries 500 \
  --warmup 30 \
  --output-json paper/inference_time_results.json \
  --output-csv paper/inference_time_results.csv
```
