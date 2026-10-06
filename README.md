# Harbor eval harness

Offline eval systems for a support agent: datasets, judges, retrieval and generation gates, trajectory contracts, regression control, and the reports around them. Nothing in this repo calls a hosted model or trains weights.

The two source threads are [the 12-stage plan](https://x.com/suraj_sharma14/status/2106353586288685419) and [the 15 projects](https://x.com/suraj_sharma14/status/2106715961403273433). Each practice project is a module under `src/evals_engineer`. Simulated raters and scripted models are labeled as such in the methodology notes.

## Run

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m pytest
py -3 -m evals_engineer.cli demo
py -3 -m evals_engineer.cli golden --out data/golden/v1
```

`demo` writes `reports/demo/portfolio.md` and `reports/demo/pareto.html`. The golden command writes `cases.jsonl`, `VERSION.json`, and `CHANGELOG.md` with Cohen's kappa.

## Map

| Stage or project | Module | Command |
| --- | --- | --- |
| JSONL plugin, precision/recall/F1, intervals, kappa | `plugin.py`, `metrics.py` | `pytest -p evals_engineer.plugin --eval-dataset cases.jsonl` |
| 500 failure classifications and reproduction steps | `failures.py` | `evals failures` |
| 300-case golden set, two raters, changelog | `golden.py` | `evals golden` |
| Rubric judge, 500 anchors, bias gates | `judge.py`, `rubrics.py` | `evals judge` |
| RAG retrieval gate and generation gate | `rag.py` | `evals rag` |
| Trajectory DAG, safety checks, hallucinated parameters | `trajectory.py` | `evals trajectory` |
| promptfoo, DeepEval, and RAGAS adapters, one report | `harness.py` | used by `evals demo` |
| "wins by X% ± Y% (p<0.05)" | `compare.py` | `evals teardown` |
| Merge gate, hysteresis, cache, flaky cases | `gate.py` | `evals gate --current a.json --baseline b.json` |
| 5% shadow sample and contained failures | `shadow.py` | used by `evals demo` |
| Nightly 5% drift alerts | `drift.py` | used by `evals demo` |
| Thumbs-down pairs and a dry-run LoRA manifest | `flywheel.py` | used by `evals demo` |
| Cost-quality frontier | `pareto.py` | `evals pareto` |
| Swap one hop and replay the rest | `replay.py` | `evals replay` |
| Edge-case operators, optional frontier model | `synthetic.py` | `evals synthetic` |
| Salience eviction versus FIFO | `context_memory.py` | `evals context` |
| N-gram and embedding leak scan | `contamination.py` | used by `evals demo` |
| Canary, schema fuzz, loop guard | `redteam.py` | `evals fuzz` |
| Careful agent versus naive baseline | `teardown.py` | `evals teardown` |

## Reading the methodology

1. [Trajectory grading](docs/methodology/01-trajectory-grading.md) covers the tool contract and why a single golden narration is the wrong gate.
2. [Judge calibration](docs/methodology/02-judge-calibration.md) has the rubric, the prompt, and the bias gates.
3. [Statistical framework](docs/methodology/03-statistical-framework.md) has the paired interval, the McNemar test, the sample-size formula, and the merge rule.

Annotation rules for the golden labels are in [docs/annotation_guidelines.md](docs/annotation_guidelines.md).

## Limits

- Ada and Ben are simulated from the guideline file. The kappa is real for that rule. It is not a human agreement study.
- The judge, the failure model, and the RAG generators are scripted. `complete(prompt, sample)` is the seam for a hosted model.
- The framework adapters implement the assertion styles locally. They do not import promptfoo, DeepEval, or RAGAS.
- The LoRA step writes a manifest and stops. It does not update weights.
- The red-team probes are canaries, invalid tool payloads, and repeated calls. The repo does not contain jailbreak or exfiltration procedures.
- The default embedder is lexical. Pass a neural embedder into `contamination.scan` when wording changes but meaning does not.
