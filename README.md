# EGC-TECQA

Executable Graph-Connected Temporal Evidence Chains for LLM-based Temporal Knowledge Graph Question Answering.

This project explores a stricter alternative to proximity-ordered temporal evidence lists. The core idea is to construct graph-connected evidence chains and execute temporal rules over them, so that answers are verifiable rather than only inferred by an LLM from sorted facts.

## Current Scope

This is the initial scaffold. It focuses on a small, dependency-free prototype:

- normalized temporal fact representation;
- temporal KG indexes;
- graph connectivity checks;
- executable temporal operators;
- simple chain search;
- Hits@1 and chain-level metrics;
- a runnable toy demo.

Full MultiTQ/CronQuestions adapters will be added after the core prototype is stable.

## Quick Start

```bash
python3 scripts/run_toy_demo.py
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

The current toy demo verifies this reasoning pattern:

```text
anchor fact: John Baird visited Thailand on 2012-03-28
rule: first valid visit to Thailand after the anchor
answer: Malaysia
```

## MultiTQ Data

The official MultiTQ benchmark is expected at:

```text
data/raw/MultiTQ/
```

The downloaded archive is intentionally ignored by git:

```text
data/raw/MultiTQ_Dataset.zip
```

Run a small real-data sanity check:

```bash
python3 scripts/run_multitq_debug.py --split test --per-type 2
```

Run the same path with LLM parsing and schema grounding:

```bash
python3 scripts/run_multitq_debug.py --split test --per-type 1 --parser llm
```

Run the repeatable evaluation runner with parse caching and per-type metrics:

```bash
python3 scripts/evaluate_multitq.py --split test --limit 100 --parser llm
```

For quick local smoke tests without LLM calls:

```bash
python3 scripts/evaluate_multitq.py --split test --limit 10 --parser heuristic
```

Current status of this script:

- loads the full MultiTQ KG;
- links entities and relations with a lightweight heuristic parser;
- builds a simple graph-connected chain;
- executes temporal rules;
- writes debug predictions to `outputs/cases/multitq_debug_predictions.jsonl`.

This is a scaffold baseline, not the final method. The next major work items are stronger anchor extraction, multi-candidate pivot search, and beam-search chain construction.

## LLM Configuration

LLM access is configured through local environment variables. Do not commit real API keys.

```bash
cp .env.example .env
# edit .env locally
```

For a DeepSeek OpenAI-compatible endpoint:

```text
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=...
LLM_MODEL=deepseek-chat
```

Run a small parser smoke test:

```bash
python3 scripts/parse_multitq_with_llm.py --split test --limit 3
```

The parser currently only extracts semantic structure. Grounding extracted mentions to KG schema is handled separately and will be strengthened next.

## Project Layout

```text
src/egc_tecqa/
  data/        dataset loading and normalization
  kg/          fact schema and indexes
  parser/      rule-based and LLM-based question parsing
  retrieval/   structure-guided and TECQA-style retrieval
  chain/       graph-connected chain construction
  executor/    temporal rule execution and verification
  eval/        QA and chain-level metrics
scripts/       runnable experiments
tests/         lightweight tests
examples/      toy examples and fixtures
```

## First Milestone

The first milestone is to run a small end-to-end prototype:

```text
question -> parsed intent -> candidate retrieval -> graph-connected chain
-> executable temporal rule -> answer -> verification -> Hits@1
```
