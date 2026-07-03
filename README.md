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
