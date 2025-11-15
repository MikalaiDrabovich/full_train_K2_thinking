#!/usr/bin/env bash
set -e

python scripts/prepare_corpus.py --config configs/env.yaml --mode domain
python scripts/prepare_corpus.py --config configs/env.yaml --mode reasoning

# Optional: shard large corpora
# python scripts/shard_corpus.py --config configs/env.yaml --corpus_key domain_corpus
# python scripts/shard_corpus.py --config configs/env.yaml --corpus_key reasoning_corpus
