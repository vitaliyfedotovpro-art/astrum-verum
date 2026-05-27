# VSA Memory Benchmark — Results

## Run Parameters

- **date_utc**: 2026-05-27 18:22:57 UTC
- **embedder_model**: paraphrase-multilingual-MiniLM-L12-v2
- **n_seeds**: 30
- **D**: 10000
- **max_N**: 16000
- **normalize_threshold**: 0.82
- **data_seed**: 12345
- **quick_mode**: False
- **ns**: (1000, 2000, 4000, 8000, 16000)
- **ls**: (50, 100, 200, 500, 1000)
- **e3_m**: 500
- **machine**: Apple M4, 16 GB RAM, arm64
- **corpus_triples**: 16000
- **corpus_unique_texts**: 581
- **corpus_entities**: 502
- **corpus_relations**: 80
- **emb_dim**: 384
- **embedding_time_s**: 7.2
- **e1_time_s**: 1294.4
- **e2_time_s**: 231.0
- **e3_time_s**: 16.6
- **total_time_s**: 1549.3

## E1 — Facts structural recall vs N (Capacity)

| N (facts) | Mean Acc | Std | Min | Max | n_seeds |
|-----------|----------|-----|-----|-----|---------|
|  1000 | 0.9333 | 0.0167 | 0.8900 | 0.9600 | 30 |
|  2000 | 0.9225 | 0.0165 | 0.8900 | 0.9500 | 30 |
|  4000 | 0.8825 | 0.0221 | 0.8250 | 0.9450 | 30 |
|  8000 | 0.8420 | 0.0226 | 0.8050 | 0.9050 | 30 |
| 16000 | 0.7468 | 0.0321 | 0.6950 | 0.8050 | 30 |

## E2 — Episode order-recall vs length

### Full episode (accuracy_full)

| L (length) | Mean Acc | Std | Min | Max | n_seeds |
|-----------|----------|-----|-----|-----|---------|
|    50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 30 |
|   100 | 0.9900 | 0.0000 | 0.9900 | 0.9900 | 30 |
|   200 | 0.9507 | 0.0090 | 0.9350 | 0.9650 | 30 |
|   500 | 0.5567 | 0.0271 | 0.4720 | 0.6060 | 30 |
|  1000 | 0.2365 | 0.0202 | 0.1910 | 0.2710 | 30 |

### Window W=150 (accuracy_window_W150)

| L (full length) | Mean Acc | Std | Min | Max | n_seeds |
|------------------|----------|-----|-----|-----|---------|
|    50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 30 |
|   100 | 0.9900 | 0.0000 | 0.9900 | 0.9900 | 30 |
|   200 | 0.9780 | 0.0031 | 0.9733 | 0.9800 | 30 |
|   500 | 0.9649 | 0.0035 | 0.9533 | 0.9667 | 30 |
|  1000 | 0.9649 | 0.0035 | 0.9533 | 0.9667 | 30 |

## E3 — SimHash grounding fidelity

| Metric | Mean Corr | Std | Min | Max | n_seeds |
|---------|-----------|-----|-----|-----|---------|
| pearson | 0.992122 | 0.000176 | 0.991792 | 0.992492 | 30 |
| spearman | 0.990727 | 0.000225 | 0.990383 | 0.991217 | 30 |

## Honesty Notes

- All seeds (including worst-case) are included in the report.
- Raw per-seed data is recorded in CSV for verification.
- Embeddings: real, via sentence-transformers.
- CORPUS: deterministically generated from real words (data_seed=12345).
- VSAMemory code was not modified.
