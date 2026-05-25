# GUI Candidate A/B Report

- Input: `/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl`
- Samples: 500
- Variants: `trace_pipeline, current, current_ocr, omniparser, current_omniparser, current_ocr_omniparser`
- Hit definition: IoU >= 0.3 or candidate center inside GT bbox.

| Variant | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@30 | Oracle Est. | Avg Candidates | Avg Best IoU | Avg Hit Rank | Avg Latency | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trace_pipeline` | 19.20% | 31.60% | 37.80% | 52.40% | 80.20% | 19.80% | 29.88 | 0.3561 | 8.80 | 0.000s | 0 |
| `current` | 8.20% | 18.00% | 23.00% | 36.20% | 62.20% | 37.80% | 29.82 | 0.2906 | 10.26 | 0.670s | 0 |
| `current_ocr` | 42.20% | 50.00% | 53.00% | 57.60% | 70.20% | 29.80% | 29.98 | 0.2889 | 4.98 | 3.695s | 0 |
| `omniparser` | 9.20% | 21.60% | 34.40% | 53.80% | 82.60% | 17.40% | 20.95 | 0.5717 | 9.27 | 0.096s | 0 |
| `current_omniparser` | 8.20% | 18.00% | 24.20% | 37.80% | 65.80% | 34.20% | 29.92 | 0.3594 | 10.68 | 0.849s | 0 |
| `current_ocr_omniparser` | 42.20% | 50.00% | 53.00% | 57.40% | 70.40% | 29.60% | 29.99 | 0.2951 | 5.08 | 3.874s | 0 |

## Notes

- `omniparser` variants require local weights under `third_party/OmniParser/weights`; otherwise they report zero candidates or unchanged current-detect behavior.
- Use this same report before and after installing external detectors to decide whether a provider enters the training data pipeline.

## Findings

- OmniParser-only is strong at candidate recall: Recall@30 is 82.60%, better than `trace_pipeline` 80.20% and much better than runtime `current_ocr` 70.20%. It also has the highest Avg Best IoU, 0.5717.
- The current fusion path does not yet convert that strength into top-30 gains. `current_ocr_omniparser` is 70.40% Recall@30, only +0.20 pp over `current_ocr` 70.20%, with slightly worse Avg Hit Rank.
- `current_omniparser` improves over `current` from 62.20% to 65.80% Recall@30, but remains far below OmniParser-only because the fused top-30 is dominated by existing heuristic/query/OCR candidates.
- Decision: do not put the current fused OmniParser variant directly into the training data pipeline yet. First add a better fusion/reranking policy, or use OmniParser as a separate candidate source for an oracle/union/reranker experiment.
- Implementation note: the upstream OmniParser `util.utils` import segfaults in this environment because it initializes OCR stacks at import time. This run used the OmniParser v2 YOLO icon detector directly without captioning.

