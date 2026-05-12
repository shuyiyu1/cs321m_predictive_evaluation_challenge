# Predictive Evaluation Challenge Submission

This submission contains a compact, dependency-free ensemble predictor.

Files:

- `model.py`: required Codabench entry point.
- `labeling.py`: optional adaptive-label acquisition function.
- `baseline_artifact.json`: fitted weights, smoothed priors, interaction
  tables, and an item-difficulty text model.

The model was trained offline with `../train_baseline.py` and enhanced with
`../fit_enhanced_artifact.py` on the public `aims-foundations/measurement-db`
parquet files. Runtime imports use only the Python standard library, so no
`requirements.txt` or `models.txt` is needed.

To smoke-test locally from this directory:

```bash
python3 - <<'PY'
from model import predict
ex = {
    "benchmark": "MMLU-Pro",
    "condition": "none",
    "subject_content": "GPT-4o",
    "item_content": "What is the derivative of x^2?",
}
print(predict(ex, labeled=[]))
PY
```
