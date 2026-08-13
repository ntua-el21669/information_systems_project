# Statistical analysis summary

## Method

Accuracy intervals are 95% Wilson binomial confidence intervals. GPT and Qwen are compared using an exact two-sided paired McNemar test on the same 306 questions. A trivial empty-result match is counted as incorrect in the strict metric.

## Overall results

| Model | Metric | Result | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict execution accuracy | 41/306 (13.4%) | 10.0%–17.7% |
| GPT-4o-mini | Lenient execution accuracy | 123/306 (40.2%) | 34.9%–45.8% |
| Qwen2.5-Coder-7B-Instruct | Strict execution accuracy | 31/306 (10.1%) | 7.2%–14.0% |
| Qwen2.5-Coder-7B-Instruct | Lenient execution accuracy | 91/306 (29.7%) | 24.9%–35.1% |

## GPT vs. Qwen (paired tests)

### Strict execution accuracy excluding trivial empty matches

- GPT-only correct: 18
- Qwen-only correct: 8
- Exact McNemar p-value: 0.075519
- Conclusion: the observed difference is **not statistically significant at α = 0.05**.

### Lenient execution accuracy

- GPT-only correct: 43
- Qwen-only correct: 11
- Exact McNemar p-value: 0.000014
- Conclusion: the observed difference is **statistically significant at α = 0.05**.

