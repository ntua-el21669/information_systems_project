# Statistical analysis summary

## Method

Of the 331 sampled questions, 6 are excluded as unscoreable: their gold query fails to execute, so there is no reference result to compare against. Whether a gold query fails depends only on the item and not on the model, so the same 325 items are scored for both models and the pairing is preserved. Accuracy intervals are 95% Wilson binomial confidence intervals. GPT and Qwen are compared using an exact two-sided paired McNemar test over these 325 items.

## Overall results

| Model | Metric | Result | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict execution accuracy | 50/325 (15.4%) | 11.9%–19.7% |
| GPT-4o-mini | Lenient execution accuracy | 61/325 (18.8%) | 14.9%–23.4% |
| Qwen2.5-Coder-7B-Instruct | Strict execution accuracy | 36/325 (11.1%) | 8.1%–15.0% |
| Qwen2.5-Coder-7B-Instruct | Lenient execution accuracy | 44/325 (13.5%) | 10.2%–17.7% |

## GPT vs. Qwen (paired tests)

### Strict execution accuracy (scoreable items)

- GPT-only correct: 22
- Qwen-only correct: 8
- Exact McNemar p-value: 0.016125
- Conclusion: the observed difference is **statistically significant at α = 0.05**.

### Lenient execution accuracy (scoreable items)

- GPT-only correct: 26
- Qwen-only correct: 9
- Exact McNemar p-value: 0.005988
- Conclusion: the observed difference is **statistically significant at α = 0.05**.

