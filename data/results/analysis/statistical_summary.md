# Statistical analysis summary

## Method

Of the 306 sampled questions, 83 are excluded as unscoreable: their gold query either fails to execute or returns zero rows, so no model answer can be distinguished as right or wrong against it. Whether a gold query returns rows depends only on the item and not on the model, so the same 223 items are scored for both models and the pairing is preserved. Accuracy intervals are 95% Wilson binomial confidence intervals. GPT and Qwen are compared using an exact two-sided paired McNemar test over these 223 items.

## Overall results

| Model | Metric | Result | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict execution accuracy | 41/223 (18.4%) | 13.9%–24.0% |
| GPT-4o-mini | Lenient execution accuracy | 51/223 (22.9%) | 17.8%–28.8% |
| Qwen2.5-Coder-7B-Instruct | Strict execution accuracy | 31/223 (13.9%) | 10.0%–19.1% |
| Qwen2.5-Coder-7B-Instruct | Lenient execution accuracy | 37/223 (16.6%) | 12.3%–22.0% |

## GPT vs. Qwen (paired tests)

### Strict execution accuracy (scoreable items)

- GPT-only correct: 18
- Qwen-only correct: 8
- Exact McNemar p-value: 0.075519
- Conclusion: the observed difference is **not statistically significant at α = 0.05**.

### Lenient execution accuracy (scoreable items)

- GPT-only correct: 23
- Qwen-only correct: 9
- Exact McNemar p-value: 0.020062
- Conclusion: the observed difference is **statistically significant at α = 0.05**.

