# Statistical analysis summary

## Method

Of the 331 sampled questions, 87 are excluded and 244 are scored. 6 are excluded because the gold query fails to execute, so there is no reference result to compare against. A further 81 are excluded because the gold query executes but returns no rows: a both-empty result is treated as a trivial match and is not credited as a success, and an empty gold result can only be matched by an equally empty generated result, so such an item is a guaranteed failure for every model and would inflate the denominator without ever being able to contribute to the numerator. Both conditions depend only on the item and not on the model, so the same 244 items are scored for both models and the pairing is preserved. Note that the exclusions are not spread evenly across the strata, so the scored subset is no longer a stratified sample of the source corpus. Accuracy intervals are 95% Wilson binomial confidence intervals. GPT and Qwen are compared using an exact two-sided paired McNemar test over these 244 items.

## Overall results

| Model | Metric | Result | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict execution accuracy | 50/244 (20.5%) | 15.9%–26.0% |
| GPT-4o-mini | Lenient execution accuracy | 61/244 (25.0%) | 20.0%–30.8% |
| Qwen2.5-Coder-7B-Instruct | Strict execution accuracy | 36/244 (14.8%) | 10.9%–19.7% |
| Qwen2.5-Coder-7B-Instruct | Lenient execution accuracy | 44/244 (18.0%) | 13.7%–23.3% |

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

