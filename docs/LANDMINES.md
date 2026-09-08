# Landmines

Operational notes for GLM-5.3-Flash on DGX Spark.

## Memory Edge

`UD-IQ3_XXS` is documented around 120GB. On a 128GB unified-memory system, background services and oversized KV/batch settings can push the server into host pressure or OOM behavior.

Start conservative:

- context: `131072`
- parallel slots: `1`
- batch/ubatch: leave default first, then tune
- MTP: off for first boot, `n=2` after baseline stability

## Runtime Drift

The GLM-5.3-Flash llama.cpp support path is evolving. Every benchmark result must include:

- llama.cpp branch
- commit hash
- exact server command
- quant
- reasoning effort
- MTP setting

## Sampling

The documented default for most tasks is:

- `temperature=1.0`
- `top_p=0.95`

DeepSWE-style settings are:

- `temperature=0.95`
- `top_p=1.0`

## Reasoning Effort

GLM-5.3-Flash supports `low`, `high`, and `max`. Treat `max` as the baseline for hard agentic/coding work, then benchmark latency and quality deltas for lower modes.

