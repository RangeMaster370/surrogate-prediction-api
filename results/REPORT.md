# Results

Model 4deec4b25c, trained on 40 synthetic simulator runs at start-up.

- Accuracy on 300 unseen points: RMSE 0.82 degC against the noise-free simulator.
- 95 % interval coverage on noisy values: 0.99.
- Single /predict request: p50 1.0 ms, p95 1.8 ms (in process).
- /predict/batch with 100 items: p50 2.4 ms.
- Invalid inputs rejected with HTTP 422: 4 of 4.

Latency is in-process (no network, container or cloud) and only indicates relative cost.
