# Surrogate prediction API

A small FastAPI service that serves a Gaussian-process surrogate with uncertainty. Built October 2026 as a practice project, not employer work.

## What it does
- `GET /health`, `GET /model`, `POST /predict`, `POST /predict/batch` (1 to 100 items).
- Pydantic validation: out-of-range, missing, wrongly typed or extra fields return HTTP 422.
- The model (NumPy GP, ARD RBF kernel) trains once at start-up on 40 runs of a synthetic transformer hot-spot simulator and returns a mean and a 95 % interval half-width.
- `pytest` with `httpx` tests (14 pass), a `Dockerfile` and a GitHub Actions workflow.

## Results (from `results/summary.json`)
- RMSE 0.82 degC on 300 unseen points; 95 % interval coverage 0.99 (conservative on this test set).
- Single request p50 1.0 ms, p95 1.8 ms; batch of 100 p50 2.4 ms. These are in-process timings (TestClient), not network or container timings.
- 4 of 4 invalid inputs rejected with 422.

## Not done
- **Not deployed to any cloud.**
- The Dockerfile and the CI workflow were written but **not built or run**: no Docker daemon was available where this was built, and the workflow has not run on GitHub.
- No authentication, rate limiting, monitoring, model registry or load test over a network.

## Run
```
pip install -r requirements.txt
python3 -m pytest -q tests
python3 src/run_all.py
uvicorn app.main:app --port 8000
```
See `docs/METHOD.md`.
