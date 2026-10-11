"""Benchmark the service in process (FastAPI TestClient) and write results/."""
import csv, datetime, hashlib, json, os, platform, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from fastapi.testclient import TestClient
from app.main import app
from app import simulator as sim
from src.svgplot import line_plot
OUT = os.path.join(ROOT, "results")

def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    X = sim.from_unit(rng.random((300, 3))); ytrue = sim.true_hotspot(X); ynoisy = sim.simulate(X, np.random.default_rng(8))
    with TestClient(app) as c:
        items = [dict(ambient_degC=float(a), n_evs=float(n), shifted_share=float(s)) for a, n, s in X]
        preds = []
        for i in range(0, 300, 100):
            preds += c.post("/predict/batch", json={"items": items[i:i + 100]}).json()["predictions"]
        mu = np.array([p["hotspot_degC"] for p in preds]); hw = np.array([p["interval95_halfwidth_degC"] for p in preds])
        rmse = float(np.sqrt(np.mean((mu - ytrue) ** 2))); cov = float(np.mean(np.abs(ynoisy - mu) <= hw))
        rows = []
        for b in (1, 5, 10, 25, 50, 100):
            ts = []
            for _ in range(60):
                t = time.perf_counter(); c.post("/predict/batch", json={"items": items[:b]}); ts.append((time.perf_counter() - t) * 1000)
            rows.append(dict(batch_size=b, p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95))))
        single = []
        for _ in range(200):
            t = time.perf_counter(); c.post("/predict", json=items[0]); single.append((time.perf_counter() - t) * 1000)
        bad = [{**items[0], "n_evs": -5}, {**items[0], "shifted_share": 2}, {**items[0], "ambient_degC": "x"}, {"n_evs": 1}]
        rejected = sum(c.post("/predict", json=b).status_code == 422 for b in bad)
        version = c.get("/health").json()["model_version"]
    with open(f"{OUT}/latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    line_plot(f"{OUT}/latency.svg", "In-process request latency vs batch size (ms)", "items per request", "ms",
              [("p50", [r["batch_size"] for r in rows], [r["p50_ms"] for r in rows]), ("p95", [r["batch_size"] for r in rows], [r["p95_ms"] for r in rows])])
    summary = dict(model_version=version, training_runs=40, test_points=300, rmse_degC=rmse, interval95_coverage=cov,
                   single_request_p50_ms=float(np.percentile(single, 50)), single_request_p95_ms=float(np.percentile(single, 95)),
                   batch100_p50_ms=rows[-1]["p50_ms"], invalid_inputs_sent=len(bad), invalid_inputs_rejected_422=int(rejected),
                   note="Latency measured in process with FastAPI TestClient on the build machine: no network, no container, no cloud.")
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    rep = ["# Results", "", f"Model {version}, trained on 40 synthetic simulator runs at start-up.", "",
           f"- Accuracy on 300 unseen points: RMSE {rmse:.2f} degC against the noise-free simulator.",
           f"- 95 % interval coverage on noisy values: {cov:.2f}.",
           f"- Single /predict request: p50 {summary['single_request_p50_ms']:.1f} ms, p95 {summary['single_request_p95_ms']:.1f} ms (in process).",
           f"- /predict/batch with 100 items: p50 {summary['batch100_p50_ms']:.1f} ms.",
           f"- Invalid inputs rejected with HTTP 422: {rejected} of {len(bad)}.", "",
           "Latency is in-process (no network, container or cloud) and only indicates relative cost."]
    open(f"{OUT}/REPORT.md", "w").write("\n".join(rep) + "\n")
    man = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(), "python": platform.python_version(),
           "results": {fn: hashlib.sha256(open(f"{OUT}/{fn}", "rb").read()).hexdigest() for fn in sorted(os.listdir(OUT)) if fn != "manifest.json"}}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(open(f"{OUT}/REPORT.md").read())

if __name__ == "__main__":
    main()
