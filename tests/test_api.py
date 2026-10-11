import os, sys
import numpy as np, pytest
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app
from app import simulator as sim

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

OK = {"ambient_degC": 20, "n_evs": 100, "shifted_share": 0.0}

def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"

def test_model_info_lists_inputs(client):
    j = client.get("/model").json(); assert j["training_runs"] == 40 and len(j["inputs"]) == 3

def test_predict_close_to_simulator_truth(client):
    j = client.post("/predict", json=OK).json()
    assert abs(j["hotspot_degC"] - sim.true_hotspot([[20, 100, 0.0]])[0]) < 4.0
    assert j["interval95_halfwidth_degC"] > 0

def test_more_evs_means_hotter(client):
    a = client.post("/predict", json={**OK, "n_evs": 50}).json()["hotspot_degC"]
    b = client.post("/predict", json={**OK, "n_evs": 200}).json()["hotspot_degC"]
    assert b > a

@pytest.mark.parametrize("bad", [{**OK, "n_evs": -1}, {**OK, "n_evs": 400}, {**OK, "shifted_share": 1.5},
                                 {**OK, "ambient_degC": 80}, {"ambient_degC": 20, "n_evs": 100},
                                 {**OK, "n_evs": "many"}, {**OK, "extra": 1}])
def test_invalid_input_rejected_with_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422

def test_batch_matches_single_calls(client):
    items = [{**OK, "n_evs": n} for n in (10, 80, 160)]
    batch = client.post("/predict/batch", json={"items": items}).json()["predictions"]
    singles = [client.post("/predict", json=i).json() for i in items]
    assert [round(b["hotspot_degC"], 6) for b in batch] == [round(s["hotspot_degC"], 6) for s in singles]

def test_batch_limits(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422
    assert client.post("/predict/batch", json={"items": [OK] * 101}).status_code == 422
    assert client.post("/predict/batch", json={"items": [OK] * 100}).status_code == 200

def test_model_is_reproducible():
    from app.model import Surrogate
    assert Surrogate(seed=0).version == Surrogate(seed=0).version != Surrogate(seed=1).version
