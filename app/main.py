from contextlib import asynccontextmanager
from fastapi import FastAPI
from .model import Surrogate
from .schemas import Operating, Prediction, BatchRequest, BatchResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Surrogate()          # trained once at start-up, about a second
    yield

app = FastAPI(title="Surrogate prediction API", version="0.1.0", lifespan=lifespan)

def _pred(model, rows):
    mu, hw = model.predict(rows)
    return [Prediction(hotspot_degC=float(m), interval95_halfwidth_degC=float(h), model_version=model.version) for m, h in zip(mu, hw)]

@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.model.version}

@app.get("/model")
def model_info():
    m = app.state.model
    return {"model_version": m.version, "training_runs": m.n_runs, "inputs": ["ambient_degC", "n_evs", "shifted_share"],
            "length_scales_unit_cube": [float(v) for v in m.gp.ls], "data": "synthetic simulator"}

@app.post("/predict", response_model=Prediction)
def predict(x: Operating):
    return _pred(app.state.model, [[x.ambient_degC, x.n_evs, x.shifted_share]])[0]

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_pred(app.state.model, [[i.ambient_degC, i.n_evs, i.shifted_share] for i in req.items]))
