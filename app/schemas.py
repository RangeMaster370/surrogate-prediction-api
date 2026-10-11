from typing import List
from pydantic import BaseModel, Field, ConfigDict

class Operating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambient_degC: float = Field(ge=-10, le=35)
    n_evs: float = Field(ge=0, le=250)
    shifted_share: float = Field(ge=0, le=1)

class Prediction(BaseModel):
    hotspot_degC: float
    interval95_halfwidth_degC: float
    model_version: str

class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[Operating] = Field(min_length=1, max_length=100)

class BatchResponse(BaseModel):
    predictions: List[Prediction]
