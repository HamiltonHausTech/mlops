from __future__ import annotations
import bentoml
import mlflow.sklearn
from pydantic import BaseModel
from typing import List

class IrisRequest(BaseModel):
    data: List[List[float]]

@bentoml.service()
class IrisService:
    def __init__(self):
        self.model = None

    @bentoml.api
    def predict(self, payload: dict) -> dict:
        if self.model is None:
            self.model = mlflow.sklearn.load_model("runs:/72ac6c89a966497482251810a7594c39/model")
        preds = self.model.predict(payload["data"])
        return {"predictions": preds.tolist()}