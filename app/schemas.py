from pydantic import BaseModel
from typing import Dict, Any

class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionRequest(BaseModel):
    features: Dict[str, Any]