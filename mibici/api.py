from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mibici.modeling.predict import (
    DEFAULT_MODEL_URI,
    load_model,
    predict_demand,
)


class PredictionRequest(BaseModel):
    origen_id: int = Field(gt=0)
    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=23)
    mes: int = Field(ge=1, le=12)
    demanda_lag_1h: float = Field(ge=0)
    demanda_lag_24h: float = Field(ge=0)
    es_fin_de_semana: int = Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    predicted_demand: float
    model_uri: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(DEFAULT_MODEL_URI)
    yield

app = FastAPI(
    title="MiBici Demand Prediction API",
    version="1.0.0",
    description="API for hourly MiBici demand predictions.",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predictions = predict_demand(
            model=app.state.model,
            data=request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    return PredictionResponse(
        predicted_demand=predictions[0],
        model_uri=DEFAULT_MODEL_URI,
    )