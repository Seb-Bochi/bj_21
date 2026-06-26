import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import torch
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from hydra import compose, initialize
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

try:
    from aiofiles import open as async_open

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from blackjack_predictor.models.ffnn import SimpleFNN

try:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
except Exception as e:
    raise RuntimeError(f"Failed to load Hydra configuration globally: {e}")

model = None
MODEL_PATH = Path(cfg.data_config.model_path)
LOG_FILE = Path("data/production_logs.jsonl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model state during application startup."""
    global model
    try:
        model = SimpleFNN(
            input_dim=cfg.model_config.input_dim,
            hidden_dim=cfg.model_config.hidden_dim,
            output_dim=cfg.model_config.output_dim,
        )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model weights missing at '{MODEL_PATH}'. Please run training pipelines first.")

        # Cloud Run serves this model on CPU-only instances.
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully pinned model runtime weights from {MODEL_PATH}")

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        yield

    finally:
        pass


app = FastAPI(
    title="Blackjack Predictor API",
    description="Production API serving blackjack game state predictions with live data drift telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


class InferenceRequest(BaseModel):
    """Input payload for blackjack inference."""

    dealt_card_1: int = Field(..., ge=0, le=11, description="Value of player initial card 1")
    dealt_card_2: int = Field(..., ge=0, le=11, description="Value of player initial card 2")
    dealer_card: int = Field(..., ge=0, le=11, description="Value of dealer face-up card")


class InferenceResponse(BaseModel):
    """Prediction payload returned by the API."""

    loss_probability: float
    win_probability: float
    prediction: bool


@app.get("/", include_in_schema=False)
def redirect_to_gui():
    """Redirect the root route to the API docs."""
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=InferenceResponse)
async def predict(payload: InferenceRequest):
    """Run inference and append a production log row."""
    if model is None:
        raise HTTPException(status_code=503, detail="Serving model is uninitialized.")

    input_features = [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]
    if len(input_features) != cfg.model_config.input_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Dimension mismatch. Model requires {cfg.model_config.input_dim} features, received {len(input_features)}.",
        )

    input_tensor = torch.tensor([input_features], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).tolist()
        predicted_class = outputs.argmax(dim=1).item()

    response_data = InferenceResponse(
        loss_probability=probabilities[0],
        win_probability=probabilities[1],
        prediction=bool(predicted_class),
    )

    log_entry = {"input": input_features, "prediction": int(predicted_class), "win_prob": probabilities[1]}

    if AIOFILES_AVAILABLE:
        async with async_open(LOG_FILE, mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")
    else:
        with open(LOG_FILE, mode="a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return response_data


@app.get("/monitoring/drift")
def detect_production_drift():
    """Compare production inputs against the reference dataset."""
    reference_path = Path("data/raw/blkjckhands.csv")

    if not LOG_FILE.exists():
        return {"status": "insufficient_data", "detail": "Production logs empty. Send requests first."}
    if not reference_path.exists():
        return {"status": "error", "detail": "Baseline reference dataset file absent from server environment."}

    ref_df = pd.read_csv(reference_path)
    ref_features = ref_df[["card1", "card2", "dealcard1"]].rename(columns={"card1": "f1", "card2": "f2", "dealcard1": "f3"})

    prod_records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                prod_records.append(json.loads(line)["input"])

    if len(prod_records) < 5:
        return {"status": "collecting", "detail": f"Need more log variations. Currently logged: {len(prod_records)}/5"}

    prod_features = pd.DataFrame(prod_records, columns=["f1", "f2", "f3"])

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_features, current_data=prod_features)

    raw_data = json.loads(my_eval.json())
    metrics_list = raw_data.get("metrics", [])

    summary = {
        "drift_detected": False,
        "metrics_evaluated": {"total_drifted_columns": 0, "drift_share": 0.0},
        "drift_by_feature": {
            "player_card_1": {"drift_detected": False, "score": 0.0},
            "player_card_2": {"drift_detected": False, "score": 0.0},
            "dealer_card": {"drift_detected": False, "score": 0.0},
        },
    }

    for m in metrics_list:
        metric_type = m.get("config", {}).get("type", "")

        if "DriftedColumnsCount" in metric_type:
            summary["metrics_evaluated"]["total_drifted_columns"] = m.get("value", {}).get("count", 0)
            summary["metrics_evaluated"]["drift_share"] = m.get("value", {}).get("share", 0.0)

            if summary["metrics_evaluated"]["total_drifted_columns"] > 0:
                summary["drift_detected"] = True

        elif "ValueDrift" in metric_type:
            col_name = m.get("config", {}).get("column")
            score = m.get("value", 0.0)
            threshold = m.get("config", {}).get("threshold", 0.1)
            is_drifted = bool(score > threshold)

            if col_name == "f1":
                summary["drift_by_feature"]["player_card_1"] = {"drift_detected": is_drifted, "score": round(score, 4)}
            elif col_name == "f2":
                summary["drift_by_feature"]["player_card_2"] = {"drift_detected": is_drifted, "score": round(score, 4)}
            elif col_name == "f3":
                summary["drift_by_feature"]["dealer_card"] = {"drift_detected": is_drifted, "score": round(score, 4)}

    return summary


@app.get("/health")
def health_check():
    """Return API health status."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="localhost", port=8000, reload=True)
