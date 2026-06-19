import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from hydra import initialize, compose

# Prometheus & Evidently AI imports
from prometheus_fastapi_instrumentator import Instrumentator
from evidently import Report
from evidently.presets import DataDriftPreset

# Handle importing aiofiles safely for asynchronous logging
try:
    from aiofiles import open as async_open
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from blackjack_predictor.models.ffnn import SimpleFNN


# 1. Safely Load Hydra Configuration without CLI overrides
try:
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
except Exception:
    try:
        # Docker fallback: Search inside the packaged src structure explicitly
        with initialize(version_base=None, config_path="src/blackjack_predictor/configs"):
            cfg = compose(config_name="config")
    except Exception as e:
        raise RuntimeError(f"Failed to load Hydra configuration globally: {e}")

# Global state handles for resource lifecycle management
model = None
MODEL_PATH = Path(cfg.data_config.model_path)
LOG_FILE = Path("data/production_logs.jsonl")


# 2. Lifecycle Manager handles container startup and shutdown states
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to load model parameters safely on startup."""
    global model
    try:
        # Initialize internal network weights with config dimensions
        model = SimpleFNN(
            input_dim=cfg.model_config.input_dim,
            hidden_dim=cfg.model_config.hidden_dim,
            output_dim=cfg.model_config.output_dim,
        )
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model weights missing at '{MODEL_PATH}'. Please run training pipelines first."
            )
            
        # Map parameters explicitly to CPU since Cloud Run instances lack GPU hardware
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully pinned model runtime weights from {MODEL_PATH}")
        
        # Create directory space for telemetry logging safely
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        yield  # Service traffic active
        
    finally:
        pass


# 3. Initialize FastAPI framework App instance
app = FastAPI(
    title="Blackjack Predictor API",
    description="Production API serving blackjack game state predictions with live data drift telemetry.",
    version="1.0.0",
    lifespan=lifespan
)

# ================= M28: PROMETHEUS INSTRUMENTATION =================
# Attach Prometheus metrics sensors to the FastAPI app and expose /metrics
Instrumentator().instrument(app).expose(app)
# ===================================================================

# 4. Define structural Request/Response JSON serialization schemes
class InferenceRequest(BaseModel):
    dealt_card_1: int = Field(..., description="Value of player initial card 1")
    dealt_card_2: int = Field(..., description="Value of player initial card 2")
    dealer_card: int = Field(..., description="Value of dealer face-up card")

class InferenceResponse(BaseModel):
    loss_probability: float
    win_probability: float
    prediction: bool


@app.get("/", include_in_schema=False)
def redirect_to_gui():
    """Redirect home route traffic straight to automated swagger interactive docs."""
    return RedirectResponse(url="/docs")


# 5. Live Serving & Telemetry Inference endpoint
@app.post("/predict", response_model=InferenceResponse)
async def predict(payload: InferenceRequest):
    """Evaluates blackjack probability tables and appends data logs asynchronously."""
    if model is None:
        raise HTTPException(status_code=503, detail="Serving model is uninitialized.")
        
    # Boundary threshold input validation
    if not all(0 <= card <= 11 for card in [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]):
        raise HTTPException(
            status_code=400, 
            detail="Card boundary distributions must be integer weights between 0 and 11 inclusive."
        )
        
    input_features = [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]
    if len(input_features) != cfg.model_config.input_dim:
        raise HTTPException(
            status_code=400, 
            detail=f"Dimension mismatch. Model requires {cfg.model_config.input_dim} features, received {len(input_features)}."
        )

    # Transform numerical array into a 2D float tensor batch layer
    input_tensor = torch.tensor([input_features], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).tolist()
        predicted_class = outputs.argmax(dim=1).item()

    # Formulate prediction structure mapping
    response_data = InferenceResponse(
        loss_probability=probabilities[0],
        win_probability=probabilities[1],
        prediction=bool(predicted_class),
    )

    # Log operational payload profiles for data drift checks
    log_entry = {
        "input": input_features,
        "prediction": int(predicted_class),
        "win_prob": probabilities[1]
    }
    
    # Commit asynchronous I/O writes safely to dodge thread blocking under query bursts
    if AIOFILES_AVAILABLE:
        async with async_open(LOG_FILE, mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")
    else:
        with open(LOG_FILE, mode="a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return response_data


# 6. Evidently AI Monitoring & Drift Evaluation API Route (M27)
@app.get("/monitoring/drift")
def detect_production_drift():
    """Compiles operational telemetry against the baseline and prints statistical drift indices."""
    reference_path = Path("data/raw/blkjckhands.csv")
    
    if not LOG_FILE.exists():
        return {"status": "insufficient_data", "detail": "Production logs empty. Send requests first."}
    if not reference_path.exists():
        return {"status": "error", "detail": "Baseline reference dataset file absent from server environment."}
        
    # Pull reference frame properties
    ref_df = pd.read_csv(reference_path)
    # Ensure validation targets features match features parsed by payload arrays
    ref_features = ref_df[["card1", "card2", "dealcard1"]].rename(
        columns={"card1": "f1", "card2": "f2", "dealcard1": "f3"}
    )
    
    # Compile text logs back into structured memory lists
    prod_records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                prod_records.append(json.loads(line)["input"])
                
    if len(prod_records) < 5:
        return {"status": "collecting", "detail": f"Need more log variations. Currently logged: {len(prod_records)}/5"}
        
    prod_features = pd.DataFrame(prod_records, columns=["f1", "f2", "f3"])

    # Generate the Evidently statistical framework metrics under the hood
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_features, current_data=prod_features)
    
    # Parse the new v0.5+ JSON array structure
    raw_data = json.loads(my_eval.json())
    metrics_list = raw_data.get("metrics", [])
    
    # Initialize default summary
    summary = {
        "drift_detected": False,
        "metrics_evaluated": {
            "total_drifted_columns": 0,
            "drift_share": 0.0
        },
        "drift_by_feature": {
            "player_card_1": {"drift_detected": False, "score": 0.0},
            "player_card_2": {"drift_detected": False, "score": 0.0},
            "dealer_card": {"drift_detected": False, "score": 0.0}
        }
    }
    
    # Loop through the metrics array and map the values
    for m in metrics_list:
        metric_type = m.get("config", {}).get("type", "")
        
        # Check overall drift count
        if "DriftedColumnsCount" in metric_type:
            summary["metrics_evaluated"]["total_drifted_columns"] = m.get("value", {}).get("count", 0)
            summary["metrics_evaluated"]["drift_share"] = m.get("value", {}).get("share", 0.0)
            
            # If any columns drifted, flag the dataset
            if summary["metrics_evaluated"]["total_drifted_columns"] > 0:
                summary["drift_detected"] = True
                
        # Check individual feature drift (Threshold is usually 0.1 for Jensen-Shannon)
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
    """Liveness probe targeting system check metrics."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)