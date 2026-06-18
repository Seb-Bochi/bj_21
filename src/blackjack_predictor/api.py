import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from hydra import initialize, compose
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse

from blackjack_predictor.models.ffnn import SimpleFNN


# 2. Safely Load Hydra Configuration without CLI overrides
# Assumes config is at: ./configs/config.yaml relative to where you run this script
try:
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
except Exception as e:
    raise RuntimeError(f"Failed to load Hydra configuration: {e}")

# 3. Global Variables for Model Management
model = None
MODEL_PATH = Path(cfg.data_config.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to load the model on startup and clean up if needed."""
    global model
    try:
        # Load the model weights during application startup
        model = SimpleFNN(
            input_dim=cfg.model_config.input_dim,
            hidden_dim=cfg.model_config.hidden_dim,
            output_dim=cfg.model_config.output_dim,
        )
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model weights not found at '{MODEL_PATH}'. Please run training first."
            )
            
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
        
        yield  # Run the application
        
    finally:
        # Optional cleanup code can go here (e.g., closing database connections)
        pass

# 1. Initialize FastAPI App
app = FastAPI(
    title="Blackjack Predictor API",
    description="Production API for predicting optimal blackjack moves using an FNN.",
    version="1.0.0",
    lifespan=lifespan
)


# 4. Define Request and Response Schemas using Pydantic
class InferenceRequest(BaseModel):
    dealt_card_1: int 
    dealt_card_2: int
    dealer_card: int

class InferenceResponse(BaseModel):
    loss_probability: float
    win_probability: float
    prediction: bool


@app.get("/", include_in_schema=False)
def redirect_to_gui():
    """Redirect the root URL to the interactive API documentation."""
    return RedirectResponse(url="/docs")

# 5. Define the Inference Endpoint
@app.post("/predict", response_model=InferenceResponse)
def predict(payload: InferenceRequest):
    """Predict the optimal action given a state representation."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")
        
    # Validate input dimensions match what the model expects
    if not all(0 <= card <= 11 for card in [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]):
        raise HTTPException(
            status_code=400, 
            detail="Card values must be between 0 and 11 (inclusive)."
        )
    input_features = [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]
    if len(input_features) != cfg.model_config.input_dim:
        raise HTTPException(
            status_code=400, 
            detail=f"Input dimensions mismatch. Expected {cfg.model_config.input_dim} features, got {len(input_features)}."
        )

    # Convert features to torch tensor and add batch dimension (1, input_dim)
    input_tensor = torch.tensor([input_features], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).tolist()
       
        predicted_class = outputs.argmax(dim=1).item()

    return InferenceResponse(
        loss_probability=probabilities[0],
        win_probability=probabilities[1],
        prediction=bool(predicted_class),
    )


@app.get("/health")
def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)