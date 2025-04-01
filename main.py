from fastapi import FastAPI, HTTPException, Request
import joblib
import torch
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertConfig,
    DistilBertPreTrainedModel,
    DistilBertModel
)
import torch.nn as nn
from pydantic import BaseModel
import os
from typing import List, Dict, Any
import time
from datetime import datetime

app = FastAPI(
    title="Phishing Email Detector API",
    description="API for detecting phishing emails using custom DistilBERT model",
    version="1.0.0"
)

# ========== MODEL DEFINITION ==========
class PhishingBERT(DistilBertPreTrainedModel):
    """Custom DistilBERT model for phishing detection with numeric features"""
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.numeric_proj = nn.Linear(2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

# ========== PATHS & CONFIG ==========
BASE_DIR = r"C:\Users\salia\Documents\NLP\model\phishing_detector_pipeline"
PATHS = {
    "weights": os.path.join(BASE_DIR, "model_weights.pt"),
    "tokenizer": os.path.join(BASE_DIR, "tokenizer"),
    "assets": os.path.join(BASE_DIR, "pipeline_assets.joblib")
}

# ========== LOAD PIPELINE ==========
def load_pipeline():
    """Load and validate all pipeline components"""
    print("🔥 Loading pipeline components...")
    
    # Verify files exist
    for name, path in PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} at {path}")
        print(f"✅ Found {name} at {path}")

    # Load assets
    assets = joblib.load(PATHS["assets"])
    required_assets = ['scaler', 'clean_text', 'enhance_phishing_tags', 'class_names']
    for asset in required_assets:
        if asset not in assets:
            raise KeyError(f"Missing {asset} in pipeline assets")
    
    # Initialize components
    tokenizer = DistilBertTokenizerFast.from_pretrained(PATHS["tokenizer"])
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    model = PhishingBERT(config)
    model.load_state_dict(torch.load(PATHS["weights"], weights_only=True))
    model.eval()
    
    print("🚀 Pipeline loaded successfully!")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "scaler": assets['scaler'],
        "clean_text": assets['clean_text'],
        "enhance_tags": assets['enhance_phishing_tags'],
        "class_names": assets['class_names']
    }

# Load pipeline at startup
try:
    pipeline = load_pipeline()
except Exception as e:
    print(f"❌ Critical pipeline loading error: {str(e)}")
    raise

# ========== API SCHEMAS ==========
class EmailRequest(BaseModel):
    text: str
    threshold: float = 0.5  # Default decision threshold

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    processing_time: float
    model_version: str = "1.0.0"

# ========== MIDDLEWARE ==========
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ========== API ENDPOINTS ==========
@app.get("/", include_in_schema=False)
def root():
    """Redirect root to docs"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=Dict[str, Any])
def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(email: EmailRequest):
    """
    Predict if an email is phishing
    
    Parameters:
    - text: Email content to analyze
    - threshold: Decision threshold (default: 0.5)
    
    Returns:
    - prediction: 'Phishing' or 'Legitimate'
    - probability: Confidence score (0-1)
    - confidence: 'high' (>0.8 or <0.2), 'medium' otherwise
    """
    start_time = time.time()
    
    try:
        # 1. Preprocess
        cleaned = pipeline["clean_text"](email.text)
        enhanced = pipeline["enhance_tags"](cleaned)
        
        # 2. Prepare numeric features
        urgency_words = ['urgent', 'verify', 'click', 'account', 'suspend']
        urgency_score = sum(enhanced.lower().count(w) for w in urgency_words)
        numeric = pipeline["scaler"].transform([[len(email.text), urgency_score]])
        
        # 3. Tokenize
        inputs = pipeline["tokenizer"](
            enhanced,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        
        # 4. Predict
        with torch.no_grad():
            numeric_tensor = torch.tensor(numeric, dtype=torch.float32)
            outputs = pipeline["model"](**inputs, numeric_features=numeric_tensor)
            prob = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        
        # Determine confidence level
        if prob > 0.8 or prob < 0.2:
            confidence = "high"
        else:
            confidence = "medium"
            
        return {
            "prediction": "Phishing" if prob >= email.threshold else "Legitimate",
            "probability": round(prob, 4),
            "confidence": confidence,
            "processing_time": round(time.time() - start_time, 4)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ========== STARTUP ==========
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print(f"🕒 Server started at {datetime.now().isoformat()}")
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ Using CPU")
