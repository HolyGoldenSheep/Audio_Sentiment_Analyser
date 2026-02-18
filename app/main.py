# main.py
from dotenv import load_dotenv
load_dotenv()

import base64
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
import time
from datetime import datetime
from fastapi import Body
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from app.schemas import (
    UserCreate,
    UserResponse,
    AudioUploadResponse,
    Token,
    TokenData,
    UserLogin,
    Base64Audio,
    PredictionResponse,
    HealthResponse,
    ModelInfo,
)
from app.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    authenticate_user,   
)

from app.db import (
    create_user,
    get_user_by_username,
    save_prediction,
    connect_db,
    close_db,
    MONGO_URL,
    DB_NAME,
    db
)

from app.model import SentimentModel
from bson import ObjectId
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request


# Collecter Registry prometheus not liking double metrics
registry = CollectorRegistry()

# CONFIG
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PROD_123")
MODEL_PATH = os.getenv("MODEL_PATH", "app/Model/emotion_model")

# Lifespan 
@asynccontextmanager 
async def lifespan(app: FastAPI): 
    await connect_db()

    yield

    await close_db()

# APP INIT
app = FastAPI(
    title="Audio Sentiment Analysis API",
    version="1.0.0",
    description="REST API exposing an audio sentiment model with JWT authentication.",
    lifespan=lifespan
)
# instrumentator

Instrumentator().instrument(app).expose(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files correctly
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Example homepage route
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Metrics Prometheus

Predict_Requests = Counter(
    "predict_Requests_total",
    "Total number of failed prediction requests",
    registry=registry
)
Predict_Errors = Counter(
    "predict_errors_total",
    "Total number of failed predictions",
    registry=registry
)
Predict_Duration = Histogram(
    "predict_duration_seconds",
    "Prediction latency in seconds",
    registry=registry
)

Audio_Size = Histogram(
    "predict_audio_size_bytes",
    "Audio file size",
    registry=registry
)
Model_Confidence = Gauge(
    "Predict_last_confidence",
    "Confidence of last prediction",
    registry=registry
)
Model_predictions_total = Counter(
    "Model_predictions_total",
    "Total number of Model predictions performed"
)

@app.get("/debug/create-test-user")
async def debug_create_user():
    test_user = {"username": "test_user_debug", "password": "123"}
    result = await db["users"].insert_one(test_user)
    return {"inserted_id": str(result.inserted_id)}

@app.get("/debug/where-am-i-writing")
async def where_am_i_writing():
    return {
        "mongo_url_used": MONGO_URL,
        "db_name_used": DB_NAME,
        "full_db_name_motor_sees": db.name,
        "collections_in_this_db": await db.list_collection_names()
    }
@app.get("/debug/list-all-databases")
async def list_all_dbs():
    client = db.client
    dbs = await client.list_database_names()
    return {"databases": dbs}


# LOAD MODEL
print("Loading Sentiment Model...")
sentiment_model = SentimentModel.load(MODEL_PATH)
print("Model loaded successfully.")


# ENDPOINTS

@app.post("/auth/signup")
async def signup(user: UserCreate):
    try:
        user_doc = await create_user(user.username, user.password, user.email)

        return {
            "id": str(user_doc.get("_id", "")),
            "username": user_doc["username"],
            "email": user_doc["email"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authentication endpoint:
    Returns a JWT token if username & password are correct.
    """
    user = await authenticate_user(form_data.username, form_data.password) 

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model", response_model=ModelInfo, tags=["Model"])
async def model_metadata(user=Depends(get_current_user)):
    """
    Show model architecture & version.
    Protected endpoint.
    """
    return {
        "model_path": MODEL_PATH,
        "embedding_model": "facebook/wav2vec2-base",
        "classifier": "Keras Dense Classifier",
        "version": "1.0",
        "last_loaded": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(
    file: UploadFile = File(...),
    current_user: dict | None = Depends(get_current_user)
):
    Predict_Requests.inc()
    start_time = time.time()

    try:
        # Input validation
        ALLOWED_AUDIO_TYPES = {
            "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/flac"
        }

        if file.content_type not in ALLOWED_AUDIO_TYPES:
            Predict_Errors.inc()
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        audio_bytes = await file.read()

        Audio_Size.observe(len(audio_bytes))

        MAX_AUDIO_SIZE = 20 * 1024 * 1024
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            Predict_Errors.inc()
            raise HTTPException(status_code=413, detail="Audio file too large max 20MB")

        if not current_user:
            Predict_Errors.inc()
            raise HTTPException(status_code=401, detail="Not authenticated")

        result = sentiment_model.predict_bytes(audio_bytes)
    
        # monitoring pred total
        Model_predictions_total.inc()

        prediction_doc = {
            "user_id": ObjectId(current_user["_id"]),
            "label": result["label"],
            "confidence": result["confidence"],
            "filename": file.filename,
            "created_at": datetime.utcnow()
        }

        await db["predictions"].insert_one(prediction_doc)

        return {
            "label": result["label"],
            "confidence": result["confidence"]
        }

    except Exception:
        Predict_Errors.inc()
        raise

    finally:
        Predict_Duration.observe(time.time() - start_time)


