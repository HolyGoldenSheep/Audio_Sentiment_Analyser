# main.py
import base64
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import jwt, JWTError
from app.model import SentimentModel

# CONFIGURATION

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Dummy user (replace with DB or user store)
FAKE_USER = {
    "username": "Karl",
    "password": "Karl2703"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# FastAPI Initialization

app = FastAPI(
    title="Audio Sentiment Analysis API",
    version="1.0.0",
    description="REST API exposing an audio sentiment model with JWT authentication."
)

# CORS (locked down for OWASP compliance)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # change for production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# LOAD MODEL

MODEL_PATH = "Model/emotion_model.h5"
model = SentimentModel.load(MODEL_PATH)

# AUTHENTICATION UTILITIES

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_user(username: str, password: str):
    """Very simple auth — replace with hashed pw + DB."""
    return username == FAKE_USER["username"] and password == FAKE_USER["password"]

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()

    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Decode & validate JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(401, "Invalid authentication")
        if username != FAKE_USER["username"]:
            raise HTTPException(401, "User not recognized")
        return username
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")


# SCHEMAS

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class Base64Audio(BaseModel):
    audio_base64: str


# ENDPOINTS

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate a JWT access token."""
    if not verify_user(form_data.username, form_data.password):
        raise HTTPException(401, "Incorrect username or password")

    access_token = create_access_token({"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/model", tags=["Model"], dependencies=[Depends(get_current_user)])
async def model_info():
    """Return metadata about the loaded model."""
    return {
        "model_path": MODEL_PATH,
        "embedding_model": "facebook/wav2vec2-base",
        "classifier": "Keras Dense NN",
        "version": "1.0",
        "last_loaded": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_audio(
    file: UploadFile | None = File(default=None),
    body: Base64Audio | None = None,
    user: str = Depends(get_current_user)
):
    """
    Predict emotion from an audio file.
    Accepts either:
     - multipart file upload
     - JSON { "audio_base64": "..." }
    """
    # Validation
    if file is None and (body is None or not body.audio_base64):
        raise HTTPException(400, "No audio file or base64 audio provided")

    # Case 1: Uploaded file
    if file:
        audio_bytes = await file.read()
    else:
        # Case 2: Base64 audio
        try:
            audio_bytes = base64.b64decode(body.audio_base64)
        except:
            raise HTTPException(400, "Invalid base64 audio")

    # Size limit: max 10 MB
    if len(audio_bytes) > 10_000_000:
        raise HTTPException(413, "Audio file too large (max 10MB)")

    # Run model prediction
    try:
        result = model.predict_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(500, f"Model error: {str(e)}")

    return PredictionResponse(label=result["label"], confidence=result["confidence"])


# ROOT ENDPOINT

@app.get("/", tags=["System"])
def index():
    return {
        "message": "Audio Sentiment Analysis API",
        "docs": "/docs",
        "auth": "/auth/token",
        "predict": "/predict"
    }

