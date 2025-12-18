# schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# Authentication Schemas

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Type of token returned")


class TokenData(BaseModel):
    username: Optional[str] = Field(None, description="Username encoded inside JWT")


class UserLogin(BaseModel):
    username: str = Field(..., example="Karl")
    password: str = Field(..., example="YourPassword123")


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    email: EmailStr = Field(...)


class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr

    class Config:
        orm_mode = True

class AudioUploadResponse(BaseModel):
    sentiment: str
    confidence: float


class UserPublic(BaseModel):
    username: str = Field(..., description="Public username")


# Audio Prediction Schemas

class Base64Audio(BaseModel):
    audio_base64: str = Field(
        ...,
        description="Base64 encoded audio file (wav/mp3)",
        example="UklGRiQAAABXQVZFZm10IBAAAAAB..."
    )


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


# System / Model Info

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    timestamp: str


class ModelInfo(BaseModel):
    model_path: str
    embedding_model: str
    classifier: str
    version: str
    last_loaded: str
