# db.py
import os
from datetime import datetime
from bson import ObjectId
from passlib.context import CryptContext

TESTING = os.getenv("TESTING") == "1"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test mode
if TESTING:

    _fake_users = []
    _fake_predictions = []

    async def get_user_by_username(username: str):
        for user in _fake_users:
            if user["username"] == username:
                return user
        return None

    async def create_user(username: str, password: str, email: str):
        existing = await get_user_by_username(username)
        if existing:
            raise ValueError("Username already exists")

        hashed_pw = pwd_context.hash(password)

        user_doc = {
            "_id": ObjectId(),
            "username": username,
            "hashed_password": hashed_pw,
            "email": email
        }

        _fake_users.append(user_doc)
        return user_doc

    def verify_password(plain_password: str, hashed_password: str):
        return pwd_context.verify(plain_password, hashed_password)

    async def save_prediction(
        user_id: ObjectId,
        username: str,
        emotion: str,
        confidence: float,
        filename: str | None = None
    ):
        doc = {
            "_id": ObjectId(),
            "user_id": user_id,
            "username": username,
            "emotion": emotion,
            "confidence": confidence,
            "filename": filename,
            "created_at": datetime.utcnow()
        }
        _fake_predictions.append(doc)
        return doc["_id"]

    async def connect_db():
        return

    async def close_db():
        return

    db = None
    MONGO_URL = None
    DB_NAME = None

# Production mode

else:
    from motor.motor_asyncio import AsyncIOMotorClient

    MONGO_URL = os.getenv("MONGO_URL")
    DB_NAME = os.getenv("DB_NAME", "audio_sentiment_db")

    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]

    users_collection = db["users"]

    async def get_user_by_username(username: str):
        return await users_collection.find_one({"username": username})

    async def create_user(username: str, password: str, email: str):
        existing = await get_user_by_username(username)
        if existing:
            raise ValueError("Username already exists")

        hashed_pw = pwd_context.hash(password)

        user_doc = {
            "username": username,
            "hashed_password": hashed_pw,
            "email": email
        }

        result = await users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        return user_doc

    def verify_password(plain_password: str, hashed_password: str):
        return pwd_context.verify(plain_password, hashed_password)

    async def save_prediction(
        user_id: ObjectId,
        username: str,
        emotion: str,
        confidence: float,
        filename: str | None = None
    ):
        doc = {
            "user_id": user_id,
            "username": username,
            "emotion": emotion,
            "confidence": confidence,
            "filename": filename,
            "created_at": datetime.utcnow()
        }

        result = await db["predictions"].insert_one(doc)
        return result.inserted_id

    async def connect_db():
        return

    async def close_db():
        client.close()
