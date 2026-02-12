# db.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime
from bson import ObjectId

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "audio_sentiment_db")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

users_collection = db["users"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def get_user_by_username(username: str):
    return await users_collection.find_one({"username": username})


async def create_user(username: str, password: str, email: str):
    print(" create_user() CALLED")

    existing = await get_user_by_username(username)
    print(" existing user:", existing)

    if existing:
        raise ValueError("Username already exists")

    hashed_pw = pwd_context.hash(password)
    print(" hashed password generated")

    user_doc = {
        "username": username,
        "hashed_password": hashed_pw,
        "email": email
    }

    print(" inserting into MongoDB...")
    result = await users_collection.insert_one(user_doc)
    print(" insert result:", result.inserted_id)

    user_doc["_id"] = result.inserted_id
    print(" create_user() DONE")
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
    # Force initialization
    await client.admin.command("ping")


async def close_db():
    client.close()
