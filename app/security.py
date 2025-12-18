# security.py

import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

from .db import get_user_by_username, verify_password

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a plaintext password for storage."""
    return pwd_context.hash(password)


# JWT CONFIGURATION
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# JWT CREATION
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create a JWT access token."""
    to_encode = data.copy()

    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


# AUTHENTICATION
async def authenticate_user(username: str, password: str):
    """
    Validate username + password from MongoDB asynchronously.
    Returns user dict if OK, else False.
    """
    user = await get_user_by_username(username) 

    if not user:
        return False

    hashed = user.get("hashed_password")
    if not verify_password(password, hashed):
        return False

    return user


# JWT VALIDATION
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Verify the JWT token and return authenticated username.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username = payload.get("sub")

    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await get_user_by_username(username)  

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
