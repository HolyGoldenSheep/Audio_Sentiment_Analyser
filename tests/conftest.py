import pytest
import os
os.environ["TESTING"] = "1"
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """
    Activate test mode to disable Mongo connection.
    """
    os.environ["TESTING"] = "1"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client