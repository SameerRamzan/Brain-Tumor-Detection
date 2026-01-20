import pytest
import httpx
import time

# The API URL when running in Docker Compose locally or in CI
API_URL = "http://localhost:8000"

def test_api_documentation_accessible():
    """
    Verifies that the FastAPI documentation endpoint is reachable.
    This confirms the container started and the app is running.
    """
    # Retry logic to wait for container startup
    max_retries = 5
    for _ in range(max_retries):
        try:
            response = httpx.get(f"{API_URL}/docs")
            if response.status_code == 200:
                assert response.status_code == 200
                return
        except httpx.ConnectError:
            time.sleep(2)
    
    pytest.fail("Could not connect to API docs after multiple retries.")

def test_token_endpoint_exists():
    """
    Verifies the auth endpoint handles requests (even with bad credentials).
    """
    try:
        response = httpx.post(
            f"{API_URL}/token", 
            data={"username": "test", "password": "wrongpassword"}
        )
        # Should return 401 Unauthorized (meaning the endpoint works)
        # or 503 if DB is initializing, but we accept 401 as success for connectivity.
        assert response.status_code in [401, 503]
    except httpx.ConnectError:
        pytest.fail("Could not connect to Token endpoint.")