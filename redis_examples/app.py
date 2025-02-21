from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import redis
import logging
from typing import Callable
import requests


app = FastAPI()

redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

RATE_LIMIT = 5
TIME_WINDOW = 60


def check_country(client_ip):
    response = requests.get(
        f"http://ip-api.com/json/{client_ip}",
        params={"fields": "status,message,country,countryCode"},
    )
    data = response.json()

    if response.status_code == 200 and data.get("status") == "success":
        return {
            "country_code": data.get("countryCode"),
            "country_name": data.get("country"),
            "success": True,
        }
    else:
        return {"success": False, "error": data.get("message", "Unknown error")}


async def check_rate_limit(request: Request):
    client_ip = request.client.host
    redis_key = f"rate__limit:{client_ip}"
    data = check_country(client_ip)
    if data["country_name"].strip().lower() in ("india"):
        raise HTTPException(status_code=404, detail="Not Authorized")

    try:
        current_count = redis_client.get(redis_key)
        if current_count is None:
            redis_client.setex(redis_key, TIME_WINDOW, 1)
        else:
            current_count = int(current_count)
            if current_count >= RATE_LIMIT:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(TIME_WINDOW),
                        "X-RateLimit-Limit": str(RATE_LIMIT),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            redis_client.incr(redis_key)
    except redis.RedisError as e:
        logger.error(f"Redis error: {e}")
        pass


@app.get("/test")
async def test_endpoint(rate_limit: None = Depends(check_rate_limit)):
    return {"message": "This is a test endpoint"}


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=429, content={"detail": exc.detail})


@app.exception_handler(404)
async def rate_limit_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"detail": exc.detail})
