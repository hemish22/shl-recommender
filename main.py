"""
FastAPI service: GET /health, POST /chat
"""

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import agent
import retriever

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SHL Assessment Recommender")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(key: str = Depends(_API_KEY_HEADER)):
    expected = os.environ.get("API_KEY")
    if expected and key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.on_event("startup")
def _startup():
    retriever._load()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 20:
            raise ValueError("too many messages")
        for m in v:
            if m.role not in ("user", "assistant"):
                raise ValueError(f"invalid role: {m.role}")
            if len(m.content) > 2000:
                raise ValueError("message content exceeds 2000 characters")
            if not m.content.strip():
                raise ValueError("message content cannot be blank")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_verify_api_key)])
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        result = agent.chat(messages)
    except agent.PromptInjectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", ""),
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )
