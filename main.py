"""
FastAPI service: GET /health, POST /chat
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import agent
import retriever

app = FastAPI(title="SHL Assessment Recommender")


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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = agent.chat(messages)
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
