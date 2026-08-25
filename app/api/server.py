from __future__ import annotations

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import SupportAgent
from app.models import AgentResponse, CustomerSafeOrder
from app.tools.order_lookup import OrderLookupService


app = FastAPI(
    title="Aster & Row Customer Support Agent API",
    description="Deterministic, grounded RAG customer support service.",
    version="1.0.0"
)

agent = SupportAgent()
order_service = OrderLookupService()


class ChatRequest(BaseModel):
    message: str = Field(..., example="Where is ORD-1007?")
    session_id: str = Field(default="default_web_session", example="user_123")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    handoff_recommended: bool
    handoff_reason: Optional[str] = None


class OrderRequest(BaseModel):
    order_id: str = Field(..., example="ORD-1007")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "aster-row-support-agent", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    res: AgentResponse = agent.process_message(req.message, session_id=req.session_id)
    return ChatResponse(
        answer=res.answer,
        sources=res.sources,
        handoff_recommended=res.handoff_recommended,
        handoff_reason=res.handoff_reason
    )


@app.post("/orders/lookup", response_model=CustomerSafeOrder)
def lookup_order_status(req: OrderRequest):
    res = order_service.lookup(req.order_id)
    if not res.success or not res.order:
        raise HTTPException(status_code=404, detail=res.error_message or "Order not found")
    return res.order
