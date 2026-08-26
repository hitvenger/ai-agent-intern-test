from __future__ import annotations

from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import SupportAgent
from app.models import AgentResponse, AgentTrace, CustomerSafeOrder
from app.tools.order_lookup import OrderLookupService

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Aster & Row Customer Support Agent API",
    description="Deterministic, grounded RAG customer support service.",
    version="1.0.0"
)

# Enable CORS for local demo frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

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
    trace: Optional[AgentTrace] = None


class OrderRequest(BaseModel):
    order_id: str = Field(..., example="ORD-1007")


@app.get("/", include_in_schema=False)
def serve_ui():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Aster & Row Customer Support API is running. Access /docs for OpenAPI specifications."}


@app.get("/style.css", include_in_schema=False)
def serve_css_fallback():
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js", include_in_schema=False)
def serve_js_fallback():
    js_file = FRONTEND_DIR / "app.js"
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


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
        handoff_reason=res.handoff_reason,
        trace=res.trace
    )


@app.post("/orders/lookup", response_model=CustomerSafeOrder)
def lookup_order_status(req: OrderRequest):
    res = order_service.lookup(req.order_id)
    if not res.success or not res.order:
        raise HTTPException(status_code=404, detail=res.error_message or "Order not found")
    return res.order

