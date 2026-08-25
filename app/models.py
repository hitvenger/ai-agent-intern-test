from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from knowledge-base markdown front-matter."""
    document_id: str
    title: str
    status: str = "active"  # "active", "superseded", "draft"
    effective_date: Optional[str] = None
    superseded_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    audience: str = "customer"  # "customer", "internal"
    policy_authority: str = "official"  # "official", "none"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True


class PolicyChunk(BaseModel):
    """A granular chunk of policy/product content with source provenance."""
    chunk_id: str
    file_name: str
    heading: str
    citation: str  # e.g., "01-returns-policy-current.md#Standard return window"
    content: str
    metadata: DocumentMetadata
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class OrderItem(BaseModel):
    """Customer-safe representation of an ordered item."""
    sku: Optional[str] = None
    name: str
    quantity: int = 1
    final_sale: bool = False


class CustomerSafeOrder(BaseModel):
    """
    Air-gapped, sanitized customer order representation.
    Guarantees that customer PII (name, email, address) and internal fields
    (risk_score, warehouse_note, support_tags) are completely excluded.
    """
    order_id: str
    membership_tier: str  # "standard", "trailplus"
    items: List[OrderItem] = Field(default_factory=list)
    placed_at: str
    status: str  # "pending", "processing", "shipped", "delayed", "delivered", "cancelled", "returned", "exception"
    status_updated_at: str
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    
    # Deterministic metadata computed by tool
    requires_handoff: bool = False
    cancellation_eligible: bool = False
    stale_fields_cleared: bool = False
    sanitized_note: Optional[str] = None


class OrderLookupResult(BaseModel):
    """Result of an order lookup operation."""
    success: bool
    order_id: str
    order: Optional[CustomerSafeOrder] = None
    error_message: Optional[str] = None
    requires_handoff: bool = False


class RetrievalResult(BaseModel):
    """Result of a policy/product knowledge base retrieval."""
    chunks: List[PolicyChunk] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_description: Optional[str] = None
    requires_handoff: bool = False
    handoff_reason: Optional[str] = None


class AgentTrace(BaseModel):
    """Observability trace capturing the complete lifecycle of a request."""
    session_id: str
    user_query: str
    intent: Optional[str] = None
    order_id_extracted: Optional[str] = None
    tool_called: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[dict[str, Any]] = None
    retrieved_citations: List[str] = Field(default_factory=list)
    retrieved_scores: List[float] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_details: Optional[str] = None
    handoff_recommended: bool = False
    handoff_reason: Optional[str] = None
    generation_mode: str = "deterministic"  # "deterministic" or "llm"


class AgentResponse(BaseModel):
    """Final structured response returned to the customer/interface."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    handoff_recommended: bool = False
    handoff_reason: Optional[str] = None
    trace: Optional[AgentTrace] = None
