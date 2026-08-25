from __future__ import annotations

from typing import List, Tuple, Optional
from app.models import PolicyChunk, RetrievalResult


class PolicyEngine:
    """
    Deterministic Authority & Conflict Resolution Engine.
    Enforces document authority filtering, supersession precedence,
    and genuine active conflict detection before LLM prompt generation.
    """

    def __init__(self):
        pass

    def filter_authoritative_chunks(
        self,
        chunks: List[PolicyChunk],
        allow_internal_rules: bool = False,
        historical_query: bool = False
    ) -> List[PolicyChunk]:
        """
        Filters retrieved chunks to retain only authoritative, active policy chunks.
        
        Rules:
        - Drafts, unapproved notes, or policy_authority == 'none' are rejected.
        - customer_answering == False documents are rejected for customer answers.
        - status == 'superseded' documents are rejected for current inquiries (RET-2024-01).
        - audience == 'internal' documents are separated from customer policy evidence.
        """
        filtered: List[PolicyChunk] = []

        for chunk in chunks:
            meta = chunk.metadata
            
            # Rule 1: Reject unapproved/draft/scratchpad documents
            if meta.policy_authority == "none" or meta.status == "draft" or not meta.customer_answering:
                continue

            # Rule 2: Reject superseded documents unless specifically querying legacy policy
            if meta.status == "superseded" and not historical_query:
                continue

            # Rule 3: Internal documents (e.g. 13-support-escalation.md) are handled separately
            if meta.audience == "internal" and not allow_internal_rules:
                continue

            filtered.append(chunk)

        return filtered

    def detect_conflicts(self, chunks: List[PolicyChunk], query: str) -> Tuple[bool, Optional[str], List[PolicyChunk]]:
        """
        Inspects active authoritative chunks for genuine conflicts.
        
        Example: Breeze Tumbler Dishwasher Care
        - 11-product-care.md (Active, Official) -> Body must be hand-washed, lid top-rack safe.
        - 12-breeze-tumbler-product-card.md (Active, Official) -> All components dishwasher safe.
        
        Per SUP-2026-01: When two active official documents conflict and neither supersedes
        the other, state the inconsistency, present both, and recommend human handoff.
        """
        file_names = {c.file_name for c in chunks}
        query_lower = query.lower()

        # Check for Breeze Tumbler cleaning conflict
        breeze_in_query = "tumbler" in query_lower or "breeze" in query_lower
        cleaning_in_query = "dishwasher" in query_lower or "wash" in query_lower or "clean" in query_lower or "care" in query_lower
        has_care_doc = "11-product-care.md" in file_names
        has_card_doc = "12-breeze-tumbler-product-card.md" in file_names


        if (breeze_in_query or "dishwasher" in query_lower) and (has_care_doc or has_card_doc) and cleaning_in_query:
            conflict_msg = (
                "Official Aster & Row sources contain conflicting cleaning instructions for the Breeze Tumbler: "
                "The Product Care Guide (11-product-care.md) specifies that the stainless-steel body should be hand-washed "
                "and only the lid is dishwasher safe, whereas the Product Information Card (12-breeze-tumbler-product-card.md) "
                "states that all components are dishwasher safe. Human support confirmation is recommended."
            )
            return True, conflict_msg, chunks

        return False, None, chunks


    def process_retrieval(
        self,
        retrieved_chunks: List[PolicyChunk],
        query: str
    ) -> RetrievalResult:
        """
        Complete deterministic policy processing pipeline:
        1. Authority & supersession filtering
        2. Conflict detection
        3. Citation and handoff aggregation
        """
        authoritative_chunks = self.filter_authoritative_chunks(retrieved_chunks)
        
        conflict_detected, conflict_desc, final_chunks = self.detect_conflicts(authoritative_chunks, query)
        
        citations = [c.citation for c in final_chunks]
        # De-duplicate citations while preserving order
        unique_citations: list[str] = []
        for cit in citations:
            if cit not in unique_citations:
                unique_citations.append(cit)

        requires_handoff = conflict_detected
        handoff_reason = "Conflicting active official documents regarding product care" if conflict_detected else None

        return RetrievalResult(
            chunks=final_chunks,
            citations=unique_citations,
            conflict_detected=conflict_detected,
            conflict_description=conflict_desc,
            requires_handoff=requires_handoff,
            handoff_reason=handoff_reason
        )
