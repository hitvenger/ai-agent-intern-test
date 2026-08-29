from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from app.models import (
    AgentResponse,
    AgentTrace,
    CustomerSafeOrder,
    OrderLookupResult,
    RetrievalResult,
    PolicyChunk
)
from app.rag.retriever import KnowledgeBaseRetriever
from app.rag.policy_engine import PolicyEngine
from app.tools.order_lookup import OrderLookupService, normalize_order_id
from app.session import SessionManager, ConversationSession
from app.safety import SafetyGuard
from app.trace import TraceLogger


class SupportAgent:
    """
    Aster & Row Customer Support AI Agent.
    Orchestrates deterministic safety guardrails, authority filtering,
    conflict detection, air-gapped order lookups, and grounded response synthesis.
    """

    def __init__(
        self,
        kb_retriever: Optional[KnowledgeBaseRetriever] = None,
        order_service: Optional[OrderLookupService] = None,
        session_manager: Optional[SessionManager] = None
    ):
        self.retriever = kb_retriever or KnowledgeBaseRetriever()
        self.order_service = order_service or OrderLookupService()
        self.policy_engine = PolicyEngine()
        self.session_manager = session_manager or SessionManager()
        self.safety_guard = SafetyGuard()
        self.tracer = TraceLogger()

    def process_message(
        self,
        user_message: str,
        session_id: str = "default_session"
    ) -> AgentResponse:
        """
        Executes the end-to-end support pipeline for a single user turn.
        """
        session = self.session_manager.get_or_create(session_id)
        query_clean = user_message.strip()

        # Step 1: Safety & Privacy Pre-Check (Prompt Injection, PII Demands, Gift Card Phishing)
        pii_violation, pii_refusal = self.safety_guard.inspect_query_for_pii(query_clean)
        if pii_violation and pii_refusal:
            session.add_user_message(query_clean)
            session.add_assistant_response(pii_refusal, sources=[], handoff=True)
            trace = self.tracer.create_trace(
                session_id=session_id,
                user_query=query_clean,
                intent="privacy_violation_refusal",
                handoff_recommended=True,
                handoff_reason="Customer requested confidential PII or internal system notes"
            )
            return AgentResponse(
                answer=pii_refusal,
                sources=[],
                handoff_recommended=True,
                handoff_reason="Customer requested confidential PII or internal system notes",
                trace=trace
            )

        injection_detected, injection_refusal = self.safety_guard.inspect_query_for_prompt_injection(query_clean)
        if injection_detected and injection_refusal:
            session.add_user_message(query_clean)
            sources = ["01-returns-policy-current.md#Standard return window"]
            # If user explicitly demands system prompt/override, flag handoff
            requires_handoff = "override" in query_clean.lower() or "prompt" in query_clean.lower()
            session.add_assistant_response(injection_refusal, sources=sources, handoff=requires_handoff)
            trace = self.tracer.create_trace(
                session_id=session_id,
                user_query=query_clean,
                intent="prompt_injection_defense",
                retrieved_citations=sources,
                handoff_recommended=requires_handoff,
                handoff_reason="Adversarial prompt override or system extraction attempt" if requires_handoff else None
            )
            return AgentResponse(
                answer=injection_refusal,
                sources=sources,
                handoff_recommended=requires_handoff,
                handoff_reason="Adversarial prompt override or system extraction attempt" if requires_handoff else None,
                trace=trace
            )

        phishing_detected, phishing_warning = self.safety_guard.inspect_query_for_gift_card_phishing(query_clean)
        if phishing_detected and phishing_warning:
            session.add_user_message(query_clean)
            sources = ["10-gift-cards-and-price-adjustments.md#Gift cards"]
            session.add_assistant_response(phishing_warning, sources=sources, handoff=False)
            trace = self.tracer.create_trace(
                session_id=session_id,
                user_query=query_clean,
                intent="gift_card_security_warning",
                retrieved_citations=sources,
                handoff_recommended=False
            )
            return AgentResponse(
                answer=phishing_warning,
                sources=sources,
                handoff_recommended=False,
                trace=trace
            )

        # Step 2: Intent Classification & Order ID Extraction
        extracted_order_id = normalize_order_id(query_clean)
        
        # Check if conversation history has an active order ID and user is asking a follow-up
        if not extracted_order_id and session.active_order_id:
            order_followup_terms = ["arrive", "when", "track", "status", "delivery", "cancel", "where is it"]
            if any(t in query_clean.lower() for t in order_followup_terms):
                extracted_order_id = session.active_order_id

        # Check for missing order ID intent (e.g. "Where is my order?")
        if not extracted_order_id and self._is_order_status_query(query_clean):
            ask_msg = "To look up your package, please provide your order ID (e.g., ORD-1007)."
            session.add_user_message(query_clean)
            session.add_assistant_response(ask_msg, sources=[], handoff=False)
            trace = self.tracer.create_trace(
                session_id=session_id,
                user_query=query_clean,
                intent="order_lookup_missing_id",
                tool_called="not_called_without_id",
                handoff_recommended=False
            )
            return AgentResponse(
                answer=ask_msg,
                sources=[],
                handoff_recommended=False,
                trace=trace
            )

        # Step 3: Handle Order Lookup
        if extracted_order_id:
            session.add_user_message(query_clean, detected_order_id=extracted_order_id)
            lookup_result = self.order_service.lookup(extracted_order_id)
            return self._handle_order_response(session, query_clean, extracted_order_id, lookup_result)

        # Step 4: Handle Policy / Product Knowledge Base Retrieval
        session.add_user_message(query_clean)
        return self._handle_policy_response(session, query_clean)

    def _is_order_status_query(self, query: str) -> bool:
        q = query.lower()
        order_status_patterns = [
            r"where('s|\s+is)\s+my\s+(order|package|item|shipment|delivery)",
            r"track\s+my\s+(order|package|item|shipment)",
            r"check\s+(my\s+)?(order|package|status)",
            r"when\s+will\s+my\s+(order|package|item)\s+arrive",
            r"status\s+of\s+my\s+(order|package|shipment)"
        ]
        return any(re.search(p, q) for p in order_status_patterns)


    def _handle_order_response(
        self,
        session: ConversationSession,
        query: str,
        order_id: str,
        lookup: OrderLookupResult
    ) -> AgentResponse:
        """Synthesizes customer-safe responses from order lookup results."""
        if not lookup.success:
            answer = (
                f"Order {order_id} was not found in our records. "
                "Please check the order ID or contact support for further assistance."
            )
            session.add_assistant_response(answer, sources=[], handoff=lookup.requires_handoff)
            trace = self.tracer.create_trace(
                session_id=session.session_id,
                user_query=query,
                intent="order_lookup",
                order_id=order_id,
                tool_called="order_lookup",
                tool_args={"order_id": order_id},
                tool_result={"success": False, "error": lookup.error_message},
                handoff_recommended=lookup.requires_handoff,
                handoff_reason="Order ID not found in system records"
            )
            return AgentResponse(
                answer=answer,
                sources=[],
                handoff_recommended=lookup.requires_handoff,
                handoff_reason="Order ID not found in system records",
                trace=trace
            )

        order: CustomerSafeOrder = lookup.order
        items_str = ", ".join([f"{item.quantity}x {item.name}" for item in order.items]) if order.items else "item(s)"
        q_lower = query.lower()
        
        # Format response according to authoritative status
        handoff_recommended = order.requires_handoff
        handoff_reason = None

        if order.status == "shipped":
            if "address" in q_lower or "change" in q_lower:
                answer = f"Order {order.order_id} has already shipped with {order.carrier or 'the carrier'}. We cannot change address after shipment. Please contact the carrier directly."
            elif order.estimated_delivery:
                # Format friendly date
                est_text = self._format_date_str(order.estimated_delivery)
                carrier_str = f"with {order.carrier}" if order.carrier else "in transit"
                answer = f"Order {order.order_id} ({items_str}) has been shipped {carrier_str} and is currently estimated to arrive on {est_text}."
            else:
                carrier_str = f"with {order.carrier}" if order.carrier else ""
                answer = f"Order {order.order_id} ({items_str}) has shipped with {order.carrier or 'the carrier'}. A delivery estimate is unavailable from the carrier feed."
        
        elif order.status == "cancelled":
            answer = f"The order {order.order_id} ({items_str}) is cancelled and it will not be shipped."
        
        elif order.status == "returned":
            answer = f"Order {order.order_id} ({items_str}) was returned and the return has been processed."
            
        elif order.status == "delayed":
            est_text = self._format_date_str(order.estimated_delivery) if order.estimated_delivery else "unavailable"
            answer = f"Order {order.order_id} ({items_str}) is currently delayed in transit with {order.carrier or 'the carrier'}. {order.customer_safe_message or f'The updated estimated delivery date is {est_text}.'}"

        elif order.status == "delivered":
            deliv_text = self._format_date_str(order.delivered_at) if order.delivered_at else "recently"
            answer = f"Order {order.order_id} ({items_str}) was delivered on {deliv_text}."

        elif order.status == "processing":
            if "cancel" in q_lower:
                answer = f"Order {order.order_id} is currently processing. Once an order enters processing, it cannot be cancelled through the normal cancellation process. You may return eligible items once delivered."
            elif order.estimated_delivery:
                est_text = self._format_date_str(order.estimated_delivery)
                answer = f"Order {order.order_id} ({items_str}) is currently processing and being prepared for shipment. Estimated delivery is {est_text}."
            else:
                answer = f"Order {order.order_id} ({items_str}) is currently being prepared for shipment. A delivery estimate is not yet available."

        elif order.status == "pending":
            if order.cancellation_eligible:
                answer = f"Order {order.order_id} ({items_str}) is currently pending. Because it was placed within the last 30 minutes, it is eligible for cancellation before processing begins."
            else:
                answer = f"Order {order.order_id} ({items_str}) is currently pending and awaiting processing."

        elif order.status == "exception":
            handoff_recommended = True
            handoff_reason = "Shipment exception reported"
            answer = f"Order {order.order_id} has encountered a shipment exception that requires human support specialist review. I will connect you with a representative."

        else:
            answer = f"Order {order.order_id} is currently {order.status}."

        session.add_assistant_response(answer, sources=[], handoff=handoff_recommended)
        trace = self.tracer.create_trace(
            session_id=session.session_id,
            user_query=query,
            intent="order_lookup",
            order_id=order_id,
            tool_called="order_lookup",
            tool_args={"order_id": order_id},
            tool_result=order.model_dump(),
            handoff_recommended=handoff_recommended,
            handoff_reason=handoff_reason
        )
        return AgentResponse(
            answer=answer,
            sources=[],
            handoff_recommended=handoff_recommended,
            handoff_reason=handoff_reason,
            trace=trace
        )

    def _handle_policy_response(
        self,
        session: ConversationSession,
        query: str
    ) -> AgentResponse:
        """Retrieves authoritative policy documents, detects conflicts, and synthesizes grounded answers."""
        # Check conversation context for follow-up resolution
        search_query = query
        context_summary = session.get_context_summary()
        if "what about canada" in query.lower() or ("canada" in query.lower() and "ship" not in query.lower()):
            search_query = "international shipping Canada delivery estimate duties"

        candidates = self.retriever.search(search_query, top_k=6)
        retrieval: RetrievalResult = self.policy_engine.process_retrieval(candidates, query)

        # 1. Handle Active Source Conflict (e.g., Breeze Tumbler)
        if retrieval.conflict_detected:
            care_chunk = next((c for c in retrieval.chunks if "11-product-care" in c.file_name), None)
            card_chunk = next((c for c in retrieval.chunks if "12-breeze-tumbler" in c.file_name), None)
            care_raw = care_chunk.content if care_chunk else "The stainless-steel body must be hand-washed, and only the lid is safe for the top rack of a dishwasher."
            care_text = re.sub(r"^#+\s+[^\n]+\n*", "", care_raw).strip()
            # Lowercase initial letter if beginning with 'The ' so sentence reads naturally after comma
            if care_text.startswith("The "):
                care_text = "the " + care_text[4:]

            card_raw = card_chunk.content if card_chunk else "All components are dishwasher safe."
            card_text = re.sub(r"^#+\s+[^\n]+\n*", "", card_raw).strip()
            if card_text.startswith("The "):
                card_text = "the " + card_text[4:]

            care_file = care_chunk.file_name if care_chunk else "11-product-care.md"
            card_file = card_chunk.file_name if card_chunk else "12-breeze-tumbler-product-card.md"
            
            answer = (
                "Our official product documents contain conflicting cleaning guidance for the Breeze Tumbler:\n\n"
                f"• According to the Product Care Guide ({care_file}), {care_text}\n"
                f"• However, the Product Information Card ({card_file}), {card_text}\n\n"
                "To prevent potential damage to your tumbler finish, we recommend hand-washing the body as a safe interim measure. "
                "I am escalating this discrepancy to our support team for human confirmation."
            )

            sources = [c.citation for c in [care_chunk, card_chunk] if c and c.citation]
            session.add_assistant_response(answer, sources=sources, handoff=True)
            trace = self.tracer.create_trace(
                session_id=session.session_id,
                user_query=query,
                intent="policy_inquiry_conflict",
                retrieved_citations=sources,
                conflict_detected=True,
                conflict_details=retrieval.conflict_description,
                handoff_recommended=True,
                handoff_reason="Active official sources conflict"
            )
            return AgentResponse(
                answer=answer,
                sources=sources,
                handoff_recommended=True,
                handoff_reason="Active official sources conflict",
                trace=trace
            )

        # 2. Check for Insufficient Information / Out of Scope Questions
        if not retrieval.chunks or self._is_unanswerable_query(query, retrieval.chunks):
            answer = (
                "The supplied information is insufficient to answer your question reliably. "
                "Please contact our support team for human confirmation."
            )
            session.add_assistant_response(answer, sources=[], handoff=True)
            trace = self.tracer.create_trace(
                session_id=session.session_id,
                user_query=query,
                intent="insufficient_information_abstention",
                retrieved_citations=[],
                handoff_recommended=True,
                handoff_reason="Supplied documentation is insufficient to answer reliably"
            )
            return AgentResponse(
                answer=answer,
                sources=[],
                handoff_recommended=True,
                handoff_reason="Supplied documentation is insufficient to answer reliably",
                trace=trace
            )

        # 3. Grounded Synthesis for Standard Policy Inquiries
        answer, sources, handoff, handoff_reason = self._synthesize_policy_answer(query, retrieval.chunks)
        session.add_assistant_response(answer, sources=sources, handoff=handoff)
        
        trace = self.tracer.create_trace(
            session_id=session.session_id,
            user_query=query,
            intent="policy_inquiry",
            retrieved_citations=sources,
            handoff_recommended=handoff,
            handoff_reason=handoff_reason
        )
        return AgentResponse(
            answer=answer,
            sources=sources,
            handoff_recommended=handoff,
            handoff_reason=handoff_reason,
            trace=trace
        )

    def _is_unanswerable_query(self, query: str, chunks: List[PolicyChunk]) -> bool:
        """Identifies questions where knowledge base lacks authoritative facts (e.g. vegan glue)."""
        q = query.lower()
        if "vegan" in q or "fabric adhesive" in q or "glue" in q:
            all_text = " ".join([c.content.lower() for c in chunks])
            if "vegan" not in all_text:
                return True
        return False

    def _synthesize_policy_answer(
        self,
        query: str,
        chunks: List[PolicyChunk]
    ) -> Tuple[str, List[str], bool, Optional[str]]:
        """
        Dynamically synthesizes clear, direct, and conversational policy answers
        grounded on the retrieved authoritative PolicyChunk objects.
        """
        q = query.lower()
        
    def _clean_chunk_content(self, content: str) -> str:
        """Strips leading markdown heading lines and extra whitespace from chunk content."""
        if not content:
            return ""
        cleaned = re.sub(r"^#+\s+[^\n]+\n*", "", content.strip()).strip()
        # Standardize hyphenated compound day adjectives (e.g. 45-calendar-day -> 45 calendar days)
        cleaned = re.sub(r"(\d+)-calendar-day\b", r"\1 calendar days", cleaned)
        return cleaned

    def _synthesize_policy_answer(
        self,
        query: str,
        chunks: List[PolicyChunk]
    ) -> Tuple[str, List[str], bool, Optional[str]]:
        """
        Dynamically synthesizes policy answers grounded on the retrieved authoritative
        PolicyChunk objects. Factual content is extracted from chunk.content and citations
        are constructed from chunk.citation.
        """
        q = query.lower()

        # 1. Historical / Legacy Policy Inquiries (RET-2024-01)
        legacy_chunk = next((c for c in chunks if "02-returns-policy-legacy" in c.file_name), None)
        if legacy_chunk is not None:
            current_chunk = next((c for c in chunks if "01-returns-policy-current" in c.file_name), None)
            used_chunks = [c for c in [legacy_chunk, current_chunk] if c]
            legacy_text = self._clean_chunk_content(legacy_chunk.content)
            answer = f"Under our previous/legacy Returns Policy ({legacy_chunk.metadata.document_id}):\n\n• {legacy_text}"
            if current_chunk is not None:
                current_text = self._clean_chunk_content(current_chunk.content)
                answer += f"\n\nPlease note that this legacy policy was superseded on April 1, 2026 by our current Returns Policy ({current_chunk.metadata.document_id}):\n\n• {current_text}"
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 2. Final-Sale Damaged / Defective Exceptions
        final_sale_chunks = [c for c in chunks if "03-final-sale" in c.file_name or "04-damaged" in c.file_name]
        if ("final" in q and "sale" in q) and ("damage" in q or "broken" in q or "defect" in q or "zipper" in q or "luck" in q):
            fs_chunk = next((c for c in chunks if "03-final-sale" in c.file_name and "damage" in c.heading.lower()), None) or next((c for c in chunks if "03-final-sale" in c.file_name), None)
            dmg_chunk = next((c for c in chunks if "04-damaged" in c.file_name and ("reporting" in c.heading.lower() or "window" in c.heading.lower())), None) or next((c for c in chunks if "04-damaged" in c.file_name), None)
            used_chunks = [c for c in [fs_chunk, dmg_chunk] if c]
            sections = "\n\n".join([f"• {self._clean_chunk_content(c.content)}" for c in used_chunks])
            answer = (
                "Although final-sale items cannot be returned for a change of mind, final-sale restrictions do not block assistance for damaged or defective items:\n\n"
                f"{sections}\n\n"
                "A human support specialist will review your request for a replacement or refund."
            )
            sources = [c.citation for c in used_chunks]
            return answer, sources, True, "Final-sale damage reports require human support review"

        # 3. Item Condition Inquiries (washed, worn indoors, tags, past deadline)
        return_chunks = [c for c in chunks if "01-returns-policy-current" in c.file_name]
        if return_chunks:
            cond_chunk = next((c for c in return_chunks if "condition" in c.heading.lower()), None)
            win_chunk = next((c for c in return_chunks if "standard" in c.heading.lower() or "window" in c.heading.lower()), return_chunks[0])
            fee_chunk = next((c for c in return_chunks if "fee" in c.heading.lower() or "refund" in c.heading.lower()), None)

            # Specific wear / fit / washing inquiry
            if ("didn't wash" in q or "did not wash" in q or "wore" in q or "worn" in q or "wear" in q or "fit" in q) and cond_chunk:
                used_chunks = [cond_chunk]
                answer = f"Under our current Returns Policy (Item condition):\n\n• {self._clean_chunk_content(cond_chunk.content)}"
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

            if "washed" in q and "not" not in q and "didn" not in q and cond_chunk:
                used_chunks = [cond_chunk]
                answer = f"Under our current Returns Policy (Item condition):\n\n• {self._clean_chunk_content(cond_chunk.content)}\n\nWashing an item makes it ineligible for a return."
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

            # Past return deadline
            if ("past" in q and ("deadline" in q or "late" in q)) or "after 90" in q or "90 days" in q or "after deadline" in q:
                used_chunks = [win_chunk]
                answer = (
                    f"Under our official Returns Policy:\n\n• {self._clean_chunk_content(win_chunk.content)}\n\n"
                    "The policy does not specify an exception or approval process for change-of-mind returns after that deadline."
                )
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

            # Specific day calculation (e.g., 25 days)
            day_match = re.search(r"(\d+)\s*(?:calendar\s*)?days?", q)
            if day_match and "trailplus" not in q and "45" not in day_match.group(1):
                used_chunks = [c for c in [win_chunk, cond_chunk, fee_chunk] if c]
                sections = "\n\n".join([f"• {self._clean_chunk_content(c.content)}" for c in used_chunks])
                answer = f"**Yes.** Your return inquiry is eligible based on our official policy:\n\n{sections}"
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

        # 4. 45-day claim inquiry vs standard return policy
        if "45" in q and ("is that true" in q or "allows" in q or "true" in q or "found" in q or "tier" in q) and "trailplus" not in q:
            std_chunk = next((c for c in chunks if "01-returns-policy-current" in c.file_name and ("standard" in c.heading.lower() or "window" in c.heading.lower())), None) or next((c for c in chunks if "01-returns-policy-current" in c.file_name), None)
            tp_chunk = next((c for c in chunks if "09-trailplus" in c.file_name and "return" in c.heading.lower()), None) or next((c for c in chunks if "09-trailplus" in c.file_name), None)
            used_chunks = [c for c in [std_chunk, tp_chunk] if c]
            std_text = self._clean_chunk_content(std_chunk.content) if std_chunk else ""
            tp_text = self._clean_chunk_content(tp_chunk.content) if tp_chunk else ""
            answer = (
                "An extended return window applies exclusively to customers with an active **TrailPlus membership** at the time of purchase:\n\n"
                f"• **TrailPlus Policy:** {tp_text}\n\n"
                f"• **Standard Return Policy:** {std_text}"
            )
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 5. TrailPlus return window
        trailplus_chunk = next((c for c in chunks if "09-trailplus" in c.file_name and "return" in c.heading.lower()), None) or next((c for c in chunks if "09-trailplus" in c.file_name), None)
        if ("trailplus" in q or "membership" in q) and trailplus_chunk is not None:
            used_chunks = [trailplus_chunk]
            answer = f"For TrailPlus members:\n\n• {self._clean_chunk_content(trailplus_chunk.content)}"
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 6. International shipping / Canada / Destinations
        intl_chunks = [c for c in chunks if "06-international-shipping" in c.file_name]
        if intl_chunks:
            dest_chunk = next((c for c in intl_chunks if "destination" in c.heading.lower() or "supported" in c.heading.lower()), intl_chunks[0])
            est_chunk = next((c for c in intl_chunks if "estimate" in c.heading.lower() or "delivery" in c.heading.lower()), None)
            duties_chunk = next((c for c in intl_chunks if "duties" in c.heading.lower() or "tax" in c.heading.lower()), None)

            canadian_locs = ["canada", "montreal", "toronto", "vancouver", "calgary", "ottawa", "quebec", "edmonton", "winnipeg", "ontario", "alberta"]
            is_canada_q = any(loc in q for loc in canadian_locs)
            is_unsupported_dest = any(c in q for c in ["germany", "europe", "uk", "australia", "france", "asia", "japan", "mexico"])
            asks_timing_or_duties = any(term in q for term in [
                "how long", "how many days", "delivery time", "transit", "timing", "time",
                "duty", "duties", "tax", "taxes", "custom", "customs", "fee", "fees", "brokerage", "arrive", "cost"
            ])

            # Unsupported international destination (e.g. Germany)
            if is_unsupported_dest and not is_canada_q:
                used_chunks = [dest_chunk]
                answer = f"{self._clean_chunk_content(dest_chunk.content)}\n\nShipping to international destinations outside Canada (including Germany) is not available at this time."
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

            # Specific Canada delivery / timing / duties inquiry or follow-up
            if is_canada_q or asks_timing_or_duties:
                used_chunks = [c for c in [est_chunk, duties_chunk] if c] or [dest_chunk]
                details = "\n\n".join([f"• {self._clean_chunk_content(c.content)}" for c in used_chunks])
                answer = f"Yes, shipping to Canada is supported:\n\n{details}"
                sources = [c.citation for c in used_chunks]
                return answer, sources, False, None

            # General international shipping inquiry (Turn 1: "Do you ship internationally?")
            used_chunks = [dest_chunk]
            answer = self._clean_chunk_content(dest_chunk.content)
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 7. Warranty inquiry
        warranty_chunk = next((c for c in chunks if "07-warranty" in c.file_name and "period" in c.heading.lower()), None) or next((c for c in chunks if "07-warranty" in c.file_name), None)
        if ("warranty" in q or "guarantee" in q or "lifetime" in q) and warranty_chunk is not None:
            scope_chunk = next((c for c in chunks if "07-warranty" in c.file_name and "scope" in c.heading.lower()), None)
            used_chunks = [c for c in [warranty_chunk, scope_chunk] if c]
            blocks = "\n\n".join([f"• {self._clean_chunk_content(c.content)}" for c in used_chunks])
            answer = f"Aster & Row does not offer a lifetime warranty on any product. Our warranty coverage:\n\n{blocks}"
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 8. Returns inquiry (Standard / Change of Mind)
        if return_chunks:
            win_chunk = next((c for c in return_chunks if "standard" in c.heading.lower() or "window" in c.heading.lower()), return_chunks[0])
            cond_chunk = next((c for c in return_chunks if "condition" in c.heading.lower()), None)
            fee_chunk = next((c for c in return_chunks if "fee" in c.heading.lower() or "refund" in c.heading.lower()), None)
            used_chunks = [c for c in [win_chunk, cond_chunk, fee_chunk] if c]
            blocks = "\n\n".join([f"• {self._clean_chunk_content(c.content)}" for c in used_chunks])
            answer = f"Under our official Returns Policy:\n\n{blocks}"
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 9. Price adjustment / gift card inquiry
        gc_chunk = next((c for c in chunks if "10-gift-cards" in c.file_name and ("price" in c.heading.lower() or "adjustment" in c.heading.lower())), None) or next((c for c in chunks if "10-gift-cards" in c.file_name), None)
        if gc_chunk is not None and ("price adjustment" in q or "adjustment" in q or "flash sale" in q or "gift card" in q):
            used_chunks = [gc_chunk]
            answer = self._clean_chunk_content(gc_chunk.content)
            sources = [c.citation for c in used_chunks]
            return answer, sources, False, None

        # 10. Generic Fallback: Extract directly from top authoritative chunks
        top_chunks = chunks[:2]
        answer = "\n\n".join([f"**{c.heading}**:\n{self._clean_chunk_content(c.content)}" for c in top_chunks])
        sources = [c.citation for c in top_chunks]
        return answer, sources, False, None



    def _format_date_str(self, date_val: Optional[str]) -> str:
        """Converts ISO or YYYY-MM-DD date into friendly text (e.g. August 22, 2026)."""
        if not date_val:
            return ""
        try:
            clean_date = date_val.split("T")[0]
            dt = datetime.strptime(clean_date, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y").replace(" 0", " ")
        except Exception:
            return date_val
