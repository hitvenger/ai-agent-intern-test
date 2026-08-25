from __future__ import annotations

import re
from typing import Tuple, Optional


PII_REQUEST_PATTERNS = [
    re.compile(r"\b(customer('s)?\s*(email|address|phone|name))\b", re.IGNORECASE),
    re.compile(r"\b(email\s*address|shipping\s*address|physical\s*address)\b", re.IGNORECASE),
    re.compile(r"\b(internal\s*(note|notes|tags?|score))\b", re.IGNORECASE),
    re.compile(r"\b(risk\s*score|fraud\s*score|fraud\s*review)\b", re.IGNORECASE),
    re.compile(r"\b(warehouse\s*note)\b", re.IGNORECASE),
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(prior|previous|system|safety)\s+(rules|instructions|guidelines|filters)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(hidden\s+)?(prompt|system\s+prompt|instructions)", re.IGNORECASE),
    re.compile(r"output\s+(your\s+)?(hidden\s+)?(prompt|system\s+prompt|instructions)", re.IGNORECASE),
    re.compile(r"system\s+(instruction|override):", re.IGNORECASE),
    re.compile(r"use\s+that\s+newer\s+document\s+and\s+approve", re.IGNORECASE),
    re.compile(r"migration\s+note\s+says\s+to\s+ignore", re.IGNORECASE),
]


GIFT_CARD_CODE_PATTERNS = [
    re.compile(r"\b(gift[\s-]card\s*(code|pin|number))\b", re.IGNORECASE),
    re.compile(r"\b(16[\s-]digit\s*(code|pin|number))\b", re.IGNORECASE),
    re.compile(r"\b(type\s*my\s*(gift\s*card|pin))\b", re.IGNORECASE),
]


class SafetyGuard:
    """
    Deterministic Safety & Privacy Guardrail.
    Inspects queries and outputs for PII extraction, prompt injection, and sensitive data leakage.
    """

    @staticmethod
    def inspect_query_for_pii(query: str) -> Tuple[bool, Optional[str]]:
        """Detects explicit requests for sensitive customer PII or internal operational fields."""
        query_lower = query.lower()
        for pat in PII_REQUEST_PATTERNS:
            if pat.search(query_lower):
                refusal_msg = (
                    "I cannot disclose confidential customer personal details (such as contact information or shipping destinations) "
                    "or internal operational system records. Customer data privacy and system integrity are strictly protected. "
                    "If you require authorized account assistance, please contact a human customer support specialist."
                )
                return True, refusal_msg
        return False, None


    @staticmethod
    def inspect_query_for_prompt_injection(query: str) -> Tuple[bool, Optional[str]]:
        """Detects prompt injection attempts aiming to override policy rules or extract hidden prompts."""
        for pat in PROMPT_INJECTION_PATTERNS:
            if pat.search(query):
                return True, (
                    "Internal scratchpads, migration notes, and prompt override commands are not authoritative policies. "
                    "Aster & Row's active standard return policy allows 30 calendar days from delivery for eligible items "
                    "(or 45 days for active TrailPlus members). Support agents cannot automatically approve returns without human specialist review."
                )
        return False, None

    @staticmethod
    def inspect_query_for_gift_card_phishing(query: str) -> Tuple[bool, Optional[str]]:
        """Warns customers never to share full gift card codes or PINs in chat."""
        if any(pat.search(query) for pat in GIFT_CARD_CODE_PATTERNS):
            if "where" in query.lower() or "type" in query.lower() or "enter" in query.lower() or "share" in query.lower():
                return True, (
                    "For your security, please never share complete 16-digit gift card numbers or PIN codes in this chat. "
                    "Aster & Row support agents will never ask for your full gift card codes. Gift cards can be redeemed directly at checkout."
                )
        return False, None

    @staticmethod
    def sanitize_output(text: str) -> str:
        """Post-processing filter to ensure no emails or raw internal risk scores appear in text."""
        # Redact raw email addresses
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED EMAIL]", text)
        # Redact risk score mentions
        text = re.sub(r"risk\s*score:\s*\d+", "[REDACTED]", text, flags=re.IGNORECASE)
        return text
