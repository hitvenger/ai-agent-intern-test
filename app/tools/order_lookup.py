from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

from app.models import CustomerSafeOrder, OrderItem, OrderLookupResult


ORDER_ID_PATTERN = re.compile(r"\b(ORD-\d{4})\b", re.IGNORECASE)
DEFAULT_SNAPSHOT_TIME = "2026-08-15T12:00:00Z"


def normalize_order_id(input_str: str) -> Optional[str]:
    """
    Extracts and normalizes order IDs from user input (e.g., ' ord-1007 ' -> 'ORD-1007').
    Returns normalized uppercase ID or None if no valid ID pattern matches.
    """
    if not input_str:
        return None
    cleaned = input_str.strip()
    match = ORDER_ID_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    return None


class OrderLookupService:
    """
    Deterministic Order Lookup Service.
    Enforces privacy air-gapping, status precedence, stale-field neutralization,
    and cancellation eligibility calculations.
    """

    def __init__(self, data_path: Union[str, Path] = "data/orders.json"):
        self.data_path = Path(data_path)
        self.snapshot_at_str = DEFAULT_SNAPSHOT_TIME
        self.snapshot_at = datetime.fromisoformat(self.snapshot_at_str.replace("Z", "+00:00"))
        self.orders: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Orders dataset not found at {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.snapshot_at_str = data.get("snapshot_at", DEFAULT_SNAPSHOT_TIME)
            self.snapshot_at = datetime.fromisoformat(self.snapshot_at_str.replace("Z", "+00:00"))
            
            for ord_dict in data.get("orders", []):
                oid = ord_dict.get("order_id", "").upper()
                if oid:
                    self.orders[oid] = ord_dict

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        """
        Looks up an order by ID and returns a strictly sanitized CustomerSafeOrder.
        Guarantees that raw internal notes and customer PII are never returned.
        """
        normalized_id = normalize_order_id(raw_order_id)
        if not normalized_id:
            return OrderLookupResult(
                success=False,
                order_id=raw_order_id.strip() if raw_order_id else "",
                error_message="Invalid order ID format. Expected format: ORD-XXXX (e.g. ORD-1007).",
                requires_handoff=False
            )

        raw_order = self.orders.get(normalized_id)
        if not raw_order:
            return OrderLookupResult(
                success=False,
                order_id=normalized_id,
                error_message=f"Order {normalized_id} was not found in our records.",
                requires_handoff=True
            )

        # Build sanitized customer-safe order
        status = raw_order.get("status", "").lower()
        placed_at_str = raw_order.get("placed_at", "")
        
        # Check cancellation eligibility based on 30-min window and pending status
        cancellation_eligible = False
        if status == "pending" and placed_at_str:
            try:
                placed_dt = datetime.fromisoformat(placed_at_str.replace("Z", "+00:00"))
                delta_minutes = (self.snapshot_at - placed_dt).total_seconds() / 60.0
                if 0 <= delta_minutes <= 30:
                    cancellation_eligible = True
            except Exception:
                cancellation_eligible = False

        # Status precedence & stale delivery field neutralization
        stale_cleared = False
        carrier = raw_order.get("carrier")
        tracking_number = raw_order.get("tracking_number")
        estimated_delivery = raw_order.get("estimated_delivery")
        
        if status in ("cancelled", "returned"):
            # Per data dictionary, cancelled/returned orders retain stale carrier/ETA in operational logs.
            # Neutralize them so the model cannot hallucinate an active arrival date.
            carrier = None
            tracking_number = None
            estimated_delivery = None
            stale_cleared = True

        requires_handoff = (status == "exception")

        # Map items safely
        items: list[OrderItem] = []
        for item in raw_order.get("items", []):
            items.append(OrderItem(
                sku=item.get("sku"),
                name=item.get("name", "Unknown Item"),
                quantity=item.get("quantity", 1),
                final_sale=bool(item.get("final_sale", False))
            ))

        safe_order = CustomerSafeOrder(
            order_id=normalized_id,
            membership_tier=raw_order.get("membership_tier", "standard"),
            items=items,
            placed_at=placed_at_str,
            status=status,
            status_updated_at=raw_order.get("status_updated_at", placed_at_str),
            shipped_at=raw_order.get("shipped_at"),
            delivered_at=raw_order.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=raw_order.get("customer_safe_message"),
            requires_handoff=requires_handoff,
            cancellation_eligible=cancellation_eligible,
            stale_fields_cleared=stale_cleared,
            sanitized_note=None
        )

        return OrderLookupResult(
            success=True,
            order_id=normalized_id,
            order=safe_order,
            requires_handoff=requires_handoff
        )
