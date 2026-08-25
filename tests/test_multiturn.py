import pytest
from app.agent import SupportAgent


@pytest.fixture(scope="module")
def agent():
    return SupportAgent()


def test_canada_multiturn_flow(agent):
    session_id = "multi_turn_canada"
    # Turn 1
    res1 = agent.process_message("Do you ship internationally?", session_id=session_id)
    assert "06-international-shipping.md" in " ".join(res1.sources)

    # Turn 2
    res2 = agent.process_message("What about Canada, and how long does it take?", session_id=session_id)
    ans2_lower = res2.answer.lower()
    assert "canada" in ans2_lower
    assert "5–9 business days" in ans2_lower or "5-9 business days" in ans2_lower or "5–9" in ans2_lower
    assert "duties" in ans2_lower or "taxes" in ans2_lower
    assert "06-international-shipping.md" in " ".join(res2.sources)


def test_order_id_context_preservation(agent):
    session_id = "multi_turn_order"
    # Turn 1
    res1 = agent.process_message("Where is ORD-1007?", session_id=session_id)
    assert "shipped" in res1.answer.lower()

    # Turn 2: Follow-up without explicit ID
    res2 = agent.process_message("When will it arrive?", session_id=session_id)
    assert "August 22, 2026" in res2.answer or "august 22" in res2.answer.lower()
    assert res2.trace.order_id_extracted == "ORD-1007"


def test_session_state_isolation(agent):
    session_a = "session_iso_a"
    session_b = "session_iso_b"

    # In Session A, ask about ORD-1007
    agent.process_message("Where is ORD-1007?", session_id=session_a)

    # In Session B, ask generic followup
    res_b = agent.process_message("Where is my package?", session_id=session_b)
    # Must ask for order ID, not bleed Session A
    assert "provide your order id" in res_b.answer.lower()


def test_paraphrased_colloquial_return_e2e(agent):
    res = agent.process_message("I bought a daypack 25 days ago and haven't opened it, can I send it back?", session_id="test_colloquial_ret")
    ans_lower = res.answer.lower()
    assert "30 calendar days" in ans_lower
    assert "01-returns-policy-current.md#Standard return window" in res.sources
    assert "10-gift-cards-and-price-adjustments.md" not in " ".join(res.sources)


def test_multi_intent_canada_shipping_e2e(agent):
    res = agent.process_message("I live in Montreal, how long will shipping take and do you cover custom fees?", session_id="test_montreal_multi")
    ans_lower = res.answer.lower()
    assert "5–9 business days" in ans_lower or "5-9 business days" in ans_lower or "5–9" in ans_lower
    assert "duties" in ans_lower or "taxes" in ans_lower
    assert "06-international-shipping.md#Canada delivery estimate" in res.sources
    assert "06-international-shipping.md#Duties and taxes" in res.sources

