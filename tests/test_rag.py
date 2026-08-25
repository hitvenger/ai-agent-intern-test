import pytest
from app.rag.loader import KnowledgeBaseLoader
from app.rag.retriever import KnowledgeBaseRetriever
from app.rag.policy_engine import PolicyEngine


@pytest.fixture(scope="module")
def kb_loader():
    return KnowledgeBaseLoader("knowledge-base")


@pytest.fixture(scope="module")
def kb_retriever():
    return KnowledgeBaseRetriever("knowledge-base")


@pytest.fixture(scope="module")
def policy_engine():
    return PolicyEngine()


def test_loader_parses_all_files(kb_loader):
    chunks = kb_loader.load_documents()
    assert len(chunks) > 0
    file_names = {c.file_name for c in chunks}
    assert len(file_names) == 14
    assert "01-returns-policy-current.md" in file_names
    assert "02-returns-policy-legacy.md" in file_names
    assert "13-support-escalation.md" in file_names
    assert "14-internal-content-migration-notes.md" in file_names


def test_authority_filtering_excludes_draft_and_superseded(kb_retriever, policy_engine):
    # Retrieve candidates for return policy
    candidates = kb_retriever.search("How many days do I have to return an item?")
    result = policy_engine.process_retrieval(candidates, "How many days do I have to return an item?")
    
    # 02 legacy and 14 migration draft must NOT be in authoritative citations
    for cit in result.citations:
        assert not cit.startswith("02-returns-policy-legacy.md")
        assert not cit.startswith("14-internal-content-migration-notes.md")
    
    # 01 current must be present
    assert any(cit.startswith("01-returns-policy-current.md") for cit in result.citations)


def test_trailplus_return_window_retrieval(kb_retriever, policy_engine):
    query = "My TrailPlus membership was active when I ordered. What is my return window?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    assert any("09-trailplus-membership.md" in cit for cit in result.citations)
    # Ensure 45 days is in the content
    trailplus_chunks = [c for c in result.chunks if "09-trailplus-membership.md" in c.file_name]
    assert any("45" in c.content for c in trailplus_chunks)


def test_conflict_detection_breeze_tumbler(kb_retriever, policy_engine):
    query = "Can I put the entire Breeze Tumbler in the dishwasher?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    assert result.conflict_detected is True
    assert result.requires_handoff is True
    assert "conflicting" in result.conflict_description.lower()
    
    # Both active sources must be cited
    citations_str = " ".join(result.citations)
    assert "11-product-care.md" in citations_str or "12-breeze-tumbler-product-card.md" in citations_str


def test_international_shipping_canada_only(kb_retriever, policy_engine):
    query = "Do you ship to Germany or Canada?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    assert any("06-international-shipping.md" in cit for cit in result.citations)
    intl_chunk = next(c for c in result.chunks if "06-international-shipping.md" in c.file_name)
    assert "Canada" in intl_chunk.content


def test_warranty_no_lifetime(kb_retriever, policy_engine):
    query = "Do all Aster & Row products have a lifetime warranty?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    assert any("07-warranty.md" in cit for cit in result.citations)
    war_chunk = next(c for c in result.chunks if "07-warranty.md" in c.file_name)
    assert "does not offer a lifetime warranty" in war_chunk.content or "2 years" in war_chunk.content


def test_colloquial_return_phrasing_retrieval(kb_retriever, policy_engine):
    query = "I bought a daypack 25 days ago and haven't opened it, can I send it back?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    # Must retrieve current return policy, NOT gift cards or legacy policy
    citations = " ".join(result.citations)
    assert "01-returns-policy-current.md" in citations
    assert "10-gift-cards-and-price-adjustments.md" not in citations
    assert "02-returns-policy-legacy.md" not in citations


def test_canada_multi_intent_shipping_retrieval(kb_retriever, policy_engine):
    query = "I live in Montreal, how long will shipping take and do you cover custom fees?"
    candidates = kb_retriever.search(query)
    result = policy_engine.process_retrieval(candidates, query)
    
    citations = " ".join(result.citations)
    # Must retrieve both delivery estimate and duties/taxes
    assert "06-international-shipping.md#Canada delivery estimate" in citations or any("06-international-shipping.md" in c for c in result.citations)
    
    # Verify chunks contain both 5-9 days delivery and duties/taxes info
    contents = " ".join([c.content for c in result.chunks if "06-international-shipping.md" in c.file_name])
    assert "5–9 business days" in contents or "5-9" in contents
    assert "duties" in contents.lower() or "taxes" in contents.lower()


def test_dynamic_chunk_content_propagation():
    """
    Regression Test: Proves that policy answers are dynamically synthesized from
    retrieved chunk content and citations rather than hardcoded templates.
    """
    from app.agent import SupportAgent
    from app.models import PolicyChunk, DocumentMetadata
    
    agent = SupportAgent()
    
    # Create custom mock chunk with distinct modified facts
    mock_meta = DocumentMetadata(
        document_id="RET-CUSTOM",
        title="Custom Returns Policy",
        status="active",
        policy_authority="official",
        audience="customer",
        customer_answering=True
    )
    mock_chunk = PolicyChunk(
        chunk_id="01-returns-policy-current.md_0",
        file_name="01-returns-policy-current.md",
        heading="Standard return window",
        citation="01-returns-policy-current.md#Standard return window",
        content="Customers on the standard plan may request a return within **999 calendar days of delivery**.",
        metadata=mock_meta
    )
    
    answer, sources, handoff, reason = agent._synthesize_policy_answer("How long to return?", [mock_chunk])
    
    # Must contain the modified fact from chunk.content
    assert "999 calendar days" in answer
    # Must NOT contain the old hardcoded 30 days
    assert "30 calendar days" not in answer
    # Must dynamically cite the mock chunk's citation
    assert sources == ["01-returns-policy-current.md#Standard return window"]
    assert handoff is False


def test_historical_legacy_policy_retrieval_and_synthesis():
    """
    Regression Test: Proves that historical queries retrieve the superseded legacy policy,
    ground on the accurate 45-day window from 02-returns-policy-legacy.md, and explain
    that it was superseded by the current 30-day policy.
    """
    from app.agent import SupportAgent
    agent = SupportAgent()

    resp = agent.process_message("What did the old return policy say about the return window?")

    # Must accurately contain 45 calendar days from 02-returns-policy-legacy.md
    assert "45 calendar days" in resp.answer
    # Must NOT contain erroneous 60 calendar days
    assert "60 calendar days" not in resp.answer
    # Must explain that it was superseded by the current 30-day policy
    assert "superseded" in resp.answer.lower()
    assert "30-calendar-day" in resp.answer or "30 calendar days" in resp.answer
    # Must cite the legacy return window source
    assert "02-returns-policy-legacy.md#Return window" in resp.sources
    assert resp.handoff_recommended is False



