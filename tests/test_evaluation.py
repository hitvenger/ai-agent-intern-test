import json
from pathlib import Path
import pytest

from app.agent import SupportAgent
from evaluation.run_eval import EvaluationHarness


@pytest.fixture(scope="module")
def agent():
    return SupportAgent()


@pytest.fixture(scope="module")
def harness(agent):
    return EvaluationHarness(agent)


def load_cases(file_path: str):
    p = Path(file_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("cases", [])


@pytest.mark.parametrize("case", load_cases("evaluation/visible-cases.json"), ids=lambda c: c["id"])
def test_visible_case(harness, case):
    passed, failures = harness.run_case(case)
    assert passed, f"Visible case '{case['id']}' failed: {'; '.join(failures)}"


@pytest.mark.parametrize("case", load_cases("evaluation/custom-cases.json"), ids=lambda c: c["id"])
def test_custom_case(harness, case):
    passed, failures = harness.run_case(case)
    assert passed, f"Custom case '{case['id']}' failed: {'; '.join(failures)}"
