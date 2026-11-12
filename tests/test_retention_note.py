"""Tests for the LLM retention note layer (API mocked, no key needed)."""

from types import SimpleNamespace

from src.llm.retention_note import SYSTEM_PROMPT, RetentionNoteGenerator, build_prompt

PREDICTION = {
    "churn_probability": 0.8716,
    "will_churn": True,
    "reasons": [
        {
            "feature": "tenure",
            "value": 1,
            "impact": 0.96,
            "effect": "increases churn risk",
        },
        {
            "feature": "Contract",
            "value": "Month-to-month",
            "impact": 0.5,
            "effect": "increases churn risk",
        },
        {
            "feature": "InternetService",
            "value": "Fiber optic",
            "impact": 0.46,
            "effect": "increases churn risk",
        },
    ],
}


class FakeMessages:
    """Stands in for client.messages; records the call and returns a canned reply."""

    def __init__(self, text="Test note.", error=None):
        self.text, self.error, self.kwargs = text, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


def fake_client(messages):
    return SimpleNamespace(messages=messages)


def test_build_prompt_contains_facts():
    prompt = build_prompt(PREDICTION)
    assert "87%" in prompt
    assert "1 months" in prompt
    assert "Month-to-month" in prompt
    assert "Fiber optic" in prompt


def test_generate_returns_note_and_sends_prompt():
    messages = FakeMessages(text="  Offer a 12-month contract.  ")
    note = RetentionNoteGenerator(client=fake_client(messages)).generate(PREDICTION)

    assert note == "Offer a 12-month contract."
    assert messages.kwargs["system"] == SYSTEM_PROMPT
    assert "87%" in messages.kwargs["messages"][0]["content"]


def test_generate_returns_none_on_api_failure():
    messages = FakeMessages(error=RuntimeError("API down"))
    note = RetentionNoteGenerator(client=fake_client(messages)).generate(PREDICTION)
    assert note is None
