from __future__ import annotations

import json
from pathlib import Path

from credhunter_x.classifiers.agentic import AgenticClassifier
from credhunter_x.gitleaks.parser import parse_gitleaks_report
from credhunter_x.llm.client import LLMToolResponse, ToolCall
from credhunter_x.masking.masker import build_raw_context, mask_context_window
from credhunter_x.models.classification import Label

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"
REPORT_PATH = FIXTURES_DIR / "gitleaks_report.json"

GITHUB_SECRET = "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"
SLACK_SECRET = "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"

FAKE_ANSWER_JSON = (
    '{"label": "true_secret", "confidence": 0.9, "severity": "high", '
    '"explanation": "looks real", "remediation": "rotate it"}'
)


class _ScriptedClient:
    """Returns each entry in `responses` in order, one per call to
    generate_with_tools. Records every call's messages/tools so tests can
    inspect exactly what the classifier sent on each turn."""

    def __init__(self, responses: list[LLMToolResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_with_tools(
        self,
        messages,
        tools,
        *,
        candidate_id: str = "",
        rule_id: str = "",
        raw_permit_candidate_id: str | None = None,
    ) -> LLMToolResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "candidate_id": candidate_id,
                "rule_id": rule_id,
                "raw_permit_candidate_id": raw_permit_candidate_id,
            }
        )
        return self._responses.pop(0)


def load_candidates():
    findings = json.loads(REPORT_PATH.read_text())
    return parse_gitleaks_report(findings, SAMPLE_REPO, repo_id="test-repo", context_lines=10)


def _final_answer(text: str = FAKE_ANSWER_JSON) -> LLMToolResponse:
    return LLMToolResponse(
        text=text, tool_calls=[], input_tokens=5, output_tokens=5, latency_ms=1.0
    )


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMToolResponse:
    return LLMToolResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        input_tokens=5,
        output_tokens=2,
        latency_ms=1.0,
    )


def test_answers_immediately_when_no_tool_call_is_needed():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient([_final_answer()])
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    result = classifier.classify(github, mask_context_window(github, []))

    assert len(client.calls) == 1
    assert result.label == Label.TRUE_SECRET
    assert result.arm == "agentic"
    assert result.turns == 1
    assert result.tool_calls == []


def test_calls_get_gitleaks_rule_then_answers():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient(
        [
            _tool_call_response("get_gitleaks_rule", {"rule_id": "github-pat"}),
            _final_answer(),
        ]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    result = classifier.classify(github, mask_context_window(github, []))

    assert len(client.calls) == 2
    assert result.turns == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_gitleaks_rule"
    assert result.tool_calls[0].arguments == {"rule_id": "github-pat"}

    # the second call's message history includes a "tool" role message with
    # the rule's real description
    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "GitHub" in tool_messages[0]["content"]


def test_search_file_result_is_masked_before_reaching_the_next_turn():
    """The classic Arm B leak risk: a tool call pulls content from
    elsewhere in the repo that contains a DIFFERENT candidate's real
    secret. search_file itself returns raw content -- masking must happen
    before it's appended to the conversation."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    slack = next(c for c in candidates if c.rule_id == "slack-bot-token")
    client = _ScriptedClient(
        [
            _tool_call_response(
                "search_file", {"file_path": "app/auth.py", "query": "SLACK_BOT_TOKEN"}
            ),
            _final_answer(),
        ]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    classifier.classify(github, build_raw_context(github, [slack]))

    second_call_messages = client.calls[1]["messages"]
    tool_message = next(m for m in second_call_messages if m["role"] == "tool")
    assert SLACK_SECRET not in tool_message["content"]
    assert "SLACK_BOT_TOKEN" in tool_message["content"]  # surrounding code is untouched


def test_check_file_exists_tool_is_dispatched_correctly():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient(
        [
            _tool_call_response("check_file_exists", {"file_path": "app/db.py"}),
            _final_answer(),
        ]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    classifier.classify(github, mask_context_window(github, []))

    tool_message = next(m for m in client.calls[1]["messages"] if m["role"] == "tool")
    assert "exists" in tool_message["content"]


def test_unknown_tool_name_is_handled_without_crashing():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient(
        [
            _tool_call_response("delete_repository", {}),
            _final_answer(),
        ]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    result = classifier.classify(github, mask_context_window(github, []))

    tool_message = next(m for m in client.calls[1]["messages"] if m["role"] == "tool")
    assert "Unknown tool" in tool_message["content"]
    assert result.label == Label.TRUE_SECRET  # the loop still recovers and finishes


def test_turn_cap_produces_uncertain_when_the_model_never_answers():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    # every one of the 4 allowed turns keeps calling a tool, even the last
    # (forced no-tools) turn -- an adversarial/misbehaving-model scenario
    client = _ScriptedClient(
        [_tool_call_response("get_gitleaks_rule", {"rule_id": "github-pat"}) for _ in range(4)]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    result = classifier.classify(github, mask_context_window(github, []))

    assert len(client.calls) == 4
    assert result.label == Label.UNCERTAIN
    assert result.turns == 4
    assert len(result.tool_calls) == 4


def test_last_turn_gets_a_text_nudge_but_keeps_tools_available():
    """Withdrawing the tools list on the last turn caused Groq to reject
    the request outright (verified live: "Tool choice is none, but model
    called a tool") once earlier turns had used a tool. Tools must stay
    available on every turn; the last turn instead gets an explicit
    "answer now" instruction appended to the conversation."""
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient(
        [
            _tool_call_response("get_gitleaks_rule", {"rule_id": "github-pat"}),
            _tool_call_response("get_gitleaks_rule", {"rule_id": "github-pat"}),
            _tool_call_response("get_gitleaks_rule", {"rule_id": "github-pat"}),
            _final_answer(),
        ]
    )
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    classifier.classify(github, mask_context_window(github, []))

    assert client.calls[0]["tools"] == client.calls[3]["tools"] != []
    last_messages = client.calls[3]["messages"]
    assert "final turn" in last_messages[-1]["content"]


def test_masked_treatment_never_sets_raw_permit():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient([_final_answer()])
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    classifier.classify(github, mask_context_window(github, []))

    assert client.calls[0]["raw_permit_candidate_id"] is None


def test_raw_treatment_sets_raw_permit_to_the_candidates_own_id():
    candidates = load_candidates()
    github = next(c for c in candidates if c.rule_id == "github-pat")
    client = _ScriptedClient([_final_answer()])
    classifier = AgenticClassifier(client, SAMPLE_REPO, candidates)

    classifier.classify(github, build_raw_context(github, []))

    assert client.calls[0]["raw_permit_candidate_id"] == github.id
