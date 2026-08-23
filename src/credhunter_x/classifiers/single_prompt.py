from __future__ import annotations

from credhunter_x.llm.client import LLMClient
from credhunter_x.llm.parsing import parse_classification
from credhunter_x.llm.prompts import build_classification_prompt
from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult
from credhunter_x.models.treatment import SanitisedContext, Treatment


class SinglePromptClassifier:
    """Arm A: one LLM call per candidate, no tools, no follow-up turns."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def classify(self, candidate: Candidate, context: SanitisedContext) -> ClassificationResult:
        prompt = build_classification_prompt(candidate, context)
        # Only permits this candidate's own value; the guard still catches
        # any other candidate's secret in the same window.
        raw_permit = candidate.id if context.treatment == Treatment.RAW else None
        response = self._client.generate(
            prompt,
            candidate_id=candidate.id,
            rule_id=candidate.rule_id,
            raw_permit_candidate_id=raw_permit,
        )
        return parse_classification(
            response, candidate_id=candidate.id, arm="single", treatment=context.treatment
        )
