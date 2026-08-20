from __future__ import annotations

from typing import Protocol

from credhunter_x.models.candidate import Candidate
from credhunter_x.models.classification import ClassificationResult
from credhunter_x.models.treatment import SanitisedContext


class Classifier(Protocol):
    """Lets gitleaks-only, Arm A, and Arm B be interchangeable in the
    evaluation harness."""

    def classify(self, candidate: Candidate, context: SanitisedContext) -> ClassificationResult: ...
