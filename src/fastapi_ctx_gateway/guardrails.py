"""Request-side prompt-injection detection.

Heuristic and request-side only: this looks for known injection *patterns*
in the raw text the client sent, not for a jailbroken *response* — it's a
cheap first line of defense, not a substitute for output-side moderation.
Deliberately mirrors the pruning/rate-limit shape (a stateless checker over
`turns`/`system`, no request-scoped state) so it slots into the same
pre-proxy sequencing.
"""

import re
from typing import Protocol

from fastapi_ctx_gateway.schemas.neutral import Part, TextPart, Turn

__all__ = ["HeuristicInjectionDetector", "InjectionDetector"]


class InjectionDetector(Protocol):
    """Checks a request's raw content for prompt-injection patterns."""

    def detect(self, turns: list[Turn], system: list[Part] | None) -> bool:
        """Return True if any turn or system content matches a known pattern."""
        ...


# Deliberately simple substring/regex heuristics, not an ML classifier: this
# is meant to catch known, common injection phrasing cheaply, not to be a
# complete jailbreak detector. Three categories per the spec, each matched
# case-insensitively and anywhere in the text (not just at the start).
_INSTRUCTION_OVERRIDE_PATTERNS = [
    r"ignore (?:all )?(?:the )?(?:previous|prior|above)\s+instructions",
    r"disregard (?:all )?(?:the )?(?:previous|prior|above)\s+(?:instructions|prompt)",
    r"forget (?:all )?(?:your|the)\s+(?:previous\s+)?instructions",
    r"override (?:your|the|all)\s+(?:previous\s+)?instructions",
]

_ROLEPLAY_JAILBREAK_PATTERNS = [
    r"you are (?:now )?dan\b",
    r"act as if you (?:have no|had no)\s+restrictions",
    r"pretend (?:that )?you (?:are not|aren't)\s+an ai",
    r"developer mode(?: enabled)?",
    r"\bjailbreak\b",
]

_FAKE_DELIMITER_PATTERNS = [
    r"\[\s*system\s*\]",
    r"<\s*system\s*>",
    r"<\|\s*system\s*\|>",
    r"#{2,}\s*system\b",
    r"end of system prompt",
]


class HeuristicInjectionDetector:
    """Regex-based detector over three known prompt-injection categories.

    detect() only needs a yes/no match, so the three category lists above
    are flattened into one compiled list rather than kept as a dict — there's
    no consumer for *which* category matched (yet; the metric/log are
    request-level, not per-category).
    """

    def __init__(self) -> None:
        """Compile every pattern once, case-insensitively."""
        raw_patterns = (
            _INSTRUCTION_OVERRIDE_PATTERNS + _ROLEPLAY_JAILBREAK_PATTERNS + _FAKE_DELIMITER_PATTERNS
        )
        self._patterns: list[re.Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns
        ]

    def detect(self, turns: list[Turn], system: list[Part] | None) -> bool:
        """Check every turn's and the system prompt's text parts for a match."""
        text = self._collect_text(turns, system)
        return any(pattern.search(text) for pattern in self._patterns)

    @staticmethod
    def _collect_text(turns: list[Turn], system: list[Part] | None) -> str:
        parts = [part for turn in turns for part in turn.parts if isinstance(part, TextPart)]
        if system:
            parts += [part for part in system if isinstance(part, TextPart)]
        return "\n".join(part.text for part in parts)
