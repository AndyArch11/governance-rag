"""Data Loss Prevention (DLP) utilities for masking/redacting sensitive data.

Supported patterns (default):
- Email addresses (requires @ and domain - not just the word 'email')
- Credit card numbers (13-19 digits with separators, Luhn validated, requires 4-group pattern)
    - Credit cards: Requires separator pattern to avoid matching 'creditCardId' or numeric identifiers
- AWS access keys (AKIA/ASIA-prefixed, 20 chars total)
- API keys (lowercase prefixes only: sk_, pk_, api_, token_, secret_)
    - API keys: Excludes SQL KEY_ columns and camelCase identifiers, uses case-sensitive matching

Usage:
    from scripts.security.dlp import DLPScanner
    scanner = DLPScanner()
    redacted = scanner.redact(text)

Config:
    - toggle patterns on/off
    - choose mask token
    - choose whether to keep last 4 digits for credit cards

TODO: Tune based on real-world ingestion scenarios to reduce false positives and improve on real positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Pattern, Tuple


@dataclass
class DLPConfig:
    mask_token: str = "[REDACTED]"
    keep_last4_cc: bool = True
    enable_email: bool = True
    enable_credit_card: bool = True
    enable_aws_key: bool = True
    enable_api_key: bool = True


class DLPScanner:
    """Scans and redacts sensitive patterns from text."""

    def __init__(self, config: DLPConfig | None = None):
        self.config = config or DLPConfig()
        self.patterns: List[Tuple[str, Pattern[str], Callable[[re.Match[str]], str]]] = []
        self._register_patterns()

    def _register_patterns(self) -> None:
        c = self.config

        if c.enable_email:
            email_regex = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
            self.patterns.append(("email", email_regex, lambda m: c.mask_token))

        if c.enable_credit_card:
            # Matches 13-19 digits with separators (space/hyphen). Must have at least 3 groups
            # to avoid matching single numeric IDs. Luhn checked in replacer.
            # Pattern requires spaces or hyphens to avoid matching concatenated identifiers.
            cc_regex = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b")

            def cc_replacer(match: re.Match[str]) -> str:
                digits = re.sub(r"[^0-9]", "", match.group())
                if not self._luhn_valid(digits):
                    return match.group()  # leave non-CC numbers unchanged
                if c.keep_last4_cc and len(digits) >= 4:
                    return (
                        f"{c.mask_token[:-1]}-{digits[-4:]}"
                        if c.mask_token.endswith("]")
                        else f"{c.mask_token}{digits[-4:]}"
                    )
                return c.mask_token

            self.patterns.append(("credit_card", cc_regex, cc_replacer))

        if c.enable_aws_key:
            # AWS access key ID: AKIA/ASIA followed by 16 alphanumerics
            aws_regex = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
            self.patterns.append(("aws_key", aws_regex, lambda m: c.mask_token))

        if c.enable_api_key:
            # Generic API key: must start with known lowercase prefix (sk_, pk_, api_, token_, secret_)
            # followed by 15-35 alphanumeric/special chars. Excludes SQL KEY_ patterns (foreign keys).
            # Note: Uses case-sensitive matching to avoid false positives on camelCase variables
            # like 'apiKey', 'publicKey', etc. which are field names, not actual API keys.
            api_regex = re.compile(r"\b(?:sk_|pk_|api_|token_|secret_)[a-zA-Z0-9+/=_-]{15,35}\b")
            self.patterns.append(("api_key", api_regex, lambda m: c.mask_token))

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        """Check number with Luhn algorithm."""
        total = 0
        reverse_digits = list(map(int, reversed(number)))
        for idx, digit in enumerate(reverse_digits):
            if idx % 2:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    def redact(self, text: str) -> str:
        """Redact sensitive data in text."""
        redacted = text
        for name, pattern, replacer in self.patterns:
            redacted = pattern.sub(replacer, redacted)
        return redacted

    def find(self, text: str) -> Dict[str, List[str]]:
        """Return found sensitive items by pattern name."""
        results: Dict[str, List[str]] = {}
        for name, pattern, _ in self.patterns:
            matches = pattern.findall(text)
            if matches:
                results.setdefault(name, []).extend(matches)
        return results


if __name__ == "__main__":
    scanner = DLPScanner()
    sample = """
    Contact: jane.doe@example.com or admin@foo.co.uk
    CC: 4111 1111 1111 1111, 5500-0000-0000-0004, not-a-cc 1234 5678 9123 4567
    AWS: AKIA1234567890ABCD12, ASIAABCDEFGHIJKLMNOP
    API: sk-abc123xyz987tokenval, 9f86d081884c7d659a2feaa0c55ad015
    """

    print("Original:\n", sample)
    print("Found:", scanner.find(sample))
    print("Redacted:\n", scanner.redact(sample))
