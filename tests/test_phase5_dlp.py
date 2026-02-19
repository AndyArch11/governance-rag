"""Tests for DLP masking and detection."""

import pytest
from scripts.security.dlp import DLPScanner, DLPConfig


def test_email_redaction():
    scanner = DLPScanner()
    text = "Contact me at user@example.com"
    redacted = scanner.redact(text)
    assert "example.com" not in redacted
    assert "[REDACTED]" in redacted


def test_credit_card_redaction_keep_last4():
    scanner = DLPScanner(DLPConfig(keep_last4_cc=True))
    text = "Pay with 4111 1111 1111 1111"
    redacted = scanner.redact(text)
    assert "1111" in redacted
    assert "4111" not in redacted


def test_credit_card_non_luhn_not_redacted():
    scanner = DLPScanner()
    text = "Number 1234 5678 9123 4567 is not a cc"
    redacted = scanner.redact(text)
    # Should remain unchanged because Luhn fails
    assert text == redacted


def test_aws_key_redaction():
    scanner = DLPScanner()
    text = "Key AKIA1234567890ABCD12 is secret"
    redacted = scanner.redact(text)
    assert "AKIA" not in redacted


def test_generic_api_key_redaction():
    scanner = DLPScanner()
    text = "API key sk_abc123xyz987tokenval is secret"
    redacted = scanner.redact(text)
    assert "tokenval" not in redacted
    assert "[REDACTED]" in redacted


def test_find_matches():
    scanner = DLPScanner()
    text = "Email a@b.com, CC 4111-1111-1111-1111"
    found = scanner.find(text)
    assert "email" in found
    assert "credit_card" in found
    assert len(found["email"]) == 1
    assert len(found["credit_card"]) == 1
