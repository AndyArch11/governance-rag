"""Unit tests for Phase 1 features in ingest_git.py

Tests for:
- DLP redaction
- Code summary generation
- Argument sanitisation
- Advanced test file detection
"""

import argparse
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import functions to test
from scripts.ingest.ingest_git import (
    obfuscate_password,
    sanitise_args_for_logging,
    is_test_file,
    generate_code_summary,
    redact_code_content,
)


# =========================
# TESTS: obfuscate_password
# =========================

class TestObfuscatePassword:
    """Test password obfuscation for logging."""
    
    def test_empty_password(self):
        """Empty password returns obfuscated string."""
        assert obfuscate_password("") == "****"
    
    def test_none_password(self):
        """None password returns obfuscated string."""
        assert obfuscate_password(None) == "****"
    
    def test_short_password(self):
        """Passwords <= 4 chars return obfuscated string."""
        assert obfuscate_password("abc") == "****"
        assert obfuscate_password("1234") == "****"
    
    def test_long_password(self):
        """Long password shows first 2 and last 2 chars."""
        result = obfuscate_password("abcdefghij")
        assert result == "ab******ij"
        assert result.count("*") == 6
    
    def test_token_like_string(self):
        """Token-like string is properly obfuscated."""
        token = "ghp_1234567890abcdefghijklmnopqrst"
        result = obfuscate_password(token)
        assert result.startswith("gh")
        assert result.endswith("st")
        assert "1234" not in result
        assert "***" in result


# =========================
# TESTS: sanitise_args_for_logging
# =========================

class TestSanitiseArgsForLogging:
    """Test argument sanitisation for safe logging."""
    
    def test_sanitises_password(self):
        """Password field is obfuscated."""
        args = argparse.Namespace(
            password="secretpassword123",
            host="https://example.com"
        )
        result = sanitise_args_for_logging(args)
        assert "secretpassword123" not in str(result)
        assert "se" in result["password"] and "23" in result["password"]
        assert "*" in result["password"]
    
    def test_sanitises_token(self):
        """Token field is obfuscated."""
        args = argparse.Namespace(
            token="ghp_1234567890abcdefghijklmnopqrst",
            host="https://api.github.com"
        )
        result = sanitise_args_for_logging(args)
        assert "1234567890" not in str(result)
        assert "****" in result["token"]
    
    def test_sanitises_api_username(self):
        """API username (email) is redacted."""
        args = argparse.Namespace(
            api_username="user@example.com",
            username="myusername"
        )
        result = sanitise_args_for_logging(args)
        assert result["api_username"] == "***REDACTED***"
        assert "user@example.com" not in str(result)
    
    def test_preserves_safe_fields(self):
        """Safe fields like host and provider are preserved."""
        args = argparse.Namespace(
            password="secret",
            host="https://bitbucket.org",
            provider="bitbucket",
            project="MYPROJ"
        )
        result = sanitise_args_for_logging(args)
        assert result["host"] == "https://bitbucket.org"
        assert result["provider"] == "bitbucket"
        assert result["project"] == "MYPROJ"
    
    def test_handles_none_values(self):
        """None values don't cause errors."""
        args = argparse.Namespace(
            password=None,
            token=None,
            api_username=None,
            host="https://example.com"
        )
        result = sanitise_args_for_logging(args)
        assert result["password"] is None
        assert result["token"] is None


# =========================
# TESTS: is_test_file
# =========================

class TestIsTestFile:
    """Test advanced test file detection."""
    
    def test_detects_test_directory(self):
        """Files in test directories detected."""
        assert is_test_file("src/test/java/MyTest.java")
        assert is_test_file("tests/unit/service_test.py")
        assert is_test_file("__tests__/unit.js")
    
    def test_detects_test_suffix_java(self):
        """Java test patterns detected."""
        assert is_test_file("MyServiceTest.java")
        assert is_test_file("MyServiceTests.java")
        assert is_test_file("MyServiceIT.java")
        assert is_test_file("MyServiceSpec.java")
    
    def test_detects_test_suffix_groovy(self):
        """Groovy test patterns detected."""
        assert is_test_file("MyServiceTest.groovy")
        assert is_test_file("MyServiceIT.groovy")
        assert is_test_file("MyServiceSpec.groovy")
    
    def test_detects_test_suffix_javascript(self):
        """JavaScript test patterns detected."""
        assert is_test_file("service.test.js")
        assert is_test_file("service.test.ts")
        assert is_test_file("service.spec.js")
        assert is_test_file("utils.spec.tsx")
    
    def test_detects_test_suffix_python(self):
        """Python test patterns detected."""
        # Python tests typically use test_ prefix in directory, not filename
        assert is_test_file("tests/test_service.py")
        assert is_test_file("test/service_test.py")
    
    def test_detects_test_suffix_ruby(self):
        """Ruby test patterns detected."""
        assert is_test_file("service_spec.rb")
    
    def test_ignores_non_test_files(self):
        """Regular files not detected as tests."""
        assert not is_test_file("src/main/java/MyService.java")
        assert not is_test_file("utils/helper.js")
        assert not is_test_file("models/user.py")
    
    def test_case_insensitive_detection(self):
        """Detection is case-insensitive."""
        assert is_test_file("SRC/TEST/MyTest.JAVA")
        assert is_test_file("Tests/Integration/TestSuite.py")
    
    def test_handles_windows_paths(self):
        """Windows path separators handled."""
        assert is_test_file("src\\test\\java\\MyTest.java")
        assert is_test_file("tests\\unit\\test_service.py")
    
    def test_handles_malformed_paths(self):
        """Malformed paths don't crash."""
        assert is_test_file("") is False
        assert is_test_file(None) is False


# =========================
# TESTS: redact_code_content
# =========================

class TestRedactCodeContent:
    """Test DLP redaction functionality."""
    
    def test_returns_original_when_disabled(self):
        """Original content returned when DLP disabled."""
        text = "password = 'secret123'"
        result, counts = redact_code_content(
            text=text,
            artifact_path="test.py",
            doc_id="test_1",
            repository="test-repo",
            enable_dlp=False
        )
        assert result == text
        assert counts == {}
    
    def test_returns_tuple(self):
        """Returns tuple of (text, counts) when DLP disabled."""
        text = "some code"
        result, counts = redact_code_content(
            text=text,
            artifact_path="test.py",
            doc_id="test_1",
            repository="test-repo",
            enable_dlp=False
        )
        assert isinstance(result, str)
        assert isinstance(counts, dict)
    
    @patch('scripts.ingest.ingest_git.logger')
    def test_handles_dlp_unavailable(self, mock_logger):
        """Gracefully handles missing DLP module."""
        with patch('scripts.security.dlp.DLPScanner', side_effect=ImportError):
            text = "code content"
            result, counts = redact_code_content(
                text=text,
                artifact_path="test.py",
                doc_id="test_1",
                repository="test-repo",
                enable_dlp=True
            )
            # When DLP is unavailable, content is unchanged
            assert result == text
            assert counts == {}
    
    def test_logs_redaction_info(self):
        """Redaction counts returned when DLP enabled."""
        with patch('scripts.security.dlp.DLPScanner') as mock_dlp_class:
            # Mock DLP scanner
            mock_scanner = MagicMock()
            stripe_like_key = "sk_" + "live_" + "abc123"
            mock_scanner.find.return_value = {
                "CREDIT_CARD": ["4532-1234-5678-9012"],
                "API_KEY": [stripe_like_key]
            }
            mock_scanner.redact.return_value = "REDACTED CODE"
            mock_dlp_class.return_value = mock_scanner
            
            text = f"cc 4532-1234-5678-9012 and key {stripe_like_key}"
            result, counts = redact_code_content(
                text=text,
                artifact_path="config.py",
                doc_id="proj_repo_config",
                repository="my-repo",
                enable_dlp=True
            )
            
            assert result == "REDACTED CODE"
            assert counts == {"CREDIT_CARD": 1, "API_KEY": 1}


# =========================
# TESTS: generate_code_summary
# =========================

class TestGenerateCodeSummary:
    """Test code summary generation."""
    
    def test_generates_summary_from_parse_result(self):
        """Summary generated from parse result."""
        parse_result = {
            "service_name": "UserService",
            "service_type": "REST",
            "exports": ["getUserById", "createUser"],
            "endpoints": ["/api/users", "/api/users/{id}"],
            "external_dependencies": ["postgresql", "redis"]
        }
        
        result = generate_code_summary(
            parse_result=parse_result,
            file_path="user_service.py",
            language="python",
            use_llm=False,
            llm_cache=None
        )
        
        assert "summary" in result
        assert "key_topics" in result
        assert "scores" in result
        assert "UserService" in result["summary"]
        assert "getUserById" in result["key_topics"]
    
    def test_handles_empty_parse_result(self):
        """Handles empty parse result gracefully."""
        parse_result = {}
        
        result = generate_code_summary(
            parse_result=parse_result,
            file_path="file.py",
            language="python",
            use_llm=False,
            llm_cache=None
        )
        
        assert result["summary"]
        assert isinstance(result["key_topics"], list)
        assert "scores" in result
    
    def test_extracts_key_topics(self):
        """Key topics extracted from parse result."""
        parse_result = {
            "exports": ["method1", "method2", "method3"],
            "endpoints": ["/api/v1", "/api/v2"],
            "external_dependencies": ["dep1", "dep2"]
        }
        
        result = generate_code_summary(
            parse_result=parse_result,
            file_path="api.py",
            language="python"
        )
        
        assert len(result["key_topics"]) > 0
        assert any("method" in str(t) for t in result["key_topics"])
    
    def test_calculates_scores(self):
        """Summary scores calculated."""
        parse_result = {
            "exports": ["a", "b"],
            "endpoints": ["/api/users"],
            "external_dependencies": ["db"]
        }
        
        result = generate_code_summary(
            parse_result=parse_result,
            file_path="service.py",
            language="python"
        )
        
        scores = result["scores"]
        assert "completeness" in scores
        assert "clarity" in scores
        assert "technical_depth" in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())
    
    def test_handles_parse_error(self):
        """Handles parse errors gracefully."""
        parse_result = "invalid"  # Not a dict
        
        result = generate_code_summary(
            parse_result=parse_result,
            file_path="file.py",
            language="python"
        )
        
        # Should still return valid structure
        assert "summary" in result
        assert "key_topics" in result
        assert "scores" in result


# =========================
# INTEGRATION TESTS
# =========================

class TestPhase1Integration:
    """Integration tests for Phase 1 features."""
    
    def test_sanitisation_flow(self):
        """Full argument sanitisation flow."""
        args = argparse.Namespace(
            provider="bitbucket",
            host="https://bitbucket.org",
            username="myuser",
            password="MySecurePass123!",
            api_username="user@company.com",
            token=None,
            project="PROJ",
            verbose=False
        )
        
        sanitised = sanitise_args_for_logging(args)
        
        # Check no critical secrets exposed
        args_str = str(sanitised)
        assert "MySecurePass123!" not in args_str  # password is redacted
        assert "user@company.com" not in args_str  # api_username is redacted
        assert "***REDACTED***" in args_str
        # Username can remain visible (not a secret)
        
        # Check safe fields preserved
        assert sanitised["host"] == "https://bitbucket.org"
        assert sanitised["project"] == "PROJ"
    
    def test_test_detection_comprehensive(self):
        """Comprehensive test file detection."""
        test_files = [
            "src/test/java/MyTest.java",
            "tests/integration/test_api.py",
            "spec/UserSpec.groovy",
            "components/__tests__/Button.test.js",
            "MyServiceIT.java",
            "utils.spec.tsx",
        ]
        
        non_test_files = [
            "src/main/java/MyService.java",
            "lib/utils.js",
            "models/user.py",
            "config/settings.yaml",
        ]
        
        for test_file in test_files:
            assert is_test_file(test_file), f"Failed to detect {test_file} as test"
        
        for non_test_file in non_test_files:
            assert not is_test_file(non_test_file), f"Incorrectly detected {non_test_file} as test"
    
    def test_summary_generation_with_various_languages(self):
        """Summary generation works for various languages."""
        languages = ["java", "python", "groovy", "javascript", "typescript"]
        
        for lang in languages:
            parse_result = {
                "service_name": f"Service_{lang}",
                "exports": ["export1", "export2"],
            }
            
            result = generate_code_summary(
                parse_result=parse_result,
                file_path=f"service.{lang}",
                language=lang
            )
            
            assert result["summary"]
            assert len(result["key_topics"]) > 0
            assert "scores" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
