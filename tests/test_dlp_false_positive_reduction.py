"""Tests for DLP false positive reduction in code ingestion.

Validates that DLP patterns don't trigger on:
- Variable/field names containing 'email', 'card', 'key'
- SQL column names with KEY_ prefix (foreign keys)
- CamelCase identifiers and method parameters

Tune based on real-world false positives from code ingestion scenarios.
"""

import pytest

from scripts.security.dlp import DLPScanner


class TestEmailFalsePositiveReduction:
    """Email pattern should only match actual addresses, not variable names."""

    def test_email_variable_names_not_matched(self):
        scanner = DLPScanner()
        code = """
        private String emailSubject = "Notification";
        private String emailFrom = "noreply@company.com.au";
        private String emailTemplate = "Company-SalesOrderResponse";
        String accountManagerEmailAddress;
        """

        # Should only match the actual email address
        matches = scanner.find(code)
        assert "email" in matches
        assert len(matches["email"]) == 1
        assert "noreply@company.com.au" in matches["email"]

    def test_email_method_parameters_not_matched(self):
        scanner = DLPScanner()
        code = "public void setEmailFrom(String emailFrom) {"

        matches = scanner.find(code)
        assert "email" not in matches

    def test_actual_email_addresses_matched(self):
        scanner = DLPScanner()
        code = "Contact middlewareTeam@company.com.au or support@example.org"

        matches = scanner.find(code)
        assert "email" in matches
        assert len(matches["email"]) == 2


class TestAPIKeyFalsePositiveReduction:
    """API key pattern should exclude SQL KEY_ columns and camelCase identifiers."""

    def test_sql_key_columns_not_matched(self):
        scanner = DLPScanner()
        # From DLP_False_Positives.txt - SQL foreign key columns
        sql = """
        SELECT
            CV.KEY_CONNECTED_VEHICLE_ID,
            CV.KEY_VEHICLE_ID,
            CV.KEY_CONNECTED_VEHI_EFF_START
        FROM CIM_CONNECTED_VEHICLE CV
        WHERE CV.KEY_VEHICLE_EFFECTIVE_DATE = ?
        """

        matches = scanner.find(sql)
        assert "api_key" not in matches

    def test_camelcase_identifiers_not_matched(self):
        scanner = DLPScanner()
        # CamelCase field names should not match
        code = """
        private String apiKey;
        public String publicKey;
        String accessToken;
        """

        matches = scanner.find(code)
        assert "api_key" not in matches

    def test_actual_api_keys_matched(self):
        scanner = DLPScanner()
        stripe_like_key = "sk_" + "live_" + "abc123xyz987tokenval1234567890"
        code = f"API key {stripe_like_key}"

        matches = scanner.find(code)
        assert "api_key" in matches
        assert len(matches["api_key"]) == 1


class TestCreditCardFalsePositiveReduction:
    """Credit card pattern should only match separated digit groups, not identifiers."""

    def test_credit_card_identifier_not_matched(self):
        scanner = DLPScanner()
        code = """
        String creditCardId;
        class Errors {
            String creditCardId;
        }
        """

        matches = scanner.find(code)
        assert "credit_card" not in matches

    def test_template_names_not_matched(self):
        scanner = DLPScanner()
        code = '"visa-ccard-template01": ["some.one@company.com.au"]'

        matches = scanner.find(code)
        assert "credit_card" not in matches

    def test_actual_credit_cards_matched(self):
        scanner = DLPScanner()
        code = "Pay with 4111 1111 1111 1111 or 5500-0000-0000-0004"

        matches = scanner.find(code)
        assert "credit_card" in matches
        assert len(matches["credit_card"]) == 2


class TestCodeIngestionScenarios:

    def test_groovy_transformer_email_fields(self):
        """Groovy source code with email field names should not trigger false positives."""
        scanner = DLPScanner()
        groovy_code = """
        class CommunicationTransformer {
            private Template emailTemplate
            
            String getAccountManagerEmailAddress(CustomerParty customerParty) {
                customerParty?.customerAccount?.CRMAccount?.keyAccountManager?.electronicContacts?.find()?.electronicAddress
            }
            
            Communication createCommunication(Exchange exchange) {
                String keyAccountManagerEmailError = StringUtils.EMPTY
                String accountManagerEmailAddress = exchange.properties['accountManagerEmailAddress']
                new Communication(
                    from: 'noreply@company.com.au',
                    to: recipients
                )
            }
        }
        """

        matches = scanner.find(groovy_code)
        # Should only match the actual email address
        assert "email" in matches
        assert len(matches["email"]) == 1
        assert "noreply@company.com.au" in matches["email"]

    def test_java_email_transformer(self):
        """Java class with email fields should not trigger on field names."""
        scanner = DLPScanner()
        java_code = """
        public class CreateNotificationEmailTransformer {
            private String emailSubject = "Notification";
            private String emailFrom = "noreply@company.com.au";
            private String emailTemplate = "Dept-SalesOrderResponse";
            private String emailBcc = "Dept_Team_notifications@company.com.au";
            
            public void setEmailFrom(String emailFrom) {
                this.emailFrom = emailFrom;
            }
        }
        """

        matches = scanner.find(java_code)
        # Should match both actual email addresses
        assert "email" in matches
        assert len(matches["email"]) == 2
        assert "noreply@company.com.au" in matches["email"]
        assert "Dept_Team_notifications@company.com.au" in matches["email"]

    def test_sql_foreign_keys(self):
        """SQL with KEY_ columns should not match API key pattern."""
        scanner = DLPScanner()
        sql = """
        SELECT
            STATUS,
            CON_VEH_ID,
            CV.KEY_CONNECTED_VEHICLE_ID CON_VEH_ID,
            V.ID VEHICLE_ID
        FROM CFR_VEHICLE V
        JOIN CIM_CONNECTED_VEHICLE CV ON CV.KEY_VEHICLE_ID=V.ID
        WHERE AUDIT_DATETIME > '${watermark}'
        """

        matches = scanner.find(sql)
        assert "api_key" not in matches

    def test_groovy_credit_card_class(self):
        """Groovy class with creditCardId field should not match."""
        scanner = DLPScanner()
        groovy_code = """
        class Errors {
            List<Error> errors
            String responseCode
            String creditCardId
        }
        """

        matches = scanner.find(groovy_code)
        assert "credit_card" not in matches


def test_dlp_summary_statistics():
    """Verify overall false positive reduction across all patterns."""
    scanner = DLPScanner()

    stripe_like_key = "sk_" + "live_" + "abc123tokenvalue123456789"

    # Code sample with potential false positives
    code_sample = """
    class PaymentService {
        private String emailTemplate;
        private String accountManagerEmailAddress;
        String creditCardId;
        
        SELECT CV.KEY_VEHICLE_ID, CV.KEY_CONNECTED_VEHICLE_ID
        FROM VEHICLE WHERE status = 'active';
        
        Contact support@company.com for help.
        Payment: 4111 1111 1111 1111
        API: __STRIPE_LIKE_KEY__
    }
    """
    code_sample = code_sample.replace("__STRIPE_LIKE_KEY__", stripe_like_key)

    matches = scanner.find(code_sample)

    # Should match:
    # - 1 email (support@company.com)
    # - 1 credit card (4111 1111 1111 1111)
    # - 1 API key (sk_live...)

    assert len(matches.get("email", [])) == 1
    assert len(matches.get("credit_card", [])) == 1
    assert len(matches.get("api_key", [])) == 1

    # Verify specific matches
    assert "support@company.com" in matches["email"]
    assert "4111 1111 1111 1111" in matches["credit_card"]
    assert stripe_like_key in matches["api_key"]
