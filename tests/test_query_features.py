"""Test suite for dashboard query features.

Tests Features 1-4:
1. Multi-turn conversation
2. Conversation persistence
3. Interactive code preview
4. Query templates
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.rag.conversation_manager import ConversationManager
from scripts.ui.query_templates import QueryTemplateManager


class TestConversationManager:
    """Test conversation persistence functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_conversations.db"
            yield db_path

    def test_conversation_creation(self, temp_db):
        """Test creating a new conversation."""
        manager = ConversationManager(temp_db)

        conv_id = "test-conv-001"
        assert manager.save_conversation(
            conv_id,
            title="Test Conversation",
            description="Testing conversation persistence",
            tags="test,demo",
        )

        summary = manager.get_conversation_summary(conv_id)
        assert summary is not None
        assert summary["title"] == "Test Conversation"
        assert summary["turn_count"] == 0

    def test_add_turn(self, temp_db):
        """Test adding turns to a conversation."""
        manager = ConversationManager(temp_db)

        conv_id = "test-conv-002"
        manager.save_conversation(conv_id, title="Multi-turn Test")

        # Add first turn
        assert manager.add_turn(
            conv_id=conv_id,
            turn_number=1,
            query="What is governance?",
            answer="Governance is the system of rules...",
            sources=[{"source": "doc1", "score": 0.95}],
            tokens_used=150,
            cache_hit=False,
        )

        # Add second turn
        assert manager.add_turn(
            conv_id=conv_id,
            turn_number=2,
            query="Explain compliance",
            answer="Compliance means adhering to rules...",
            sources=[{"source": "doc2", "score": 0.92}],
            tokens_used=120,
            cache_hit=True,
        )

        # Load conversation
        conv = manager.load_conversation(conv_id)
        assert conv is not None
        assert conv["turn_count"] == 2
        assert len(conv["turns"]) == 2
        assert conv["total_tokens"] == 270
        assert conv["cached_results"] == 1

        # Verify turn content
        assert conv["turns"][0]["query"] == "What is governance?"
        assert conv["turns"][1]["cache_hit"] == 1

    def test_load_conversations_list(self, temp_db):
        """Test loading list of conversations."""
        manager = ConversationManager(temp_db)

        # Create multiple conversations
        for i in range(3):
            manager.save_conversation(
                f"conv-{i}", title=f"Conversation {i}", tags="test" if i % 2 == 0 else "demo"
            )

        # Load all
        conversations = manager.load_conversations(limit=10)
        assert len(conversations) == 3

        # Verify ordering (most recent first)
        assert conversations[0]["id"] == "conv-2"
        assert conversations[2]["id"] == "conv-0"

    def test_search_conversations(self, temp_db):
        """Test searching conversations."""
        manager = ConversationManager(temp_db)

        manager.save_conversation("conv-1", title="Governance Policies")
        manager.save_conversation("conv-2", title="Security Controls")
        manager.save_conversation("conv-3", title="Compliance Framework")

        # Search for governance
        results = manager.search_conversations("Governance", limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Governance Policies"

        # Search for compliance
        results = manager.search_conversations("Compliance", limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Compliance Framework"

    def test_delete_conversation(self, temp_db):
        """Test deleting a conversation."""
        manager = ConversationManager(temp_db)

        conv_id = "conv-to-delete"
        manager.save_conversation(conv_id, title="Delete Me")
        manager.add_turn(conv_id, 1, "Query", "Answer")

        # Verify it exists
        assert manager.get_conversation_summary(conv_id) is not None

        # Delete
        assert manager.delete_conversation(conv_id)

        # Verify it's gone
        assert manager.get_conversation_summary(conv_id) is None

    def test_export_conversation_json(self, temp_db):
        """Test exporting conversation as JSON."""
        manager = ConversationManager(temp_db)

        conv_id = "export-test"
        manager.save_conversation(conv_id, title="Export Test")
        manager.add_turn(conv_id, 1, "Question 1", "Answer 1")

        exported = manager.export_conversation(conv_id, format="json")
        assert exported is not None

        data = json.loads(exported)
        assert data["id"] == conv_id
        assert data["title"] == "Export Test"
        assert len(data["turns"]) == 1

    def test_export_conversation_markdown(self, temp_db):
        """Test exporting conversation as Markdown."""
        manager = ConversationManager(temp_db)

        conv_id = "markdown-test"
        manager.save_conversation(conv_id, title="Markdown Export")
        manager.add_turn(conv_id, 1, "Test Question", "Test Answer")

        exported = manager.export_conversation(conv_id, format="markdown")
        assert exported is not None
        assert "# Conversation: Markdown Export" in exported
        assert "Test Question" in exported
        assert "Test Answer" in exported


class TestQueryTemplateManager:
    """Test query template functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_templates.db"
            yield db_path

    def test_default_templates_created(self, temp_db):
        """Test that default templates are created."""
        manager = QueryTemplateManager(temp_db)

        categories = manager.get_all_categories()
        assert len(categories) > 0
        assert "governance" in categories
        assert "security" in categories
        assert "code" in categories

    def test_get_templates_by_category(self, temp_db):
        """Test retrieving templates by category."""
        manager = QueryTemplateManager(temp_db)

        governance_templates = manager.get_templates_by_category("governance")
        assert len(governance_templates) > 0

        # Verify template structure
        template = governance_templates[0]
        assert "name" in template
        assert "template_text" in template
        assert "k_results" in template

    def test_get_template_and_track_usage(self, temp_db):
        """Test getting a template and usage tracking."""
        manager = QueryTemplateManager(temp_db)

        # Get a template (should increment usage)
        template = manager.get_template("What policies exist?")
        assert template is not None
        initial_usage = template["usage_count"]

        # Get it again
        template = manager.get_template("What policies exist?")
        assert template["usage_count"] > initial_usage

    def test_save_custom_template(self, temp_db):
        """Test saving custom templates."""
        manager = QueryTemplateManager(temp_db)

        assert manager.save_template(
            name="Custom Architecture Query",
            category="architecture",
            template_text="Describe the architecture of the {} system",
            description="Custom template for architecture questions",
            k_results=10,
            temperature=0.5,
            tags="custom,architecture",
        )

        template = manager.get_template("Custom Architecture Query")
        assert template is not None
        assert template["category"] == "architecture"
        assert template["template_text"] == "Describe the architecture of the {} system"

    def test_search_templates(self, temp_db):
        """Test searching templates."""
        manager = QueryTemplateManager(temp_db)

        results = manager.search_templates("security")
        assert len(results) > 0

        # At least one result should match the query in some field
        any_match = False
        for template in results:
            match_found = (
                "security" in template["name"].lower()
                or "security" in template.get("description", "").lower()
                or "security" in template.get("tags", "").lower()
            )
            if match_found:
                any_match = True
                break
        assert any_match, "At least one security-related template should be found"

    def test_delete_template(self, temp_db):
        """Test deleting templates."""
        manager = QueryTemplateManager(temp_db)

        # Create a template
        manager.save_template(name="Delete Me", category="test", template_text="Test template")

        # Verify it exists
        assert manager.get_template("Delete Me") is not None

        # Delete
        assert manager.delete_template("Delete Me")

        # Verify it's gone
        assert manager.get_template("Delete Me") is None

    def test_get_recent_templates(self, temp_db):
        """Test getting recently used templates."""
        manager = QueryTemplateManager(temp_db)

        # Use some templates
        manager.get_template("What policies exist?")
        manager.get_template("Architecture overview")

        recent = manager.get_recent_templates(limit=5)
        assert len(recent) > 0

    def test_get_popular_templates(self, temp_db):
        """Test getting most-used templates."""
        manager = QueryTemplateManager(temp_db)

        # Use some templates multiple times
        for _ in range(3):
            manager.get_template("What policies exist?")
        for _ in range(2):
            manager.get_template("Architecture overview")

        popular = manager.get_popular_templates(limit=5)
        assert len(popular) > 0

        # Most popular should be first
        if len(popular) > 1:
            assert popular[0]["usage_count"] >= popular[1]["usage_count"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
