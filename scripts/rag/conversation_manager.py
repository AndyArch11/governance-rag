"""Conversation Persistence for RAG Dashboard.

Provides storage and retrieval of conversation history with full context preservation.
Conversations are stored in SQLite with complete turn history, allowing users to
resume conversations and view their full interaction history.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConversationManager:
    """Manage conversation history in SQLite database."""

    def __init__(self, db_path: Path):
        """Initialise conversation manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialise database schema."""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    turn_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cached_results INTEGER DEFAULT 0,
                    tags TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens_used INTEGER DEFAULT 0,
                    cache_hit BOOLEAN DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_turns(conversation_id)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
            """
            )

            conn.commit()

    def save_conversation(
        self, conv_id: str, title: str = "", description: str = "", tags: str = ""
    ) -> bool:
        """Save a new conversation.

        Args:
            conv_id: Unique conversation ID
            title: Conversation title
            description: Conversation description
            tags: Comma-separated tags

        Returns:
            True if saved successfully
        """
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations
                    (id, title, description, updated_at, tags)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (conv_id, title, description, datetime.now(timezone.utc).isoformat(), tags),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False

    def add_turn(
        self,
        conv_id: str,
        turn_number: int,
        query: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        tokens_used: int = 0,
        cache_hit: bool = False,
    ) -> bool:
        """Add a turn to a conversation.

        Args:
            conv_id: Conversation ID
            turn_number: Turn number (1-indexed)
            query: User query
            answer: RAG answer
            sources: List of source documents
            metadata: Additional metadata
            tokens_used: Number of tokens used
            cache_hit: Whether result was from cache

        Returns:
            True if added successfully
        """
        try:
            sources_json = json.dumps(sources) if sources else None
            metadata_json = json.dumps(metadata) if metadata else None

            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_turns
                    (conversation_id, turn_number, query, answer, sources, metadata, tokens_used, cache_hit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        conv_id,
                        turn_number,
                        query,
                        answer,
                        sources_json,
                        metadata_json,
                        tokens_used,
                        cache_hit,
                    ),
                )

                # Update conversation metadata
                conn.execute(
                    """
                    UPDATE conversations
                    SET turn_count = ?,
                        total_tokens = total_tokens + ?,
                        cached_results = cached_results + ?,
                        updated_at = ?
                    WHERE id = ?
                """,
                    (
                        turn_number,
                        tokens_used,
                        1 if cache_hit else 0,
                        datetime.now(timezone.utc).isoformat(),
                        conv_id,
                    ),
                )

                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding turn: {e}")
            return False

    def load_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Load a complete conversation with all turns.

        Args:
            conv_id: Conversation ID

        Returns:
            Conversation dict with turns, or None
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get conversation metadata
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            conv_row = cursor.fetchone()
            if not conv_row:
                return None

            # Get all turns
            cursor.execute(
                "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_number",
                (conv_id,),
            )
            turns = []
            for turn_row in cursor.fetchall():
                turn = dict(turn_row)
                if turn["sources"]:
                    turn["sources"] = json.loads(turn["sources"])
                if turn["metadata"]:
                    turn["metadata"] = json.loads(turn["metadata"])
                turns.append(turn)

            conv = dict(conv_row)
            conv["turns"] = turns
            return conv

    def load_conversations(
        self, limit: int = 20, offset: int = 0, tags: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load list of conversations.

        Args:
            limit: Maximum number of conversations
            offset: Offset for pagination
            tags: Filter by tags (comma-separated)

        Returns:
            List of conversations
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if tags:
                # Filter by tags
                tag_filter = " OR ".join(
                    [f"tags LIKE '%{tag.strip()}%'" for tag in tags.split(",")]
                )
                cursor.execute(
                    f"""
                    SELECT id, title, description, created_at, updated_at, turn_count, total_tokens, cached_results
                    FROM conversations
                    WHERE {tag_filter}
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, title, description, created_at, updated_at, turn_count, total_tokens, cached_results
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_summary(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get quick summary of a conversation.

        Args:
            conv_id: Conversation ID

        Returns:
            Summary dict or None
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, description, created_at, updated_at, turn_count, total_tokens, cached_results
                FROM conversations
                WHERE id = ?
            """,
                (conv_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_conversation_title(self, conv_id: str, title: str) -> bool:
        """Update conversation title.

        Args:
            conv_id: Conversation ID
            title: New title

        Returns:
            True if updated
        """
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute(
                    """
                    UPDATE conversations SET title = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (title, datetime.now(timezone.utc).isoformat(), conv_id),
                )
                conn.commit()
            return True
        except Exception:
            return False

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its turns.

        Args:
            conv_id: Conversation ID

        Returns:
            True if deleted
        """
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute("DELETE FROM conversation_turns WHERE conversation_id = ?", (conv_id,))
                conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
                conn.commit()
            return True
        except Exception:
            return False

    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations by title or description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching conversations
        """
        search_term = f"%{query.lower()}%"
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, description, created_at, updated_at, turn_count, total_tokens, cached_results
                FROM conversations
                WHERE LOWER(title) LIKE ? OR LOWER(description) LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            """,
                (search_term, search_term, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recently updated conversations.

        Args:
            limit: Number of conversations

        Returns:
            List of conversations
        """
        return self.load_conversations(limit=limit, offset=0)

    def export_conversation(self, conv_id: str, format: str = "json") -> Optional[str]:
        """Export conversation to JSON or Markdown.

        Args:
            conv_id: Conversation ID
            format: Export format ('json' or 'markdown')

        Returns:
            Exported content string or None
        """
        conv = self.load_conversation(conv_id)
        if not conv:
            return None

        if format == "json":
            return json.dumps(conv, indent=2, default=str)

        elif format == "markdown":
            md = f"# Conversation: {conv.get('title', 'Untitled')}\n\n"
            md += f"**Started:** {conv.get('created_at', 'Unknown')}\n"
            md += f"**Last Updated:** {conv.get('updated_at', 'Unknown')}\n"
            md += f"**Turns:** {conv.get('turn_count', 0)}\n"
            md += f"**Total Tokens:** {conv.get('total_tokens', 0)}\n\n"

            if conv.get("description"):
                md += f"**Description:** {conv['description']}\n\n"

            md += "## Conversation History\n\n"
            for turn in conv.get("turns", []):
                md += f"### Turn {turn['turn_number']}\n"
                md += f"**Q ({turn['timestamp']}):** {turn['query']}\n\n"
                md += f"**A:** {turn['answer']}\n\n"

                if turn.get("sources"):
                    md += "**Sources:**\n"
                    for src in turn["sources"]:
                        md += f"- {src.get('source', 'Unknown')}\n"
                    md += "\n"

            return md

        return None
