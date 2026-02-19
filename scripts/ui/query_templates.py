"""Query Templates for RAG Dashboard.

Provides reusable query templates for common governance, architecture, and code analysis questions.
Templates are stored in SQLite and can be saved/loaded from the dashboard.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QueryTemplateManager:
    """Manage query templates in SQLite database."""

    def __init__(self, db_path: Path):
        """Initialise template manager.

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
                CREATE TABLE IF NOT EXISTS query_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    description TEXT,
                    template_text TEXT NOT NULL,
                    k_results INTEGER DEFAULT 5,
                    temperature REAL DEFAULT 0.3,
                    code_aware BOOLEAN DEFAULT 1,
                    persona TEXT DEFAULT NULL,
                    tags TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_template_category ON query_templates(category)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_template_tags ON query_templates(tags)
            """
            )

            # Insert default templates
            self._insert_default_templates(conn)
            conn.commit()

    def _insert_default_templates(self, conn: sqlite3.Connection):
        """Insert default templates if they don't exist."""
        default_templates = [
            {
                "name": "What policies exist?",
                "category": "governance",
                "description": "Find all governance policies",
                "template_text": "What governance policies and procedures are documented?",
                "k_results": 5,
                "tags": "governance,policies",
            },
            {
                "name": "Security controls",
                "category": "governance",
                "description": "Find security control documentation",
                "template_text": "What are the security controls and requirements for {}?",
                "k_results": 7,
                "tags": "security,controls",
            },
            {
                "name": "Architecture overview",
                "category": "architecture",
                "description": "Get architecture and system design",
                "template_text": "Describe the architecture and main components of the {} system",
                "k_results": 8,
                "tags": "architecture,design",
            },
            {
                "name": "API endpoints",
                "category": "code",
                "description": "Find API endpoints and services",
                "template_text": "Show me the API endpoints and services for {} with their methods and parameters",
                "k_results": 10,
                "code_aware": True,
                "tags": "api,code,services",
            },
            {
                "name": "Database schema",
                "category": "code",
                "description": "Find database schema and models",
                "template_text": "Show the database schema and data models for the {} service",
                "k_results": 8,
                "code_aware": True,
                "tags": "database,schema,models",
            },
            {
                "name": "Authentication flow",
                "category": "security",
                "description": "Explain authentication and authorisation",
                "template_text": "Explain the authentication and authorisation flow for {}",
                "k_results": 6,
                "tags": "auth,security,flow",
            },
            {
                "name": "Deployment process",
                "category": "operations",
                "description": "Find deployment and release procedures",
                "template_text": "What is the deployment and release process for {}?",
                "k_results": 5,
                "tags": "deployment,operations,release",
            },
            {
                "name": "Error handling",
                "category": "code",
                "description": "Find error handling patterns",
                "template_text": "How are errors and exceptions handled in {} module?",
                "k_results": 6,
                "code_aware": True,
                "tags": "error,handling,exceptions",
            },
            # Academic templates
            {
                "name": "Find high-quality references",
                "category": "academic",
                "description": "Search for high-quality academic references (strict quality filtering)",
                "template_text": "Find high-quality academic references about {}",
                "k_results": 8,
                "temperature": 0.2,
                "code_aware": False,
                "persona": "supervisor",
                "tags": "academic,references,quality",
            },
            {
                "name": "Find recent research",
                "category": "academic",
                "description": "Search for recent research papers (broad discovery focus)",
                "template_text": "What are the recent research papers on {}?",
                "k_results": 15,
                "temperature": 0.5,
                "code_aware": False,
                "persona": "researcher",
                "tags": "academic,research,recent",
            },
            {
                "name": "Literature review",
                "category": "academic",
                "description": "Find papers for literature review (balanced comprehensive)",
                "template_text": "Provide a literature review on {} including key theories and findings",
                "k_results": 10,
                "temperature": 0.3,
                "code_aware": False,
                "persona": "assessor",
                "tags": "academic,literature,review",
            },
            {
                "name": "Find methodology",
                "category": "academic",
                "description": "Find research methodologies (discovery-focused)",
                "template_text": "What research methodologies have been used to study {}?",
                "k_results": 15,
                "temperature": 0.5,
                "code_aware": False,
                "persona": "researcher",
                "tags": "academic,methodology,research",
            },
            {
                "name": "Citation analysis",
                "category": "academic",
                "description": "Analyse citations and influences (quality and authority)",
                "template_text": "What are the most cited works on {} and their key contributions?",
                "k_results": 8,
                "temperature": 0.2,
                "code_aware": False,
                "persona": "supervisor",
                "tags": "academic,citations,analysis",
            },
        ]

        cursor = conn.cursor()
        for template in default_templates:
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO query_templates
                    (name, category, description, template_text, k_results, temperature, code_aware, persona, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        template["name"],
                        template["category"],
                        template.get("description", ""),
                        template["template_text"],
                        template.get("k_results", 5),
                        template.get("temperature", 0.3),
                        template.get("code_aware", False),
                        template.get("persona"),
                        template.get("tags", ""),
                    ),
                )
            except Exception:
                pass  # Template already exists

    def save_template(
        self,
        name: str,
        category: str,
        template_text: str,
        description: str = "",
        k_results: int = 5,
        temperature: float = 0.3,
        code_aware: bool = False,
        tags: str = "",
    ) -> bool:
        """Save a new query template.

        Args:
            name: Template name (must be unique)
            category: Template category
            template_text: Query template text (can include {})
            description: Template description
            k_results: Default k value
            temperature: Default temperature
            code_aware: Enable code-aware context
            tags: Comma-separated tags

        Returns:
            True if saved successfully
        """
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO query_templates
                    (name, category, description, template_text, k_results, temperature, code_aware, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        name,
                        category,
                        description,
                        template_text,
                        k_results,
                        temperature,
                        code_aware,
                        tags,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving template: {e}")
            return False

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template dict or None
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM query_templates WHERE name = ?
            """,
                (name,),
            )
            row = cursor.fetchone()

            if row:
                # Increment usage count
                cursor.execute(
                    """
                    UPDATE query_templates SET usage_count = usage_count + 1, last_used = ?
                    WHERE name = ?
                """,
                    (datetime.now(timezone.utc).isoformat(), name),
                )
                conn.commit()

                return dict(row)
            return None

    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all templates in a category.

        Args:
            category: Template category

        Returns:
            List of template dicts
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, description, category, template_text, k_results, temperature
                FROM query_templates
                WHERE category = ?
                ORDER BY usage_count DESC, name ASC
            """,
                (category,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_categories(self) -> List[str]:
        """Get all template categories.

        Returns:
            List of category names
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT category FROM query_templates
                ORDER BY category
            """
            )
            return [row[0] for row in cursor.fetchall()]

    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """Search templates by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching templates
        """
        search_term = f"%{query.lower()}%"
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, description, category, template_text, k_results, temperature
                FROM query_templates
                WHERE LOWER(name) LIKE ?
                   OR LOWER(description) LIKE ?
                   OR LOWER(tags) LIKE ?
                ORDER BY usage_count DESC, name ASC
            """,
                (search_term, search_term, search_term),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_template(self, name: str) -> bool:
        """Delete a template.

        Args:
            name: Template name

        Returns:
            True if deleted
        """
        try:
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                conn.execute("DELETE FROM query_templates WHERE name = ?", (name,))
                conn.commit()
            return True
        except Exception:
            return False

    def get_recent_templates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recently used templates.

        Args:
            limit: Number of templates to return

        Returns:
            List of templates
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, description, category, template_text, k_results, temperature
                FROM query_templates
                WHERE last_used IS NOT NULL
                ORDER BY last_used DESC
                LIMIT ?
            """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_popular_templates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most-used templates.

        Args:
            limit: Number of templates to return

        Returns:
            List of templates
        """
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, description, category, template_text, k_results, temperature, usage_count
                FROM query_templates
                WHERE usage_count > 0
                ORDER BY usage_count DESC
                LIMIT ?
            """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
