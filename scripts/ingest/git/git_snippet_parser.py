"""Shared code parsing and extraction utilities for Git repositories.

Provides platform-agnostic utilities for:
- Language detection and syntax highlighting
- Code chunking with semantic awareness
- Dependency extraction
- Metadata generation for code snippets
- Comment/docstring extraction

Used by all Git connector implementations (Bitbucket, GitHub, GitLab, Azure).

Note:
    This module intentionally stays separate from `code_parser.py`.
    It provides generic snippet and dependency extraction used by the Git
    framework tests, while `code_parser.py` focuses on higher-level service
    and architecture metadata for ingestion.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from scripts.utils.logger import create_module_logger

get_logger, _ = create_module_logger("ingest")
logger = get_logger()


class CodeLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    KOTLIN = "kotlin"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    SHELL = "shell"
    UNKNOWN = "unknown"


# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": CodeLanguage.PYTHON,
    ".js": CodeLanguage.JAVASCRIPT,
    ".jsx": CodeLanguage.JAVASCRIPT,
    ".ts": CodeLanguage.TYPESCRIPT,
    ".tsx": CodeLanguage.TYPESCRIPT,
    ".java": CodeLanguage.JAVA,
    ".cs": CodeLanguage.CSHARP,
    ".cpp": CodeLanguage.CPP,
    ".cc": CodeLanguage.CPP,
    ".cxx": CodeLanguage.CPP,
    ".h": CodeLanguage.CPP,
    ".hpp": CodeLanguage.CPP,
    ".go": CodeLanguage.GO,
    ".rs": CodeLanguage.RUST,
    ".rb": CodeLanguage.RUBY,
    ".php": CodeLanguage.PHP,
    ".kt": CodeLanguage.KOTLIN,
    ".sql": CodeLanguage.SQL,
    ".yaml": CodeLanguage.YAML,
    ".yml": CodeLanguage.YAML,
    ".json": CodeLanguage.JSON,
    ".xml": CodeLanguage.XML,
    ".html": CodeLanguage.HTML,
    ".htm": CodeLanguage.HTML,
    ".css": CodeLanguage.CSS,
    ".md": CodeLanguage.MARKDOWN,
    ".sh": CodeLanguage.SHELL,
    ".bash": CodeLanguage.SHELL,
}


@dataclass
class CodeChunk:
    """Represents a semantic chunk of code."""

    code: str
    language: CodeLanguage
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "module", "block"
    name: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: Optional[Set[str]] = None
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.metadata is None:
            self.metadata = {}


class CodeParser:
    """Parse and extract semantic units from code files."""

    @staticmethod
    def detect_language(file_path: str) -> CodeLanguage:
        """Detect programming language from file extension.

        Args:
            file_path: Path to code file

        Returns:
            CodeLanguage enum value
        """
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext, CodeLanguage.UNKNOWN)

    @staticmethod
    def extract_functions(code: str, language: CodeLanguage) -> List[CodeChunk]:
        """Extract function definitions from code.

        Args:
            code: Source code content
            language: Programming language

        Returns:
            List of CodeChunk objects representing functions
        """
        chunks = []

        if language == CodeLanguage.PYTHON:
            chunks.extend(CodeParser._extract_python_functions(code))
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            chunks.extend(CodeParser._extract_js_functions(code))
        elif language == CodeLanguage.JAVA:
            chunks.extend(CodeParser._extract_java_methods(code))
        elif language == CodeLanguage.CSHARP:
            chunks.extend(CodeParser._extract_csharp_methods(code))
        elif language == CodeLanguage.GO:
            chunks.extend(CodeParser._extract_go_functions(code))

        return chunks

    @staticmethod
    def _extract_python_functions(code: str) -> List[CodeChunk]:
        """Extract Python function definitions."""
        chunks = []
        lines = code.split("\n")

        # Simple regex for function definitions (doesn't handle nested functions perfectly)
        pattern = r"^\s*(?:async\s+)?def\s+(\w+)\s*\("

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                func_name = match.group(1)
                # Extract docstring if present
                docstring = CodeParser._extract_python_docstring(lines, i + 1)

                chunks.append(
                    CodeChunk(
                        code=line,
                        language=CodeLanguage.PYTHON,
                        start_line=i + 1,
                        end_line=i + 1,  # Simplified: single line
                        chunk_type="function",
                        name=func_name,
                        docstring=docstring,
                    )
                )

        return chunks

    @staticmethod
    def _extract_python_docstring(lines: List[str], start_idx: int) -> Optional[str]:
        """Extract Python docstring following a function definition."""
        if start_idx >= len(lines):
            return None

        line = lines[start_idx].strip()
        if not (line.startswith('"""') or line.startswith("'''")):
            return None

        quote = '"""' if line.startswith('"""') else "'''"
        if line.count(quote) == 2:  # Single-line docstring
            return line[3:-3].strip()

        # Multi-line docstring
        docstring = []
        for i in range(start_idx + 1, len(lines)):
            if quote in lines[i]:
                break
            docstring.append(lines[i])

        return "\n".join(docstring).strip() if docstring else None

    @staticmethod
    def _extract_js_functions(code: str) -> List[CodeChunk]:
        """Extract JavaScript/TypeScript function definitions."""
        chunks = []
        # Simple pattern for function declarations
        pattern = r"(?:async\s+)?(?:function\s+)?(\w+)\s*(?:\([^)]*\))?\s*(?::\s*\w+)?\s*\{"

        for match in re.finditer(pattern, code):
            func_name = match.group(1)
            line_num = code[: match.start()].count("\n") + 1

            chunks.append(
                CodeChunk(
                    code=match.group(0),
                    language=CodeLanguage.JAVASCRIPT,
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="function",
                    name=func_name,
                )
            )

        return chunks

    @staticmethod
    def _extract_java_methods(code: str) -> List[CodeChunk]:
        """Extract Java method definitions."""
        chunks = []
        # Pattern for Java method signatures
        pattern = r"((?:public|private|protected|static|final)\s+)*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{"

        for match in re.finditer(pattern, code):
            method_name = match.group(2)
            line_num = code[: match.start()].count("\n") + 1

            chunks.append(
                CodeChunk(
                    code=match.group(0),
                    language=CodeLanguage.JAVA,
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="method",
                    name=method_name,
                )
            )

        return chunks

    @staticmethod
    def _extract_csharp_methods(code: str) -> List[CodeChunk]:
        """Extract C# method definitions."""
        chunks = []
        # Pattern for C# method signatures
        pattern = r"(?:public|private|protected|static|async)\s+(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{"

        for match in re.finditer(pattern, code):
            method_name = match.group(1)
            line_num = code[: match.start()].count("\n") + 1

            chunks.append(
                CodeChunk(
                    code=match.group(0),
                    language=CodeLanguage.CSHARP,
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="method",
                    name=method_name,
                )
            )

        return chunks

    @staticmethod
    def _extract_go_functions(code: str) -> List[CodeChunk]:
        """Extract Go function definitions."""
        chunks = []
        # Pattern for Go function definitions
        pattern = r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*\{"

        for match in re.finditer(pattern, code):
            func_name = match.group(1)
            line_num = code[: match.start()].count("\n") + 1

            chunks.append(
                CodeChunk(
                    code=match.group(0),
                    language=CodeLanguage.GO,
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="function",
                    name=func_name,
                )
            )

        return chunks

    @staticmethod
    def extract_imports(code: str, language: CodeLanguage) -> Set[str]:
        """Extract import/require statements from code.

        Args:
            code: Source code content
            language: Programming language

        Returns:
            Set of imported module names
        """
        imports = set()

        if language == CodeLanguage.PYTHON:
            # Python: import X, from X import Y
            patterns = [
                r"^(?:from|import)\s+([\w.]+)",
                r"^from\s+[\w.]+\s+import\s+([\w, ]+)",
            ]
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            # JS/TS: import X from Y, require(X)
            patterns = [
                r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
                r"require\(['\"]([^'\"]+)['\"]\)",
            ]
        elif language == CodeLanguage.JAVA:
            # Java: import X.Y.Z;
            patterns = [r"^import\s+([\w.]+);"]
        elif language == CodeLanguage.CSHARP:
            # C#: using X;
            patterns = [r"^using\s+([\w.]+);"]
        elif language == CodeLanguage.GO:
            # Go: import "X" or import ( "X" )
            patterns = [r'import\s+["\']([^"\']+)["\']']
        else:
            return imports

        for pattern in patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                import_name = match.group(1)
                # Clean up import names (remove spaces, handle multiple imports)
                for name in re.split(r"[,\s]+", import_name):
                    name = name.strip()
                    if name and not name.startswith("("):
                        imports.add(name)

        return imports

    @staticmethod
    def extract_classes(code: str, language: CodeLanguage) -> List[CodeChunk]:
        """Extract class definitions from code.

        Args:
            code: Source code content
            language: Programming language

        Returns:
            List of CodeChunk objects representing classes
        """
        chunks = []

        if language == CodeLanguage.PYTHON:
            pattern = r"^class\s+(\w+)\s*(?:\([^)]*\))?:"
            for match in re.finditer(pattern, code, re.MULTILINE):
                class_name = match.group(1)
                line_num = code[: match.start()].count("\n") + 1
                chunks.append(
                    CodeChunk(
                        code=match.group(0),
                        language=CodeLanguage.PYTHON,
                        start_line=line_num,
                        end_line=line_num,
                        chunk_type="class",
                        name=class_name,
                    )
                )

        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            pattern = r"class\s+(\w+)\s*(?:extends\s+\w+)?\s*\{"
            for match in re.finditer(pattern, code):
                class_name = match.group(1)
                line_num = code[: match.start()].count("\n") + 1
                chunks.append(
                    CodeChunk(
                        code=match.group(0),
                        language=language,
                        start_line=line_num,
                        end_line=line_num,
                        chunk_type="class",
                        name=class_name,
                    )
                )

        elif language in (CodeLanguage.JAVA, CodeLanguage.CSHARP):
            pattern = r"(?:public\s+)?class\s+(\w+)\s*(?:extends|implements|:\s*[^{]+)?\s*\{"
            for match in re.finditer(pattern, code):
                class_name = match.group(1)
                line_num = code[: match.start()].count("\n") + 1
                chunks.append(
                    CodeChunk(
                        code=match.group(0),
                        language=language,
                        start_line=line_num,
                        end_line=line_num,
                        chunk_type="class",
                        name=class_name,
                    )
                )

        return chunks


class DependencyExtractor:
    """Extract and analyse code dependencies."""

    @staticmethod
    def extract_dependencies_from_files(
        files: Dict[str, str], language: CodeLanguage
    ) -> Dict[str, Set[str]]:
        """Extract all dependencies from multiple code files.

        Args:
            files: Mapping of file paths to file contents
            language: Programming language

        Returns:
            Mapping of file paths to sets of imported modules
        """
        dependencies = {}

        for file_path, content in files.items():
            imports = CodeParser.extract_imports(content, language)
            if imports:
                dependencies[file_path] = imports

        return dependencies

    @staticmethod
    def extract_requirements_file(content: str, language: CodeLanguage) -> Set[str]:
        """Extract package dependencies from requirements/manifest files.

        Args:
            content: Contents of requirements/package.json/pom.xml/etc
            language: Type of requirements file

        Returns:
            Set of required packages
        """
        packages = set()

        if language == CodeLanguage.PYTHON:
            # Parse requirements.txt format
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Handle version specifiers: pkg>=1.0, pkg[extra]==1.0, etc.
                pkg_name = re.split(r"[<>=!#;[]", line)[0].strip()
                if pkg_name:
                    packages.add(pkg_name)

        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            # Parse package.json format (simplified)
            import json

            try:
                data = json.loads(content)
                for key in ["dependencies", "devDependencies", "peerDependencies"]:
                    if key in data:
                        packages.update(data[key].keys())
            except json.JSONDecodeError:
                pass

        elif language == CodeLanguage.JAVA:
            # Parse pom.xml format (simplified)
            for match in re.finditer(
                r"<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>", content
            ):
                group_id = match.group(1)
                artifact_id = match.group(2)
                packages.add(f"{group_id}:{artifact_id}")

        elif language == CodeLanguage.CSHARP:
            # Parse .csproj or packages.config format
            for match in re.finditer(r'<PackageReference\s+Include="([^"]+)"', content):
                packages.add(match.group(1))

        elif language == CodeLanguage.GO:
            # Parse go.mod format
            in_require = False
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("require"):
                    in_require = True
                    continue
                if in_require and line == ")":
                    break
                if in_require and line:
                    pkg = line.split()[0]
                    if pkg:
                        packages.add(pkg)

        return packages
