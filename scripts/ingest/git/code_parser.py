"""Code-aware parser for extracting dependencies, exports, and service definitions.

This module provides language-aware parsing for multi-language codebases to extract:
- External dependencies (Maven, Gradle, npm coordinates)
- Internal service calls and cross-repo references
- Public API exports (classes, functions, services)
- Configuration patterns (endpoints, message queues, databases)
- Architectural patterns (controllers, processors, listeners)

Supports: Java, Groovy, Mule, XML configs, pom.xml, build.gradle

The parser integrates with the ingestion pipeline to enrich chunk metadata with
dependency and architectural information for migration analysis and impact tracking.

TODO: Map imports and function calls to actual services/classes for better internal call graph construction
TODO: Capture enough endpoint detail to be able to build both dataflow graphs, control flow graphs, and sequence diagrams of service interactions and API dependencies
TODO: Enhance export detection to better identify public APIs and service interfaces, especially in dynamic languages like Groovy
TODO: Improve handling of configuration files (e.g., Spring Boot properties, YAML) to extract more meaningful metadata
TODO: Add support for more languages and frameworks (eg Express, ASP.NET)
TODO: Support for semantic contextualising markdown files describing code/services, extracting service names, dependencies, and architectural patterns from documentation
TODO: Semantic capture of code comments and Javadoc/KDoc for better understanding of service responsibilities and dependencies
TODO: Implement confidence scoring for extracted metadata based on heuristics and patterns

Example Usage:
    from scripts.ingest.git.code_parser import CodeParser

    parser = CodeParser()
    result = parser.parse_file("path/to/Service.java")

    print(result.external_dependencies)  # ['com.example:rest-client:1.0']
    print(result.internal_calls)         # ['PaymentService', 'AuditService']
    print(result.exports)                # ['OrderProcessor', 'OrderListener']
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

from scripts.utils.logger import create_module_logger

get_logger, _ = create_module_logger("ingest")

logger = get_logger()


class FileType(Enum):
    """Supported file types for parsing."""

    JAVA = "java"
    GROOVY = "groovy"
    GRADLE = "gradle"
    MAVEN_POM = "pom.xml"
    MULE_XML = "mule.xml"
    XML = "xml"
    PROPERTIES = "properties"
    YAML = "yaml"
    CSHARP = "cs"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    POWERSHELL = "powershell"
    TERRAFORM = "terraform"
    SQL = "sql"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """Result of parsing a code file."""

    file_path: str
    file_type: FileType
    language: str

    # Dependencies
    external_dependencies: List[str] = field(
        default_factory=list
    )  # e.g., "com.example:service:1.0"
    internal_imports: List[str] = field(default_factory=list)  # e.g., "com.mycompany.auth"

    # Service references
    internal_calls: List[str] = field(default_factory=list)  # Services/classes called

    # Public APIs / Exports
    exports: List[str] = field(default_factory=list)  # Classes, functions, services

    # Architecture patterns
    service_name: Optional[str] = None  # e.g., "PaymentService"
    service_type: Optional[str] = None  # e.g., "controller", "processor", "listener"

    # Configuration points
    endpoints: List[str] = field(default_factory=list)  # REST endpoints, message topics
    database_refs: List[str] = field(default_factory=list)  # Database names, JPA entities
    message_queues: List[str] = field(default_factory=list)  # JMS queues, Kafka topics
    external_services: List[str] = field(default_factory=list)  # External service URLs

    # Metadata
    confidence: float = 1.0  # Confidence score for extraction
    errors: List[str] = field(default_factory=list)  # Parsing errors encountered

    def to_dict(self) -> Dict:
        """Convert to dictionary for ChromaDB metadata storage."""
        return {
            "file_type": self.file_type.value,
            "language": self.language,
            "external_dependencies": self.external_dependencies,
            "internal_imports": self.internal_imports,
            "internal_calls": self.internal_calls,
            "exports": self.exports,
            "service_name": self.service_name or "",
            "service_type": self.service_type or "",
            "endpoints": self.endpoints,
            "database_refs": self.database_refs,
            "message_queues": self.message_queues,
            "external_services": self.external_services,
            "confidence": self.confidence,
        }


class CodeParser:
    """Language-aware code parser for dependency and architecture extraction."""

    # Common English stopwords to filter out from exports/endpoints
    # These often appear in Camel route URIs or comments and should not be treated as valid exports
    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "which",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "can",
        "must",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "as",
        "with",
        "about",
        "into",
        "up",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "where",
        "when",
        "why",
        "how",
        "what",
        "who",
        "whom",
        "whose",
        "your",
        "our",
        "their",
        "my",
        "his",
        "her",
        "its",
        "we",
        "me",
        "him",
        "her",
        "you",
        "he",
        "she",
        "it",
        "i",
        "not",
        "no",
        "nor",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "am",
        "pm",
        "get",
        "put",
        "post",
        "delete",
        "patch",
        "head",
        "options",
        "trace",
        "connect",
    }

    # Patterns for Java/Groovy
    JAVA_IMPORT_PATTERN = re.compile(r"^\s*import\s+([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)
    JAVA_PACKAGE_PATTERN = re.compile(r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)

    # Class/Interface definitions
    CLASS_PATTERN = re.compile(
        r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
    )
    INTERFACE_PATTERN = re.compile(
        r"(?:public\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
    )
    ENUM_PATTERN = re.compile(r"(?:public\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

    # Method/Function definitions
    METHOD_PATTERN = re.compile(
        r"(?:public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )

    # Service annotations and patterns
    ANNOTATION_PATTERNS = {
        "controller": re.compile(r"@(?:Rest)?Controller|@RequestMapping"),
        "service": re.compile(r"@Service\b"),
        "component": re.compile(r"@Component\b"),
        "processor": re.compile(r"@Processor"),
        "listener": re.compile(r"@(?:Jms)?Listener|@RabbitListener"),
        "repository": re.compile(r"@Repository"),
    }

    # REST endpoint patterns
    ENDPOINT_PATTERNS = {
        "mapping": re.compile(r'@(?:Request)?Mapping\("([^"]+)"'),
        "get": re.compile(r'@GetMapping\("([^"]+)"'),
        "post": re.compile(r'@PostMapping\("([^"]+)"'),
        "put": re.compile(r'@PutMapping\("([^"]+)"'),
        "delete": re.compile(r'@DeleteMapping\("([^"]+)"'),
    }

    # Service call patterns
    SERVICE_CALL_PATTERNS = [
        re.compile(r"(@Autowired|@Inject)\s+(?:private\s+)?(\w+)\s+(\w+)"),  # Injected dependencies
        re.compile(r'restTemplate\.(?:get|post|put|delete)\("([^"]+)"'),  # RestTemplate calls
        re.compile(r'template\.convertAndSend\("([^"]+)"'),  # JMS template sends
    ]

    # Mule patterns
    MULE_PATTERNS = {
        "flow": re.compile(r'<flow\s+name="([^"]+)"'),
        "connector": re.compile(r"<(?:http|jms|kafka|db):(\w+)"),
        "endpoint": re.compile(r'path="([^"]*)"'),
    }

    # Camel patterns (Java/Groovy DSL and XML) - handle both single and double quotes
    CAMEL_PATTERNS = {
        "route": re.compile(
            r'(?:from|\.from)\(["\']([^"\']*)["\']?\)', re.MULTILINE
        ),  # from("endpoint") or from('endpoint')
        "to": re.compile(
            r'(?:\.to|\.to)\(["\']([^"\']*)["\']?\)', re.MULTILINE
        ),  # .to("endpoint") or .to('endpoint')
        "process": re.compile(
            r"\.process\s*\(\s*([A-Za-z_]\w*)", re.MULTILINE
        ),  # .process(Processor)
        "route_id": re.compile(
            r'\.routeId\(["\']([^"\']*)["\']?\)', re.MULTILINE
        ),  # .routeId("name") or .routeId('name')
        "xml_route": re.compile(r'<route\s+id="([^"]+)"', re.MULTILINE),  # XML routes
        "xml_from": re.compile(r'<from\s+uri="([^"]+)"', re.MULTILINE),  # XML from
        "xml_to": re.compile(r'<to\s+uri="([^"]+)"', re.MULTILINE),  # XML to
        "component": re.compile(r"@(?:camel)?Component\b", re.IGNORECASE),  # Camel component bean
    }

    # Spring Boot Camel properties
    CAMEL_PROPERTIES_PATTERNS = [
        re.compile(
            r"camel\.component\.([a-z0-9]+)\.([a-z0-9.]+)\s*=\s*(.+)"
        ),  # camel.component.http.*
        re.compile(r"camel\.route\.([a-z0-9.]+)\s*=\s*(.+)"),  # camel.route.*
    ]

    def __init__(self):
        """Initialise the code parser."""
        self.logger = get_logger()

    def _is_valid_export(self, value: str) -> bool:
        """Check if a value is a valid export (filters stopwords, short strings, etc).

        Args:
            value: Export/endpoint value to validate

        Returns:
            True if value should be kept, False if it's a stopword or invalid
        """
        if not value or len(value.strip()) == 0:
            return False

        # Strip whitespace and lowercase for checking
        clean_value = value.strip().lower()

        # Reject single characters
        if len(clean_value) == 1:
            return False

        # Reject pure stopwords
        if clean_value in self.STOPWORDS:
            return False

        # Reject values that are only stopwords and hyphens/underscores
        tokens = re.split(r"[-_/:]", clean_value)
        non_stopword_tokens = [t for t in tokens if t and t not in self.STOPWORDS]
        if not non_stopword_tokens:
            return False

        return True

    @staticmethod
    def _extract_service_name_from_path(file_path: str) -> Optional[str]:
        """Extract service name from file path as fallback.

        Examples:
            'src/main/java/com/example/PaymentService.java' -> 'PaymentService'
            'src/main/groovy/services/AuthService.groovy' -> 'AuthService'
            'config/database-config.properties' -> 'database-config'

        Args:
            file_path: Path to the file

        Returns:
            Service name derived from filename (without extension) or None
        """
        path = Path(file_path)
        stem = path.stem  # Filename without extension

        # Remove common prefixes/suffixes
        service_name = stem
        for prefix in ["test_", "Test"]:
            if service_name.startswith(prefix):
                service_name = service_name[len(prefix) :]

        for suffix in ["Test", "Tests", "Spec", "Specs", "Config", "Configuration"]:
            if service_name.endswith(suffix) and service_name != suffix:
                service_name = service_name[: -len(suffix)]

        # Only return if meaningful (not empty after cleanup and longer than 2 chars)
        if service_name and len(service_name) > 2:
            return service_name

        return None

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a code file and extract metadata.

        Args:
            file_path: Path to the file to parse.

        Returns:
            ParseResult containing extracted metadata.
        """
        path = Path(file_path)

        # Determine file type
        file_type = self._detect_file_type(path)

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                file_type=file_type,
                language="unknown",
                errors=[f"Read error: {str(e)}"],
            )

        # Route to appropriate parser
        # TODO: Refactor to have separate parser classes for each file type for better modularity and testability
        # TODO: Add markdown parser that can extract service names, dependencies, and architectural patterns from documentation files
        if file_type == FileType.JAVA or file_type == FileType.GROOVY:
            return self._parse_java_groovy(file_path, content, file_type)
        elif file_type == FileType.MAVEN_POM:
            return self._parse_maven_pom(file_path, content)
        elif file_type == FileType.GRADLE:
            return self._parse_gradle(file_path, content)
        elif file_type == FileType.MULE_XML:
            return self._parse_mule_xml(file_path, content)
        elif file_type == FileType.XML:
            return self._parse_generic_xml(file_path, content)
        elif file_type == FileType.PROPERTIES:
            return self._parse_properties(file_path, content)
        elif file_type == FileType.CSHARP:
            return self._parse_csharp(file_path, content)
        elif file_type in (FileType.JAVASCRIPT, FileType.TYPESCRIPT):
            return self._parse_javascript_like(file_path, content, file_type)
        elif file_type == FileType.HTML:
            return self._parse_html(file_path, content)
        elif file_type == FileType.POWERSHELL:
            return ParseResult(file_path=file_path, file_type=file_type, language="powershell")
        elif file_type == FileType.TERRAFORM:
            return ParseResult(file_path=file_path, file_type=file_type, language="terraform")
        elif file_type == FileType.SQL:
            return self._parse_sql(file_path, content)
        else:
            return ParseResult(
                file_path=file_path, file_type=file_type, language="unknown", confidence=0.0
            )

    def _detect_file_type(self, path: Path) -> FileType:
        """Detect file type from extension and name."""
        suffix = path.suffix.lower()
        name = path.name.lower()

        if name == "pom.xml":
            return FileType.MAVEN_POM
        elif name.endswith("-mule.xml"):
            return FileType.MULE_XML
        elif suffix == ".xml":
            return FileType.XML
        elif suffix == ".java":
            return FileType.JAVA
        elif suffix in [".groovy", ".gvy", ".gsp"]:
            return FileType.GROOVY
        elif suffix == ".gradle":
            return FileType.GRADLE
        elif suffix in [".properties", ".yml", ".yaml"]:
            return FileType.PROPERTIES
        elif suffix == ".cs":
            return FileType.CSHARP
        elif suffix in [".js", ".jsx"]:
            return FileType.JAVASCRIPT
        elif suffix in [".ts", ".tsx"]:
            return FileType.TYPESCRIPT
        elif suffix in [".html", ".htm"]:
            return FileType.HTML
        elif suffix in [".ps1", ".psm1"]:
            return FileType.POWERSHELL
        elif suffix == ".tf":
            return FileType.TERRAFORM
        elif suffix == ".sql":
            return FileType.SQL
        else:
            return FileType.UNKNOWN

    def _parse_java_groovy(self, file_path: str, content: str, file_type: FileType) -> ParseResult:
        """Parse Java or Groovy file for dependencies and services."""
        result = ParseResult(
            file_path=file_path,
            file_type=file_type,
            language="java" if file_type == FileType.JAVA else "groovy",
        )

        try:
            # Extract package
            package_match = self.JAVA_PACKAGE_PATTERN.search(content)
            if package_match:
                package = package_match.group(1)
                result.internal_imports.append(package)

            # Extract imports
            for match in self.JAVA_IMPORT_PATTERN.finditer(content):
                import_path = match.group(1)
                if import_path.startswith("com.") or import_path.startswith("org."):
                    result.internal_imports.append(import_path)

            # Extract public classes/interfaces/enums
            for match in self.CLASS_PATTERN.finditer(content):
                export = match.group(1)
                if self._is_valid_export(export):
                    result.exports.append(export)
            for match in self.INTERFACE_PATTERN.finditer(content):
                export = match.group(1)
                if self._is_valid_export(export):
                    result.exports.append(export)
            for match in self.ENUM_PATTERN.finditer(content):
                export = match.group(1)
                if self._is_valid_export(export):
                    result.exports.append(export)

            # Detect service type from annotations
            service_type = self._detect_service_type(content)
            if service_type:
                result.service_type = service_type
                # Try to extract service name from first class
                class_match = self.CLASS_PATTERN.search(content)
                if class_match:
                    result.service_name = class_match.group(1)

            # Extract REST endpoints
            endpoints = self._extract_endpoints(content)
            result.endpoints.extend(endpoints)

            # Extract injected dependencies and service calls
            for pattern in self.SERVICE_CALL_PATTERNS:
                for match in pattern.finditer(content):
                    if match.lastindex and match.lastindex >= 2:
                        # This is an injected dependency
                        dep_name = match.group(match.lastindex)
                        if dep_name and dep_name[0].isupper():
                            result.internal_calls.append(dep_name)

            # Extract database references (actual DB connections, not table mappings)
            # Look for DataSource bean names
            datasource_beans = re.findall(
                r"@(?:Autowired|Inject)\s+(?:private\s+)?DataSource\s+(\w+)", content
            )
            result.database_refs.extend(datasource_beans)

            # Look for JDBC URL patterns (extract DB type/service name)
            jdbc_urls = re.findall(r'jdbc:(\w+):[^\s;"]+', content)
            result.database_refs.extend([f"jdbc-{db}" for db in set(jdbc_urls)])

            # Look for database property references
            db_props = re.findall(
                r"\$\{([^}]*(?:database|db|datasource|oracle|postgres|mysql)[^}]*)\}",
                content,
                re.IGNORECASE,
            )
            result.database_refs.extend([prop for prop in db_props if self._is_valid_export(prop)])

            # Extract message queue references
            queue_refs = re.findall(
                r'@(?:JmsListener|RabbitListener|KafkaListener)\s*\(\s*(?:destination|topics?)\s*=\s*"([^"]+)"',
                content,
            )
            result.message_queues.extend(queue_refs)

            # Extract external service URLs
            urls = re.findall(r'(?:url|baseUrl|endpoint)\s*=\s*"(https?://[^"]+)"', content)
            result.external_services.extend(urls)

            # ===== CAMEL-SPECIFIC PARSING =====
            # Extract Camel routes (Java/Groovy DSL)
            camel_from_endpoints = re.findall(self.CAMEL_PATTERNS["route"], content)
            result.endpoints.extend(camel_from_endpoints)

            camel_to_endpoints = re.findall(self.CAMEL_PATTERNS["to"], content)
            result.endpoints.extend(camel_to_endpoints)

            # Extract Camel route IDs
            route_ids = re.findall(self.CAMEL_PATTERNS["route_id"], content)
            if route_ids:
                result.exports.extend([f"CamelRoute:{rid}" for rid in route_ids])

            # Extract Camel processors
            processors = re.findall(self.CAMEL_PATTERNS["process"], content)
            result.internal_calls.extend(processors)

            # Detect if this is a Camel component
            if self.CAMEL_PATTERNS["component"].search(content):
                result.service_type = result.service_type or "camel-component"

            # Fallback: if we found exports but no name, use first export as service name
            if result.exports and not result.service_name:
                result.service_name = result.exports[0]

            # Final fallback: extract service name from file path
            if not result.service_name:
                result.service_name = self._extract_service_name_from_path(file_path) or ""

        except Exception as e:
            self.logger.warning(f"Error parsing {file_path}: {e}")

            result.errors.append(str(e))
            result.confidence = 0.7  # Partial success

        return result

    def _detect_service_type(self, content: str) -> Optional[str]:
        """Detect service type from annotations."""
        for service_type, pattern in self.ANNOTATION_PATTERNS.items():
            if pattern.search(content):
                return service_type
        return None

    def _extract_endpoints(self, content: str) -> List[str]:
        """Extract REST endpoints from annotations."""
        endpoints = []
        for pattern in self.ENDPOINT_PATTERNS.values():
            for match in pattern.finditer(content):
                endpoints.append(match.group(1))
        return endpoints

    def _parse_maven_pom(self, file_path: str, content: str) -> ParseResult:
        """Parse Maven pom.xml for dependencies."""
        result = ParseResult(file_path=file_path, file_type=FileType.MAVEN_POM, language="xml")

        try:
            root = ET.fromstring(content)
            # Define namespace - handle both with and without namespace
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}

            # Extract project name/service name
            name_elem = root.find("m:name", ns)
            if name_elem is None:
                name_elem = root.find("name")
            if name_elem is not None and name_elem.text:
                result.service_name = name_elem.text.strip()

            # Extract dependencies - try with namespace first, then without
            deps = root.findall(".//m:dependency", ns)
            if not deps:
                deps = root.findall(".//dependency")

            for dep in deps:
                # Try with namespace
                group = dep.find("m:groupId", ns)
                if group is None:
                    group = dep.find("groupId")

                artifact = dep.find("m:artifactId", ns)
                if artifact is None:
                    artifact = dep.find("artifactId")

                version = dep.find("m:version", ns)
                if version is None:
                    version = dep.find("version")

                if group is not None and artifact is not None:
                    group_text = (group.text or "").strip()
                    artifact_text = (artifact.text or "").strip()
                    version_text = (version.text or "?").strip() if version is not None else "?"

                    if group_text and artifact_text:
                        dep_str = f"{group_text}:{artifact_text}:{version_text}"
                        result.external_dependencies.append(dep_str)

        except Exception as e:
            self.logger.warning(f"Error parsing Maven pom {file_path}: {e}")
            result.errors.append(str(e))
            result.confidence = 0.6

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_gradle(self, file_path: str, content: str) -> ParseResult:
        """Parse Gradle build.gradle for dependencies (Maven and Camel Spring Boot)."""
        result = ParseResult(file_path=file_path, file_type=FileType.GRADLE, language="groovy")

        try:
            # Extract dependencies block - handle single and double quotes and various scopes
            # Matches: implementation 'org.example:name:1.0' or runtimeOnly "org.example:name:1.0"
            dep_pattern = re.compile(
                r"(?:api|implementation|compile|testImplementation|runtimeOnly)\s+['\"]([^'\"]+)['\"]",
                re.MULTILINE,
            )

            for match in dep_pattern.finditer(content):
                dep = match.group(1).strip()
                if dep and ":" in dep:  # Ensure it looks like a gradle dependency
                    result.external_dependencies.append(dep)

            # ===== CAMEL-SPECIFIC GRADLE PARSING =====
            # Categorise Camel Spring Boot dependencies
            camel_starters = [
                dep
                for dep in result.external_dependencies
                if "camel" in dep.lower() and "springboot" in dep.lower()
            ]
            if camel_starters:
                result.service_type = result.service_type or "camel-springboot"
                # Add as annotations for tracking
                for starter in camel_starters:
                    result.exports.append(f"CamelStarter:{starter.split(':')[1]}")

        except Exception as e:
            self.logger.warning(f"Error parsing Gradle {file_path}: {e}")
            result.errors.append(str(e))
            result.confidence = 0.7

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_mule_xml(self, file_path: str, content: str) -> ParseResult:
        """Parse Mule XML configuration file."""
        result = ParseResult(file_path=file_path, file_type=FileType.MULE_XML, language="xml")

        try:
            # Extract flow names (service boundaries)
            flows = re.findall(self.MULE_PATTERNS["flow"], content)
            result.exports.extend(flows)

            # Extract connectors (service integrations)
            connectors = re.findall(self.MULE_PATTERNS["connector"], content)
            result.internal_calls.extend(set(connectors))

            # Extract HTTP endpoints
            endpoints = re.findall(self.MULE_PATTERNS["endpoint"], content)
            result.endpoints.extend(endpoints)

            # Extract message flow references
            flow_refs = re.findall(r'<flow-ref\s+name="([^"]+)"', content)
            result.internal_calls.extend(flow_refs)

            # Extract HTTP outbound endpoints (service calls)
            http_urls = re.findall(r'<http:request\s+(?:[^>]*?)path="([^"]+)"', content)
            result.external_services.extend(http_urls)

            # Detect service name from file name
            service_match = re.search(r"(?:^|\W)(\w+)-mule\.xml$", file_path)
            if service_match:
                result.service_name = service_match.group(1)
                result.service_type = "mule-flow"

        except Exception as e:
            self.logger.warning(f"Error parsing Mule XML {file_path}: {e}")
            result.errors.append(str(e))
            result.confidence = 0.75

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_generic_xml(self, file_path: str, content: str) -> ParseResult:
        """Parse generic XML configuration file (including Camel routes)."""
        result = ParseResult(file_path=file_path, file_type=FileType.XML, language="xml")

        try:
            # Check if this is a Camel route file
            if "<route" in content or "<camelContext" in content:
                return self._parse_camel_xml(file_path, content)

            # Extract bean names (Spring context)
            beans = re.findall(r'<bean\s+id="([^"]+)"', content)
            result.exports.extend(beans)

            # Extract bean class references
            classes = re.findall(r'class="([^"]+)"', content)
            result.internal_calls.extend(classes)

        except Exception as e:
            self.logger.warning(f"Error parsing XML {file_path}: {e}")
            result.errors.append(str(e))

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_csharp(self, file_path: str, content: str) -> ParseResult:
        """Lightweight C# parser to avoid UNKNOWN language and capture basics."""
        result = ParseResult(file_path=file_path, file_type=FileType.CSHARP, language="csharp")

        try:
            # Namespace as internal import for grouping
            ns_match = re.search(r"^\s*namespace\s+([A-Za-z0-9_.]+)", content, re.MULTILINE)
            if ns_match:
                result.internal_imports.append(ns_match.group(1))

            # Classes / interfaces
            for match in self.CLASS_PATTERN.finditer(content):
                result.exports.append(match.group(1))

            # Field injections (readonly services/clients)
            for match in re.finditer(
                r"(?:private|protected|internal)\s+(?:readonly\s+)?([A-Z][A-Za-z0-9_]*)\s+[A-Za-z_][A-Za-z0-9_]*\s*;",
                content,
            ):
                dep = match.group(1)
                result.internal_calls.append(dep)

            # Constructor injection: capture parameter types
            for match in re.finditer(r"\b[A-Z][A-Za-z0-9_]*\s*\(([^)]*)\)", content):
                params = match.group(1)
                for p in params.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    # Split type and name
                    parts = p.split()
                    if len(parts) >= 1 and parts[0][0].isupper():
                        result.internal_calls.append(parts[0])

            # ASP.NET style endpoints
            for match in re.finditer(r"\[Http(?:Get|Post|Put|Delete|Patch)\(\"([^\"]+)\"", content):
                result.endpoints.append(match.group(1))

            if result.endpoints:
                result.service_type = "controller"
            elif result.internal_calls:
                result.service_type = "service"

            # Prefer first export as service name when annotation missing
            if result.exports and not result.service_name:
                result.service_name = result.exports[0]

        except Exception as e:
            self.logger.warning(f"Error parsing C# {file_path}: {e}")
            result.errors.append(str(e))
            result.confidence = 0.7

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_javascript_like(
        self, file_path: str, content: str, file_type: FileType
    ) -> ParseResult:
        """Lightweight JS/TS parser: capture exports and language."""
        lang = "javascript" if file_type == FileType.JAVASCRIPT else "typescript"
        result = ParseResult(file_path=file_path, file_type=file_type, language=lang)

        try:
            for match in re.finditer(
                r"export\s+(?:class|function|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)", content
            ):
                result.exports.append(match.group(1))

            # Simple endpoint capture from fetch/axios calls
            for match in re.finditer(r"https?://[^\"'\s)]+", content):
                result.external_services.append(match.group(0))

            if result.exports:
                result.service_name = result.exports[0]
                result.service_type = "module"

        except Exception as e:
            self.logger.warning(f"Error parsing {lang} {file_path}: {e}")
            result.errors.append(str(e))
            result.confidence = 0.7

        return result

    def _parse_html(self, file_path: str, content: str) -> ParseResult:
        """Minimal HTML parsing to tag language and capture title."""
        result = ParseResult(file_path=file_path, file_type=FileType.HTML, language="html")
        title_match = re.search(r"<title>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            result.exports.append(title_match.group(1).strip())
            result.service_name = result.exports[0]

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""
        return result

    def _parse_sql(self, file_path: str, content: str) -> ParseResult:
        """Minimal SQL parsing to record tables referenced."""
        result = ParseResult(file_path=file_path, file_type=FileType.SQL, language="sql")
        tables = re.findall(r"(?:from|join)\s+([A-Za-z0-9_\.]+)", content, flags=re.IGNORECASE)
        if tables:
            result.exports.extend(list({t.strip() for t in tables}))
            result.service_name = result.exports[0]
            result.service_type = "sql-script"

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""
        return result

    def _parse_camel_xml(self, file_path: str, content: str) -> ParseResult:
        """Parse Camel XML route configuration file."""
        result = ParseResult(file_path=file_path, file_type=FileType.XML, language="xml")

        try:
            # Extract route IDs
            route_ids = re.findall(self.CAMEL_PATTERNS["xml_route"], content)
            result.exports.extend([f"CamelRoute:{rid}" for rid in route_ids])

            # Extract from endpoints (sources)
            from_endpoints = re.findall(self.CAMEL_PATTERNS["xml_from"], content)
            result.endpoints.extend([ep for ep in from_endpoints if self._is_valid_export(ep)])

            # Extract to endpoints (targets/sinks)
            to_endpoints = re.findall(self.CAMEL_PATTERNS["xml_to"], content)
            result.endpoints.extend([ep for ep in to_endpoints if self._is_valid_export(ep)])

            # Extract bean class references (processors, error handlers)
            classes = re.findall(r'ref="([^"]+)"|bean="([^"]+)"', content)
            result.internal_calls.extend([c[0] or c[1] for c in classes if c[0] or c[1]])

            result.service_type = "camel-xml-route"

        except Exception as e:
            self.logger.warning(f"Error parsing Camel XML {file_path}: {e}")
            result.errors.append(str(e))

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result

    def _parse_properties(self, file_path: str, content: str) -> ParseResult:
        """Parse properties/YAML configuration file (including Spring Boot Camel config)."""
        result = ParseResult(
            file_path=file_path, file_type=FileType.PROPERTIES, language="properties"
        )

        try:
            # Extract service endpoints
            endpoints = re.findall(r"(?:url|endpoint|baseUrl)[.\w]*\s*=\s*([\w:/.]+)", content)
            result.external_services.extend(endpoints)

            # Extract database URLs
            db_urls = re.findall(r"(?:jdbc|datasource)[:\w.]*url\s*=\s*([^\n]+)", content)
            result.database_refs.extend(db_urls)

            # Extract message queue references
            queue_refs = re.findall(r"(?:queue|topic|channel)[:\w.]*\s*=\s*([^\n]+)", content)
            result.message_queues.extend(queue_refs)

            # ===== CAMEL-SPECIFIC PROPERTIES PARSING =====
            # Check for Spring Boot Camel configuration indicators first
            has_camel_config = "camel." in content or "camel:" in content
            if has_camel_config:
                result.service_type = result.service_type or "spring-boot-camel"

            # Extract Camel component configuration
            for pattern in self.CAMEL_PROPERTIES_PATTERNS:
                for match in pattern.finditer(content):
                    if len(match.groups()) >= 2:
                        component = match.group(1)
                        result.endpoints.append(f"camel-{component}")

        except Exception as e:
            self.logger.warning(f"Error parsing properties {file_path}: {e}")
            result.errors.append(str(e))

        # Final fallback: extract service name from file path
        if not result.service_name:
            result.service_name = self._extract_service_name_from_path(file_path) or ""

        return result
