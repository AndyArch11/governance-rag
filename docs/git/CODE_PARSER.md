"""CodeParser Module - Documentation and Usage Guide

The CodeParser module extends the RAG ingestion pipeline to understand and extract
architectural information from multi-language codebases. This is critical for
analysing a large number of repositories for cross-service dependencies for
understanding legacy codebases or for activities such as migration planning.

## Overview

The CodeParser enables:
- **Language support**: Java, Groovy, Mule, XML configs, pom.xml, build.gradle
- **Dependency extraction**: External packages, internal service calls
- **Architecture detection**: Controllers, services, processors, listeners
- **Integration points**: REST endpoints, message queues, database references
- **Service mapping**: Identifies services and their cross-repo dependencies

## Quick Start

### Basic Usage

```python
from scripts.ingest.git.code_parser import CodeParser

parser = CodeParser()
result = parser.parse_file("path/to/PaymentService.java")

print(result.service_name)              # "PaymentService"
print(result.service_type)              # "service"
print(result.external_dependencies)     # ["com.example:auth:1.0", ...]
print(result.internal_calls)            # ["AuthService", "NotificationService", ...]
print(result.endpoints)                 # ["/api/payments", ...]
print(result.message_queues)            # ["orders.processed", ...]
```

### Repository Analysis

```bash
python3 examples/git/code_analysis_example.py ~/projects/payment-service analysis.json
```

This generates a comprehensive JSON report showing:
- All detected services and their types
- External dependencies by service
- Cross-service method/class calls
- REST API endpoints
- Message queue integrations

## Supported File Types

### Java Files (.java)
Extracts:
- Package and import statements
- Public classes, interfaces, enums
- Spring annotations (@Service, @Controller, @RestController)
- REST endpoints (@GetMapping, @PostMapping, etc.)
- Injected dependencies (@Autowired, @Inject)
- JPA entities and database tables
- JMS/Kafka message listeners

Example:
```java
@Service
public class PaymentService {
    @Autowired
    private AuthService authService;
    
    @PostMapping("/process")
    public void processPayment(Order order) { }
}
```
Extracted: `service_type=service`, `internal_calls=["AuthService"]`, `endpoints=["/process"]`

### Groovy Files (.groovy)
Similar to Java, also handles:
- Groovy-specific syntax
- DSL expressions

### Gradle Files (build.gradle)
Extracts:
- Implementation, api, and compile dependencies
- Handles both `implementation 'org:artifact:version'` and `implementation("org:artifact:version")`

Example:
```gradle
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web:2.7.0'
    api 'com.example:payment-client:1.2.3'
}
```
Extracted: All dependencies in `external_dependencies`

### Maven Files (pom.xml)
Extracts:
- Project name/service name from `<name>` tag
- All dependencies with groupId:artifactId:version
- Handles both namespaced and non-namespaced XML

### Mule XML (muleservice-mule.xml)
Extracts:
- Flow definitions
- Connected components (HTTP, JMS, Kafka, DB)
- Flow references (service calls)
- HTTP request paths
- Message endpoints

Example:
```xml
<flow name="processOrder">
    <http:listener path="/orders" />
    <flow-ref name="validateOrder" />
    <http:request path="/api/payments" />
</flow>
```
Extracted: `service_name=muleservice`, `exports=["processOrder"]`, `endpoints=["/orders"]`

### Configuration Files
- **properties/yaml**: Extracts URLs, database references, queue names
- **Generic XML**: Extracts Spring beans and their class references

## ParseResult Structure

```python
@dataclass
class ParseResult:
    file_path: str                          # Path to parsed file
    file_type: FileType                     # java, groovy, gradle, pom.xml, etc.
    language: str                           # java, groovy, xml, properties
    
    # Dependencies
    external_dependencies: List[str]        # Maven/npm packages: org.example:lib:1.0
    internal_imports: List[str]             # Java packages: com.mycompany.auth
    
    # Service references
    internal_calls: List[str]               # Classes/services called: AuthService, etc.
    
    # Public APIs
    exports: List[str]                      # Public classes, functions, flows
    
    # Architecture
    service_name: Optional[str]             # Detected service name
    service_type: Optional[str]             # controller, service, processor, listener
    
    # Integration points
    endpoints: List[str]                    # REST endpoints: /api/orders
    database_refs: List[str]                # Database names, JPA entities
    message_queues: List[str]               # JMS/Kafka topics: orders.queue
    external_services: List[str]            # External URLs called
    
    # Quality metrics
    confidence: float                       # Confidence of extraction (0.0-1.0)
    errors: List[str]                       # Parsing errors encountered
```

## Integration with RAG Ingestion

To use CodeParser within the ingestion pipeline:

```python
from scripts.ingest.git.code_parser import CodeParser
from scripts.ingest.vectors import store_chunks_in_chroma

# 1. Parse the file for architectural information
parser = CodeParser()
code_info = parser.parse_file(file_path)

# 2. Enhance chunk metadata with extracted information
metadata["external_dependencies"] = code_info.external_dependencies
metadata["internal_calls"] = code_info.internal_calls
metadata["service_name"] = code_info.service_name
metadata["service_type"] = code_info.service_type
metadata["endpoints"] = code_info.endpoints
metadata["message_queues"] = code_info.message_queues

# 3. Store chunks with enriched metadata
store_chunks_in_chroma(..., metadata=metadata, ...)
```

## Migration Analysis Queries

With code metadata stored in ChromaDB, you can run sophisticated queries:

```python
# Query: "What services depend on the auth module?"
# Answer: Returns all chunks with internal_calls containing "AuthService"

# Query: "Show me all REST endpoints in the payment service"
# Answer: Returns all payment service files with endpoints in metadata

# Query: "Which services call the ComplianceChecker class?"
# Answer: Searches for internal_calls="ComplianceChecker"

# Query: "What are all downstream services of the order processor?"
# Answer: Builds dependency graph from message_queues and internal_calls
```

## Test Coverage

Run tests with:
```bash
pytest tests/test_code_parser.py -v
```

## Performance Notes

- **Typical file processing**: 1-10ms per file (regex-based parsing)
- **Batch analysis**: 1000s of files in seconds
- **Memory**: Minimal footprint, suitable for analysing large repositories
- **Confidence scores**: Extracted from pattern matching accuracy

## Limitations

1. **Reflection-based calls**: Cannot detect runtime service calls via reflection
2. **Dynamic imports**: Only captures static imports and declarations
3. **External services**: Must have explicit URLs/endpoints in config/code
4. **Message flows**: Limited to declarative configurations, not runtime behaviour
5. **Namespace ambiguity**: May report both "Service" and "ServiceImpl" as separate services

## TODO: Future Enhancements

1. **AST parsing**: Move beyond regex to semantic understanding
2. **Call graph analysis**: Track method call chains across files
3. **Library dependency trees**: Resolve transitive dependencies
4. **Service topology**: Generate visual service dependency graphs
5. **Migration recommendations**: Suggest chunking strategy based on graph analysis
6. **Go/Python/Node/C# support**: Extend language coverage
7. **Artifact registry**: Cross-reference against Maven Central, npm registry
8. **Data Flow diagrams**: Generate data flow diagrams for threat modelling such as STRIDE
9. **Semantic parsing**: Parse comments and *.md files to derive semantic meaning of application
10. **Compliance validation**: Validate code against company guidelines and standards

## Examples

See `examples/git/code_analysis_example.py` for complete working example that:
1. Analyses a directory recursively
2. Extracts service structure
3. Generates dependency report
4. Exports to JSON

Run:
```bash
python3 examples/git/code_analysis_example.py ~/my-service analysis.json
```

Output will show:
- Services detected and their types
- All external dependencies
- Cross-service method calls
- Public endpoints
- Message queue flows
"""
