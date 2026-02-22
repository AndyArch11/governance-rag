"""Test suite for code_parser module.

Run with: python -m pytest tests/test_code_parser.py -v
"""

import tempfile
from pathlib import Path

import pytest

from scripts.ingest.git.code_parser import CodeParser, FileType


@pytest.fixture
def parser():
    """Create a code parser instance."""
    return CodeParser()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestJavaParser:
    """Test Java file parsing."""

    def test_java_imports_and_exports(self, parser, temp_dir):
        """Test extraction of imports and class exports."""
        java_file = temp_dir / "PaymentService.java"
        java_file.write_text("""
package com.example.payment;

import com.example.auth.AuthService;
import com.example.notification.NotificationService;
import org.springframework.stereotype.Service;

@Service
public class PaymentService {
    private AuthService authService;
    
    public void processPayment(String orderId) {
        // Process payment
    }
}
""")

        result = parser.parse_file(str(java_file))

        assert result.file_type == FileType.JAVA
        assert "com.example.payment" in result.internal_imports
        assert "com.example.auth.AuthService" in result.internal_imports
        assert "PaymentService" in result.exports
        assert result.service_type == "service"

    def test_java_rest_endpoints(self, parser, temp_dir):
        """Test extraction of REST endpoints."""
        java_file = temp_dir / "OrderController.java"
        java_file.write_text("""
package com.example.order;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/orders")
public class OrderController {
    
    @GetMapping("/{id}")
    public Order getOrder(@PathVariable String id) {
        return null;
    }
    
    @PostMapping
    public Order createOrder(@RequestBody Order order) {
        return null;
    }
}
""")

        result = parser.parse_file(str(java_file))

        assert result.service_type == "controller"
        assert "/{id}" in result.endpoints
        assert "/api/orders" in result.endpoints

    def test_java_message_queue(self, parser, temp_dir):
        """Test extraction of message queue references."""
        java_file = temp_dir / "OrderListener.java"
        java_file.write_text("""
package com.example.order;

import org.springframework.jms.annotation.JmsListener;

public class OrderListener {
    
    @JmsListener(destination = "orders.queue")
    public void onOrderMessage(String message) {
        // Handle order
    }
    
    @JmsListener(destination = "payments.queue")
    public void onPaymentMessage(String message) {
        // Handle payment
    }
}
""")

        result = parser.parse_file(str(java_file))

        assert "orders.queue" in result.message_queues
        assert "payments.queue" in result.message_queues


class TestGroovyParser:
    """Test Groovy file parsing."""

    def test_groovy_gradle_parsing(self, parser, temp_dir):
        """Test parsing of Groovy gradle file."""
        gradle_file = temp_dir / "build.gradle"
        gradle_file.write_text("""
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web:2.7.0'
    implementation 'com.example:payment-client:1.2.3'
    api 'org.apache.kafka:kafka-clients:3.0.0'
}
""")

        result = parser.parse_file(str(gradle_file))

        assert result.file_type == FileType.GRADLE
        assert (
            "org.springframework.boot:spring-boot-starter-web:2.7.0" in result.external_dependencies
        )
        assert "com.example:payment-client:1.2.3" in result.external_dependencies


class TestMavenParser:
    """Test Maven pom.xml parsing."""

    def test_maven_pom_dependencies(self, parser, temp_dir):
        """Test extraction of Maven dependencies."""
        pom_file = temp_dir / "pom.xml"
        pom_file.write_text("""
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <name>Payment Service</name>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.7.0</version>
        </dependency>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>auth-client</artifactId>
            <version>1.0.0</version>
        </dependency>
    </dependencies>
</project>
""")

        result = parser.parse_file(str(pom_file))

        assert result.file_type == FileType.MAVEN_POM
        assert result.service_name == "Payment Service"
        assert (
            "org.springframework.boot:spring-boot-starter-web:2.7.0" in result.external_dependencies
        )
        assert "com.example:auth-client:1.0.0" in result.external_dependencies


class TestMuleParser:
    """Test Mule XML parsing."""

    def test_mule_xml_flows(self, parser, temp_dir):
        """Test extraction of Mule flows and connectors."""
        mule_file = temp_dir / "order-mule.xml"
        mule_file.write_text("""
<mule xmlns="http://www.mulesoft.org/schema/mule/core">
    <flow name="processOrder">
        <http:listener path="/orders" config-ref="HTTP_Listener_config"/>
        <flow-ref name="validateOrder" />
        <http:request method="POST" path="/api/payments"/>
        <jms:outbound-endpoint queue="orders.processed"/>
    </flow>
    
    <flow name="validateOrder">
        <logger message="Validating order"/>
    </flow>
</mule>
""")

        result = parser.parse_file(str(mule_file))

        assert result.file_type == FileType.MULE_XML
        assert result.service_name == "order"
        assert "processOrder" in result.exports
        assert "validateOrder" in result.exports
        assert "validateOrder" in result.internal_calls
        assert "/orders" in result.endpoints


class TestResultConversion:
    """Test ParseResult to dict conversion."""

    def test_parse_result_to_dict(self, parser, temp_dir):
        """Test converting ParseResult to dictionary."""
        java_file = temp_dir / "Service.java"
        java_file.write_text("""
package com.example;

import org.springframework.stereotype.Service;

@Service
public class MyService {
}
""")

        result = parser.parse_file(str(java_file))
        result_dict = result.to_dict()

        assert "file_type" in result_dict
        assert "language" in result_dict
        assert "external_dependencies" in result_dict
        assert "exports" in result_dict
        assert "service_type" in result_dict
        assert isinstance(result_dict["exports"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
