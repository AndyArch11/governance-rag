#!/usr/bin/env python
"""Simple test runner for code parser Camel tests (bypasses pytest config issues)."""

import sys
import tempfile

from scripts.ingest.git.code_parser import CodeParser, FileType


def test_gradle_mixed_dependencies():
    """Test parsing mixed Maven and Camel dependencies."""
    parser = CodeParser()
    gradle_content = """
    dependencies {
        implementation 'org.springframework.boot:spring-boot-starter-web:2.7.0'
        implementation 'org.apache.camel.springboot:camel-core-starter:3.20.0'
        implementation 'org.apache.camel.springboot:camel-rest-starter:3.20.0'
        runtimeOnly 'com.h2database:h2:2.1.0'
    }
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".gradle", delete=False) as f:
        f.write(gradle_content)
        f.flush()
        result = parser.parse_file(f.name)

        assert (
            result.service_type == "camel-springboot"
        ), f"Expected camel-springboot, got {result.service_type}"
        assert (
            len(result.external_dependencies) >= 3
        ), f"Expected >= 3 deps, got {len(result.external_dependencies)}"
        assert result.file_type == FileType.GRADLE


def test_camel_groovy_dsl_routes():
    """Test parsing Camel routes from Groovy DSL."""
    parser = CodeParser()
    groovy_content = """
    package com.example.routes
    
    import org.apache.camel.builder.RouteBuilder
    
    class PaymentRoutes extends RouteBuilder {
        void configure() {
            from('jms:queue:payments')
                .routeId('paymentProcessing')
                .process(validatePayment)
                .to('http://gateway.payment.com/charge')
                .to('kafka:topic:payments-processed')
        }
    }
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".groovy", delete=False) as f:
        f.write(groovy_content)
        f.flush()
        result = parser.parse_file(f.name)

        assert result.language == "groovy"
        assert (
            "jms:queue:payments" in result.endpoints
        ), f"jms:queue:payments not in {result.endpoints}"
        assert "http://gateway.payment.com/charge" in result.endpoints
        assert "kafka:topic:payments-processed" in result.endpoints
        assert any("CamelRoute:paymentProcessing" in exp for exp in result.exports)


def test_spring_boot_camel_properties():
    """Test parsing Spring Boot Camel configuration from properties."""
    parser = CodeParser()
    properties_content = """
    # Camel configuration
    camel.springboot.name=my-camel-app
    camel.component.http.connection-timeout=5000
    camel.component.jms.broker-url=tcp://localhost:61616
    camel.component.kafka.brokers=localhost:9092
    camel.route.error-handler=log
    
    # Spring configuration
    spring.jms.broker-url=tcp://localhost:61616
    spring.kafka.bootstrap-servers=localhost:9092
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".properties", delete=False) as f:
        f.write(properties_content)
        f.flush()
        result = parser.parse_file(f.name)

        assert (
            result.service_type == "spring-boot-camel"
        ), f"Expected spring-boot-camel, got {result.service_type}"


def test_spring_boot_camel_yaml():
    """Test parsing Spring Boot Camel configuration from YAML."""
    parser = CodeParser()
    yaml_content = """
    camel:
      springboot:
        name: my-camel-service
      component:
        http:
          connection-timeout: 5000
          connect-timeout: 3000
        jms:
          broker-url: tcp://localhost:61616
        kafka:
          brokers: localhost:9092
      route:
        error-handler: log
    
    spring:
      application:
        name: my-camel-service
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        result = parser.parse_file(f.name)

        assert (
            result.service_type == "spring-boot-camel"
        ), f"Expected spring-boot-camel, got {result.service_type}"


def test_camel_xml_routes():
    """Test parsing Camel XML route configuration."""
    parser = CodeParser()
    xml_content = """
    <?xml version="1.0" encoding="UTF-8"?>
    <beans xmlns="http://www.springframework.org/schema/beans"
           xmlns:camel="http://camel.apache.org/schema/spring"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:schemaLocation="http://www.springframework.org/schema/beans
                               http://www.springframework.org/schema/beans/spring-beans.xsd
                               http://camel.apache.org/schema/spring
                               http://camel.apache.org/schema/spring/camel-spring.xsd">
    
        <camelContext id="camelContext" xmlns="http://camel.apache.org/schema/spring">
            <route id="orderRoute">
                <from uri="jms:queue:orders"/>
                <process ref="orderValidator"/>
                <choice>
                    <when>
                        <simple>${header.status} == 'valid'</simple>
                        <to uri="http://api.example.com/orders"/>
                        <to uri="kafka:topic:valid-orders"/>
                    </when>
                    <otherwise>
                        <to uri="jms:queue:invalid-orders"/>
                    </otherwise>
                </choice>
            </route>
        </camelContext>
    </beans>
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        f.flush()
        result = parser.parse_file(f.name)

        assert result.service_type == "camel-xml-route"
        assert any("CamelRoute:orderRoute" in exp for exp in result.exports)
        assert "jms:queue:orders" in result.endpoints
        assert "http://api.example.com/orders" in result.endpoints


def main():
    """Run all tests."""
    tests = [
        ("Gradle Mixed Dependencies", test_gradle_mixed_dependencies),
        ("Camel Groovy DSL Routes", test_camel_groovy_dsl_routes),
        ("Spring Boot Camel Properties", test_spring_boot_camel_properties),
        ("Spring Boot Camel YAML", test_spring_boot_camel_yaml),
        ("Camel XML Routes", test_camel_xml_routes),
    ]

    print("\n" + "=" * 70)
    print("Running Code Parser Camel Integration Tests")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            print(f"✓ {test_name:<45} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name:<45} FAILED")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name:<45} ERROR")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
