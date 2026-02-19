"""Camel Parser Demo

Demonstrates parsing Camel-related files using CodeParser.
Prints extracted metadata instead of using pytest assertions.
"""

import tempfile
from pathlib import Path

from scripts.ingest.git.code_parser import CodeParser, FileType


def demo_gradle_mixed_dependencies(parser: CodeParser) -> None:
    print("\n=== Demo: Gradle Mixed Dependencies ===")
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
    print(f"file_type: {result.file_type}")
    print(f"service_type: {result.service_type}")
    print(f"external_dependencies (first 5): {result.external_dependencies[:5]}")
    Path(f.name).unlink(missing_ok=True)


def demo_camel_groovy_dsl_routes(parser: CodeParser) -> None:
    print("\n=== Demo: Camel Groovy DSL Routes ===")
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
    print(f"language: {result.language}")
    print(f"exports: {result.exports}")
    print(f"endpoints: {result.endpoints}")
    Path(f.name).unlink(missing_ok=True)


def demo_spring_boot_camel_properties(parser: CodeParser) -> None:
    print("\n=== Demo: Spring Boot Camel Properties ===")
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
    print(f"service_type: {result.service_type}")
    print(f"endpoints: {result.endpoints}")
    Path(f.name).unlink(missing_ok=True)


def demo_spring_boot_camel_yaml(parser: CodeParser) -> None:
    print("\n=== Demo: Spring Boot Camel YAML ===")
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
    print(f"service_type: {result.service_type}")
    print(f"external_dependencies: {result.external_dependencies}")
    Path(f.name).unlink(missing_ok=True)


def demo_camel_xml_routes(parser: CodeParser) -> None:
    print("\n=== Demo: Camel XML Routes ===")
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
    print(f"service_type: {result.service_type}")
    print(f"exports: {result.exports}")
    print(f"endpoints: {result.endpoints}")
    Path(f.name).unlink(missing_ok=True)


def main() -> int:
    parser = CodeParser()
    demo_gradle_mixed_dependencies(parser)
    demo_camel_groovy_dsl_routes(parser)
    demo_spring_boot_camel_properties(parser)
    demo_spring_boot_camel_yaml(parser)
    demo_camel_xml_routes(parser)
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
