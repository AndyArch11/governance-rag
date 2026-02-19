"""Unit tests for code parser and Bitbucket integration.

TODO: Move examples.git.bitbucket_code_ingestion to a better location.

Tests cover:
- Gradle dependency parsing (Maven, Camel Spring Boot)
- Camel Java/Groovy DSL route parsing
- Camel XML route parsing
- Spring Boot properties/YAML parsing
- Bitbucket connector (mocked)
- Repository walker
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from examples.git.bitbucket_code_ingestion import BitbucketCodeIngestion
from scripts.ingest.git.bitbucket_connector import (
    BitbucketConnector,
    BitbucketProject,
    BitbucketRepository,
    PullRequest,
    RepositoryWalker,
)
from scripts.ingest.git.code_parser import CodeParser, FileType, ParseResult


class TestCodeParserGradle:
    """Test Gradle build.gradle parsing."""

    def test_gradle_maven_dependencies(self):
        """Test parsing Maven dependencies from Gradle."""
        parser = CodeParser()

        gradle_content = """
        dependencies {
            implementation 'org.springframework.boot:spring-boot-starter-web:2.7.0'
            implementation 'com.fasterxml.jackson.core:jackson-databind:2.13.0'
            testImplementation 'junit:junit:4.13.2'
        }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".gradle", delete=False) as f:
            f.write(gradle_content)
            f.flush()

            result = parser.parse_file(f.name)

            assert len(result.external_dependencies) == 3
            assert (
                "org.springframework.boot:spring-boot-starter-web:2.7.0"
                in result.external_dependencies
            )
            assert (
                "com.fasterxml.jackson.core:jackson-databind:2.13.0" in result.external_dependencies
            )
            assert "junit:junit:4.13.2" in result.external_dependencies

    def test_gradle_camel_springboot_dependencies(self):
        """Test parsing Camel Spring Boot dependencies from Gradle."""
        parser = CodeParser()

        gradle_content = """
        dependencies {
            implementation 'org.apache.camel.springboot:camel-core-starter:3.20.0'
            implementation 'org.apache.camel.springboot:camel-jms-starter:3.20.0'
            implementation 'org.apache.camel.springboot:camel-http-starter:3.20.0'
            implementation 'org.apache.camel.springboot:camel-kafka-starter:3.20.0'
        }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".gradle", delete=False) as f:
            f.write(gradle_content)
            f.flush()

            result = parser.parse_file(f.name)

            assert result.service_type == "camel-springboot"
            assert len(result.external_dependencies) == 4
            assert any("camel-core-starter" in dep for dep in result.external_dependencies)
            assert any("camel-jms-starter" in dep for dep in result.external_dependencies)

            # Check exports contain Camel starters
            assert any("CamelStarter:camel-core-starter" in exp for exp in result.exports)

    def test_gradle_mixed_dependencies(self):
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

            assert result.service_type == "camel-springboot"
            assert (
                len(result.external_dependencies) >= 3
            ), f"Expected >= 3 deps, got {len(result.external_dependencies)}: {result.external_dependencies}"
            assert result.file_type == FileType.GRADLE


class TestCodeParserCamelJavaDSL:
    """Test Camel Java/Groovy DSL route parsing."""

    def test_camel_java_dsl_routes(self):
        """Test parsing Camel routes from Java DSL."""
        parser = CodeParser()

        java_content = """
        package com.example.routes;
        
        import org.apache.camel.builder.RouteBuilder;
        
        public class OrderRoutes extends RouteBuilder {
            @Override
            public void configure() throws Exception {
                from("jms:queue:orders")
                    .routeId("orderProcessing")
                    .process(validateOrder)
                    .to("http://api.payment.com/process")
                    .choice()
                        .when(simple("${header.status} == 'approved'"))
                            .to("kafka:topic:orders-approved")
                        .otherwise()
                            .to("jms:queue:rejected-orders")
                    .end();
            }
        }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
            f.write(java_content)
            f.flush()

            result = parser.parse_file(f.name)

            # Check endpoints extracted
            assert "jms:queue:orders" in result.endpoints
            assert "http://api.payment.com/process" in result.endpoints
            assert "kafka:topic:orders-approved" in result.endpoints
            assert "jms:queue:rejected-orders" in result.endpoints

            # Check route ID exported
            assert any("CamelRoute:orderProcessing" in exp for exp in result.exports)

            # Check processor tracked
            assert "validateOrder" in result.internal_calls

    def test_camel_groovy_dsl_routes(self):
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


class TestCodeParserCamelXML:
    """Test Camel XML route parsing."""

    def test_camel_xml_routes(self):
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
                
                <route id="auditRoute">
                    <from uri="kafka:topic:orders-processed"/>
                    <to uri="jms:queue:audit-log"/>
                </route>
            </camelContext>
        </beans>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()

            result = parser.parse_file(f.name)

            assert result.service_type == "camel-xml-route"

            # Check route IDs
            assert any("CamelRoute:orderRoute" in exp for exp in result.exports)
            assert any("CamelRoute:auditRoute" in exp for exp in result.exports)

            # Check endpoints
            assert "jms:queue:orders" in result.endpoints
            assert "http://api.example.com/orders" in result.endpoints
            assert "kafka:topic:valid-orders" in result.endpoints
            assert "jms:queue:invalid-orders" in result.endpoints
            assert "kafka:topic:orders-processed" in result.endpoints
            assert "jms:queue:audit-log" in result.endpoints

            # Check processor refs
            assert "orderValidator" in result.internal_calls


class TestCodeParserSpringBootCamel:
    """Test Spring Boot Camel properties parsing."""

    def test_spring_boot_camel_properties(self):
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

            # Primary check: service type should be detected as spring-boot-camel
            assert (
                result.service_type == "spring-boot-camel"
            ), f"Expected spring-boot-camel, got {result.service_type}"

    def test_spring_boot_camel_yaml(self):
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


class TestCodeParserJavaServices:
    """Test parsing Java service annotations and dependencies."""

    def test_spring_controller(self):
        """Test parsing Spring REST controller."""
        parser = CodeParser()

        java_content = """
        package com.example.api;
        
        import org.springframework.web.bind.annotation.*;
        
        @RestController
        @RequestMapping("/api/orders")
        public class OrderController {
            @GetMapping("/{id}")
            public Order getOrder(@PathVariable String id) {
                return orderService.findById(id);
            }
        }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
            f.write(java_content)
            f.flush()

            result = parser.parse_file(f.name)

            assert result.service_type == "controller"
            assert "/api/orders/{id}" in result.endpoints or "/api/orders" in result.endpoints

    def test_spring_service(self):
        """Test parsing Spring service."""
        parser = CodeParser()

        java_content = """
        package com.example.service;
        
        import org.springframework.stereotype.Service;
        
        @Service
        public class OrderService {
            public Order findById(String id) {
                return repository.findById(id);
            }
        }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
            f.write(java_content)
            f.flush()

            result = parser.parse_file(f.name)

            assert result.service_type == "service"
            assert "OrderService" in result.exports


class TestBitbucketConnector:
    """Test Bitbucket connector (mocked)."""

    @patch("scripts.ingest.git.bitbucket_connector.requests.Session")
    def test_list_projects_server(self, mock_session_class):
        """Test listing projects from Bitbucket Server."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "values": [
                {"key": "PROJ1", "name": "Project One", "description": "First project"},
                {"key": "PROJ2", "name": "Project Two"},
            ],
            "isLastPage": True,
        }
        mock_session.request.return_value = mock_response

        connector = BitbucketConnector(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
            is_cloud=False,
        )

        projects = connector.list_projects()

        assert len(projects) == 2
        assert projects[0].key == "PROJ1"
        assert projects[0].name == "Project One"

    @patch("scripts.ingest.git.bitbucket_connector.requests.Session")
    def test_list_repositories(self, mock_session_class):
        """Test listing repositories in a project."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {
            "values": [
                {
                    "slug": "repo-1",
                    "name": "Repository One",
                    "description": "First repository",
                    "links": {
                        "clone": [
                            {
                                "name": "http",
                                "href": "https://bitbucket.example.com/scm/proj/repo-1.git",
                            },
                            {
                                "name": "ssh",
                                "href": "ssh://git@bitbucket.example.com/proj/repo-1.git",
                            },
                        ]
                    },
                },
                {"slug": "repo-2", "name": "Repository Two", "links": {"clone": []}},
            ],
            "isLastPage": True,
        }
        mock_session.request.return_value = mock_response

        connector = BitbucketConnector(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
            is_cloud=False,
        )

        repos = connector.list_repositories("PROJ")

        assert len(repos) == 2
        assert repos[0].slug == "repo-1"
        assert repos[0].clone_url == "https://bitbucket.example.com/scm/proj/repo-1.git"

    @patch("scripts.ingest.git.bitbucket_connector.subprocess.run")
    def test_clone_repository(self, mock_run):
        """Test cloning a repository."""
        mock_session = Mock()
        with patch(
            "scripts.ingest.git.bitbucket_connector.requests.Session", return_value=mock_session
        ):
            mock_response = Mock()
            mock_response.json.return_value = {
                "values": [
                    {
                        "slug": "my-repo",
                        "name": "My Repository",
                        "links": {
                            "clone": [
                                {
                                    "name": "http",
                                    "href": "https://bitbucket.example.com/scm/proj/my-repo.git",
                                },
                            ]
                        },
                    },
                ],
                "isLastPage": True,
            }
            mock_session.request.return_value = mock_response

            connector = BitbucketConnector(
                host="https://bitbucket.example.com",
                username="user",
                password="pass",
                is_cloud=False,
            )

            mock_run.return_value = Mock(returncode=0)

            with tempfile.TemporaryDirectory() as tmpdir:
                repo_path = connector.clone_repository(
                    "PROJ", "my-repo", target_dir=f"{tmpdir}/my-repo"
                )

                assert mock_run.called
                call_args = mock_run.call_args
                assert "git" in call_args[0][0]
                assert "clone" in call_args[0][0]


class TestRepositoryWalker:
    """Test repository file walker."""

    def test_walk_groovy_files(self):
        """Test walking Groovy files in a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create test files
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
            (repo_path / "src" / "test" / "groovy").mkdir(parents=True)

            groovy_file = repo_path / "src" / "main" / "groovy" / "Route.groovy"
            groovy_file.write_text("class Route {}")

            gradle_file = repo_path / "build.gradle"
            gradle_file.write_text("dependencies {}")

            java_file = repo_path / "src" / "main" / "java" / "Service.java"
            java_file.parent.mkdir(parents=True)
            java_file.write_text("class Service {}")

            walker = RepositoryWalker(str(repo_path))

            groovy_files = list(walker.walk_groovy_files())
            assert len(groovy_files) == 2  # Route.groovy and build.gradle

            file_paths = [fp for fp, _ in groovy_files]
            assert any("Route.groovy" in fp for fp in file_paths)
            assert any("build.gradle" in fp for fp in file_paths)

    def test_walk_java_files(self):
        """Test walking Java files in a repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            (repo_path / "src" / "main" / "java").mkdir(parents=True)

            java_file = repo_path / "src" / "main" / "java" / "Service.java"
            java_file.write_text("class Service {}")

            walker = RepositoryWalker(str(repo_path))

            java_files = list(walker.walk_java_files())
            assert len(java_files) == 1
            assert "Service.java" in java_files[0][0]

    def test_get_directory_structure(self):
        """Test getting directory structure summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create various files
            (repo_path / "src" / "main" / "java").mkdir(parents=True)
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
            (repo_path / "src" / "test").mkdir(parents=True)

            (repo_path / "src" / "main" / "java" / "Service.java").write_text("class Service {}")
            (repo_path / "src" / "main" / "groovy" / "Route.groovy").write_text("class Route {}")
            (repo_path / "build.gradle").write_text("dependencies {}")
            (repo_path / "pom.xml").write_text("<project></project>")
            (repo_path / "src" / "test" / "TestService.java").write_text("class TestService {}")

            walker = RepositoryWalker(str(repo_path))
            struct = walker.get_directory_structure()

            assert struct["total_files"] >= 4
            assert len(struct["java_files"]) >= 1
            assert len(struct["groovy_files"]) >= 1
            assert len(struct["gradle_files"]) >= 1

    def test_walk_groovy_files_dated(self):
        """Test date-based filtering for Groovy files."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create test files
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
            groovy_file = repo_path / "src" / "main" / "groovy" / "Route.groovy"
            groovy_file.write_text("class Route {}")

            walker = RepositoryWalker(str(repo_path))

            # Without date filtering
            all_files = list(walker.walk_groovy_files_dated())
            assert len(all_files) > 0

            # With date filtering - before future date should include file
            future_date = datetime.now() + timedelta(days=1)
            dated_files = list(walker.walk_groovy_files_dated(modified_before=future_date))
            assert len(dated_files) > 0

            # Verify date is included in results
            for file_path, content, mod_date in dated_files:
                assert isinstance(mod_date, datetime)

    def test_walk_java_files_dated(self):
        """Test date-based filtering for Java files."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create test files
            (repo_path / "src" / "main" / "java").mkdir(parents=True)
            java_file = repo_path / "src" / "main" / "java" / "Service.java"
            java_file.write_text("class Service {}")

            walker = RepositoryWalker(str(repo_path))

            # With future date - should include file
            future_date = datetime.now() + timedelta(days=1)
            dated_files = list(walker.walk_java_files_dated(modified_before=future_date))
            assert len(dated_files) > 0

            # Verify structure
            for file_path, content, mod_date in dated_files:
                assert "Service.java" in file_path
                assert isinstance(mod_date, datetime)

    def test_walk_all_code_files_dated(self):
        """Test date-based filtering for all code files."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create multiple file types
            (repo_path / "src" / "main" / "java").mkdir(parents=True)
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)

            (repo_path / "src" / "main" / "java" / "Service.java").write_text("class Service {}")
            (repo_path / "src" / "main" / "groovy" / "Route.groovy").write_text("class Route {}")
            (repo_path / "build.gradle").write_text("dependencies {}")

            walker = RepositoryWalker(str(repo_path))

            # All files before future date
            future_date = datetime.now() + timedelta(days=1)
            dated_files = list(walker.walk_all_code_files_dated(modified_before=future_date))
            assert len(dated_files) >= 2  # At least Java and Groovy files

            # Verify dates are reasonable
            now = datetime.now()
            for file_path, content, mod_date in dated_files:
                assert mod_date <= now

    def test_compare_versions(self):
        """Test version comparison between two dates."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create initial files
            (repo_path / "src" / "main" / "java").mkdir(parents=True)
            file1 = repo_path / "src" / "main" / "java" / "Service.java"
            file1.write_text("class Service {}")

            walker = RepositoryWalker(str(repo_path))

            # Compare versions with nearby dates
            v1_date = datetime.now() - timedelta(days=1)
            v2_date = datetime.now() + timedelta(days=1)

            comparison = walker.compare_versions(
                v1_date=v1_date, v2_date=v2_date, extensions=[".java"]
            )

            assert "v1_files" in comparison
            assert "v2_files" in comparison
            assert "summary" in comparison

            summary = comparison["summary"]
            assert "added" in summary
            assert "removed" in summary
            assert "modified" in summary
            assert "unchanged" in summary
            assert "drift_percentage" in summary

            # Drift should be a reasonable percentage
            assert 0 <= summary["drift_percentage"] <= 100


class TestBitbucketCodeIngestion:
    """Test end-to-end Bitbucket code ingestion."""

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    def test_list_projects_integration(self, mock_connector_class):
        """Test listing projects through ingestion pipeline."""
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector

        mock_connector.list_projects.return_value = [
            BitbucketProject(key="PROJ1", name="Project One"),
            BitbucketProject(key="PROJ2", name="Project Two"),
        ]

        ingestion = BitbucketCodeIngestion(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
        )

        projects = ingestion.list_projects()

        assert len(projects) == 2
        assert projects[0]["key"] == "PROJ1"

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    def test_list_repositories_integration(self, mock_connector_class):
        """Test listing repositories through ingestion pipeline."""
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector

        mock_connector.list_repositories.return_value = [
            BitbucketRepository(
                slug="repo-1",
                name="Repository One",
                project_key="PROJ",
            ),
        ]

        ingestion = BitbucketCodeIngestion(
            host="https://bitbucket.example.com",
            username="user",
            password="pass",
        )

        repos = ingestion.list_repositories("PROJ")

        assert len(repos) == 1
        assert repos[0]["slug"] == "repo-1"

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    def test_ingest_repository_dated(self, mock_connector_class):
        """Test dated repository ingestion with parsing."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create test files
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
            groovy_file = repo_path / "src" / "main" / "groovy" / "Route.groovy"
            groovy_file.write_text(
                """
            from('jms:queue:input')
                .to('http://service/process')
                .to('kafka:topic:output')
            """
            )

            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.clone_repository.return_value = str(repo_path)

            ingestion = BitbucketCodeIngestion(
                host="https://bitbucket.example.com",
                username="user",
                password="pass",
            )

            future_date = datetime.now() + timedelta(days=1)
            result = ingestion.ingest_repository_dated(
                "PROJ", "test-repo", modified_before=future_date, file_types=["groovy"]
            )

            assert "parsed_files" in result
            assert len(result["parsed_files"]) > 0

            # Check that files have modification dates
            for parsed_file in result["parsed_files"]:
                assert "modified_date" in parsed_file
                # modified_date is a string ISO format, not datetime object
                assert isinstance(parsed_file["modified_date"], (str, datetime))

    @patch("examples.git.bitbucket_code_ingestion.BitbucketConnector")
    def test_analyse_version_drift(self, mock_connector_class):
        """Test version drift analysis."""
        from datetime import datetime, timedelta

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create test files
            (repo_path / "src" / "main" / "groovy").mkdir(parents=True)
            groovy_file = repo_path / "src" / "main" / "groovy" / "Route.groovy"
            groovy_file.write_text(
                """
            from('kafka:input')
                .routeId('processor')
                .to('http://service.example.com/api')
            """
            )

            gradle_file = repo_path / "build.gradle"
            gradle_file.write_text(
                """
            dependencies {
                implementation 'org.apache.camel.springboot:camel-core-starter:3.14.0'
            }
            """
            )

            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector
            mock_connector.clone_repository.return_value = str(repo_path)

            ingestion = BitbucketCodeIngestion(
                host="https://bitbucket.example.com",
                username="user",
                password="pass",
            )

            v1_date = datetime.now() - timedelta(days=1)
            v2_date = datetime.now() + timedelta(days=1)

            drift_report = ingestion.analyse_version_drift(
                "PROJ", "test-repo", v1_date=v1_date, v2_date=v2_date
            )

            assert "summary" in drift_report
            assert "parsed_changes" in drift_report

            summary = drift_report["summary"]
            assert "total_added" in summary
            assert "total_removed" in summary
            assert "total_modified" in summary
            assert "drift_percentage" in summary

            parsed_changes = drift_report["parsed_changes"]
            assert "added" in parsed_changes
            assert "removed" in parsed_changes
            assert "modified" in parsed_changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
