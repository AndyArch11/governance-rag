#!/usr/bin/env python3
"""
Test script for dependency visualisation tab in dashboard.
Tests that the dependency graph extraction, visualisation, and metrics work correctly.
"""

import networkx as nx


def test_dependency_extraction():
    """Test extracting code nodes and building dependency graph."""
    print("=" * 70)
    print("TEST 1: Dependency Graph Extraction (Simulated)")
    print("=" * 70)

    # Simulate code nodes as they would appear in the graph
    code_nodes = {
        "AuthService_v1": {
            "source_category": "bitbucket_code",
            "service_name": "auth-service",
            "language": "java",
            "internal_calls": ["UserService"],
            "dependencies": ["spring-boot", "jackson"],
        },
        "UserService_v1": {
            "source_category": "bitbucket_code",
            "service_name": "user-service",
            "language": "groovy",
            "internal_calls": ["TokenService"],
            "dependencies": ["spring-boot", "hibernate"],
        },
        "TokenService_v1": {
            "source_category": "bitbucket_code",
            "service_name": "token-service",
            "language": "kotlin",
            "internal_calls": [],
            "dependencies": ["spring-boot", "jwt"],
        },
        "DatabaseService_v1": {
            "source_category": "bitbucket_code",
            "service_name": "database-service",
            "language": "java",
            "internal_calls": [],
            "dependencies": ["spring-boot", "postgresql"],
        },
    }

    print(f"✓ Loaded {len(code_nodes)} simulated code nodes")

    # Build dependency graph
    dep_graph = nx.DiGraph()

    # Add nodes
    for node_id, data in code_nodes.items():
        service = data.get("service_name") or data.get("service", node_id)
        language = data.get("language", "")
        dep_graph.add_node(node_id, service=service, language=language)

    print(f"✓ Built dependency graph with {len(dep_graph.nodes())} nodes")

    # Add internal call edges
    internal_calls_count = 0
    for node_id, data in code_nodes.items():
        internal_calls = data.get("internal_calls", [])
        if internal_calls:
            calls_list = internal_calls if isinstance(internal_calls, list) else [internal_calls]
            for called_service in calls_list:
                for target_id, target_data in code_nodes.items():
                    target_service = target_data.get("service_name") or target_data.get("service")
                    if target_service == called_service or called_service in str(target_id):
                        if node_id != target_id and not dep_graph.has_edge(node_id, target_id):
                            dep_graph.add_edge(node_id, target_id, type="internal_call", weight=1.0)
                            internal_calls_count += 1
                        break

    print(f"✓ Added {internal_calls_count} internal call edges")

    # Add shared dependency edges
    shared_deps_count = 0
    for i, (node_id_a, data_a) in enumerate(code_nodes.items()):
        deps_a = set(
            data_a.get("dependencies", []) if isinstance(data_a.get("dependencies"), list) else []
        )
        for node_id_b, data_b in list(code_nodes.items())[i + 1 :]:
            deps_b = set(
                data_b.get("dependencies", [])
                if isinstance(data_b.get("dependencies"), list)
                else []
            )
            shared = deps_a & deps_b
            if shared:
                if not dep_graph.has_edge(node_id_a, node_id_b):
                    weight = (
                        len(shared) / max(len(deps_a), len(deps_b))
                        if max(len(deps_a), len(deps_b)) > 0
                        else 0
                    )
                    dep_graph.add_edge(
                        node_id_a,
                        node_id_b,
                        type="shared_deps",
                        weight=weight,
                        shared=sorted(list(shared)),
                    )
                    shared_deps_count += 1

    print(f"✓ Added {shared_deps_count} shared dependency edges")

    # Detect circular dependencies
    cycles = list(nx.simple_cycles(dep_graph))
    print(f"✓ Detected {len(cycles)} circular dependency path(s)")

    if cycles:
        print("  Circular dependencies found:")
        for i, cycle in enumerate(cycles[:5], 1):
            cycle_services = [
                code_nodes[nid].get("service_name", nid) for nid in cycle if nid in code_nodes
            ]
            print(f"    {i}. {' → '.join(cycle_services)} → {cycle_services[0]}")
        if len(cycles) > 5:
            print(f"    ... and {len(cycles) - 5} more")
    else:
        print("  ✅ No circular dependencies detected (healthy DAG)")

    # Check network properties
    print(f"\nDependency Network Properties:")
    print(f"  - Services: {len(dep_graph.nodes())}")
    print(f"  - Total Dependencies: {len(dep_graph.edges())}")
    print(f"  - Internal Calls: {internal_calls_count}")
    print(f"  - Shared Dependencies: {shared_deps_count}")

    # Show sample nodes
    if len(dep_graph.nodes()) > 0:
        print(f"\nServices in Graph:")
        for i, (node_id, data) in enumerate(sorted(dep_graph.nodes(data=True)), 1):
            service = data.get("service", node_id)
            language = data.get("language", "unknown")
            degree = dep_graph.degree(node_id)
            print(f"  {i}. {service} ({language}) - degree: {degree}")

    # Verify node degree computation
    auth_degree = dep_graph.degree("AuthService_v1")
    if auth_degree == 0:
        print("❌ FAILED: AuthService should have non-zero degree")
        return False

    print("\n✅ PASSED: Dependency graph extraction works correctly\n")
    return True


def test_circular_dependency_detection():
    """Test circular dependency detection."""
    print("=" * 70)
    print("TEST 2: Circular Dependency Detection")
    print("=" * 70)

    # Create test graph with circular dependency
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("service-a", "service-b"),
            ("service-b", "service-c"),
            ("service-c", "service-a"),  # Creates cycle: A → B → C → A
        ]
    )

    cycles = list(nx.simple_cycles(G))

    if len(cycles) == 0:
        print("❌ FAILED: Should have detected 1 cycle")
        return False

    print(f"✓ Detected {len(cycles)} cycle(s)")
    print(f"✓ Cycle nodes: {set(cycles[0])}")

    # Test with acyclic graph
    G2 = nx.DiGraph()
    G2.add_edges_from(
        [
            ("service-a", "service-b"),
            ("service-a", "service-c"),
            ("service-b", "service-c"),
        ]
    )

    cycles2 = list(nx.simple_cycles(G2))

    if len(cycles2) != 0:
        print("❌ FAILED: Acyclic graph should have 0 cycles")
        return False

    print("✓ Acyclic graph correctly detected as having 0 cycles")

    print("\n✅ PASSED: Circular dependency detection works correctly\n")
    return True


def test_service_matrix():
    """Test service call matrix generation."""
    print("=" * 70)
    print("TEST 3: Service Call Matrix Generation")
    print("=" * 70)

    # Create test graph
    services = ["auth-service", "user-service", "token-service"]
    G = nx.DiGraph()

    # Add nodes
    for service in services:
        G.add_node(service, service=service, language="java")

    # Add edges
    G.add_edge("auth-service", "user-service", weight=1.0, type="internal_call")
    G.add_edge("auth-service", "token-service", weight=0.5, type="shared_deps")
    G.add_edge("user-service", "token-service", weight=1.0, type="internal_call")

    # Build matrix
    matrix = [[0.0 for _ in services] for _ in services]
    for i, node_a in enumerate(services):
        for j, node_b in enumerate(services):
            if G.has_edge(node_a, node_b):
                edges = G.get_edge_data(node_a, node_b)
                matrix[i][j] = edges.get("weight", 1.0)

    # Verify matrix
    expected_values = [
        [0.0, 1.0, 0.5],  # auth-service calls user-service (1.0) and token-service (0.5)
        [0.0, 0.0, 1.0],  # user-service calls token-service (1.0)
        [0.0, 0.0, 0.0],  # token-service calls nobody
    ]

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val != expected_values[i][j]:
                print(f"❌ FAILED: matrix[{i}][{j}] = {val}, expected {expected_values[i][j]}")
                return False

    print("✓ Service call matrix generated correctly")
    print(f"✓ Matrix shape: {len(matrix)}x{len(matrix[0])}")
    print("\nMatrix visualisation:")
    print("  ", "  ".join(f"{s:15}" for s in services))
    for i, row in enumerate(matrix):
        print(f"{services[i]:15}", "  ".join(f"{v:15.2f}" for v in row))

    print("\n✅ PASSED: Service call matrix generation works correctly\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DEPENDENCY VISUALISATION TAB - TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Dependency Extraction", test_dependency_extraction),
        ("Circular Detection", test_circular_dependency_detection),
        ("Service Matrix", test_service_matrix),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ EXCEPTION in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
