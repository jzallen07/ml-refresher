#!/usr/bin/env python3
"""One-time script to build/validate the concept graph.

Usage:
    uv run python scripts/build_concept_graph.py          # validate existing graph
    uv run python scripts/build_concept_graph.py --stats   # show statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.graph import ConceptGraph, GRAPH_PATH


def validate_graph(cg: ConceptGraph) -> list[str]:
    """Check for issues in the concept graph."""
    errors: list[str] = []

    # Check for cycles in PREREQUISITE edges
    import networkx as nx

    prereq_subgraph = nx.DiGraph()
    for u, v, data in cg._graph.edges(data=True):
        if data["type"] in ("PREREQUISITE", "BUILDS_ON"):
            prereq_subgraph.add_edge(u, v)

    try:
        nx.find_cycle(prereq_subgraph)
        errors.append("CYCLE detected in PREREQUISITE/BUILDS_ON edges!")
    except nx.NetworkXNoCycle:
        pass

    # Check for dangling references
    all_ids = set(cg._node_attrs.keys())
    for u, v, _ in cg._graph.edges(data=True):
        if u not in all_ids:
            errors.append(f"Edge source '{u}' not in node list")
        if v not in all_ids:
            errors.append(f"Edge target '{v}' not in node list")

    # Check question nodes have topic attribute
    for nid, attrs in cg._node_attrs.items():
        if attrs.get("type") == "question" and not attrs.get("topic"):
            errors.append(f"Question node '{nid}' missing 'topic' attribute")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Build/validate concept graph")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()

    if not GRAPH_PATH.exists():
        print(f"Graph file not found: {GRAPH_PATH}")
        sys.exit(1)

    cg = ConceptGraph.load(GRAPH_PATH)

    if cg.is_empty():
        print("Graph is empty!")
        sys.exit(1)

    errors = validate_graph(cg)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Graph validation passed!")

    summary = cg.summary()
    print(f"\nNodes: {summary['total_nodes']}")
    for ntype, count in sorted(summary["nodes_by_type"].items()):
        print(f"  {ntype}: {count}")
    print(f"Edges: {summary['total_edges']}")
    for etype, count in sorted(summary["edges_by_type"].items()):
        print(f"  {etype}: {count}")

    if args.stats:
        # Show topological order
        topo = cg.topological_sort_prerequisites()
        print(f"\nTopological order ({len(topo)} nodes in prereq subgraph):")
        for i, nid in enumerate(topo, 1):
            node = cg.get_node(nid)
            label = node.get("label", nid) if node else nid
            print(f"  {i}. {label} ({nid})")


if __name__ == "__main__":
    main()
