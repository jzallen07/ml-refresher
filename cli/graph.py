from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "data" / "concept_graph.json"

_instance: ConceptGraph | None = None


class ConceptGraph:
    """Wraps a concept relationship graph stored as a NetworkX DiGraph.

    Node types: topic, question, concept
    Edge types: PREREQUISITE, BUILDS_ON, PART_OF, RELATED_TO
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_attrs: dict[str, dict] = {}

    # -- Loading --

    @classmethod
    def load(cls, path: Path = GRAPH_PATH) -> ConceptGraph:
        cg = cls()
        if not path.exists():
            return cg

        data = json.loads(path.read_text())

        for node in data.get("nodes", []):
            node_id = node["id"]
            cg._graph.add_node(node_id)
            cg._node_attrs[node_id] = node

        for edge in data.get("edges", []):
            cg._graph.add_edge(
                edge["source"],
                edge["target"],
                type=edge["type"],
            )

        return cg

    def is_empty(self) -> bool:
        return len(self._graph) == 0

    # -- Node queries --

    def get_node(self, node_id: str) -> dict | None:
        return self._node_attrs.get(node_id)

    def get_prerequisites(self, node_id: str) -> list[dict]:
        """Direct predecessors via PREREQUISITE or BUILDS_ON edges."""
        results = []
        for pred in self._graph.predecessors(node_id):
            edge_data = self._graph.edges[pred, node_id]
            if edge_data["type"] in ("PREREQUISITE", "BUILDS_ON"):
                node = self._node_attrs.get(pred, {"id": pred})
                results.append({**node, "edge_type": edge_data["type"]})
        return results

    def get_dependents(self, node_id: str) -> list[dict]:
        """Direct successors via PREREQUISITE or BUILDS_ON edges."""
        results = []
        for succ in self._graph.successors(node_id):
            edge_data = self._graph.edges[node_id, succ]
            if edge_data["type"] in ("PREREQUISITE", "BUILDS_ON"):
                node = self._node_attrs.get(succ, {"id": succ})
                results.append({**node, "edge_type": edge_data["type"]})
        return results

    def get_neighbors(self, node_id: str, hops: int = 1) -> dict[str, list[dict]]:
        """Get neighbors grouped by edge type, up to N hops."""
        grouped: dict[str, list[dict]] = {}
        visited = {node_id}
        frontier = {node_id}

        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                # Predecessors
                for pred in self._graph.predecessors(nid):
                    if pred not in visited:
                        visited.add(pred)
                        next_frontier.add(pred)
                        edge_type = self._graph.edges[pred, nid]["type"]
                        node = self._node_attrs.get(pred, {"id": pred})
                        entry = {**node, "direction": "incoming", "edge_type": edge_type}
                        grouped.setdefault(edge_type, []).append(entry)
                # Successors
                for succ in self._graph.successors(nid):
                    if succ not in visited:
                        visited.add(succ)
                        next_frontier.add(succ)
                        edge_type = self._graph.edges[nid, succ]["type"]
                        node = self._node_attrs.get(succ, {"id": succ})
                        entry = {**node, "direction": "outgoing", "edge_type": edge_type}
                        grouped.setdefault(edge_type, []).append(entry)
            frontier = next_frontier

        return grouped

    def get_concepts_for_topic(self, topic_slug: str) -> list[dict]:
        """Get concepts connected to a topic via PART_OF edges."""
        topic_id = f"topic:{topic_slug}"
        results = []
        for pred in self._graph.predecessors(topic_id):
            edge_data = self._graph.edges[pred, topic_id]
            if edge_data["type"] == "PART_OF":
                node = self._node_attrs.get(pred, {"id": pred})
                results.append(node)
        return results

    def get_all_prerequisites_recursive(self, node_id: str) -> list[dict]:
        """Transitive ancestors via PREREQUISITE/BUILDS_ON edges."""
        visited = set()
        result = []

        def _walk(nid: str):
            for pred in self._graph.predecessors(nid):
                edge_data = self._graph.edges[pred, nid]
                if edge_data["type"] in ("PREREQUISITE", "BUILDS_ON") and pred not in visited:
                    visited.add(pred)
                    node = self._node_attrs.get(pred, {"id": pred})
                    result.append(node)
                    _walk(pred)

        _walk(node_id)
        return result

    def topological_sort_prerequisites(self) -> list[str]:
        """Topological sort over the PREREQUISITE/BUILDS_ON subgraph (foundations first)."""
        subgraph = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            if data["type"] in ("PREREQUISITE", "BUILDS_ON"):
                subgraph.add_edge(u, v)

        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            return []

    def generate_learning_path(
        self,
        target_topic: str | None = None,
        target_concept: str | None = None,
    ) -> list[dict]:
        """Generate an ordered learning path for a target, prerequisites first.

        Returns a list of dicts with node info, order index, and classification
        (foundation / intermediate / target).
        """
        # Determine target nodes
        targets: set[str] = set()
        if target_topic:
            topic_id = f"topic:{target_topic}"
            # Include all question nodes for this topic
            for nid, attrs in self._node_attrs.items():
                if attrs.get("type") == "question" and attrs.get("topic") == target_topic:
                    targets.add(nid)
            if not targets:
                targets.add(topic_id)
        elif target_concept:
            concept_id = f"concept:{target_concept}"
            targets.add(concept_id)
        else:
            return []

        # Collect all needed nodes: targets + transitive prereqs
        needed: set[str] = set(targets)
        for t in targets:
            for prereq in self.get_all_prerequisites_recursive(t):
                needed.add(prereq["id"])

        # Build subgraph of PREREQUISITE/BUILDS_ON edges among needed nodes
        subgraph = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            if data["type"] in ("PREREQUISITE", "BUILDS_ON") and u in needed and v in needed:
                subgraph.add_edge(u, v)
        # Add isolated needed nodes
        for n in needed:
            subgraph.add_node(n)

        # Topological sort
        try:
            ordered = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            ordered = sorted(needed)

        # Classify each node
        path = []
        for i, nid in enumerate(ordered):
            node = self._node_attrs.get(nid, {"id": nid})
            preds_in_sub = list(subgraph.predecessors(nid))
            succs_in_sub = list(subgraph.successors(nid))

            if nid in targets:
                classification = "target"
            elif not preds_in_sub:
                classification = "foundation"
            else:
                classification = "intermediate"

            path.append({
                "order": i + 1,
                "id": nid,
                "label": node.get("label", nid),
                "type": node.get("type", "unknown"),
                "classification": classification,
                "prerequisites": [self._node_attrs.get(p, {"id": p}).get("label", p) for p in preds_in_sub],
            })

        return path

    def summary(self) -> dict:
        """Node/edge counts by type."""
        node_counts: dict[str, int] = {}
        for attrs in self._node_attrs.values():
            ntype = attrs.get("type", "unknown")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            etype = data.get("type", "unknown")
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "total_nodes": len(self._graph),
            "total_edges": self._graph.number_of_edges(),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }


def get_concept_graph() -> ConceptGraph:
    """Module-level singleton accessor."""
    global _instance
    if _instance is None:
        _instance = ConceptGraph.load()
    return _instance
