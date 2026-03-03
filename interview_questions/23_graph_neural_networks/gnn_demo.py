"""
GNN Demo — Pure PyTorch implementations (no torch_geometric dependency).

Demonstrates:
1. SimpleGCNLayer: message passing with adjacency matrix
2. demo_message_passing(): visualize neighbor aggregation
3. demo_transe(): learn TransE embeddings on a toy knowledge graph
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── GCN Layer ────────────────────────────────────────────────────────────────


class SimpleGCNLayer(nn.Module):
    """Graph Convolutional Network layer using adjacency matrix multiplication.

    Implements: H' = σ(D̃⁻¹/²ÃD̃⁻¹/²HW)
    Simplified: H' = σ(ÂHW) where Â = D̃⁻¹/²ÃD̃⁻¹/²
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Add self-loops: Ã = A + I
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

        # Symmetric normalization: D̃⁻¹/²ÃD̃⁻¹/²
        degree = adj_hat.sum(dim=1)
        d_inv_sqrt = torch.diag(degree.pow(-0.5))
        adj_norm = d_inv_sqrt @ adj_hat @ d_inv_sqrt

        # Message passing + transform: σ(ÂXW)
        return F.relu(adj_norm @ x @ self.weight)


class SimpleGCN(nn.Module):
    """Two-layer GCN for node classification."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(in_features, hidden)
        self.gcn2 = SimpleGCNLayer(hidden, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adj)
        h = F.dropout(h, p=0.5, training=self.training)
        return self.gcn2(h, adj)


# ── Demo: Message Passing ────────────────────────────────────────────────────


def demo_message_passing():
    """Visualize how message passing aggregates neighbor information."""
    print("=" * 60)
    print("DEMO: Message Passing in GCN")
    print("=" * 60)

    # Build a small graph:
    #   0 -- 1 -- 2
    #   |         |
    #   3 ------- 4
    adj = torch.tensor([
        [0, 1, 0, 1, 0],  # node 0 connects to 1, 3
        [1, 0, 1, 0, 0],  # node 1 connects to 0, 2
        [0, 1, 0, 0, 1],  # node 2 connects to 1, 4
        [1, 0, 0, 0, 1],  # node 3 connects to 0, 4
        [0, 0, 1, 1, 0],  # node 4 connects to 2, 3
    ], dtype=torch.float32)

    # Each node has a 3-dim feature vector
    features = torch.tensor([
        [1.0, 0.0, 0.0],  # node 0: "red"
        [0.0, 1.0, 0.0],  # node 1: "green"
        [0.0, 0.0, 1.0],  # node 2: "blue"
        [1.0, 1.0, 0.0],  # node 3: "yellow"
        [0.0, 1.0, 1.0],  # node 4: "cyan"
    ])

    print("\nGraph edges: 0-1, 1-2, 0-3, 2-4, 3-4")
    print(f"\nInitial features:\n{features}")

    # Manual message passing (without learned weights)
    adj_hat = adj + torch.eye(5)
    degree = adj_hat.sum(dim=1)
    d_inv_sqrt = torch.diag(degree.pow(-0.5))
    adj_norm = d_inv_sqrt @ adj_hat @ d_inv_sqrt

    aggregated = adj_norm @ features
    print(f"\nAfter 1 round of message passing (no weights):")
    print(f"  Node 0: {features[0].tolist()} -> {aggregated[0].tolist()}")
    print(f"  (mixed in features from neighbors 1, 3 and self)")
    print(f"\n  Node 2: {features[2].tolist()} -> {aggregated[2].tolist()}")
    print(f"  (mixed in features from neighbors 1, 4 and self)")

    # Now with a GCN layer (learned weights)
    torch.manual_seed(42)
    gcn = SimpleGCNLayer(3, 2)
    output = gcn(features, adj)
    print(f"\nAfter GCN layer (3 -> 2 dims with learned weights):")
    print(f"  Output shape: {output.shape}")
    print(f"  Node 0: {output[0].detach().tolist()}")
    print(f"  Node 1: {output[1].detach().tolist()}")


# ── Demo: TransE ─────────────────────────────────────────────────────────────


def demo_transe():
    """Learn TransE embeddings on a toy knowledge graph.

    Goal: h + r ≈ t for valid triples.
    """
    print("\n" + "=" * 60)
    print("DEMO: TransE Knowledge Graph Embeddings")
    print("=" * 60)

    # Toy knowledge graph
    entities = ["Einstein", "Germany", "Physics", "Berlin", "Europe"]
    relations = ["born_in", "field", "capital_of", "located_in"]

    # Triples: (head, relation, tail)
    triples = [
        (0, 0, 1),  # Einstein born_in Germany
        (0, 1, 2),  # Einstein field Physics
        (3, 2, 1),  # Berlin capital_of Germany (reversed: Germany has capital Berlin)
        (1, 3, 4),  # Germany located_in Europe
    ]

    entity_to_idx = {e: i for i, e in enumerate(entities)}
    relation_to_idx = {r: i for i, r in enumerate(relations)}

    print(f"\nEntities: {entities}")
    print(f"Relations: {relations}")
    print(f"Triples:")
    for h, r, t in triples:
        print(f"  ({entities[h]}, {relations[r]}, {entities[t]})")

    # TransE model
    embed_dim = 8
    torch.manual_seed(42)
    entity_emb = nn.Embedding(len(entities), embed_dim)
    relation_emb = nn.Embedding(len(relations), embed_dim)

    # Normalize entity embeddings
    with torch.no_grad():
        entity_emb.weight.data = F.normalize(entity_emb.weight.data, dim=1)

    optimizer = torch.optim.Adam(
        list(entity_emb.parameters()) + list(relation_emb.parameters()),
        lr=0.01,
    )

    heads = torch.tensor([h for h, r, t in triples])
    rels = torch.tensor([r for h, r, t in triples])
    tails = torch.tensor([t for h, r, t in triples])

    # Training loop
    print(f"\nTraining TransE (embed_dim={embed_dim})...")
    for epoch in range(500):
        optimizer.zero_grad()

        h_emb = entity_emb(heads)
        r_emb = relation_emb(rels)
        t_emb = entity_emb(tails)

        # Positive score: ||h + r - t||
        pos_score = torch.norm(h_emb + r_emb - t_emb, dim=1)

        # Negative sampling: corrupt tails
        neg_tails = torch.randint(0, len(entities), (len(triples),))
        neg_t_emb = entity_emb(neg_tails)
        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, dim=1)

        # Margin-based loss: max(0, margin + pos - neg)
        margin = 1.0
        loss = F.relu(margin + pos_score - neg_score).mean()

        loss.backward()
        optimizer.step()

        # Re-normalize entity embeddings
        with torch.no_grad():
            entity_emb.weight.data = F.normalize(entity_emb.weight.data, dim=1)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")

    # Test: does h + r ≈ t?
    print("\nVerification (lower distance = better):")
    with torch.no_grad():
        for h, r, t in triples:
            h_e = entity_emb(torch.tensor(h))
            r_e = relation_emb(torch.tensor(r))
            t_e = entity_emb(torch.tensor(t))
            dist = torch.norm(h_e + r_e - t_e).item()
            print(f"  {entities[h]} + {relations[r]} ≈ {entities[t]}  (dist={dist:.4f})")

    # Analogy: Germany + capital_of = ?
    print("\nLink prediction: Germany + capital_of = ?")
    with torch.no_grad():
        query = entity_emb(torch.tensor(entity_to_idx["Germany"])) + \
                relation_emb(torch.tensor(relation_to_idx["capital_of"]))
        # Note: capital_of goes FROM city TO country in our triples,
        # so we search for the closest entity that could be the head
        distances = torch.norm(entity_emb.weight - query, dim=1)
        ranked = distances.argsort()
        print(f"  Closest entities:")
        for i in range(3):
            idx = ranked[i].item()
            print(f"    {entities[idx]} (dist={distances[idx].item():.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    demo_message_passing()
    demo_transe()
