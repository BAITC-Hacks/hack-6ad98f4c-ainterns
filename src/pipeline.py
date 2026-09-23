from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from .analytics import compute_node_features
except ImportError:
    from analytics import compute_node_features


DEFAULT_DATA = Path("data")
DEFAULT_OUTPUT = Path("output")
RANDOM_SEED = 42
REQUIRED_NODE_COLUMNS = [
    "gid",
    "role",
    "role_score",
    "cluster_id",
    "priority_score",
    "evidence",
]


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(data_dir / "nodes.parquet"),
        pd.read_parquet(data_dir / "edges.parquet"),
        pd.read_parquet(data_dir / "transactions.parquet"),
    )


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["gid"].tolist())
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src, row.dst, sum_kzt=float(row.sum_kzt))
    return graph


def cluster_nodes(graph: nx.Graph) -> dict[object, int]:
    """Cluster every component; Louvain is deterministic with the fixed seed."""
    communities = nx.community.louvain_communities(graph, weight="sum_kzt", seed=RANDOM_SEED)
    ordered = sorted(communities, key=lambda members: min(map(str, members)))
    return {gid: cluster_id for cluster_id, members in enumerate(ordered) for gid in members}


def _normalise(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    maximum = float(numeric.max())
    return numeric / maximum if maximum > 0 else pd.Series(0.0, index=values.index)


def add_priority(features: pd.DataFrame) -> pd.DataFrame:
    """Priority = .35 role + .25 turnover + .20 degree + .20 pass-through.

    Each metric is scaled to [0, 1] by the dataset maximum. Turnover uses
    log1p to prevent a few large KZT flows dominating the score. ``pass_ratio``
    is clipped at 1 because unusually large ratios are already represented by
    turnover and role. The final score is clipped to [0, 1].
    """
    result = features.copy()
    turnover = _normalise(np.log1p(result["turnover"].clip(lower=0)))
    degree = _normalise(result["in_degree"] + result["out_degree"])
    pass_through = pd.to_numeric(result["pass_ratio"], errors="coerce").fillna(0).clip(0, 1)
    result["priority_score"] = (
        0.35 * result["role_score"].clip(0, 1)
        + 0.25 * turnover
        + 0.20 * degree
        + 0.20 * pass_through
    ).clip(0, 1).round(6)
    result["why"] = result.apply(
        lambda row: (
            f"role={row.role} ({row.role_score:.2f}); turnover={row.turnover:.2f} KZT; "
            f"degree={int(row.in_degree + row.out_degree)}; pass_ratio={row.pass_ratio:.2f}"
        ),
        axis=1,
    )
    return result


def make_cluster_table(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    cluster_ids = sorted(features["cluster_id"].unique())
    internal_amount = {cluster_id: 0.0 for cluster_id in cluster_ids}
    gid_to_cluster = features.set_index("gid")["cluster_id"].to_dict()
    for row in edges.itertuples(index=False):
        source_cluster = gid_to_cluster.get(row.src)
        if source_cluster == gid_to_cluster.get(row.dst):
            internal_amount[source_cluster] += float(row.sum_kzt)

    rows = []
    for cluster_id in cluster_ids:
        members = features.loc[features["cluster_id"] == cluster_id]
        top_gids = members.sort_values("priority_score", ascending=False)["gid"].head(5)
        seed_count = int(members["is_seed"].astype(bool).sum()) if "is_seed" in members else 0
        roles = ", ".join(members["role"].value_counts().head(2).index.astype(str))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_nodes": int(len(members)),
                "n_seed": seed_count,
                "sum_kzt_internal": round(internal_amount[cluster_id], 2),
                "top_gids": ",".join(map(str, top_gids.tolist())),
                "hypothesis": f"Наблюдаемая {roles or 'смешанная'}-структура; требует проверки контекста.",
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(nodes: pd.DataFrame, node_path: Path, cluster_path: Path, top_path: Path) -> None:
    written_nodes = pd.read_csv(node_path)
    written_clusters = pd.read_csv(cluster_path)
    written_top = pd.read_csv(top_path)
    assert len(written_nodes) == len(nodes), "nodes_roles.csv must contain every input node"
    assert written_nodes["gid"].is_unique, "gid values must be unique"
    assert not written_nodes[REQUIRED_NODE_COLUMNS].isna().any().any(), "required fields cannot be null"
    assert set(written_nodes["cluster_id"]).issubset(set(written_clusters["cluster_id"]))
    for column in ("role_score", "priority_score"):
        assert written_nodes[column].between(0, 1).all(), f"{column} must be in [0, 1]"
    assert len(written_top) >= min(20, len(nodes)), "top_nodes.csv must contain at least 20 rows"
    assert written_top["priority_score"].is_monotonic_decreasing, "top_nodes.csv must be sorted"
    assert not written_top[["gid", "role", "priority_score", "why"]].isna().any().any()


def run(data_dir: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT) -> None:
    nodes, edges, transactions = load_data(data_dir)
    features = compute_node_features(nodes, edges, transactions)
    if features["gid"].duplicated().any():
        raise ValueError("compute_node_features returned duplicate gid values")
    features["cluster_id"] = features["gid"].map(cluster_nodes(build_graph(nodes, edges)))
    if features["cluster_id"].isna().any():
        raise ValueError("every node must belong to a cluster")
    features = add_priority(features)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_columns = REQUIRED_NODE_COLUMNS
    features[node_columns].to_csv(output_dir / "nodes_roles.csv", index=False)
    make_cluster_table(features, edges).to_csv(output_dir / "clusters.csv", index=False)
    top = features.sort_values(["priority_score", "gid"], ascending=[False, True]).head(20).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top[["rank", "gid", "role", "priority_score", "why"]].to_csv(output_dir / "top_nodes.csv", index=False)
    validate_outputs(nodes, output_dir / "nodes_roles.csv", output_dir / "clusters.csv", output_dir / "top_nodes.csv")
    print(f"created {len(nodes)} nodes, {features['cluster_id'].nunique()} clusters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible Graph of Money exports")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
