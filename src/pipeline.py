from pathlib import Path
import pandas as pd
import networkx as nx

DATA = Path("data")
OUTPUT = Path("output")


def load_data():
    edges = pd.read_parquet(DATA / "edges.parquet")
    nodes = pd.read_parquet(DATA / "nodes.parquet")
    transactions = pd.read_parquet(DATA / "transactions.parquet")
    return nodes, edges, transactions


def build_graph(edges):
    G = nx.DiGraph()

    for row in edges.itertuples(index=False):
        G.add_edge(
            row.src,
            row.dst,
            sum_kzt=row.sum_kzt,
            n_tx=row.n_tx,
            depth=row.depth,
        )

    return G


def calculate_metrics(nodes, edges, G):
    result = nodes.copy()

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    in_amount = edges.groupby("dst")["sum_kzt"].sum()
    out_amount = edges.groupby("src")["sum_kzt"].sum()

    result["in_degree"] = result["gid"].map(in_degree).fillna(0).astype(int)
    result["out_degree"] = result["gid"].map(out_degree).fillna(0).astype(int)

    result["in_amount"] = result["gid"].map(in_amount).fillna(0)
    result["out_amount"] = result["gid"].map(out_amount).fillna(0)

    result["turnover"] = result["in_amount"] + result["out_amount"]

    result["pass_ratio"] = (
        result["out_amount"] / result["in_amount"].replace(0, pd.NA)
    ).fillna(0)

    return result


def main():
    OUTPUT.mkdir(exist_ok=True)

    nodes, edges, transactions = load_data()

    print(f"nodes: {len(nodes)}")
    print(f"edges: {len(edges)}")
    print(f"transactions: {len(transactions)}")

    G = build_graph(edges)
    metrics = calculate_metrics(nodes, edges, G)

    metrics.to_csv(OUTPUT / "nodes_metrics.csv", index=False)

    print("created: output/nodes_metrics.csv")


if __name__ == "__main__":
    main()
