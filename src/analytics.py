"""Explainable node-level flow features and heuristic roles.

Roles are deterministic descriptions of observed graph shape, not findings of
intent or wrongdoing. The sample is truncated at depth 4 and seed in-flows are
incomplete, so those limits are explicitly accounted for below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _fast_transit_features(
    gids: pd.Series, transactions: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Count outgoing transactions with an incoming transaction on the prior date.

    The source data has date-only precision. A transaction on the immediately
    following calendar date is necessarily later and less than 48 hours later;
    same-day ordering and events two dates apart cannot be established, so are
    excluded. This is a conservative date-level signal, not a money trace.
    """
    counts = pd.Series(0, index=gids.index, dtype="int64")
    shares = pd.Series(0.0, index=gids.index, dtype="float64")
    src_col = _find_column(transactions, ("src", "sender", "sender_gid", "from_gid"))
    dst_col = _find_column(transactions, ("dst", "receiver", "receiver_gid", "to_gid"))
    date_col = _find_column(transactions, ("date", "timestamp", "datetime", "created_at"))
    if not (src_col and dst_col and date_col) or transactions.empty:
        return counts, shares

    dates = pd.to_datetime(transactions[date_col], errors="coerce").dt.normalize()
    valid = transactions[src_col].notna() & transactions[dst_col].notna() & dates.notna()
    in_dates: dict[object, set[pd.Timestamp]] = {}
    out_dates: dict[object, list[pd.Timestamp]] = {}
    for sender, receiver, date in zip(
        transactions.loc[valid, src_col], transactions.loc[valid, dst_col], dates.loc[valid]
    ):
        in_dates.setdefault(receiver, set()).add(date)
        out_dates.setdefault(sender, []).append(date)

    for pos, gid in enumerate(gids):
        outgoing = out_dates.get(gid, [])
        if outgoing:
            previous_day_incoming = in_dates.get(gid, set())
            fast = sum((date - pd.Timedelta(days=1)) in previous_day_incoming for date in outgoing)
            counts.iloc[pos] = fast
            shares.iloc[pos] = fast / len(outgoing)
    return counts, shares


def compute_node_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per node, with graph measures and one explainable role.

    Role precedence: consolidator, distributor, coordinator, transit, terminal,
    peripheral. Exact thresholds are documented in ``_classify``.
    """
    if "gid" not in nodes.columns:
        raise ValueError("nodes must contain a 'gid' column")
    result = nodes.drop_duplicates("gid", keep="first").copy()
    gids = result["gid"]

    src = _find_column(edges, ("src", "source", "from_gid", "sender_gid"))
    dst = _find_column(edges, ("dst", "target", "to_gid", "receiver_gid"))
    amount_col = _find_column(edges, ("sum_kzt", "amount", "sum_amount", "value"))
    count_col = _find_column(edges, ("n_tx", "transaction_count", "tx_count"))

    if src is None or dst is None:
        edge_data = pd.DataFrame(columns=["src", "dst", "sum_kzt", "n_tx"])
    else:
        edge_data = pd.DataFrame({"src": edges[src], "dst": edges[dst]})
        edge_data["sum_kzt"] = _number(edges[amount_col]) if amount_col else 0.0
        edge_data["n_tx"] = _number(edges[count_col]) if count_col else 1.0
        edge_data = edge_data.dropna(subset=["src", "dst"])

    indeg = edge_data.groupby("dst").size()
    outdeg = edge_data.groupby("src").size()
    in_amount = edge_data.groupby("dst")["sum_kzt"].sum()
    out_amount = edge_data.groupby("src")["sum_kzt"].sum()
    incoming_tx = edge_data.groupby("dst")["n_tx"].sum()
    outgoing_tx = edge_data.groupby("src")["n_tx"].sum()

    result["in_degree"] = gids.map(indeg).fillna(0).astype(int)
    result["out_degree"] = gids.map(outdeg).fillna(0).astype(int)
    result["in_amount"] = gids.map(in_amount).fillna(0.0).astype(float)
    result["out_amount"] = gids.map(out_amount).fillna(0.0).astype(float)

    # Direct transactions provide distinct incoming/outgoing counts. If their
    # endpoints are unavailable, fall back to the edge-level aggregate counts.
    tx_count = None
    in_tx = out_tx = None
    tx_src = _find_column(transactions, ("src", "sender", "sender_gid", "from_gid"))
    tx_dst = _find_column(transactions, ("dst", "receiver", "receiver_gid", "to_gid"))
    if tx_src and tx_dst:
        out_tx = transactions[tx_src].value_counts()
        in_tx = transactions[tx_dst].value_counts()
        tx_count = out_tx.add(in_tx, fill_value=0)
    if tx_count is not None:
        result["transaction_count"] = gids.map(tx_count).fillna(0).astype(int)
        result["in_tx"] = gids.map(in_tx).fillna(0).astype(int)
        result["out_tx"] = gids.map(out_tx).fillna(0).astype(int)
    else:
        result["in_tx"] = gids.map(incoming_tx).fillna(0).astype(float)
        result["out_tx"] = gids.map(outgoing_tx).fillna(0).astype(float)
        result["transaction_count"] = result["in_tx"] + result["out_tx"]

    depth_col = _find_column(nodes, ("depth", "min_depth", "hop", "hops"))
    if depth_col:
        depth_map_nodes = nodes.drop_duplicates("gid", keep="first").set_index("gid")[depth_col]
        result["depth"] = gids.map(pd.to_numeric(depth_map_nodes, errors="coerce")).values
    else:
        result["depth"] = np.nan
    if "depth" not in nodes.columns and "depth" in edges.columns and src:
        dep = pd.to_numeric(edges["depth"], errors="coerce")
        depth_map = pd.concat([
            pd.DataFrame({"gid": edges[src], "depth": dep}),
            pd.DataFrame({"gid": edges[dst], "depth": dep + 1}),
        ]).groupby("gid")["depth"].min()
        result["depth"] = gids.map(depth_map)

    seed_col = _find_column(nodes, ("is_seed", "seed", "is_source"))
    if seed_col:
        seed_map = nodes.drop_duplicates("gid", keep="first").set_index("gid")[seed_col]
        result["is_seed"] = gids.map(seed_map).fillna(False).astype(bool).values
    else:
        result["is_seed"] = False

    result["turnover"] = result["in_amount"] + result["out_amount"]
    result["net_flow"] = result["in_amount"] - result["out_amount"]
    result["flow_ratio"] = result["out_amount"] / result["in_amount"].replace(0, np.nan)
    result["pass_ratio"] = result["flow_ratio"].fillna(0.0)
    result["neighbor_count"] = result["in_degree"] + result["out_degree"]
    result["truncated_by_depth"] = result["depth"].eq(4) & result["out_degree"].eq(0)
    result["fast_transit_count"], result["fast_transit_share"] = _fast_transit_features(
        gids, transactions
    )
    result["role"], result["role_score"], result["evidence"] = zip(*[
        _classify(row) for row in result.to_dict(orient="records")
    ]) if len(result) else ([], [], [])
    return result


def _classify(row: dict) -> tuple[str, float, str]:
    """Apply fixed rules with precedence: consolidator > distributor >
    coordinator > transit > terminal > peripheral.

    Consolidator: >=2 incoming edges and inbound amount >=1.25x outbound.
    Distributor: >=2 outgoing edges and outbound amount >=1.25x inbound.
    Coordinator: >=4 incident edges and at least one edge in each direction.
    Transit: >=1 edge in each direction and outbound/inbound amount in [0.5, 2].
    Terminal: has incoming edge, no outgoing edge, and depth <4 (or unknown).
    Peripheral: all other cases, including isolated nodes and depth-4 cutoffs.
    Seed nodes' incomplete incoming flow cannot alone support a consolidator role.
    """
    ind = int(row["in_degree"])
    out = int(row["out_degree"])
    ia = float(row["in_amount"])
    oa = float(row["out_amount"])
    dep = row.get("depth")
    depth = float(dep) if pd.notna(dep) else np.nan
    seed = bool(row["is_seed"])
    tx = float(row["transaction_count"])
    eps = 1e-12
    if ind >= 2 and not seed and ia >= 1.25 * oa:
        role, raw = "consolidator", min(ind / 4, ia / max(1.25 * max(oa, eps), eps) / 2)
    elif out >= 2 and oa >= 1.25 * ia:
        role, raw = "distributor", min(out / 4, oa / max(1.25 * max(ia, eps), eps) / 2)
    elif ind + out >= 4 and ind > 0 and out > 0:
        role, raw = "coordinator", min((ind + out) / 8, 1.0)
    elif ind > 0 and out > 0 and (ia == 0 or 0.5 <= oa / ia <= 2.0):
        role, raw = "transit", min(1.0, 0.5 + min(ind, out) / 4)
    elif ind > 0 and out == 0 and not (pd.notna(depth) and depth >= 4):
        role, raw = "terminal", min(1.0, 0.5 + ind / 4)
    else:
        role, raw = "peripheral", 0.5 if ind + out == 0 else 0.35
    score = float(np.clip(raw, 0.0, 1.0))
    depth_text = "неизвестна" if pd.isna(depth) else f"{depth:g}"
    evidence = (
        f"{role}: вход. рёбер {ind}, исход. {out}; суммы {ia:.2f}/{oa:.2f} KZT; "
        f"транзакций {tx:g}, depth {depth_text}, seed={'да' if seed else 'нет'}"
    )
    if role == "transit":
        evidence += (
            f"; быстрый транзит {int(row['fast_transit_count'])} "
            f"(доля {float(row['fast_transit_share']):.2f}, дата +1)"
        )
    return role, score, evidence[:200]
