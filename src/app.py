"""Local AML analyst dashboard for the Money Graph project."""
from pathlib import Path
import math
import re

import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "output"
DATA = ROOT.parent / "data"
ROLE_COLORS = {
    "consolidator": "#e4572e", "transit": "#7b2cbf",
    "distributor": "#f3a712", "terminal": "#168aad",
    "coordinator": "#2a9d8f", "peripheral": "#718096",
    "unknown": "#94a3b8",
}

st.set_page_config(page_title="Граф денег · AML", page_icon="◉", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:Manrope, sans-serif;}
.block-container {padding-top:1.4rem; max-width:1500px;}
[data-testid="stMetric"] {background:#111d2b;padding:14px 18px;border:1px solid #263648;border-radius:12px;}
[data-testid="stSidebar"] {background:#0b1420;}
.small-muted {color:#8fa2b7;font-size:.88rem;}
.pill {display:inline-block;padding:4px 10px;border-radius:100px;background:#203247;color:#c7d7e6;font-size:.8rem;}
</style>""", unsafe_allow_html=True)


def load_csv(name):
    path = OUT / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Не удалось прочитать {path}: {exc}")
        return pd.DataFrame()


def col(df, candidates, default=None):
    for name in candidates:
        if name in df.columns:
            return name
    return default


def fmt(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}".replace(",", " ")
    return str(value)


def gid_key(value):
    """Normalize numeric CSV/parquet IDs without turning integer gids into x.0."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


@st.cache_data(show_spinner=False)
def read_parquet(path, modified):
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def directed_graph_from_file(path, modified):
    frame = read_parquet(path, modified)
    required = {"src", "dst", "sum_kzt"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("ожидаются колонки src, dst, sum_kzt; отсутствуют " + ", ".join(missing))
    graph = nx.DiGraph()
    graph.add_edges_from((gid_key(src), gid_key(dst)) for src, dst in zip(frame["src"], frame["dst"]))
    return frame, graph


def nearest_seed_path(graph, seeds, target):
    """Return a shortest directed seed-to-target path, tie-breaking by seed gid."""
    seed_keys = {gid_key(seed) for seed in seeds}
    if target in seed_keys:
        return target, [target]
    best_seed = None
    best_distance = None
    for seed in sorted(seed_keys):
        if seed not in graph:
            continue
        try:
            distance = nx.shortest_path_length(graph, seed, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if best_distance is None or distance < best_distance:
            best_seed, best_distance = seed, distance
    if best_seed is None:
        return None
    return best_seed, nx.shortest_path(graph, best_seed, target)


def require_columns(df, filename, required):
    missing = [name for name in required if name not in df.columns]
    if missing:
        st.error(f"В {filename} не хватает колонок: {', '.join(missing)}.")
        return False
    return True


def parse_top_gids(value):
    """Parse comma-separated numeric gids stored in one CSV cell."""
    if pd.isna(value) or not str(value).strip():
        return []
    gids = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            try:
                gids.append(str(int(item)))
            except ValueError:
                continue
    return gids


def esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def draw_graph(nodes, edges, selected_gid, gid_col, role_col, title="Ближайшие связи",
               selected_neighborhood=True, include_all_nodes=False, seed_path=None):
    gid_col = col(nodes, ["gid", "node", "id"])
    src_col = col(edges, ["src", "source", "from_gid"])
    dst_col = col(edges, ["dst", "target", "to_gid"])
    if not src_col or not dst_col:
        st.error("В data/edges.parquet не найдены обязательные колонки src и dst.")
        return
    selected = str(selected_gid)
    src_keys = edges[src_col].map(gid_key)
    dst_keys = edges[dst_col].map(gid_key)
    relevant = edges[(src_keys == selected) | (dst_keys == selected)].copy() if selected_neighborhood else edges.copy()
    if selected_neighborhood and len(relevant) > 80:
        amount_col = col(relevant, ["sum_kzt", "amount", "weight"])
        relevant = relevant.assign(_rank=pd.to_numeric(relevant[amount_col], errors="coerce").fillna(0) if amount_col else 0).nlargest(80, "_rank")
    path_edges = set(zip(seed_path, seed_path[1:])) if seed_path and len(seed_path) > 1 else set()
    if path_edges:
        all_pairs = pd.Series(list(zip(src_keys, dst_keys)), index=edges.index)
        path_frame = edges.loc[all_pairs.isin(path_edges)]
        relevant = pd.concat([relevant, path_frame]).drop_duplicates(subset=[src_col, dst_col])
    node_records = nodes.to_dict(orient="records")
    node_map = {gid_key(record[gid_col]): record for record in node_records}
    neighbor_ids = set(relevant[src_col].map(gid_key)) | set(relevant[dst_col].map(gid_key))
    keep = {selected} | neighbor_ids
    if seed_path:
        keep.update(seed_path)
    if include_all_nodes:
        keep.update(nodes[gid_col].map(gid_key))
    # Cap visual clutter without changing the loaded data or adjacency shown in the card.
    if not selected_neighborhood and len(relevant) > 80:
        amount_col = col(relevant, ["sum_kzt", "amount", "weight"])
        relevant = relevant.assign(_rank=pd.to_numeric(relevant[amount_col], errors="coerce").fillna(0) if amount_col else 0).nlargest(80, "_rank")
        keep = {selected} | set(relevant[src_col].map(gid_key)) | set(relevant[dst_col].map(gid_key))
    amount_col = col(relevant, ["sum_kzt", "amount", "weight"])
    tx_col = col(relevant, ["n_tx", "tx_count", "transactions"])
    # Place the selected gid at the center and its neighbors around it.
    positions = {selected: (0.0, 0.0)}
    ordered = sorted(keep - {selected})
    for i, gid in enumerate(ordered):
        angle = 2 * math.pi * i / max(1, len(ordered))
        positions[gid] = (2.0 * math.cos(angle), 2.0 * math.sin(angle))
    fig = go.Figure()
    edge_x, edge_y, edge_labels, edge_hover = [], [], [], []
    for edge in relevant.to_dict(orient="records"):
        a, b = gid_key(edge[src_col]), gid_key(edge[dst_col])
        x1, y1 = positions[a]; x2, y2 = positions[b]
        on_seed_path = (a, b) in path_edges
        fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.5 if on_seed_path else 1.1,
                           arrowwidth=3.5 if on_seed_path else 1.7,
                           arrowcolor="#ff7a45" if on_seed_path else "#8297ad")
        edge_x.append((x1 + x2) / 2)
        edge_y.append((y1 + y2) / 2)
        edge_labels.append(fmt(edge.get(amount_col)) if amount_col else "")
        tx_value = fmt(edge.get(tx_col)) if tx_col else "не указано"
        edge_hover.append(f"{a} → {b}<br>Сумма: {fmt(edge.get(amount_col)) if amount_col else 'не указана'} KZT<br>Транзакций: {tx_value}")
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="markers+text", text=edge_labels,
            textposition="top center", textfont={"size": 10, "color": "#dce8f4"},
            marker={"size": 18, "color": "#dce8f4", "opacity": 0.01},
            customdata=edge_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False, name="Связи",
        ))
    for role_name, color in ROLE_COLORS.items():
        ids = [gid for gid in keep if ((str(node_map[gid].get(role_col, "unknown")).lower() if role_col and gid in node_map else "unknown") == role_name)]
        if not ids:
            continue
        node_hover = []
        for gid in ids:
            record = node_map.get(gid, {})
            node_role = str(record.get(role_col, "unknown")) if role_col else "unknown"
            cluster = gid_key(record.get("cluster_id", "—"))
            priority = fmt(record.get("priority_score"))
            node_hover.append(f"gid {gid}<br>Роль: {node_role}<br>Кластер: {cluster}<br>Приоритет: {priority}")
        fig.add_trace(go.Scatter(x=[positions[g][0] for g in ids], y=[positions[g][1] for g in ids],
            mode="markers+text", text=ids, textposition="top center", name=role_name,
            marker={"size":[32 if g == selected else 19 for g in ids], "color":color,
                    "line":{"width":[5 if g == selected else (3 if seed_path and g in seed_path else 1) for g in ids],
                            "color":["#f4bd50" if g == selected else ("#ff6b35" if seed_path and g in seed_path else color) for g in ids]}},
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 16}},
        height=510, paper_bgcolor="#0c1724", plot_bgcolor="#0c1724",
        font={"color":"#dbe7f2"}, margin={"l":10,"r":10,"t":55,"b":10}, showlegend=False,
        hoverlabel={"bgcolor":"#152437", "bordercolor":"#38516b", "font":{"color":"#f2f6fa"}},
        dragmode="pan", uirevision=f"{title}:{selected}",
        xaxis={"visible":False,"range":[-2.8,2.8],"fixedrange":False},
        yaxis={"visible":False,"range":[-2.8,2.8],"scaleanchor":"x","fixedrange":False},
    )
    st.plotly_chart(
        fig, width="stretch",
        config={
            "displayModeBar": True, "displaylogo": False, "scrollZoom": True,
            "doubleClick": "reset", "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {"format": "png", "filename": "money-graph-neighborhood", "scale": 2},
            "responsive": True,
        },
    )

st.title("Граф денег")
st.caption("Локальная рабочая панель AML-аналитика · признаки и гипотезы для проверки")
nodes = load_csv("nodes_roles.csv")
clusters = load_csv("clusters.csv")
top = load_csv("top_nodes.csv")

if nodes.empty:
    missing = [name for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv") if not (OUT / name).exists()]
    st.info("Для панели нужны выходные CSV из пайплайна. " +
            ("Не найдены: " + ", ".join(missing) + ". " if missing else "nodes_roles.csv пуст. ") +
            f"Ожидаемая папка: `{OUT}`. Пересчитайте результаты командой `python -m src.pipeline` из корня проекта.")
    st.stop()

required_node_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
if not require_columns(nodes, "nodes_roles.csv", required_node_cols):
    st.stop()
if nodes["gid"].isna().any() or nodes["gid"].duplicated().any():
    st.error("В nodes_roles.csv колонка gid должна быть заполнена и уникальна.")
    st.stop()
if nodes[["role", "role_score", "cluster_id", "priority_score", "evidence"]].isna().any().any():
    st.error("В nodes_roles.csv найдены незаполненные обязательные поля.")
    st.stop()
if nodes["evidence"].astype(str).str.strip().eq("").any():
    st.error("Поле evidence в nodes_roles.csv должно содержать объяснение для каждого узла.")
    st.stop()
allowed_roles = set(ROLE_COLORS) - {"unknown"}
unexpected_roles = sorted(set(nodes["role"].astype(str)) - allowed_roles)
if unexpected_roles:
    st.error("В nodes_roles.csv обнаружены неизвестные роли: " + ", ".join(unexpected_roles))
    st.stop()
for score_name in ("role_score", "priority_score"):
    score_values = pd.to_numeric(nodes[score_name], errors="coerce")
    if score_values.isna().any() or not score_values.between(0, 1).all():
        st.error(f"Значения {score_name} должны быть числами в диапазоне [0, 1].")
        st.stop()

gid_col, role_col = "gid", "role"
role_score_col, priority_score_col = "role_score", "priority_score"
cluster_col, evidence_col = "cluster_id", "evidence"
top_schema_ok = require_columns(top, "top_nodes.csv", ["rank", "gid", "role", "priority_score", "why"]) if not top.empty else False
clusters_schema_ok = require_columns(clusters, "clusters.csv", ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]) if not clusters.empty else False

edges_path, nodes_path = DATA / "edges.parquet", DATA / "nodes.parquet"
transactions_path = DATA / "transactions.parquet"
edges, graph, node_meta = pd.DataFrame(), None, pd.DataFrame()
if edges_path.exists():
    try:
        edges, graph = directed_graph_from_file(str(edges_path), edges_path.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Не удалось загрузить data/edges.parquet: {exc}")
edge_src = col(edges, ["src", "source", "from_gid"])
edge_dst = col(edges, ["dst", "target", "to_gid"])
edge_amount = col(edges, ["sum_kzt", "amount", "weight"])
edges_valid = bool(not edges.empty and edge_src and edge_dst and edge_amount and graph is not None)

seed_ids, seed_data_ok = [], False
if nodes_path.exists():
    try:
        node_meta = read_parquet(str(nodes_path), nodes_path.stat().st_mtime_ns)
        if {"gid", "is_seed"}.issubset(node_meta.columns):
            seed_ids = node_meta.loc[node_meta["is_seed"].fillna(False).astype(bool), "gid"].map(gid_key).tolist()
            seed_data_ok = True
        else:
            st.warning("Для поиска seed-пути в nodes.parquet ожидаются колонки gid и is_seed.")
    except Exception as exc:
        st.warning(f"Не удалось прочитать seed из data/nodes.parquet: {exc}")

transaction_count = None
if transactions_path.exists():
    try:
        transaction_count = len(read_parquet(str(transactions_path), transactions_path.stat().st_mtime_ns))
    except Exception as exc:
        st.warning(f"Не удалось прочитать data/transactions.parquet: {exc}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Узлы", f"{len(nodes):,}".replace(",", " "))
k2.metric("Направленные рёбра", f"{len(edges):,}".replace(",", " ") if edges_valid else "—")
k3.metric("Транзакции", f"{transaction_count:,}".replace(",", " ") if transaction_count is not None else "—")
k4.metric("Seed", f"{len(seed_ids):,}".replace(",", " ") if seed_data_ok else "—")
k5.metric("Кластеры", f"{clusters[cluster_col].nunique():,}" if clusters_schema_ok else "—")
if edges_valid:
    st.caption(f"Сумма агрегированных рёбер: {fmt(edges[edge_amount].sum())} KZT · порог выборки 5 000 KZT.")
elif not edges_path.exists():
    st.warning(f"Не найден файл связей: {edges_path}")

all_gids = nodes[gid_col].map(gid_key).tolist()
gid_set = set(all_gids)
if not all_gids:
    st.error("В nodes_roles.csv нет gid для поиска.")
    st.stop()
if "selected_gid" not in st.session_state or st.session_state["selected_gid"] not in gid_set:
    st.session_state["selected_gid"] = all_gids[0]

st.subheader("Топ узлов для проверки")
st.caption("Щёлкните строку, чтобы открыть карточку узла. Роль и приоритет — аналитические признаки.")
if top_schema_ok:
    top_view = top.rename(columns={"rank": "Ранг", "gid": "GID", "role": "Роль",
                                   "priority_score": "Приоритет", "why": "Почему в топе"})
    top_event = st.dataframe(
        top_view[["Ранг", "GID", "Роль", "Приоритет", "Почему в топе"]],
        hide_index=True, width="stretch", height=390, on_select="rerun",
        selection_mode="single-row", key="top_nodes_table",
    )
    selected_rows = list(top_event.selection.rows)
    previous_rows = st.session_state.get("_top_rows", [])
    if selected_rows and selected_rows != previous_rows:
        index = selected_rows[0]
        if 0 <= index < len(top):
            candidate = gid_key(top["gid"].iloc[index])
            if candidate in gid_set:
                st.session_state["selected_gid"] = candidate
    st.session_state["_top_rows"] = selected_rows
else:
    st.info("Топ-лист недоступен: output/top_nodes.csv отсутствует, пуст или имеет неверную схему.")

with st.sidebar:
    st.header("Поиск узла")
    with st.form("gid_search_form"):
        search = st.text_input("GID из nodes_roles.csv", placeholder="Введите любой gid")
        submitted = st.form_submit_button("Найти")
    if submitted:
        search_key = gid_key(search)
        if search_key in gid_set:
            st.session_state["selected_gid"] = search_key
        else:
            st.warning("Такой gid не найден в output/nodes_roles.csv.")
    st.caption("Поиск работает и для узла без рёбер.")
    st.selectbox("Выбранный узел", all_gids, key="selected_gid")

selected_gid = st.session_state["selected_gid"]
row = nodes.loc[nodes[gid_col].map(gid_key) == selected_gid].iloc[0]
path_result = nearest_seed_path(graph, seed_ids, selected_gid) if graph is not None and seed_data_ok else None

st.divider()
left, right = st.columns([1.5, 1])
with left:
    st.subheader("Направленное окружение")
    st.caption(f"Выбранный узел {selected_gid} · кластер {gid_key(row[cluster_col])}")
    st.caption("Колесо мыши — масштаб · перетаскивание — перемещение · двойной щелчок — сброс. Наведите курсор на узел или сумму, чтобы увидеть подробности.")
    legend = "&nbsp;&nbsp;".join(
        f'<span style="color:{color}">●</span> {esc(role)}'
        for role, color in ROLE_COLORS.items() if role != "unknown"
    )
    st.markdown(legend + ' &nbsp;&nbsp; <span style="color:#ff6b35">━</span> путь от seed', unsafe_allow_html=True)
    if not edges_valid:
        st.info("Направленный граф недоступен: нужны data/edges.parquet с колонками src, dst, sum_kzt.")
        seed_path = None
    elif not seed_data_ok:
        st.warning("Путь от seed недоступен: в data/nodes.parquet нет подтверждённого списка gid и is_seed.")
        seed_path = None
    elif path_result is None:
        st.warning("От seed до этого узла нет направленного пути в наблюдаемом графе.")
        seed_path = None
    else:
        seed, seed_path = path_result
        if len(seed_path) == 1:
            st.success(f"Выбранный узел {selected_gid} — seed.")
        else:
            st.success(f"Seed {seed} → {' → '.join(seed_path)} · {len(seed_path) - 1} рёбер. Путь выделен оранжевым.")
    if edges_valid:
        draw_graph(nodes, edges, selected_gid, gid_col, role_col,
                   "Ближайшие связи и путь от seed", seed_path=seed_path)

with right:
    role_text = str(row[role_col])
    color = ROLE_COLORS.get(role_text, ROLE_COLORS["unknown"])
    st.subheader(f"Узел {selected_gid}")
    st.markdown(f'<span class="pill" style="border:1px solid {color};color:{color}">{esc(role_text)}</span>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.metric("Role score", fmt(row[role_score_col]))
    b.metric("Priority score", fmt(row[priority_score_col]))
    st.metric("Кластер", fmt(row[cluster_col]))
    st.markdown("**Evidence — признаки и гипотеза для проверки**")
    st.write(row[evidence_col])
    top_match = top.loc[top["gid"].map(gid_key) == selected_gid] if top_schema_ok else pd.DataFrame()
    if not top_match.empty:
        st.markdown("**В приоритетном списке**")
        st.write(f"Ранг {fmt(top_match.iloc[0]['rank'])} · {top_match.iloc[0]['why']}")

    if edges_valid:
        incoming = edges.loc[edges[edge_dst].map(gid_key) == selected_gid]
        outgoing = edges.loc[edges[edge_src].map(gid_key) == selected_gid]
        tx_col = col(edges, ["n_tx", "tx_count"])
        def edge_table(frame, other):
            result = pd.DataFrame({"GID": frame[other].map(gid_key).values})
            result["Сумма KZT"] = frame[edge_amount].map(fmt).values
            if tx_col:
                result["Транзакций"] = frame[tx_col].map(fmt).values
            return result
        st.markdown("**Входящие связи**")
        st.dataframe(edge_table(incoming, edge_src).head(8), hide_index=True, width="stretch", height=190)
        st.markdown("**Исходящие связи**")
        st.dataframe(edge_table(outgoing, edge_dst).head(8), hide_index=True, width="stretch", height=190)
        if incoming.empty and outgoing.empty:
            st.info("У этого узла нет рёбер в наблюдаемой выборке.")
    else:
        st.info("Суммы связей недоступны: проверьте data/edges.parquet.")

if pd.notna(row[cluster_col]):
    st.divider()
    st.subheader(f"Кластер {gid_key(row[cluster_col])}")
    if clusters_schema_ok:
        cluster_row = clusters.loc[clusters["cluster_id"].map(gid_key) == gid_key(row[cluster_col])]
        if cluster_row.empty:
            st.warning("Для cluster_id выбранного узла нет строки в clusters.csv.")
        else:
            info = cluster_row.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Узлов", fmt(info["n_nodes"]))
            m2.metric("Seed", fmt(info["n_seed"]))
            m3.metric("Внутренний оборот, KZT", fmt(info["sum_kzt_internal"]))
            st.markdown("**Гипотеза для проверки по кластеру**")
            st.write(info["hypothesis"])
            gids = parse_top_gids(info["top_gids"])
            st.caption("Ключевые узлы: " + (", ".join(gids) if gids else "не указаны"))
            members = set(gids) | {selected_gid}
            member_nodes = nodes[nodes[gid_col].map(gid_key).isin(members)]
            st.caption("Граф показывает выбранный узел и ключевые top_gids, а не полный состав кластера.")
            if edges_valid:
                cluster_edges = edges[edges[edge_src].map(gid_key).isin(members) & edges[edge_dst].map(gid_key).isin(members)]
                draw_graph(member_nodes, cluster_edges, selected_gid, gid_col, role_col,
                           "Рёбра между ключевыми узлами кластера",
                           selected_neighborhood=False, include_all_nodes=True)
    else:
        st.info("Описание кластера недоступно: output/clusters.csv отсутствует или имеет неверную схему.")

with st.expander("Ограничения выборки"):
    st.caption("Обход обрывается на четвёртом колене; входящие переводы seed неполны; в выборке учтены суммы от 5 000 KZT. Отсутствие ребра в этой выборке не доказывает отсутствие перевода вне наблюдаемых данных.")
