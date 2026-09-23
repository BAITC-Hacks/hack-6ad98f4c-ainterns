"""Local AML analyst dashboard for the Money Graph project."""
from pathlib import Path
import math
import re

import pandas as pd
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
               selected_neighborhood=True, include_all_nodes=False):
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
    node_records = nodes.to_dict(orient="records")
    node_map = {gid_key(record[gid_col]): record for record in node_records}
    neighbor_ids = set(relevant[src_col].map(gid_key)) | set(relevant[dst_col].map(gid_key))
    keep = {selected} | neighbor_ids
    if include_all_nodes:
        keep.update(nodes[gid_col].map(gid_key))
    # Cap visual clutter without changing the loaded data or adjacency shown in the card.
    if len(relevant) > 80:
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
    for edge in relevant.to_dict(orient="records"):
        a, b = gid_key(edge[src_col]), gid_key(edge[dst_col])
        x1, y1 = positions[a]; x2, y2 = positions[b]
        fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.4, arrowcolor="#718ba5")
        if amount_col:
            fig.add_trace(go.Scatter(x=[(x1+x2)/2], y=[(y1+y2)/2], mode="text",
                text=[fmt(edge.get(amount_col))], textfont={"size":9,"color":"#b9c8d8"}, hoverinfo="skip", showlegend=False))
    for role_name, color in ROLE_COLORS.items():
        ids = [gid for gid in keep if ((str(node_map[gid].get(role_col, "unknown")).lower() if role_col and gid in node_map else "unknown") == role_name)]
        if not ids:
            continue
        fig.add_trace(go.Scatter(x=[positions[g][0] for g in ids], y=[positions[g][1] for g in ids],
            mode="markers+text", text=ids, textposition="top center", name=role_name,
            marker={"size":[32 if g == selected else 19 for g in ids], "color":color,
                    "line":{"width":[4 if g == selected else 1 for g in ids], "color":"#f4bd50"}},
            customdata=[str(node_map[g].get(role_col,"unknown")) if role_col and g in node_map else "unknown" for g in ids],
            hovertemplate="gid %{text}<br>роль %{customdata}<extra></extra>"))
    fig.update_layout(title=title, height=430, paper_bgcolor="#0c1724", plot_bgcolor="#0c1724",
        font={"color":"#dbe7f2"}, margin={"l":10,"r":10,"t":45,"b":10}, showlegend=False,
        xaxis={"visible":False,"range":[-2.8,2.8]}, yaxis={"visible":False,"range":[-2.8,2.8],"scaleanchor":"x"})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})

st.title("Граф денег")
st.caption("Локальная рабочая панель AML-аналитика · исследовательские признаки и гипотезы для проверки")
nodes = load_csv("nodes_roles.csv")
clusters = load_csv("clusters.csv")
top = load_csv("top_nodes.csv")

if nodes.empty:
    missing = [name for name in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv") if not (OUT / name).exists()]
    st.info("Для панели нужны выходные CSV из пайплайна. " +
            ("Не найдены: " + ", ".join(missing) + ". " if missing else "nodes_roles.csv пуст. ") +
            f"Ожидаемая папка: `{OUT}`. Запустите `python -m src.pipeline` из корня проекта после установки входных parquet в `{DATA}`.")
    st.stop()

gid_col = col(nodes, ["gid", "node", "id"])
if gid_col is None:
    st.error("В output/nodes_roles.csv отсутствует обязательная колонка gid.")
    st.stop()
if not require_columns(nodes, "nodes_roles.csv", ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]):
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
unexpected_roles = sorted(set(nodes["role"].dropna().astype(str)) - allowed_roles)
if unexpected_roles:
    st.error("В nodes_roles.csv обнаружены неизвестные роли: " + ", ".join(unexpected_roles))
    st.stop()
for score_name in ("role_score", "priority_score"):
    score_values = pd.to_numeric(nodes[score_name], errors="coerce")
    if score_values.isna().any() or not score_values.between(0, 1).all():
        st.error(f"Значения {score_name} в nodes_roles.csv должны быть числами в диапазоне [0, 1].")
        st.stop()
role_col = "role"
role_score_col = "role_score"
priority_score_col = "priority_score"
cluster_col = "cluster_id"
evidence_col = "evidence"
top_schema_ok = require_columns(top, "top_nodes.csv", ["rank", "gid", "role", "priority_score", "why"]) if not top.empty else False
clusters_schema_ok = require_columns(clusters, "clusters.csv", ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]) if not clusters.empty else False

k1,k2,k3,k4 = st.columns(4)
k1.metric("Узлов с ролями", f"{len(nodes):,}".replace(",", " "))
k2.metric("Кластеров", f"{clusters[cluster_col].nunique():,}" if clusters_schema_ok else "—")
k3.metric("В топ-листе", f"{len(top):,}".replace(",", " ") if top_schema_ok else "—")
k4.metric("Порог транзакций", "5 000 KZT")


with st.sidebar:
    st.header("Навигация")
    st.caption("Поиск по идентификатору узла")
    search = st.text_input("gid", placeholder="Введите gid", label_visibility="collapsed")
    if not top.empty:
        st.divider()
        st.subheader("Приоритеты")
        if top_schema_ok:
            top_view = top.rename(columns={"rank":"Ранг", "gid":"GID", "role":"Роль", "priority_score":"Приоритет", "why":"Объяснение"})
            st.dataframe(top_view[["Ранг", "GID", "Роль", "Приоритет", "Объяснение"]], hide_index=True, width="stretch", height=360)
    else:
        st.info("top_nodes.csv отсутствует или пуст: топ-лист недоступен.")

all_gids = nodes[gid_col].map(gid_key).tolist()
gid_set = set(all_gids)
search_key = gid_key(search)
default_gid = search_key if search_key in gid_set else (all_gids[0] if all_gids else "")
if search.strip() and search_key not in gid_set:
    st.warning("Такой gid не найден в nodes_roles.csv.")
if not all_gids:
    st.error("В nodes_roles.csv нет gid для поиска.")
    st.stop()
selected_gid = st.selectbox("Выбранный узел", all_gids, index=all_gids.index(default_gid), label_visibility="collapsed")
row = nodes.loc[nodes[gid_col].map(gid_key) == selected_gid].iloc[0]

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Окрестность узла")
    st.caption("Стрелка показывает направление потока; размер визуализации ограничен ближайшими связями.")
    legend = "&nbsp;&nbsp;".join(
        f'<span style="color:{color}">●</span> {esc(role)}'
        for role, color in ROLE_COLORS.items() if role != "unknown"
    )
    st.markdown(legend, unsafe_allow_html=True)
    edges_path = DATA / "edges.parquet"
    try:
        edges = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame()
    except Exception as exc:
        edges = pd.DataFrame()
        st.error(f"Ошибка чтения edges.parquet: {exc}")
    edge_src = col(edges, ["src", "source", "from_gid"])
    edge_dst = col(edges, ["dst", "target", "to_gid"])
    edge_amount = col(edges, ["sum_kzt", "amount", "weight"])
    edges_valid = bool(not edges.empty and edge_src and edge_dst and edge_amount)
    if edges.empty:
        if not edges_path.exists():
            st.warning(f"Связи недоступны: не найден файл {edges_path}.")
        else:
            st.warning("data/edges.parquet пуст или не содержит строк.")
    else:
        if not edges_valid:
            st.error("В data/edges.parquet ожидаются колонки src, dst и sum_kzt.")
        else:
            draw_graph(nodes, edges, selected_gid, gid_col, role_col, "Ближайшие направленные связи")
src, dst = edge_src, edge_dst
with right:
    role = row.get(role_col, "unknown") if role_col else "unknown"
    role_text = str(role)
    color = next((v for k,v in ROLE_COLORS.items() if k in role_text.lower()), ROLE_COLORS["unknown"])
    st.subheader(f"Узел {selected_gid}")
    st.markdown(f'<span class="pill" style="border:1px solid {color};color:{color}">{esc(role_text)}</span>', unsafe_allow_html=True)
    a,b = st.columns(2)
    a.metric("Role score", fmt(row.get(role_score_col)) if role_score_col else "—")
    b.metric("Priority score", fmt(row.get(priority_score_col)) if priority_score_col else "—")
    st.metric("Кластер", fmt(row.get(cluster_col)) if cluster_col else "—")
    evidence_cols = [c for c in ["in_degree","out_degree","in_amount","out_amount","turnover","pass_ratio","score"] if c in row.index]
    st.markdown("**Evidence — признаки и гипотеза для проверки**")
    st.write(row.get(evidence_col))
    if evidence_cols:
        st.dataframe(pd.DataFrame({"Признак": evidence_cols,"Значение":[fmt(row[c]) for c in evidence_cols]}), hide_index=True, width="stretch", height=240)
    if top_schema_ok:
        top_match = top.loc[top["gid"].map(gid_key) == selected_gid]
        if not top_match.empty:
            st.markdown("**В приоритетном списке**")
            st.write(f"Ранг {fmt(top_match.iloc[0]['rank'])} · {top_match.iloc[0]['why']}")
    st.markdown("**Входящие и исходящие связи**")
    src = col(edges, ["src","source","from_gid"]); dst = col(edges, ["dst","target","to_gid"])
    if edges_valid:
        incoming = edges.loc[edges[dst].map(gid_key) == selected_gid]
        outgoing = edges.loc[edges[src].map(gid_key) == selected_gid]
        amt = edge_amount
        tx = col(edges, ["n_tx","tx_count"])
        def edge_table(frame, other):
            result = pd.DataFrame({"gid":frame[other].astype(str)})
            if amt: result["Сумма KZT"] = frame[amt].map(fmt)
            if tx: result["Транзакций"] = frame[tx].map(fmt)
            return result
        c_in,c_out = st.columns(2)
        c_in.caption(f"Входящие · {len(incoming)}")
        c_in.dataframe(edge_table(incoming,src).head(8), hide_index=True, width="stretch", height=185)
        c_out.caption(f"Исходящие · {len(outgoing)}")
        c_out.dataframe(edge_table(outgoing,dst).head(8), hide_index=True, width="stretch", height=185)

if cluster_col and pd.notna(row.get(cluster_col)):
    st.divider()
    st.subheader(f"Кластер {row.get(cluster_col)}")
    if clusters_schema_ok:
        cluster_row = clusters.loc[clusters["cluster_id"].astype(str) == str(row[cluster_col])]
        if cluster_row.empty:
            st.warning("Для cluster_id выбранного узла нет строки в clusters.csv.")
        else:
            info = cluster_row.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Узлов", fmt(info["n_nodes"]))
            m2.metric("Seed", fmt(info["n_seed"]))
            m3.metric("Внутренний оборот, KZT", fmt(info["sum_kzt_internal"]))
            st.markdown("**Гипотеза по кластеру**")
            st.write(info["hypothesis"])
            gids = parse_top_gids(info["top_gids"])
            st.caption("Ключевые узлы кластера: " + (", ".join(gids) if gids else "не указаны"))
            members = set(gids)
            members.add(selected_gid)
            member_nodes = nodes[nodes[gid_col].astype(str).isin(members)]
            st.caption("Кластерный граф содержит выбранный узел и перечисленные top_gids — это ключевые узлы, не полный список участников.")
            if edges_valid:
                ce = edges[edges[src].astype(str).isin(members) & edges[dst].astype(str).isin(members)]
                draw_graph(member_nodes, ce, selected_gid, gid_col, role_col,
                           "Ключевые узлы кластера", selected_neighborhood=False, include_all_nodes=True)
    else:
        st.info("Описание кластера недоступно: output/clusters.csv отсутствует, пуст или имеет неверную схему.")
