#!/usr/bin/env python3
"""
LANL auth graph preprocessor — Euler-aligned with CARS extensions.

Granularity modes (--granularity):
  legacy  — pair-collapsed train; snapshot attack test; pair-collapsed benign test
            (original CARS / Euler static export)
  pair    — pair-collapsed train and test (Option A)
  window  — window-bucketed (src,dst,win) train and test (default CARS plan)

References: Euler split.py, load_lanl.py
"""

from __future__ import annotations

import argparse
import gzip
import math
import random
from pathlib import Path

import torch

EDGE_FEAT_DIM_LEGACY = 6
EDGE_FEAT_DIM_WINDOW = 7

COL_TIME = 0
COL_SRC_COMP = 3
COL_DST_COMP = 4
COL_LOGON_TYPE = 6
COL_SUCCESS = 8

ADMIN_PREFIXES = ("DC", "TGT", "KDC", "DOMAIN", "ADMIN")


def _new_pair_stats() -> dict:
    return {
        "count_pre": 0,
        "ntlm_pre": 0,
        "success_pre": 0,
        "logon_network_pre": 0,
        "first_ts_pre": None,
        "last_ts_pre": None,
        "count_post": 0,
        "ntlm_post": 0,
        "success_post": 0,
        "logon_network_post": 0,
        "anom": 0,
        "first_ts_post": None,
        "last_ts_post": None,
    }


def _new_window_stats() -> dict:
    return {
        "count": 0,
        "ntlm": 0,
        "anom": 0,
        "success": 0,
        "logon_network": 0,
        "first_ts": None,
        "last_ts": None,
    }


def euler_entity(raw: str) -> str:
    """Euler split.py stores raw auth/redteam src/dst tokens (no fmt_src in nmap)."""
    return raw.strip()


def load_redteam_anom_dict(redteam_path: Path) -> dict[tuple[str, str], list[int]]:
    """Euler split.py mark_anoms: (src,dst) -> compromise timestamps (raw tokens)."""
    anom: dict[tuple[str, str], list[int]] = {}
    opener = gzip.open if redteam_path.suffix == ".gz" else open
    with opener(redteam_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            src, dst = euler_entity(parts[2]), euler_entity(parts[3])
            if not src or not dst or src == dst:
                continue
            anom.setdefault((src, dst), []).append(ts)
    return anom


def is_anomalous(anom_dict: dict, src: str, dst: str, ts: int) -> bool:
    """Attack iff (src, dst, ts) exactly matches a redteam compromise event."""
    if (src, dst) not in anom_dict:
        return False
    return ts in anom_dict[(src, dst)]


def first_anomalous_timestamp(anom_dict: dict[tuple[str, str], list[int]]) -> int:
    return min(t for vals in anom_dict.values() for t in vals)


def node_role(name: str) -> list[float]:
    upper = name.upper()
    if any(upper.startswith(p) or p in upper for p in ADMIN_PREFIXES):
        return [0.0, 0.0, 1.0]
    if name.startswith("U"):
        return [1.0, 0.0, 0.0]
    if name.startswith("C"):
        return [0.0, 1.0, 0.0]
    return [0.0, 1.0, 0.0]


def edge_feat(
    s: dict,
    use_post: bool,
    *,
    include_ntlm: bool = True,
    count_key: str | None = None,
    ntlm_key: str | None = None,
) -> list[float]:
    """7-d edge features (or 6-d when include_ntlm=False for legacy graphs)."""
    if count_key is not None:
        count = s[count_key]
        ntlm = s.get(ntlm_key or "ntlm", 0)
        success = s["success"]
        net = s["logon_network"]
        first_ts, last_ts = s["first_ts"] or 0, s["last_ts"] or 0
    elif not use_post:
        count, ntlm = s["count_pre"], s.get("ntlm_pre", 0)
        success, net = s["success_pre"], s["logon_network_pre"]
        first_ts, last_ts = s["first_ts_pre"] or 0, s["last_ts_pre"] or 0
    else:
        count, ntlm = s["count_post"], s.get("ntlm_post", 0)
        success, net = s["success_post"], s["logon_network_post"]
        first_ts, last_ts = s["first_ts_post"] or 0, s["last_ts_post"] or 0

    duration = max(last_ts - first_ts, 0)
    hour = (first_ts // 3600) % 24
    feats = [
        math.log1p(count),
        success / max(count, 1),
        net / max(count, 1),
        math.log1p(duration),
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
    ]
    if include_ntlm:
        feats.append(ntlm / max(count, 1))
    return feats


def edge_feat_6d(s: dict, use_post: bool) -> list[float]:
    """Backward-compatible alias (no ntlm_frac)."""
    return edge_feat(s, use_post, include_ntlm=False)


def euler_weight(count: float, mean: float, std: float) -> float:
    if std < 1e-8:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(count - mean) / std))


def zscore(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t
    return (t - t.mean(dim=0, keepdim=True)) / t.std(dim=0, keepdim=True).clamp_min(1e-6)


def _parse_auth_line(
    line: str,
    ntlm_only: bool,
) -> tuple[int, str, str, str, int, int] | None:
    if ntlm_only and "NTLM" not in line.upper():
        return None
    is_ntlm = 1 if "NTLM" in line.upper() else 0
    parts = line.strip().split(",")
    if len(parts) <= COL_DST_COMP:
        return None
    try:
        ts = int(parts[COL_TIME])
    except ValueError:
        return None
    src = euler_entity(parts[COL_SRC_COMP])
    dst = euler_entity(parts[COL_DST_COMP])
    if not src or not dst or src == dst:
        return None
    logon = parts[COL_LOGON_TYPE].strip().lower() if len(parts) > COL_LOGON_TYPE else ""
    success = 1 if (
        parts[COL_SUCCESS].strip().lower() == "success"
        if len(parts) > COL_SUCCESS else True
    ) else 0
    return ts, src, dst, logon, success, is_ntlm


def stream_auth_pass(
    auth_path: Path,
    anom_dict: dict[tuple[str, str], list[int]],
    first_anom_ts: int,
    *,
    delta: int,
    ntlm_only: bool,
    granularity: str,
    progress_every: int,
) -> tuple[dict[str, int], dict, dict, int, int]:
    """Single streaming pass; returns node_map and mode-specific stat dicts."""
    node_map: dict[str, int] = {}
    pair_stats: dict[tuple[str, str], dict] = {}
    window_stats: dict[tuple[str, str, int], dict] = {}
    n_lines = n_kept = 0

    opener = gzip.open if auth_path.suffix == ".gz" else open
    with opener(auth_path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()  # auth header
        for line in f:
            n_lines += 1
            if progress_every and n_lines % progress_every == 0:
                n_atk = _count_attack_edges(granularity, pair_stats, window_stats)
                print(
                    f"      ... {n_lines:,} lines, {n_kept:,} kept, "
                    f"nodes={len(node_map):,}, attack_edges={n_atk:,}",
                    flush=True,
                )

            parsed = _parse_auth_line(line, ntlm_only)
            if parsed is None:
                continue
            ts, src, dst, logon, success, is_ntlm = parsed
            n_kept += 1
            for name in (src, dst):
                if name not in node_map:
                    node_map[name] = len(node_map)

            label = 1 if is_anomalous(anom_dict, src, dst, ts) else 0

            if granularity == "legacy":
                _accumulate_legacy(
                    pair_stats, window_stats, src, dst, ts, first_anom_ts, delta,
                    logon, success, is_ntlm, label,
                )
            elif granularity == "pair":
                _accumulate_pair(
                    pair_stats, src, dst, ts, first_anom_ts,
                    logon, success, is_ntlm, label,
                )
            else:
                _accumulate_window(
                    window_stats, src, dst, ts, first_anom_ts, delta,
                    logon, success, is_ntlm, label,
                )

    return node_map, pair_stats, window_stats, n_lines, n_kept


def _count_attack_edges(
    granularity: str,
    pair_stats: dict,
    window_stats: dict,
) -> int:
    if granularity == "window":
        return sum(
            1 for (_src, _dst, win), s in window_stats.items()
            if win >= 0 and s["anom"]
        )
    if granularity == "pair":
        return sum(1 for s in pair_stats.values() if s["anom"])
    return sum(1 for s in window_stats.values() if s["anom"])


def _accumulate_legacy(
    pair_stats: dict,
    window_stats: dict,
    src: str,
    dst: str,
    ts: int,
    first_anom_ts: int,
    delta: int,
    logon: str,
    success: int,
    is_ntlm: int,
    label: int,
) -> None:
    if ts < first_anom_ts:
        key = (src, dst)
        if key not in pair_stats:
            pair_stats[key] = _new_pair_stats()
        s = pair_stats[key]
        s["count_pre"] += 1
        s["ntlm_pre"] += is_ntlm
        s["success_pre"] += success
        if logon == "network":
            s["logon_network_pre"] += 1
        s["first_ts_pre"] = ts if s["first_ts_pre"] is None else min(s["first_ts_pre"], ts)
        s["last_ts_pre"] = ts if s["last_ts_pre"] is None else max(s["last_ts_pre"], ts)
    else:
        win = (ts - first_anom_ts) // delta
        wkey = (src, dst, win)
        if wkey not in window_stats:
            window_stats[wkey] = _new_window_stats()
        ws = window_stats[wkey]
        ws["count"] += 1
        ws["ntlm"] += is_ntlm
        ws["anom"] = max(ws["anom"], label)
        ws["success"] += success
        if logon == "network":
            ws["logon_network"] += 1
        ws["first_ts"] = ts if ws["first_ts"] is None else min(ws["first_ts"], ts)
        ws["last_ts"] = ts if ws["last_ts"] is None else max(ws["last_ts"], ts)


def _accumulate_pair(
    pair_stats: dict,
    src: str,
    dst: str,
    ts: int,
    first_anom_ts: int,
    logon: str,
    success: int,
    is_ntlm: int,
    label: int,
) -> None:
    key = (src, dst)
    if key not in pair_stats:
        pair_stats[key] = _new_pair_stats()
    s = pair_stats[key]
    if ts < first_anom_ts:
        s["count_pre"] += 1
        s["ntlm_pre"] += is_ntlm
        s["success_pre"] += success
        if logon == "network":
            s["logon_network_pre"] += 1
        s["first_ts_pre"] = ts if s["first_ts_pre"] is None else min(s["first_ts_pre"], ts)
        s["last_ts_pre"] = ts if s["last_ts_pre"] is None else max(s["last_ts_pre"], ts)
    else:
        s["count_post"] += 1
        s["ntlm_post"] += is_ntlm
        s["success_post"] += success
        if logon == "network":
            s["logon_network_post"] += 1
        s["anom"] = max(s["anom"], label)
        s["first_ts_post"] = ts if s["first_ts_post"] is None else min(s["first_ts_post"], ts)
        s["last_ts_post"] = ts if s["last_ts_post"] is None else max(s["last_ts_post"], ts)


def _accumulate_window(
    window_stats: dict,
    src: str,
    dst: str,
    ts: int,
    first_anom_ts: int,
    delta: int,
    logon: str,
    success: int,
    is_ntlm: int,
    label: int,
) -> None:
    win = (ts - first_anom_ts) // delta
    wkey = (src, dst, win)
    if wkey not in window_stats:
        window_stats[wkey] = _new_window_stats()
    ws = window_stats[wkey]
    ws["count"] += 1
    ws["ntlm"] += is_ntlm
    ws["anom"] = max(ws["anom"], label)
    ws["success"] += success
    if logon == "network":
        ws["logon_network"] += 1
    ws["first_ts"] = ts if ws["first_ts"] is None else min(ws["first_ts"], ts)
    ws["last_ts"] = ts if ws["last_ts"] is None else max(ws["last_ts"], ts)


def _finalize_legacy(
    pair_stats: dict,
    window_stats: dict,
) -> tuple[list, list, list]:
    train_edges = [
        (src, dst, s) for (src, dst), s in pair_stats.items() if s["count_pre"] > 0
    ]
    attack_test: list = []
    benign_pair_stats: dict[tuple[str, str], dict] = {}
    for (src, dst, _win), ws in window_stats.items():
        edge_stats = {
            "count_post": ws["count"],
            "ntlm_post": ws["ntlm"],
            "success_post": ws["success"],
            "logon_network_post": ws["logon_network"],
            "first_ts_post": ws["first_ts"],
            "last_ts_post": ws["last_ts"],
        }
        if ws["anom"]:
            attack_test.append((src, dst, edge_stats, ws["first_ts"]))
        else:
            key = (src, dst)
            if key not in benign_pair_stats:
                benign_pair_stats[key] = {
                    "count_post": 0,
                    "ntlm_post": 0,
                    "success_post": 0,
                    "logon_network_post": 0,
                    "first_ts_post": None,
                    "last_ts_post": None,
                }
            bp = benign_pair_stats[key]
            bp["count_post"] += ws["count"]
            bp["ntlm_post"] += ws["ntlm"]
            bp["success_post"] += ws["success"]
            bp["logon_network_post"] += ws["logon_network"]
            bp["first_ts_post"] = (
                ws["first_ts"] if bp["first_ts_post"] is None
                else min(bp["first_ts_post"], ws["first_ts"])
            )
            bp["last_ts_post"] = (
                ws["last_ts"] if bp["last_ts_post"] is None
                else max(bp["last_ts_post"], ws["last_ts"])
            )
    benign_test = [(src, dst, s) for (src, dst), s in benign_pair_stats.items()]
    return train_edges, attack_test, benign_test


def _finalize_pair(pair_stats: dict) -> tuple[list, list, list]:
    train_edges = [
        (src, dst, s) for (src, dst), s in pair_stats.items() if s["count_pre"] > 0
    ]
    attack_test = [
        (src, dst, s) for (src, dst), s in pair_stats.items()
        if s["count_post"] > 0 and s["anom"]
    ]
    benign_test = [
        (src, dst, s) for (src, dst), s in pair_stats.items()
        if s["count_post"] > 0 and not s["anom"]
    ]
    return train_edges, attack_test, benign_test


def _finalize_window(
    window_stats: dict,
    first_anom_ts: int,
    delta: int,
) -> tuple[list, list, list]:
    train_edges: list = []
    attack_test: list = []
    benign_test: list = []
    for (src, dst, win), ws in window_stats.items():
        if win < 0:
            train_edges.append((src, dst, win, ws))
        elif ws["anom"]:
            attack_test.append((src, dst, win, ws))
        else:
            benign_test.append((src, dst, win, ws))
    return train_edges, attack_test, benign_test


def stream_build_graph(
    auth_path: Path,
    anom_dict: dict[tuple[str, str], list[int]],
    *,
    granularity: str = "window",
    delta: int = 1800,
    ntlm_only: bool = False,
    val_frac: float = 0.05,
    val_seed: int = 42,
    cal_frac: float = 0.2,
    cal_seed: int = 43,
    progress_every: int = 5_000_000,
) -> dict:
    """Build graph.pt payload for the selected granularity mode."""
    first_anom_ts = first_anomalous_timestamp(anom_dict)
    node_map, pair_stats, window_stats, n_lines, n_kept = stream_auth_pass(
        auth_path,
        anom_dict,
        first_anom_ts,
        delta=delta,
        ntlm_only=ntlm_only,
        granularity=granularity,
        progress_every=progress_every,
    )

    if granularity == "legacy":
        train_edges, attack_test, benign_test = _finalize_legacy(pair_stats, window_stats)
        n_attack = sum(1 for _ in attack_test)
        print(
            f"      done: {n_lines:,} lines, {n_kept:,} kept, nodes={len(node_map):,}, "
            f"granularity=legacy, delta={delta}s, first_anom_ts={first_anom_ts}, "
            f"anom_snapshot_edges={n_attack:,} (Euler ~518), "
            f"benign_pairs={len(benign_test):,}",
            flush=True,
        )
        return build_tensors_legacy(
            node_map, train_edges, attack_test, benign_test,
            val_frac, val_seed, cal_frac=cal_frac, cal_seed=cal_seed,
            delta=delta, first_anom_ts=first_anom_ts,
        )

    if granularity == "pair":
        train_edges, attack_test, benign_test = _finalize_pair(pair_stats)
        n_attack = len(attack_test)
        print(
            f"      done: {n_lines:,} lines, {n_kept:,} kept, nodes={len(node_map):,}, "
            f"granularity=pair, first_anom_ts={first_anom_ts}, "
            f"attack_pairs={n_attack:,}, benign_pairs={len(benign_test):,}",
            flush=True,
        )
        return build_tensors_pair(
            node_map, train_edges, attack_test, benign_test,
            val_frac, val_seed, cal_frac=cal_frac, cal_seed=cal_seed,
            first_anom_ts=first_anom_ts,
        )

    train_edges, attack_test, benign_test = _finalize_window(
        window_stats, first_anom_ts, delta,
    )
    n_attack = len(attack_test)
    print(
        f"      done: {n_lines:,} lines, {n_kept:,} kept, nodes={len(node_map):,}, "
        f"granularity=window, delta={delta}s, first_anom_ts={first_anom_ts}, "
        f"attack_window_edges={n_attack:,} (Euler ~518), "
        f"benign_window_edges={len(benign_test):,}, train_window_edges={len(train_edges):,}",
        flush=True,
    )
    return build_tensors_window(
        node_map, train_edges, attack_test, benign_test,
        val_frac, val_seed, cal_frac=cal_frac, cal_seed=cal_seed,
        delta=delta, first_anom_ts=first_anom_ts,
    )


def list_to_tensors(
    edge_list: list,
    node_map: dict[str, int],
    use_post: bool,
    euler_mean: float,
    euler_std: float,
    *,
    include_ntlm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ei, ea, ew = [], [], []
    for src, dst, s in edge_list:
        ei.append([node_map[src], node_map[dst]])
        c = s["count_post"] if use_post else s["count_pre"]
        ea.append(edge_feat(s, use_post, include_ntlm=include_ntlm))
        ew.append(euler_weight(float(c), euler_mean, euler_std))
    if not ei:
        dim = EDGE_FEAT_DIM_WINDOW if include_ntlm else EDGE_FEAT_DIM_LEGACY
        z = torch.empty((2, 0), dtype=torch.long)
        return z, torch.empty((0, dim), dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    dim = len(ea[0])
    return (
        torch.tensor(ei, dtype=torch.long).t().contiguous(),
        torch.tensor(ea, dtype=torch.float32),
        torch.tensor(ew, dtype=torch.float32),
    )


def attack_events_to_tensors(
    events: list,
    node_map: dict[str, int],
    euler_mean: float,
    euler_std: float,
    *,
    include_ntlm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy attack list: (src, dst, stats, first_ts)."""
    ei, ea, ew = [], [], []
    for src, dst, s, _ts in events:
        ei.append([node_map[src], node_map[dst]])
        ea.append(edge_feat(s, use_post=True, include_ntlm=include_ntlm))
        ew.append(euler_weight(float(s["count_post"]), euler_mean, euler_std))
    if not ei:
        dim = EDGE_FEAT_DIM_WINDOW if include_ntlm else EDGE_FEAT_DIM_LEGACY
        z = torch.empty((2, 0), dtype=torch.long)
        return z, torch.empty((0, dim), dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    return (
        torch.tensor(ei, dtype=torch.long).t().contiguous(),
        torch.tensor(ea, dtype=torch.float32),
        torch.tensor(ew, dtype=torch.float32),
    )


def window_edges_to_tensors(
    edge_list: list,
    node_map: dict[str, int],
    euler_mean: float,
    euler_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Window edges: (src, dst, win, stats) -> edge_index, edge_attr, edge_weight, edge_win."""
    ei, ea, ew, wins = [], [], [], []
    for src, dst, win, s in edge_list:
        ei.append([node_map[src], node_map[dst]])
        ea.append(
            edge_feat(
                s, use_post=False, include_ntlm=True,
                count_key="count", ntlm_key="ntlm",
            )
        )
        ew.append(euler_weight(float(s["count"]), euler_mean, euler_std))
        wins.append(win)
    if not ei:
        z = torch.empty((2, 0), dtype=torch.long)
        return (
            z,
            torch.empty((0, EDGE_FEAT_DIM_WINDOW), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    return (
        torch.tensor(ei, dtype=torch.long).t().contiguous(),
        torch.tensor(ea, dtype=torch.float32),
        torch.tensor(ew, dtype=torch.float32),
        torch.tensor(wins, dtype=torch.long),
    )


def build_onehot_role_features(num_nodes: int, node_names: list[str]) -> torch.Tensor:
    eye = torch.eye(num_nodes, dtype=torch.float32)
    roles = torch.tensor([node_role(node_names[i]) for i in range(num_nodes)], dtype=torch.float32)
    return torch.cat([eye, roles], dim=1)


def train_val_mask(n_edges: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    idx = list(range(n_edges))
    rng.shuffle(idx)
    n_val = max(1, int(n_edges * val_frac))
    val_set = set(idx[:n_val])
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask = torch.ones(n_edges, dtype=torch.bool)
    for i in val_set:
        val_mask[i] = True
        train_mask[i] = False
    return train_mask, val_mask


def cal_eval_mask(n_items: int, cal_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    cal_mask = torch.zeros(n_items, dtype=torch.bool)
    eval_mask = torch.zeros(n_items, dtype=torch.bool)
    if n_items == 0:
        return cal_mask, eval_mask
    rng = random.Random(seed)
    idx = list(range(n_items))
    rng.shuffle(idx)
    n_cal = max(1, int(n_items * cal_frac)) if n_items > 1 else 1
    for i in idx[:n_cal]:
        cal_mask[i] = True
    for i in idx[n_cal:]:
        eval_mask[i] = True
    if not eval_mask.any():
        eval_mask[idx[-1]] = True
        cal_mask[idx[-1]] = False
    return cal_mask, eval_mask


def _zscore_test_from_train(train_attr: torch.Tensor, test_attr: torch.Tensor) -> torch.Tensor:
    if test_attr.numel() == 0:
        return test_attr
    mean = train_attr.mean(dim=0, keepdim=True)
    std = train_attr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (test_attr - mean) / std


def build_tensors_legacy(
    node_map: dict[str, int],
    train_edges: list,
    attack_snapshots: list,
    benign_snapshots: list,
    val_frac: float,
    val_seed: int,
    cal_frac: float = 0.2,
    cal_seed: int = 43,
    delta: int = 1800,
    first_anom_ts: int = 0,
) -> dict:
    num_nodes = len(node_map)
    node_names = [""] * num_nodes
    for name, idx in node_map.items():
        node_names[idx] = name

    train_counts = [float(s["count_pre"]) for _s, _d, s in train_edges]
    euler_mean = float(sum(train_counts) / len(train_counts)) if train_counts else 0.0
    euler_std = float(torch.tensor(train_counts).std().item()) if len(train_counts) > 1 else 1.0

    edge_index, edge_attr, edge_weight = list_to_tensors(
        train_edges, node_map, use_post=False, euler_mean=euler_mean, euler_std=euler_std,
        include_ntlm=False,
    )
    edge_attr = zscore(edge_attr)

    attack_ei, attack_ea, attack_ew = attack_events_to_tensors(
        attack_snapshots, node_map, euler_mean, euler_std, include_ntlm=False,
    )
    attack_ea = _zscore_test_from_train(edge_attr, attack_ea)

    benign_ei, benign_ea, benign_ew = list_to_tensors(
        benign_snapshots, node_map, use_post=True, euler_mean=euler_mean, euler_std=euler_std,
        include_ntlm=False,
    )
    benign_ea = _zscore_test_from_train(edge_attr, benign_ea)

    train_mask, val_mask = train_val_mask(edge_index.size(1), val_frac, val_seed)
    attack_cal_mask, attack_eval_mask = cal_eval_mask(attack_ei.size(1), cal_frac, cal_seed)
    benign_cal_mask, benign_eval_mask = cal_eval_mask(
        benign_ei.size(1), cal_frac, cal_seed + 1,
    )
    x = build_onehot_role_features(num_nodes, node_names)

    return {
        "num_nodes": num_nodes,
        "node_feat_dim": x.size(1),
        "edge_feat_dim": EDGE_FEAT_DIM_LEGACY,
        "granularity": "legacy",
        "test_granularity": "snapshot",
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_weight": edge_weight,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "attack_edge_index": attack_ei,
        "attack_edge_attr": attack_ea,
        "attack_edge_weight": attack_ew,
        "attack_cal_mask": attack_cal_mask,
        "attack_eval_mask": attack_eval_mask,
        "benign_test_edge_index": benign_ei,
        "benign_test_edge_attr": benign_ea,
        "benign_test_edge_weight": benign_ew,
        "benign_cal_mask": benign_cal_mask,
        "benign_eval_mask": benign_eval_mask,
        "node_names": node_names,
        "first_anom_ts": first_anom_ts,
        "snapshot_delta": delta,
        "directed": True,
        "euler_count_mean": euler_mean,
        "euler_count_std": euler_std,
        "n_attack_events": len(attack_snapshots),
        "n_attack_edges": len(attack_snapshots),
        "n_attack_snapshot_edges": len(attack_snapshots),
    }


def build_tensors_pair(
    node_map: dict[str, int],
    train_edges: list,
    attack_edges: list,
    benign_edges: list,
    val_frac: float,
    val_seed: int,
    cal_frac: float = 0.2,
    cal_seed: int = 43,
    first_anom_ts: int = 0,
) -> dict:
    num_nodes = len(node_map)
    node_names = [""] * num_nodes
    for name, idx in node_map.items():
        node_names[idx] = name

    train_counts = [float(s["count_pre"]) for _s, _d, s in train_edges]
    euler_mean = float(sum(train_counts) / len(train_counts)) if train_counts else 0.0
    euler_std = float(torch.tensor(train_counts).std().item()) if len(train_counts) > 1 else 1.0

    edge_index, edge_attr, edge_weight = list_to_tensors(
        train_edges, node_map, use_post=False, euler_mean=euler_mean, euler_std=euler_std,
        include_ntlm=True,
    )
    edge_attr = zscore(edge_attr)

    attack_ei, attack_ea, attack_ew = list_to_tensors(
        attack_edges, node_map, use_post=True, euler_mean=euler_mean, euler_std=euler_std,
        include_ntlm=True,
    )
    attack_ea = _zscore_test_from_train(edge_attr, attack_ea)

    benign_ei, benign_ea, benign_ew = list_to_tensors(
        benign_edges, node_map, use_post=True, euler_mean=euler_mean, euler_std=euler_std,
        include_ntlm=True,
    )
    benign_ea = _zscore_test_from_train(edge_attr, benign_ea)

    train_mask, val_mask = train_val_mask(edge_index.size(1), val_frac, val_seed)
    attack_cal_mask, attack_eval_mask = cal_eval_mask(attack_ei.size(1), cal_frac, cal_seed)
    benign_cal_mask, benign_eval_mask = cal_eval_mask(
        benign_ei.size(1), cal_frac, cal_seed + 1,
    )
    x = build_onehot_role_features(num_nodes, node_names)

    return {
        "num_nodes": num_nodes,
        "node_feat_dim": x.size(1),
        "edge_feat_dim": EDGE_FEAT_DIM_WINDOW,
        "granularity": "pair",
        "test_granularity": "pair",
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_weight": edge_weight,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "attack_edge_index": attack_ei,
        "attack_edge_attr": attack_ea,
        "attack_edge_weight": attack_ew,
        "attack_cal_mask": attack_cal_mask,
        "attack_eval_mask": attack_eval_mask,
        "benign_test_edge_index": benign_ei,
        "benign_test_edge_attr": benign_ea,
        "benign_test_edge_weight": benign_ew,
        "benign_cal_mask": benign_cal_mask,
        "benign_eval_mask": benign_eval_mask,
        "node_names": node_names,
        "first_anom_ts": first_anom_ts,
        "directed": True,
        "euler_count_mean": euler_mean,
        "euler_count_std": euler_std,
        "n_attack_edges": len(attack_edges),
    }


def build_tensors_window(
    node_map: dict[str, int],
    train_edges: list,
    attack_edges: list,
    benign_edges: list,
    val_frac: float,
    val_seed: int,
    cal_frac: float = 0.2,
    cal_seed: int = 43,
    delta: int = 1800,
    first_anom_ts: int = 0,
) -> dict:
    num_nodes = len(node_map)
    node_names = [""] * num_nodes
    for name, idx in node_map.items():
        node_names[idx] = name

    train_counts = [float(s["count"]) for _s, _d, _w, s in train_edges]
    euler_mean = float(sum(train_counts) / len(train_counts)) if train_counts else 0.0
    euler_std = float(torch.tensor(train_counts).std().item()) if len(train_counts) > 1 else 1.0

    edge_index, edge_attr, edge_weight, edge_win = window_edges_to_tensors(
        train_edges, node_map, euler_mean, euler_std,
    )
    edge_attr = zscore(edge_attr)

    attack_ei, attack_ea, attack_ew, attack_win = window_edges_to_tensors(
        attack_edges, node_map, euler_mean, euler_std,
    )
    attack_ea = _zscore_test_from_train(edge_attr, attack_ea)

    benign_ei, benign_ea, benign_ew, benign_win = window_edges_to_tensors(
        benign_edges, node_map, euler_mean, euler_std,
    )
    benign_ea = _zscore_test_from_train(edge_attr, benign_ea)

    train_mask, val_mask = train_val_mask(edge_index.size(1), val_frac, val_seed)
    attack_cal_mask, attack_eval_mask = cal_eval_mask(attack_ei.size(1), cal_frac, cal_seed)
    benign_cal_mask, benign_eval_mask = cal_eval_mask(
        benign_ei.size(1), cal_frac, cal_seed + 1,
    )
    x = build_onehot_role_features(num_nodes, node_names)

    return {
        "num_nodes": num_nodes,
        "node_feat_dim": x.size(1),
        "edge_feat_dim": EDGE_FEAT_DIM_WINDOW,
        "granularity": "window",
        "test_granularity": "window",
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_weight": edge_weight,
        "edge_win": edge_win,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "attack_edge_index": attack_ei,
        "attack_edge_attr": attack_ea,
        "attack_edge_weight": attack_ew,
        "attack_edge_win": attack_win,
        "attack_cal_mask": attack_cal_mask,
        "attack_eval_mask": attack_eval_mask,
        "benign_test_edge_index": benign_ei,
        "benign_test_edge_attr": benign_ea,
        "benign_test_edge_weight": benign_ew,
        "benign_test_edge_win": benign_win,
        "benign_cal_mask": benign_cal_mask,
        "benign_eval_mask": benign_eval_mask,
        "node_names": node_names,
        "first_anom_ts": first_anom_ts,
        "snapshot_delta": delta,
        "directed": True,
        "euler_count_mean": euler_mean,
        "euler_count_std": euler_std,
        "n_attack_edges": len(attack_edges),
        "n_attack_snapshot_edges": len(attack_edges),
    }


def main():
    parser = argparse.ArgumentParser(description="Euler-aligned LANL graph preprocess (CARS)")
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--granularity",
        choices=["legacy", "pair", "window"],
        default="window",
        help="legacy=pair train + snapshot test; pair=pair-level; window=window-level (default)",
    )
    parser.add_argument(
        "--ntlm-only",
        action="store_true",
        help="Keep only NTLM auth lines (Euler default; off = all auth + ntlm_frac feature)",
    )
    parser.add_argument(
        "--delta", type=int, default=1800,
        help="Window seconds for legacy test snapshots and window mode (Euler §VI: 1800)",
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--cal-seed", type=int, default=43)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()

    print("[1/3] Loading redteam (Euler split.py mark_anoms)...")
    anom_dict = load_redteam_anom_dict(args.redteam)
    n_ts = sum(len(v) for v in anom_dict.values())
    print(f"      {len(anom_dict)} pairs, {n_ts} compromise timestamps")

    print(f"[2/3] Streaming auth (granularity={args.granularity})...")
    data = stream_build_graph(
        args.auth, anom_dict,
        granularity=args.granularity,
        delta=args.delta,
        ntlm_only=args.ntlm_only,
        val_frac=args.val_frac,
        val_seed=args.val_seed,
        cal_frac=args.cal_frac,
        cal_seed=args.cal_seed,
        progress_every=args.progress_every,
    )

    print("[3/3] Saving graph.pt ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    n_atk = data.get("n_attack_edges", data.get("n_attack_snapshot_edges", 0))
    print(f"Saved: {args.out}")
    print(f"  granularity={data.get('granularity', '?')}  test={data.get('test_granularity', '?')}")
    print(f"  nodes={data['num_nodes']} (Euler Table V target ~17685)")
    print(f"  train_edges={data['edge_index'].size(1)}, val={data['val_mask'].sum().item()}")
    print(f"  edge_feat_dim={data['edge_feat_dim']}")
    print(f"  n_attack_edges={n_atk} (Euler Table V target ~518)")
    print(f"  attack_cal={data['attack_cal_mask'].sum().item()}, "
          f"attack_eval={data['attack_eval_mask'].sum().item()}, "
          f"benign_eval={data['benign_eval_mask'].sum().item()}")
    if "snapshot_delta" in data:
        print(f"  delta={data['snapshot_delta']}s, first_anom_ts={data['first_anom_ts']}")


if __name__ == "__main__":
    main()
