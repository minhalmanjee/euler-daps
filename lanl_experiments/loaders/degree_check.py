"""Degree-deviation diagnostics for paper (train stability + test attack signal)."""

import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from loaders.load_lanl import load_partial_lanl, DATE_OF_EVIL_LANL, TIMES, LANL_FOLDER
from loaders.tdata import TData
from loaders.edge_dev import (
    load_train_stats, snapshot_node_dev, edge_penalty, adjust_edge_scores, _degrees,
)
import loaders.edge_dev as ed

ed.ZSCORE_CAP = 2.0
DELTA = int(0.5 * 3600)
DEFAULT_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEFAULT_REDTEAM = '/home/mmanjee/CARS/datasets/lanl/redteam.txt'
DEFAULT_OPTC_LABELS = '/home/mmanjee/CARS/datasets/optc/cybergfm/optc_labels.csv'


def load_optc_redteam_node_ids(
    labels_path: str = DEFAULT_OPTC_LABELS,
    nmap_path: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Map CyberGFM optc_labels.csv hosts -> contiguous node ids via nmap.pkl."""
    from loaders.load_optc_flow import OPTC_FLOW_FOLDER

    if nmap_path is None:
        nmap_path = OPTC_FLOW_FOLDER + 'nmap.pkl'
    with open(nmap_path, 'rb') as f:
        nmap = pickle.load(f)
    # nmap[contig] = original host id (int)
    orig_to_contig = {int(orig): i for i, orig in enumerate(nmap)}

    def parse_host(h: str) -> int | None:
        h = h.strip()
        if h == 'DC1':
            return 1000
        try:
            return int(h)
        except ValueError:
            return None

    hosts: set[int] = set()
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            for raw in (parts[1], parts[2]):
                hid = parse_host(raw)
                if hid is not None:
                    hosts.add(hid)

    ids, names = [], []
    for h in sorted(hosts):
        if h in orig_to_contig:
            ids.append(orig_to_contig[h])
            names.append(str(h) if h != 1000 else 'DC1')
    return np.array(ids, dtype=np.int64), names


def run_optc_redteam_degree_plot(
    fig_dir: str,
    labels_path: str = DEFAULT_OPTC_LABELS,
    delta: int = DELTA,
) -> str:
    """Write OpTC analogue of fig_redteam_degree_timeseries.png."""
    from loaders import load_optc_flow as optc

    ed.configure(optc.OPTC_FLOW_FOLDER, optc.load_partial_optc_flow)
    label_ids, names = load_optc_redteam_node_ids(labels_path)
    evil = int(optc.DATE_OF_EVIL)
    t_end = int(optc.TIMES['all']) + 1
    print(f'Loading OpTC timeline (0 → {t_end}, δ={delta})...')
    full_data = optc.load_partial_optc_flow(
        start=0, end=t_end, delta=delta, is_test=True,
    )
    assert getattr(full_data, 'snapshot_starts', None) is not None, \
        'OpTC loader must set snapshot_starts for correct time axis'

    # Plot all hosts that touch a malicious edge, plus optc_labels set.
    atk_nodes: set[int] = set()
    for t in range(full_data.T):
        y = full_data.ys[t].numpy().astype(bool)
        if not y.any():
            continue
        ei = full_data.eis[t]
        atk_nodes.update(int(x) for x in ei[0, y].tolist())
        atk_nodes.update(int(x) for x in ei[1, y].tolist())
    plot_ids = np.array(sorted(set(label_ids.tolist()) | atk_nodes), dtype=np.int64)
    print(f'=== OpTC plot hosts: {len(plot_ids)} '
          f'(optc_labels={len(label_ids)}, attack-edge nodes={len(atk_nodes)}) ===')
    print(f'  label hosts: {", ".join(names)}')

    # Sanity: first mal snapshot start should be <= first mal edge < start+delta
    starts = full_data.snapshot_starts
    first_mal_t = None
    for t in range(full_data.T):
        if full_data.ys[t].numpy().astype(bool).any():
            first_mal_t = starts[t]
            break
    print(f'  DATE_OF_EVIL={evil}  first_mal_snapshot_start={first_mal_t}  '
          f'({None if first_mal_t is None else first_mal_t/86400:.3f}d)')

    redteam_ts = collect_redteam_timeseries(full_data, 0, plot_ids, delta=delta)
    report_redteam_trajectories(redteam_ts, 0, evil_ts=evil)
    os.makedirs(fig_dir, exist_ok=True)
    path = plot_redteam_trajectories(
        redteam_ts, fig_dir, tr_origin=0,
        evil_ts=evil, delta=delta,
        source_label='optc_labels + attack edges',
        filename='fig_redteam_degree_timeseries.png',
    )
    print(f'Wrote {path}')
    return path


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def collect_test_edge_data(data, stats: dict) -> dict:
    """Per-edge arrays for diagnostics and paper figures."""
    mean_out = stats['mean_out']
    std_out = stats['std_out'].clamp_min(stats['global_std_out'])

    mean_out_src, z_out_src, pen_chunks, out_pen_chunks, y_chunks = [], [], [], [], []
    out_atk, in_atk, out_norm, in_norm = [], [], [], []

    for t in range(data.T):
        ei = data.eis[t]
        y = data.ys[t].numpy().astype(bool)
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        z_out = (out_d - mean_out) / std_out
        out_dev, in_dev = snapshot_node_dev(ei.cpu(), stats, data.num_nodes)
        pen = edge_penalty(ei, out_dev, in_dev).numpy()

        src = ei[0].long()
        dst = ei[1].long()
        mean_out_src.append(mean_out[src].numpy())
        z_out_src.append(z_out[src].numpy())
        pen_chunks.append(pen)
        out_pen_chunks.append(out_dev[src].numpy())
        y_chunks.append(y)

        if y.any():
            out_atk.extend(out_dev[src][y].numpy().tolist())
            in_atk.extend(in_dev[dst][y].numpy().tolist())
        if (~y).any():
            out_norm.extend(out_dev[src][~y].numpy().tolist())
            in_norm.extend(in_dev[dst][~y].numpy().tolist())

    return {
        'mean_out_src': np.concatenate(mean_out_src),
        'z_out_src': np.concatenate(z_out_src),
        'pen': np.concatenate(pen_chunks),
        'out_pen': np.concatenate(out_pen_chunks),
        'y': np.concatenate(y_chunks),
        'out_atk': np.array(out_atk),
        'out_norm': np.array(out_norm),
        'in_atk': np.array(in_atk),
        'in_norm': np.array(in_norm),
    }


def report_train_stability(stats: dict) -> None:
    mean_in = stats['mean_in'].numpy()
    mean_out = stats['mean_out'].numpy()
    std_in = stats['std_in'].numpy()
    std_out = stats['std_out'].numpy()
    active = (mean_in + mean_out) > 0

    cv_in = std_in[active] / (mean_in[active] + 1e-6)
    cv_out = std_out[active] / (mean_out[active] + 1e-6)

    print('=== Train stability (benign period) ===')
    print(f'  Active nodes: {int(active.sum()):,}')
    print(f'  in-degree std  — mean={std_in[active].mean():.4f}  median={np.median(std_in[active]):.4f}')
    print(f'  out-degree std — mean={std_out[active].mean():.4f}  median={np.median(std_out[active]):.4f}')
    print(f'  CV in-degree   — mean={cv_in.mean():.4f}  median={np.median(cv_in):.4f}')
    print(f'  CV out-degree  — mean={cv_out.mean():.4f}  median={np.median(cv_out):.4f}')
    print(f'  Fraction active nodes with std_in < 1.0:  {(std_in[active] < 1.0).mean():.3f}')
    print(f'  Fraction active nodes with std_out < 1.0: {(std_out[active] < 1.0).mean():.3f}')
    print(f'  Fraction active nodes with CV_in < 0.5:  {(cv_in < 0.5).mean():.3f}')
    print(f'  Fraction active nodes with CV_out < 0.5: {(cv_out < 0.5).mean():.3f}')
    print()


def report_test_edges(edge_data: dict) -> None:
    pen = edge_data['pen']
    y = edge_data['y']
    atk, norm = pen[y], pen[~y]

    print('=== Test edges (penalty = out_dev[src], cap=2) ===')
    print(f'  Total edges: {len(pen):,}  Attack edges: {len(atk):,}')
    print(f'  Attack  mean={atk.mean():.4f}  median={np.median(atk):.4f}')
    print(f'  Normal  mean={norm.mean():.4f}  median={np.median(norm):.4f}')
    print(f'  Mean ratio attack/normal: {atk.mean() / norm.mean():.3f}')
    print(f'  Cohen d (penalty): {_cohens_d(atk, norm):.3f}')

    for pct in [0.1, 1, 5]:
        thr = np.percentile(pen, 100 - pct)
        n_top = int((pen >= thr).sum())
        n_atk_top = int(((pen >= thr) & y).sum())
        print(f'  Top {pct}% penalty: recall={n_atk_top/len(atk):.3f}  precision={n_atk_top/n_top:.5f}')

    out_atk, out_norm = edge_data['out_atk'], edge_data['out_norm']
    in_atk, in_norm = edge_data['in_atk'], edge_data['in_norm']
    print(f'  out_dev[src]: attack={out_atk.mean():.4f}  normal={out_norm.mean():.4f}')
    print(f'  in_dev[dst]:  attack={in_atk.mean():.4f}  normal={in_norm.mean():.4f}')
    print()


def _existence_to_anomaly(existence_scores: np.ndarray) -> np.ndarray:
    """Low existence score => anomalous; PR uses higher = more anomalous."""
    return 1.0 - existence_scores


def compute_pr_stats(anomaly_scores: np.ndarray, labels: np.ndarray) -> dict:
    """Precision-recall curve and AP (attack=positive, higher score = more anomalous)."""
    y = labels.astype(np.int64)
    prec, rec, _ = precision_recall_curve(y, anomaly_scores)
    ap = float(average_precision_score(y, anomaly_scores))
    return {'precision': prec, 'recall': rec, 'ap': ap}


def build_pr_curves(
    edge_data: dict,
    base_scores: np.ndarray | None = None,
    edge_dev_alpha: float = 0.1,
) -> dict[str, dict]:
    """PR curves for penalty-only and optional baseline vs edge-dev model scores."""
    y = edge_data['y']
    curves = {
        'penalty (out+in dev)': compute_pr_stats(edge_data['pen'], y),
    }
    if base_scores is None:
        return curves

    base_t = torch.tensor(base_scores, dtype=torch.float32)
    pen_t = torch.tensor(edge_data['pen'], dtype=torch.float32)
    adj = adjust_edge_scores(base_t, pen_t, edge_dev_alpha).numpy()
    curves['baseline (model)'] = compute_pr_stats(_existence_to_anomaly(base_scores), y)
    curves[f'edge-dev (α={edge_dev_alpha:g})'] = compute_pr_stats(
        _existence_to_anomaly(adj), y,
    )
    return curves


def report_pr_curves(curves: dict[str, dict]) -> None:
    print('=== Precision-recall (attack edges positive; sweep score threshold) ===')
    for name, stats in curves.items():
        print(f'  {name}: AP={stats["ap"]:.6f}')
    print()


def plot_fig3_pr_curve(curves: dict[str, dict], fig_dir: str) -> str:
    """PR curve: recall (x) vs precision (y); AP in legend."""
    plt = _plt()
    styles = {
        'penalty (out+in dev)': ('steelblue', '-'),
        'baseline (model)': ('gray', '--'),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, stats in curves.items():
        color, ls = styles.get(name, ('crimson', '-'))
        ax.plot(
            stats['recall'], stats['precision'],
            color=color, linestyle=ls, linewidth=2.0,
            label=f'{name} (AP={stats["ap"]:.4f})',
        )

    ax.set_xlabel('Recall (attack edges caught)')
    ax.set_ylabel('Precision (flagged edges that are attacks)')
    ax.set_title('Test-set precision-recall curves')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()

    path = os.path.join(fig_dir, 'fig3_pr_curve.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return path


def build_per_seed_pr_curves(
    edge_data: dict,
    score_paths: list[str],
    edge_dev_alpha: float = 0.1,
    apply_daps: bool = True,
    include_baseline: bool = True,
    seed_labels: list[str] | None = None,
) -> dict[str, dict]:
    """PR curves per NPZ: baseline and/or edge-dev from the same existence scores."""
    from loaders.edge_dev import adjust_edge_scores

    y = edge_data['y']
    pen_t = torch.tensor(edge_data['pen'], dtype=torch.float32)
    curves: dict[str, dict] = {}
    for i, path in enumerate(score_paths):
        base = load_base_scores_npz(path, edge_data)
        label = (seed_labels[i] if seed_labels and i < len(seed_labels)
                 else f'seed {i + 1}')
        if include_baseline or not apply_daps or edge_dev_alpha <= 0:
            curves[f'{label} (baseline)'] = compute_pr_stats(
                _existence_to_anomaly(base), y,
            )
        if apply_daps and edge_dev_alpha > 0:
            base_t = torch.tensor(base, dtype=torch.float32)
            adj = adjust_edge_scores(base_t, pen_t, edge_dev_alpha).numpy()
            curves[f'{label} (edge-dev α={edge_dev_alpha:g})'] = compute_pr_stats(
                _existence_to_anomaly(adj), y,
            )
    return curves


def plot_fig3_pr_per_seed(curves: dict[str, dict], fig_dir: str,
                          filename: str = 'fig3_pr_curve_per_seed.png') -> str:
    """Plot per-seed PR curves; same seed shares color (solid=edge-dev, dashed=baseline)."""
    import re

    plt = _plt()
    cmap = plt.get_cmap('tab10')
    seed_color: dict[str, tuple] = {}
    next_c = 0

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, stats in curves.items():
        m = re.search(r'seed\s+(\d+)', name, re.I)
        seed_key = m.group(1) if m else name
        if seed_key not in seed_color:
            seed_color[seed_key] = cmap(next_c % 10)
            next_c += 1
        ls = '--' if 'baseline' in name else '-'
        ax.plot(
            stats['recall'], stats['precision'],
            color=seed_color[seed_key], linestyle=ls, linewidth=2.0,
            label=f'{name} (AP={stats["ap"]:.4f})',
        )
    ax.set_xlabel('Recall (attack edges caught)', fontsize=20)
    ax.set_ylabel('Precision (flagged edges that are attacks)', fontsize=20)
    #ax.set_title('Test-set PR curves by seed (solid=edge-dev, dashed=baseline)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis='both', labelsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=14, borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.10, right=0.65, bottom=0.16, top=0.97)
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return path


def load_base_scores_npz(path: str, edge_data: dict) -> np.ndarray:
    """Load model existence scores; verify labels align with degree_check edge order."""
    data = np.load(path)
    if 'base_scores' not in data:
        raise KeyError(f'{path} must contain base_scores array')
    base = np.asarray(data['base_scores'], dtype=np.float64)
    y = edge_data['y']
    if 'y' in data:
        file_y = np.asarray(data['y']).astype(bool)
        if file_y.shape != y.shape or not np.array_equal(file_y, y):
            raise ValueError(
                f'{path} labels do not match test edges from degree_check '
                f'(file n={file_y.size:,}, expected n={y.size:,})'
            )
    if base.shape != y.shape:
        raise ValueError(
            f'{path} base_scores length {base.size:,} != test edges {y.size:,}'
        )
    return base


def train_window(tr_start: int = 0, tr_end: int = DATE_OF_EVIL_LANL, delta: int = DELTA) -> tuple[int, int]:
    """Match spinup.py val holdout: stats are built on [tr_start, tr_end) before final 5%."""
    val = max((tr_end - tr_start) // 20, delta * 2)
    return tr_start, tr_end - val


def load_redteam_node_ids(redteam_path: str = DEFAULT_REDTEAM) -> tuple[np.ndarray, list[str]]:
    """Fixed global set of compromised host IDs from redteam.txt (src/dst computers)."""
    with open(LANL_FOLDER + 'nmap.pkl', 'rb') as f:
        nmap = pickle.load(f)
    name_to_id = {name: i for i, name in enumerate(nmap) if name}

    hosts: set[str] = set()
    with open(redteam_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                hosts.add(parts[2])
                hosts.add(parts[3])

    ids, names = [], []
    for h in sorted(hosts):
        if h in name_to_id:
            ids.append(name_to_id[h])
            names.append(h)
    return np.array(ids, dtype=np.int64), names


def load_redteam_log_events(redteam_path: str = DEFAULT_REDTEAM) -> list[tuple[int, int, int]]:
    """(timestamp, src_id, dst_id) for every line in redteam.txt."""
    with open(LANL_FOLDER + 'nmap.pkl', 'rb') as f:
        nmap = pickle.load(f)
    name_to_id = {name: i for i, name in enumerate(nmap) if name}

    events: list[tuple[int, int, int]] = []
    with open(redteam_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            ts = int(parts[0])
            src, dst = parts[2], parts[3]
            if src in name_to_id and dst in name_to_id:
                events.append((ts, name_to_id[src], name_to_id[dst]))
    return events


def _benign_edge_index(ei: torch.Tensor, data, t: int) -> torch.Tensor:
    """Drop attack-labeled edges from snapshot (train snapshots unchanged)."""
    if not data.is_test:
        return ei
    y = data.ys[t].numpy().astype(bool)
    if not y.any():
        return ei
    keep = ~y
    return ei[:, keep]


def _append_redteam_snapshot(
    trajectories: dict, redteam_ids: np.ndarray,
    ts: int, in_np: np.ndarray, out_np: np.ndarray, is_train: bool,
) -> tuple[list[float], list[float]]:
    outs, ins = [], []
    for n in redteam_ids:
        od, idg = float(out_np[n]), float(in_np[n])
        if od + idg <= 0:
            continue
        trajectories[int(n)]['t'].append(ts)
        trajectories[int(n)]['out'].append(od)
        trajectories[int(n)]['in'].append(idg)
        trajectories[int(n)]['train'].append(is_train)
        outs.append(od)
        ins.append(idg)
    return outs, ins


def collect_redteam_benign_std_timeseries(
    data, time_start: int, redteam_ids: np.ndarray, stats: dict,
) -> dict:
    """Per redteam node: train-period in/out degree std (from edge-dev stats) at each active snapshot."""
    std_out = stats['std_out'].numpy()
    std_in = stats['std_in'].numpy()
    times = []
    mean_out, mean_in = [], []
    trajectories = {
        int(n): {'t': [], 'out': [], 'in': [], 'train': []} for n in redteam_ids
    }

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        is_train = ts < DATE_OF_EVIL_LANL
        ei = _benign_edge_index(data.eis[t], data, t)
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        in_np, out_np = in_d.numpy(), out_d.numpy()
        outs, ins = [], []
        for n in redteam_ids:
            if out_np[n] + in_np[n] <= 0:
                continue
            sod, sid = float(std_out[n]), float(std_in[n])
            trajectories[int(n)]['t'].append(ts)
            trajectories[int(n)]['out'].append(sod)
            trajectories[int(n)]['in'].append(sid)
            trajectories[int(n)]['train'].append(is_train)
            outs.append(sod)
            ins.append(sid)
        mean_out.append(_nanmean(np.array(outs)) if outs else float('nan'))
        mean_in.append(_nanmean(np.array(ins)) if ins else float('nan'))

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'trajectories': trajectories,
        'n_nodes': len(redteam_ids),
    }


def collect_redteam_benign_timeseries(
    data, time_start: int, redteam_ids: np.ndarray,
) -> dict:
    """Redteam node degrees from auth graph, excluding attack-labeled edges."""
    times = []
    mean_out, mean_in = [], []
    trajectories = {
        int(n): {'t': [], 'out': [], 'in': [], 'train': []} for n in redteam_ids
    }

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        is_train = ts < DATE_OF_EVIL_LANL
        ei = _benign_edge_index(data.eis[t], data, t)
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        outs, ins = _append_redteam_snapshot(
            trajectories, redteam_ids, ts, in_d.numpy(), out_d.numpy(), is_train,
        )
        mean_out.append(_nanmean(np.array(outs)) if outs else float('nan'))
        mean_in.append(_nanmean(np.array(ins)) if ins else float('nan'))

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'trajectories': trajectories,
        'n_nodes': len(redteam_ids),
    }


def _redteam_log_by_snapshot(
    data, time_start: int, events: list[tuple[int, int, int]],
) -> list[list[tuple[int, int]]]:
    by_t: list[list[tuple[int, int]]] = [[] for _ in range(data.T)]
    for ts, src, dst in events:
        if ts < time_start:
            continue
        t = (ts - time_start) // DELTA
        if t < data.T:
            by_t[t].append((src, dst))
    return by_t


def _log_degrees_at_snapshot(
    by_t: list, t: int, num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    in_d = np.zeros(num_nodes, dtype=np.float64)
    out_d = np.zeros(num_nodes, dtype=np.float64)
    for src, dst in by_t[t]:
        out_d[src] += 1
        in_d[dst] += 1
    return in_d, out_d


def _train_log_degree_std(
    by_t: list, data, time_start: int, redteam_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node std of redteam.txt log in/out degree over train snapshots (incl. zeros)."""
    n = data.num_nodes
    out_hist = {int(node): [] for node in redteam_ids}
    in_hist = {int(node): [] for node in redteam_ids}
    for t in range(data.T):
        ts = time_start + t * DELTA
        if ts >= DATE_OF_EVIL_LANL:
            break
        in_d, out_d = _log_degrees_at_snapshot(by_t, t, n)
        for node in redteam_ids:
            nid = int(node)
            out_hist[nid].append(out_d[nid])
            in_hist[nid].append(in_d[nid])
    std_out, std_in = [], []
    for node in redteam_ids:
        nid = int(node)
        o, i = out_hist[nid], in_hist[nid]
        std_out.append(float(np.std(o, ddof=1)) if len(o) > 1 else 0.0)
        std_in.append(float(np.std(i, ddof=1)) if len(i) > 1 else 0.0)
    return np.array(std_out), np.array(std_in)


def collect_redteam_log_std_timeseries(
    data, time_start: int, redteam_ids: np.ndarray,
    events: list[tuple[int, int, int]],
) -> dict:
    """Train-period std of redteam.txt log degrees, plotted at active node-snapshots."""
    by_t = _redteam_log_by_snapshot(data, time_start, events)
    std_out, std_in = _train_log_degree_std(by_t, data, time_start, redteam_ids)
    std_out_map = {int(n): std_out[i] for i, n in enumerate(redteam_ids)}
    std_in_map = {int(n): std_in[i] for i, n in enumerate(redteam_ids)}

    times = []
    mean_out, mean_in = [], []
    trajectories = {
        int(n): {'t': [], 'out': [], 'in': [], 'train': []} for n in redteam_ids
    }

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        is_train = ts < DATE_OF_EVIL_LANL
        in_d, out_d = _log_degrees_at_snapshot(by_t, t, data.num_nodes)
        outs, ins = [], []
        for node in redteam_ids:
            nid = int(node)
            if out_d[nid] + in_d[nid] <= 0:
                continue
            sod, sid = std_out_map[nid], std_in_map[nid]
            trajectories[nid]['t'].append(ts)
            trajectories[nid]['out'].append(sod)
            trajectories[nid]['in'].append(sid)
            trajectories[nid]['train'].append(is_train)
            outs.append(sod)
            ins.append(sid)
        mean_out.append(_nanmean(np.array(outs)) if outs else float('nan'))
        mean_in.append(_nanmean(np.array(ins)) if ins else float('nan'))

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'trajectories': trajectories,
        'n_nodes': len(redteam_ids),
        'n_log_events': len(events),
    }


def collect_redteam_log_timeseries(
    data, time_start: int, redteam_ids: np.ndarray,
    events: list[tuple[int, int, int]],
) -> dict:
    """Redteam node degrees counting only connections listed in redteam.txt per snapshot."""
    by_t = _redteam_log_by_snapshot(data, time_start, events)

    times = []
    mean_out, mean_in = [], []
    trajectories = {
        int(n): {'t': [], 'out': [], 'in': [], 'train': []} for n in redteam_ids
    }

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        is_train = ts < DATE_OF_EVIL_LANL
        in_d, out_d = _log_degrees_at_snapshot(by_t, t, data.num_nodes)
        outs, ins = _append_redteam_snapshot(
            trajectories, redteam_ids, ts, in_d, out_d, is_train,
        )
        mean_out.append(_nanmean(np.array(outs)) if outs else float('nan'))
        mean_in.append(_nanmean(np.array(ins)) if ins else float('nan'))

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'trajectories': trajectories,
        'n_nodes': len(redteam_ids),
        'n_log_events': len(events),
    }


def collect_redteam_timeseries(
    data, time_start: int, redteam_ids: np.ndarray, delta: int | None = None,
) -> dict:
    """Per-snapshot degrees for fixed redteam nodes (any edge, attack or benign)."""
    delta = DELTA if delta is None else int(delta)
    snapshot_starts = getattr(data, 'snapshot_starts', None)
    times = []
    mean_out, mean_in = [], []
    trajectories = {
        int(n): {'t': [], 'out': [], 'in': [], 'attack_out': [], 'attack_in': []}
        for n in redteam_ids
    }
    n_attack_out_snaps = n_attack_in_snaps = 0

    for t in range(data.T):
        if snapshot_starts is not None:
            ts = int(snapshot_starts[t])
        else:
            ts = time_start + t * delta
        times.append(ts)
        ei = data.eis[t]
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        in_np, out_np = in_d.numpy(), out_d.numpy()

        atk_src: set[int] = set()
        atk_dst: set[int] = set()
        if data.is_test:
            yv = data.ys[t].numpy().astype(bool)
            if yv.any():
                atk_src.update(int(x) for x in ei[0, yv].long().tolist())
                atk_dst.update(int(x) for x in ei[1, yv].long().tolist())

        outs, ins = [], []
        for n in redteam_ids:
            od, idg = out_np[n], in_np[n]
            if od + idg <= 0:
                continue
            on_atk_src = int(n) in atk_src
            on_atk_dst = int(n) in atk_dst
            trajectories[int(n)]['t'].append(ts)
            trajectories[int(n)]['out'].append(od)
            trajectories[int(n)]['in'].append(idg)
            trajectories[int(n)]['attack_out'].append(on_atk_src)
            trajectories[int(n)]['attack_in'].append(on_atk_dst)
            outs.append(od)
            ins.append(idg)
            if on_atk_src:
                n_attack_out_snaps += 1
            if on_atk_dst:
                n_attack_in_snaps += 1

        mean_out.append(_nanmean(np.array(outs)) if outs else float('nan'))
        mean_in.append(_nanmean(np.array(ins)) if ins else float('nan'))

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'trajectories': trajectories,
        'n_nodes': len(redteam_ids),
        'n_attack_out_snaps': n_attack_out_snaps,
        'n_attack_in_snaps': n_attack_in_snaps,
    }


def collect_dataset_degree_snapshots(data, time_start: int) -> dict:
    """Flat node-snapshot degrees for all active nodes (attack src/dst flags aligned to panels)."""
    times = []
    mean_out, mean_in = [], []
    day_chunks, snap_idx_chunks = [], []
    out_chunks, in_chunks = [], []
    atk_out_chunks, atk_in_chunks = [], []

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        day = (ts - time_start) / 86400.0
        ei = data.eis[t]
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        in_np, out_np = in_d.numpy(), out_d.numpy()
        active = (in_np + out_np) > 0

        atk_src = np.zeros(data.num_nodes, dtype=bool)
        atk_dst = np.zeros(data.num_nodes, dtype=bool)
        if getattr(data, 'is_test', False):
            yv = data.ys[t].numpy().astype(bool)
            if yv.any():
                src_idx = ei[0, yv].long().numpy()
                dst_idx = ei[1, yv].long().numpy()
                atk_src[src_idx] = True
                atk_dst[dst_idx] = True

        outs = out_np[active]
        ins = in_np[active]
        mean_out.append(_nanmean(outs) if outs.size else float('nan'))
        mean_in.append(_nanmean(ins) if ins.size else float('nan'))

        n_act = int(active.sum())
        day_chunks.append(np.full(n_act, day, dtype=np.float32))
        snap_idx_chunks.append(np.full(n_act, t, dtype=np.int32))
        out_chunks.append(outs.astype(np.float32))
        in_chunks.append(ins.astype(np.float32))
        atk_out_chunks.append(atk_src[active])
        atk_in_chunks.append(atk_dst[active])

    return {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
        'days': np.concatenate(day_chunks) if day_chunks else np.array([], dtype=np.float32),
        'snap_idx': np.concatenate(snap_idx_chunks) if snap_idx_chunks else np.array([], dtype=np.int32),
        'out': np.concatenate(out_chunks) if out_chunks else np.array([], dtype=np.float32),
        'in': np.concatenate(in_chunks) if in_chunks else np.array([], dtype=np.float32),
        'attack_out': np.concatenate(atk_out_chunks) if atk_out_chunks else np.array([], dtype=bool),
        'attack_in': np.concatenate(atk_in_chunks) if atk_in_chunks else np.array([], dtype=bool),
        'n_nodes': data.num_nodes,
        'n_snapshots': data.T,
    }


def report_redteam_trajectories(redteam_ts: dict, tr_origin: int,
                                evil_ts: int | None = None) -> None:
    evil_ts = DATE_OF_EVIL_LANL if evil_ts is None else int(evil_ts)
    days = _days_since_start(redteam_ts['times'], tr_origin)
    evil_day = (evil_ts - tr_origin) / 86400.0
    pre = days < evil_day
    post = days >= evil_day
    for key, label in (('mean_out', 'out-degree'), ('mean_in', 'in-degree')):
        y = redteam_ts[key]
        pre_y, post_y = y[pre & ~np.isnan(y)], y[post & ~np.isnan(y)]
        print(f'  {label} mean (redteam nodes active that snapshot):')
        print(f'    pre-attack snapshots:  mean={np.nanmean(pre_y):.3f}  n_snap={np.sum(~np.isnan(y[pre]))}')
        print(f'    post-attack snapshots: mean={np.nanmean(post_y):.3f}  n_snap={np.sum(~np.isnan(y[post]))}')
    active_nodes = sum(1 for tr in redteam_ts['trajectories'].values() if tr['t'])
    print(f'  Redteam nodes with >=1 appearance: {active_nodes}/{redteam_ts["n_nodes"]}')
    if 'n_attack_out_snaps' in redteam_ts:
        print(f'  Node-snapshots attack src (out): {redteam_ts["n_attack_out_snaps"]:,}  '
              f'attack dst (in): {redteam_ts["n_attack_in_snaps"]:,}')
    print()


def slice_test_data(data, tr_start: int) -> tuple[TData, int]:
    """Return test-period TData sliced from a full timeline load.

    Uses the first snapshot at or after DATE_OF_EVIL_LANL (snapshot boundaries from
  tr_start). For PR curves vs spinup exports, use load_spinup_test_data() instead.
    """
    t0 = max(0, (DATE_OF_EVIL_LANL - tr_start + DELTA - 1) // DELTA)
    ews = getattr(data, 'ews', None)
    ews = ews[t0:] if ews is not None else None
    sliced = TData(
        data.eis[t0:], data.xs, data.ys[t0:], data.masks[t0:],
        ews=ews, nmap=getattr(data, 'nmap', None),
    )
    return sliced, tr_start + t0 * DELTA


def load_spinup_test_data() -> TData:
    """Test window matching run.py / spinup.py te_times for score export alignment."""
    return load_partial_lanl(
        start=DATE_OF_EVIL_LANL, end=TIMES['all'], delta=DELTA, is_test=True,
    )


def _nanmean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else float('nan')


def collect_degree_timeseries(data, time_start: int, is_test: bool = False) -> dict:
    """Per-snapshot mean degrees; test uses attack src (out) / dst (in) to match edge penalty."""
    times = []
    mean_out, mean_in = [], []
    ben_mean_out, ben_mean_in = [], []
    atk_src_mean_out, atk_dst_mean_in = [], []
    atk_times_out, atk_deg_out = [], []
    atk_times_in, atk_deg_in = [], []
    pooled = {
        'out': [], 'in': [],
        'atk_src_out': [], 'atk_dst_in': [],
        'ben_out': [], 'ben_in': [],
    }

    for t in range(data.T):
        ts = time_start + t * DELTA
        times.append(ts)
        ei = data.eis[t]
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        active_np = (in_d + out_d).gt(0).numpy()
        out_np, in_np = out_d.numpy(), in_d.numpy()

        if not is_test:
            src_mask = out_np > 0
            dst_mask = in_np > 0
            mean_out.append(_nanmean(out_np[src_mask]))
            mean_in.append(_nanmean(in_np[dst_mask]))
            pooled['out'].extend(out_np[active_np].tolist())
            pooled['in'].extend(in_np[active_np].tolist())
            continue

        yv = data.ys[t].numpy().astype(bool)
        atk_nodes = set()
        atk_src = np.array([], dtype=np.int64)
        atk_dst = np.array([], dtype=np.int64)
        if yv.any():
            atk_src = np.unique(ei[0, yv].long().numpy())
            atk_dst = np.unique(ei[1, yv].long().numpy())
            atk_nodes.update(atk_src.tolist())
            atk_nodes.update(atk_dst.tolist())

        atk_mask = np.zeros(data.num_nodes, dtype=bool)
        for n in atk_nodes:
            atk_mask[n] = True
        ben_mask = active_np & ~atk_mask

        ben_mean_out.append(_nanmean(out_np[ben_mask]))
        ben_mean_in.append(_nanmean(in_np[ben_mask]))
        atk_src_mean_out.append(_nanmean(out_np[atk_src]))
        atk_dst_mean_in.append(_nanmean(in_np[atk_dst]))

        pooled['ben_out'].extend(out_np[ben_mask].tolist())
        pooled['ben_in'].extend(in_np[ben_mask].tolist())
        pooled['atk_src_out'].extend(out_np[atk_src].tolist())
        pooled['atk_dst_in'].extend(in_np[atk_dst].tolist())

        for n in atk_src:
            atk_times_out.append(ts)
            atk_deg_out.append(out_np[n])
        for n in atk_dst:
            atk_times_in.append(ts)
            atk_deg_in.append(in_np[n])

    result = {
        'times': np.array(times, dtype=np.float64),
        'mean_out': np.array(mean_out, dtype=np.float64),
        'mean_in': np.array(mean_in, dtype=np.float64),
    }
    if is_test:
        result.update({
            'ben_mean_out': np.array(ben_mean_out, dtype=np.float64),
            'ben_mean_in': np.array(ben_mean_in, dtype=np.float64),
            'atk_src_mean_out': np.array(atk_src_mean_out, dtype=np.float64),
            'atk_dst_mean_in': np.array(atk_dst_mean_in, dtype=np.float64),
            'atk_times_out': np.array(atk_times_out, dtype=np.float64),
            'atk_deg_out': np.array(atk_deg_out, dtype=np.float64),
            'atk_times_in': np.array(atk_times_in, dtype=np.float64),
            'atk_deg_in': np.array(atk_deg_in, dtype=np.float64),
            'pooled': {k: np.array(v) for k, v in pooled.items()},
        })
    else:
        result['pooled'] = {k: np.array(v) for k, v in pooled.items()}
    return result


def report_train_diurnal(train_ts: dict, tr_origin: int) -> None:
    """Flag cyclical swings in train mean degree (pooled stats inflate per-node std)."""
    days = _days_since_start(train_ts['times'], tr_origin)
    for key, label in (('mean_out', 'out-degree'), ('mean_in', 'in-degree')):
        y = train_ts[key]
        valid = ~np.isnan(y)
        if not valid.any():
            continue
        yv, dv = y[valid], days[valid]
        imin, imax = int(np.argmin(yv)), int(np.argmax(yv))
        swing = (yv[imax] - yv[imin]) / max(yv[imin], 1e-6)
        print(f'  {label}: min={yv[imin]:.3f} (day {dv[imin]:.2f})  '
              f'max={yv[imax]:.3f} (day {dv[imax]:.2f})  swing={100 * swing:.1f}%')
    print('  Note: diurnal/weekly cycles in train inflate pooled per-node std → wider z baseline.')
    print()


def print_attack_mean_sanity(data, time_start: int, target_day: float = 7.0) -> None:
    """Compare union atk_active mean vs attack-src/dst means at one snapshot (masks old bug)."""
    print(f'=== Sanity check near test day {target_day} ===')
    best_t, best_gap = None, float('inf')
    for t in range(data.T):
        yv = data.ys[t].numpy().astype(bool)
        if not yv.any():
            continue
        day = (time_start + t * DELTA - time_start) / 86400.0
        gap = abs(day - target_day)
        if gap < best_gap:
            best_gap, best_t = gap, t

    if best_t is None:
        print('  No attack snapshots found.')
        print()
        return

    t = best_t
    ts = time_start + t * DELTA
    day = (ts - time_start) / 86400.0
    ei = data.eis[t]
    yv = data.ys[t].numpy().astype(bool)
    in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
    out_np = out_d.numpy()

    atk_src = np.unique(ei[0, yv].long().numpy())
    atk_dst = np.unique(ei[1, yv].long().numpy())
    atk_union = np.unique(np.concatenate([atk_src, atk_dst]))
    src_out = out_np[atk_src]
    union_out = out_np[atk_union]

    print(f'  Snapshot t={t}  day={day:.2f}  attack edges={int(yv.sum())}')
    print(f'  Attack src nodes: n={len(atk_src)}  out-degree={src_out.tolist()}')
    print(f'  Attack union nodes: n={len(atk_union)}  out-degrees (sample)={union_out[:20].tolist()}...')
    print(f'  Mean out-degree — attack src: {src_out.mean():.2f}  '
          f'union (old bug): {union_out.mean():.2f}')
    print('  Orange line now uses attack-src mean (out) / attack-dst mean (in) only.')
    print()


def report_attack_dst_in_degree(stats: dict, data, time_start: int) -> None:
    """Total snapshot in-degree at attack-edge dst vs benign — what in_dev[dst] scores on."""
    mean_in = stats['mean_in']
    std_in = stats['std_in'].clamp_min(stats['global_std_in'])
    cap = ed.ZSCORE_CAP

    raw_atk, raw_ben = [], []
    z_atk, z_ben = [], []
    tr_mean_atk, tr_mean_ben = [], []

    for t in range(data.T):
        ei = data.eis[t]
        yv = data.ys[t].numpy().astype(bool)
        in_d, _ = _degrees(ei.cpu(), data.num_nodes)
        dst = ei[1].long()
        raw = in_d[dst].numpy()
        z = ((in_d[dst] - mean_in[dst]) / std_in[dst]).numpy()
        tm = mean_in[dst].numpy()

        raw_atk.extend(raw[yv].tolist())
        raw_ben.extend(raw[~yv].tolist())
        z_atk.extend(z[yv].tolist())
        z_ben.extend(z[~yv].tolist())
        tr_mean_atk.extend(tm[yv].tolist())
        tr_mean_ben.extend(tm[~yv].tolist())

    raw_atk = np.array(raw_atk)
    raw_ben = np.array(raw_ben)
    z_atk = np.array(z_atk)
    z_ben = np.array(z_ben)
    clipped_atk = np.clip(z_atk, -cap, cap)
    clipped_ben = np.clip(z_ben, -cap, cap)

    print('=== Attack-dst total in-degree (full auth graph per snapshot) ===')
    print('  Quantity scored by in_dev[dst]: total in-degree of dst node in that snapshot.')
    print(f'  Attack edges:  n={len(raw_atk):,}')
    print(f'  Benign edges:  n={len(raw_ben):,}')
    print(f'  Raw in-degree — attack dst: mean={raw_atk.mean():.2f}  median={np.median(raw_atk):.2f}')
    print(f'  Raw in-degree — benign dst: mean={raw_ben.mean():.2f}  median={np.median(raw_ben):.2f}')
    print(f"  Cohen's d (raw): {_cohens_d(raw_atk, raw_ben):.3f}")
    print(f'  Train mean_in at dst — attack: {np.mean(tr_mean_atk):.2f}  benign: {np.mean(tr_mean_ben):.2f}')
    print(f'  Unclipped z_in[dst] — attack: mean={z_atk.mean():.3f}  median={np.median(z_atk):.3f}')
    print(f'  Unclipped z_in[dst] — benign: mean={z_ben.mean():.3f}  median={np.median(z_ben):.3f}')
    print(f"  Cohen's d (z): {_cohens_d(z_atk, z_ben):.3f}")
    print(f'  Clipped in_dev[dst] — attack: mean={clipped_atk.mean():.3f}  benign: mean={clipped_ben.mean():.3f}')
    print(f'  Attack frac z<0: {(z_atk < 0).mean():.3f}   frac z<-{cap}: {(z_atk < -cap).mean():.3f}')
    print(f'  Benign frac z<0: {(z_ben < 0).mean():.3f}   frac z<-{cap}: {(z_ben < -cap).mean():.3f}')
    print(f'  Attack frac z>+{cap}: {(z_atk > cap).mean():.3f}   benign: {(z_ben > cap).mean():.3f}')

    # unique dst node-snapshots (dedupe same dst appearing on multiple attack edges)
    seen: set[tuple[int, int]] = set()
    uniq_raw_atk, uniq_z_atk = [], []
    for t in range(data.T):
        ei = data.eis[t]
        yv = data.ys[t].numpy().astype(bool)
        if not yv.any():
            continue
        in_d, _ = _degrees(ei.cpu(), data.num_nodes)
        for dst_id in np.unique(ei[1, yv].long().numpy()):
            key = (t, int(dst_id))
            if key in seen:
                continue
            seen.add(key)
            rid = int(dst_id)
            uniq_raw_atk.append(float(in_d[rid]))
            uniq_z_atk.append(float((in_d[rid] - mean_in[rid]) / std_in[rid]))
    uniq_raw_atk = np.array(uniq_raw_atk)
    uniq_z_atk = np.array(uniq_z_atk)
    print(f'  Unique attack-dst node-snapshots: n={len(uniq_raw_atk):,}')
    print(f'    raw in-degree: mean={uniq_raw_atk.mean():.2f}  median={np.median(uniq_raw_atk):.2f}')
    print(f'    z_in: mean={uniq_z_atk.mean():.3f}  frac z<0={(uniq_z_atk < 0).mean():.3f}')
    print()


def report_zcap_asymmetry(stats: dict, data) -> None:
    """Fraction of attack edge z-scores hitting cap=2 (in vs out on different raw scales)."""
    mean_in, mean_out, std_in, std_out = _degree_z_tensors(stats)
    cap = ed.ZSCORE_CAP
    z_out_atk, z_in_atk = [], []

    for t in range(data.T):
        ei = data.eis[t]
        yv = data.ys[t].numpy().astype(bool)
        if not yv.any():
            continue
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        src, dst = ei[0, yv].long(), ei[1, yv].long()
        z_out_atk.extend(((out_d[src] - mean_out[src]) / std_out[src]).numpy().tolist())
        z_in_atk.extend(((in_d[dst] - mean_in[dst]) / std_in[dst]).numpy().tolist())

    z_out_atk = np.array(z_out_atk)
    z_in_atk = np.array(z_in_atk)
    print(f'=== Z-score cap asymmetry (attack edges, cap={cap}) ===')
    for name, z in (('out_dev[src]', z_out_atk), ('in_dev[dst]', z_in_atk)):
        clipped = np.clip(z, -cap, cap)
        hit_hi = (z > cap).mean()
        hit_lo = (z < -cap).mean()
        print(f'  {name}: unclipped mean={z.mean():.2f}  max={z.max():.1f}  '
              f'frac>{cap}={hit_hi:.3f}  frac<-{cap}={hit_lo:.3f}  '
              f'clipped mean={clipped.mean():.2f}')
    print()


def _degree_z_tensors(stats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_in, mean_out = stats['mean_in'], stats['mean_out']
    std_in = stats['std_in'].clamp_min(stats['global_std_in'])
    std_out = stats['std_out'].clamp_min(stats['global_std_out'])
    return mean_in, mean_out, std_in, std_out


def _days_since_start(times: np.ndarray, origin: int) -> np.ndarray:
    return (times - origin) / 86400.0


def _print_degree_baseline(label: str, deg: np.ndarray) -> None:
    print(
        f'    {label}: n={len(deg):,}  mean={deg.mean():.4f}  median={np.median(deg):.4f}'
    )


def _print_degree_group_stats(label: str, deg_atk: np.ndarray, deg_ben: np.ndarray) -> None:
    print(f'  --- {label} ---')
    for name, deg in [('attack (src/dst aligned)', deg_atk), ('benign-only', deg_ben)]:
        print(
            f'    {name}: n={len(deg):,}  mean={deg.mean():.4f}  median={np.median(deg):.4f}'
        )
    print(f"    Cohen's d = {_cohens_d(deg_atk, deg_ben):.3f}")


def report_node_snapshot_degrees(train_ts: dict, test_ts: dict) -> None:
    tr = train_ts['pooled']
    te = test_ts['pooled']
    print('=== Node-snapshot degree (pooled; attack src=out, dst=in) ===')
    print('  --- train baseline ---')
    _print_degree_baseline('out-degree', tr['out'])
    _print_degree_baseline('in-degree', tr['in'])
    print('  --- test ---')
    _print_degree_group_stats('out-degree (attack src)', te['atk_src_out'], te['ben_out'])
    _print_degree_group_stats('in-degree (attack dst)', te['atk_dst_in'], te['ben_in'])
    print()


def _attack_above_mean_from_arrays(
    mean_out: np.ndarray, mean_in: np.ndarray, snap_idx: np.ndarray,
    out: np.ndarray, in_deg: np.ndarray,
    attack_out: np.ndarray, attack_in: np.ndarray,
) -> dict:
    snap_mean_out = mean_out[snap_idx]
    snap_mean_in = mean_in[snap_idx]
    out_total = int(attack_out.sum())
    in_total = int(attack_in.sum())
    out_above = int((attack_out & (out > snap_mean_out)).sum()) if out_total else 0
    in_above = int((attack_in & (in_deg > snap_mean_in)).sum()) if in_total else 0
    return {
        'out_above': out_above,
        'out_total': out_total,
        'in_above': in_above,
        'in_total': in_total,
    }


def compute_attack_above_mean_stats(redteam_ts: dict) -> dict:
    """Count attack src/dst node-snapshots with degree above that snapshot's redteam mean."""
    snap_idx_chunks, out_chunks, in_chunks = [], [], []
    atk_out_chunks, atk_in_chunks = [], []
    ts_to_idx = {int(t): i for i, t in enumerate(redteam_ts['times'])}

    for tr in redteam_ts['trajectories'].values():
        for i, t in enumerate(tr['t']):
            snap_idx_chunks.append(ts_to_idx[int(t)])
            out_chunks.append(tr['out'][i])
            in_chunks.append(tr['in'][i])
            atk_out_chunks.append(tr['attack_out'][i])
            atk_in_chunks.append(tr['attack_in'][i])

    if not snap_idx_chunks:
        return {'out_above': 0, 'out_total': 0, 'in_above': 0, 'in_total': 0}

    return _attack_above_mean_from_arrays(
        redteam_ts['mean_out'], redteam_ts['mean_in'],
        np.array(snap_idx_chunks, dtype=np.int32),
        np.array(out_chunks, dtype=np.float32),
        np.array(in_chunks, dtype=np.float32),
        np.array(atk_out_chunks, dtype=bool),
        np.array(atk_in_chunks, dtype=bool),
    )


def compute_dataset_attack_above_mean_stats(dataset_ts: dict) -> dict:
    """Attack src/dst above per-snapshot mean over all active nodes."""
    return _attack_above_mean_from_arrays(
        dataset_ts['mean_out'], dataset_ts['mean_in'], dataset_ts['snap_idx'],
        dataset_ts['out'], dataset_ts['in'],
        dataset_ts['attack_out'], dataset_ts['attack_in'],
    )


def report_attack_above_mean(stats: dict, scope: str = 'redteam') -> None:
    print(f'=== Attack node-snapshots above per-snapshot {scope} mean ===')
    for key, label in (('out', 'Out-degree (attack src)'), ('in', 'In-degree (attack dst)')):
        above = stats[f'{key}_above']
        total = stats[f'{key}_total']
        if total:
            pct = 100 * above / total
            print(f'  {label}: {above}/{total} above mean ({pct:.1f}%)')
        else:
            print(f'  {label}: no attack node-snapshots')
    print()


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _plot_redteam_train_test_panels(
    ts_dict: dict, tr_origin: int, suptitle: str, path: str,
    panel_titles: tuple[str, str] = ('Out-degree', 'In-degree'),
    ylabel: str = 'Node degree',
) -> str:
    """Per-node trajectories: blue=train, green=test; mean lines per period."""
    plt = _plt()
    days = _days_since_start(ts_dict['times'], tr_origin)
    evil_day = (DATE_OF_EVIL_LANL - tr_origin) / 86400.0
    train_snap = days < evil_day

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, deg_key, panel_title in (
        (axes[0], 'out', panel_titles[0]),
        (axes[1], 'in', panel_titles[1]),
    ):
        for tr in ts_dict['trajectories'].values():
            if not tr['t']:
                continue
            x = _days_since_start(np.array(tr['t']), tr_origin)
            y = np.array(tr[deg_key])
            is_tr = np.array(tr['train'], dtype=bool)
            ax.plot(x, y, color='gray', alpha=0.12, linewidth=0.4, zorder=1)
            if is_tr.any():
                ax.scatter(x[is_tr], y[is_tr], c='blue', s=8, alpha=0.4,
                           marker='o', zorder=2, linewidths=0)
            if (~is_tr).any():
                ax.scatter(x[~is_tr], y[~is_tr], c='green', s=8, alpha=0.45,
                           marker='o', zorder=2, linewidths=0)

        mean_key = f'mean_{deg_key}'
        m = ts_dict[mean_key]
        ax.plot(days[train_snap], m[train_snap], color='blue', linewidth=2.0,
                zorder=10, label='Train mean')
        ax.plot(days[~train_snap], m[~train_snap], color='green', linewidth=2.0,
                zorder=10, label='Test mean')
        ax.axvline(evil_day, color='gray', linestyle='--', linewidth=1.0, label='Attack start')
        ax.scatter([], [], c='blue', s=18, marker='o', label='Train (pre-attack)')
        ax.scatter([], [], c='green', s=18, marker='o', label='Test (post-attack)')
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel('Days since timeline start')
    axes[1].set_xlabel('Days since timeline start')
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_redteam_benign_trajectories(ts_dict: dict, fig_dir: str, tr_origin: int) -> str:
    path = os.path.join(fig_dir, 'fig_redteam_benign_degree_timeseries.png')
    return _plot_redteam_train_test_panels(
        ts_dict, tr_origin,
        f'Redteam hosts — benign auth only (δ=30 min; n={ts_dict["n_nodes"]})',
        path,
        panel_titles=('Out-degree', 'In-degree'),
        ylabel='Node degree (benign auth)',
    )


def plot_redteam_benign_std_trajectories(ts_dict: dict, fig_dir: str, tr_origin: int) -> str:
    path = os.path.join(fig_dir, 'fig_redteam_benign_std_timeseries.png')
    return _plot_redteam_train_test_panels(
        ts_dict, tr_origin,
        f'Redteam hosts — train-period degree std (benign auth; attack edges excluded; n={ts_dict["n_nodes"]})',
        path,
        panel_titles=('Out-degree std (train)', 'In-degree std (train)'),
        ylabel='Train-period degree std',
    )


def plot_redteam_log_trajectories(ts_dict: dict, fig_dir: str, tr_origin: int) -> str:
    path = os.path.join(fig_dir, 'fig_redteam_log_degree_timeseries.png')
    n_ev = ts_dict.get('n_log_events', 0)
    return _plot_redteam_train_test_panels(
        ts_dict, tr_origin,
        f'Redteam hosts — redteam.txt events only ({n_ev:,} log lines; n={ts_dict["n_nodes"]})',
        path,
        panel_titles=('Out-degree (log)', 'In-degree (log)'),
        ylabel='Node degree (redteam.txt only)',
    )


def plot_redteam_log_std_trajectories(ts_dict: dict, fig_dir: str, tr_origin: int) -> str:
    path = os.path.join(fig_dir, 'fig_redteam_log_std_timeseries.png')
    n_ev = ts_dict.get('n_log_events', 0)
    return _plot_redteam_train_test_panels(
        ts_dict, tr_origin,
        f'Redteam hosts — train-period log degree std (redteam.txt only; {n_ev:,} lines; n={ts_dict["n_nodes"]})',
        path,
        panel_titles=('Out-degree std (train, log)', 'In-degree std (train, log)'),
        ylabel='Train-period log degree std',
    )


def plot_redteam_trajectories(
    redteam_ts: dict, fig_dir: str, tr_origin: int,
    evil_ts: int | None = None,
    delta: int | None = None,
    source_label: str = 'redteam.txt',
    filename: str = 'fig_redteam_degree_timeseries.png',
) -> str:
    """Out/in degree vs time: per-node trajectories with attack vs benign points."""
    plt = _plt()
    evil_ts = DATE_OF_EVIL_LANL if evil_ts is None else int(evil_ts)
    delta = DELTA if delta is None else int(delta)
    evil_day = (evil_ts - tr_origin) / 86400.0
    delta_min = delta // 60

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, deg_key, atk_key, title, atk_label in (
        (axes[0], 'out', 'attack_out', 'Redteam nodes — out-degree', 'Attack src'),
        (axes[1], 'in', 'attack_in', 'Redteam nodes — in-degree', 'Attack dst'),
    ):
        for tr in redteam_ts['trajectories'].values():
            if not tr['t']:
                continue
            x = _days_since_start(np.array(tr['t']), tr_origin)
            y = np.array(tr[deg_key])
            atk = np.array(tr[atk_key], dtype=bool)
            ax.plot(x, y, color='gray', alpha=0.15, linewidth=0.5, zorder=1)
            if (~atk).any():
                ax.scatter(x[~atk], y[~atk], c='steelblue', s=8, alpha=0.35,
                           marker='o', zorder=2, linewidths=0)
            if atk.any():
                ax.scatter(x[atk], y[atk], c='crimson', s=22, alpha=0.85,
                           marker='^', zorder=4, edgecolors='black', linewidths=0.3)

        ax.axvline(evil_day, color='gray', linestyle='--', linewidth=1.0, label='Attack start')
        ax.scatter([], [], c='steelblue', s=18, marker='o', label='Benign snapshot')
        ax.scatter([], [], c='crimson', s=36, marker='^', label=atk_label)

        ax.set_ylabel(f'Node {deg_key}-degree', fontsize=24)
        ax.set_title(title, fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(loc='upper right', fontsize=18)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel('Days since timeline start', fontsize=24)
    axes[1].set_xlabel('Days since timeline start', fontsize=24)
    fig.tight_layout()

    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _subsample_benign_mask(benign: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Keep all attack points; randomly subsample benign node-snapshots for plotting."""
    plot_mask = benign.copy()
    n_ben = int(benign.sum())
    if n_ben > max_points:
        ben_idx = np.flatnonzero(benign)
        keep = np.random.default_rng(seed).choice(ben_idx, max_points, replace=False)
        plot_mask[:] = False
        plot_mask[keep] = True
    return plot_mask


def _train_test_snapshot_ranges(data, time_start: int) -> tuple[range, range]:
    """Snapshot index ranges for pre-attack train vs post-attack test."""
    evil_t = data.T
    for t in range(data.T):
        if time_start + t * DELTA >= DATE_OF_EVIL_LANL:
            evil_t = t
            break
    return range(0, evil_t), range(evil_t, data.T)


def _node_mean_degrees_over_snapshots(data, snap_range: range) -> dict:
    """Per-node mean in/out degree averaged over the given snapshot indices."""
    if not snap_range:
        return {
            'mean_in': np.array([], dtype=np.float64),
            'mean_out': np.array([], dtype=np.float64),
            'n_snapshots': 0,
            'n_nodes': data.num_nodes,
        }
    in_stack, out_stack = [], []
    for t in snap_range:
        ei = data.eis[t]
        in_d, out_d = _degrees(ei.cpu(), data.num_nodes)
        in_stack.append(in_d.numpy())
        out_stack.append(out_d.numpy())
    in_mat = np.stack(in_stack)
    out_mat = np.stack(out_stack)
    return {
        'mean_in': in_mat.mean(0),
        'mean_out': out_mat.mean(0),
        'n_snapshots': len(snap_range),
        'n_nodes': data.num_nodes,
    }


def collect_node_mean_degrees(data, time_start: int) -> dict[str, dict]:
    """Per-node mean in/out over train (pre-attack) and test (post-attack) snapshots."""
    train_range, test_range = _train_test_snapshot_ranges(data, time_start)
    return {
        'train': _node_mean_degrees_over_snapshots(data, train_range),
        'test': _node_mean_degrees_over_snapshots(data, test_range),
    }


def report_node_mean_degree_stats(node_means: dict[str, dict]) -> None:
    """Terminal summary of per-node mean degrees (active nodes: mean_in + mean_out > 0)."""
    print('=== Per-node mean degree (mean over snapshots in period) ===')
    for period in ('train', 'test'):
        m = node_means[period]
        active = (m['mean_in'] + m['mean_out']) > 0
        print(f'  {period}: {m["n_snapshots"]:,} snapshots, {m["n_nodes"]:,} nodes')
        for key, label in (('mean_out', 'out-degree'), ('mean_in', 'in-degree')):
            vals = m[key][active]
            print(f'    {label} — active nodes: mean={vals.mean():.4f}  '
                  f'median={np.median(vals):.4f}  n={vals.size:,}')
    print()


def plot_node_mean_degree_distribution(
    node_means: dict, fig_dir: str, period: str, filename: str,
) -> str:
    """Histogram of per-node mean out/in degree for one train or test period."""
    plt = _plt()
    mean_in = node_means['mean_in']
    mean_out = node_means['mean_out']
    active = (mean_in + mean_out) > 0
    n_snap = node_means['n_snapshots']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, title in (
        (axes[0], mean_out[active], 'Per-node mean out-degree'),
        (axes[1], mean_in[active], 'Per-node mean in-degree'),
    ):
        ax.hist(vals, bins=80, color='steelblue', alpha=0.85, edgecolor='none')
        ax.set_yscale('log')
        ax.set_xlabel('Mean degree')
        ax.set_ylabel('Node count (log)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.97, 0.97,
            f'mean={vals.mean():.3f}\nmedian={np.median(vals):.3f}\nn={vals.size:,}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

    period_label = 'train (pre-attack)' if period == 'train' else 'test (post-attack)'
    fig.suptitle(
        f'Per-node mean degree — {period_label} '
        f'({n_snap:,} snapshots, δ=30 min; active nodes only)',
        fontsize=11,
    )
    fig.tight_layout()

    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def report_dataset_degree_means(dataset_ts: dict, tr_origin: int) -> None:
    """Mean in/out degree over all active node-snapshots, split at attack start."""
    evil_day = (DATE_OF_EVIL_LANL - tr_origin) / 86400.0
    days = dataset_ts['days']
    print('=== All active node-snapshots — mean degree (train vs test) ===')
    for label, mask in (
        ('train (pre-attack)', days < evil_day),
        ('test (post-attack)', days >= evil_day),
    ):
        n = int(mask.sum())
        print(f'  {label}: n={n:,}')
        print(f'    mean out-degree: {dataset_ts["out"][mask].mean():.4f}')
        print(f'    mean in-degree:  {dataset_ts["in"][mask].mean():.4f}')
    print()


def plot_dataset_degree_trajectories(
    dataset_ts: dict, fig_dir: str, tr_origin: int,
    benign_subsample: int = 200_000,
) -> str:
    """All active nodes: scatter of out/in degree with attack src/dst coloring (no trajectories)."""
    plt = _plt()
    evil_day = (DATE_OF_EVIL_LANL - tr_origin) / 86400.0
    days = dataset_ts['days']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, deg_key, atk_key, title, atk_label in (
        (axes[0], 'out', 'attack_out', 'All nodes — out-degree', 'Attack src'),
        (axes[1], 'in', 'attack_in', 'All nodes — in-degree', 'Attack dst'),
    ):
        y = dataset_ts[deg_key]
        atk = dataset_ts[atk_key]
        ben = ~atk
        plot_mask = _subsample_benign_mask(ben, benign_subsample)
        plot_mask |= atk

        if plot_mask.any():
            ax.scatter(
                days[plot_mask & ben], y[plot_mask & ben],
                c='steelblue', s=4, alpha=0.2, marker='o', zorder=2,
                linewidths=0, rasterized=True,
            )
        if atk.any():
            ax.scatter(
                days[atk], y[atk],
                c='crimson', s=18, alpha=0.85, marker='^', zorder=4,
                edgecolors='black', linewidths=0.25, rasterized=True,
            )

        ax.axvline(evil_day, color='gray', linestyle='--', linewidth=1.0, label='Attack start')
        ax.scatter([], [], c='steelblue', s=18, marker='o', label='Benign nodes')
        ax.scatter([], [], c='crimson', s=36, marker='^', label=atk_label)
        ax.set_ylabel(f'Node {deg_key}-degree')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel('Days since timeline start')
    axes[1].set_xlabel('Days since timeline start')
    n_pts = len(days)
    fig.suptitle(
        f'LANL auth graph — (δ=30 min)',
        fontsize=11,
    )
    fig.tight_layout()

    path = os.path.join(fig_dir, 'fig_dataset_degree_timeseries.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_node_mean_degree_plots(full_data, tr_start: int, fig_dir: str) -> tuple[str, str]:
    """Collect and write train/test per-node mean degree histogram figures."""
    os.makedirs(fig_dir, exist_ok=True)
    node_mean_degrees = collect_node_mean_degrees(full_data, tr_start)
    report_node_mean_degree_stats(node_mean_degrees)
    p_train = plot_node_mean_degree_distribution(
        node_mean_degrees['train'], fig_dir, 'train', 'fig_train_node_mean_degree.png',
    )
    p_test = plot_node_mean_degree_distribution(
        node_mean_degrees['test'], fig_dir, 'test', 'fig_test_node_mean_degree.png',
    )
    return p_train, p_test


def plot_fig1_out_degree_zscore(edge_data: dict, fig_dir: str) -> str:
    """Train mean out-degree (log x) vs unclipped snapshot out-degree z-score (y)."""
    plt = _plt()
    x = np.log10(edge_data['mean_out_src'] + 1.0)
    y = edge_data['z_out_src']
    atk = edge_data['y']

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        x[~atk], y[~atk],
        gridsize=60,
        bins='log',
        cmap='Blues',
        mincnt=1,
        linewidths=0.2,
    )
    fig.colorbar(hb, ax=ax, label='Benign edge count (log scale)')

    ax.scatter(
        x[atk], y[atk],
        c='crimson',
        s=28,
        alpha=1.0,
        edgecolors='black',
        linewidths=0.4,
        label=f'Attack edges (n={atk.sum():,})',
        zorder=10,
    )

    ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, label='Train mean (z=0)')
    ax.set_xlabel(r'$\log_{10}$(train mean out-degree + 1) at source')
    ax.set_ylabel('Snapshot out-degree z-score (unclipped)')
    ax.set_title('Train vs test out-degree deviation per edge')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(fig_dir, 'fig1_out_degree_zscore.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_fig2_penalty_histogram(edge_data: dict, fig_dir: str) -> str:
    """Log-scale histogram of edge penalty: attack vs benign."""
    plt = _plt()
    pen = edge_data['pen']
    y = edge_data['y']
    atk, norm = pen[y], pen[~y]
    d = _cohens_d(atk, norm)
    p95 = np.percentile(pen, 95)

    bins = np.linspace(0.0, pen.max() + 1e-6, 50)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(norm, bins=bins, alpha=0.55, color='steelblue', label=f'Benign (n={len(norm):,})')
    ax.hist(atk, bins=bins, alpha=0.75, color='crimson', label=f'Attack (n={len(atk):,})')
    ax.axvline(p95, color='black', linestyle=':', linewidth=1.2, label='95th percentile (top 5%)')
    ax.set_yscale('log')
    ax.set_xlabel('Edge penalty (out_dev[src], cap=2)')
    ax.set_ylabel('Edge count (log scale)')
    ax.set_title('Edge penalty distribution on test set')
    ax.text(
        0.97, 0.97,
        f"Cohen's d = {d:.2f}\nattack mean = {atk.mean():.2f}\nbenign mean = {norm.mean():.2f}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    ax.legend(loc='upper center', framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(fig_dir, 'fig2_penalty_histogram.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='Degree-deviation diagnostics')
    parser.add_argument('--fig-dir', default=DEFAULT_FIG_DIR, help='Directory for figure output')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--no-edge-figures', action='store_true',
                        help='Skip edge-level figures (fig1, fig2)')
    parser.add_argument('--dataset', choices=('LANL', 'OPTC'), default='LANL',
                        help='Dataset for degree diagnostics (default: LANL)')
    parser.add_argument('--redteam-path', default=DEFAULT_REDTEAM, help='Path to redteam.txt')
    parser.add_argument('--optc-labels', default=DEFAULT_OPTC_LABELS,
                        help='CyberGFM optc_labels.csv for OpTC redteam hosts')
    parser.add_argument('--legacy-attack-plot', action='store_true',
                        help='Also write fig_redteam_degree_timeseries.png (attack-edge coloring)')
    parser.add_argument('--dataset-degree-plot', action='store_true',
                        help='Also write fig_dataset_degree_timeseries.png (all active nodes)')
    parser.add_argument('--node-mean-degree-plots', action='store_true',
                        help='Write per-node mean in/out degree histograms for train and test')
    parser.add_argument('--benign-subsample', type=int, default=200_000,
                        help='Max benign node-snapshots to draw on dataset plot (default: 200000)')
    parser.add_argument('--pr-curve', action='store_true',
                        help='Print PR stats and write fig3_pr_curve.png')
    parser.add_argument('--base-scores', action='append', default=None,
                        help='NPZ with base_scores (+ optional y); repeat for multi-seed')
    parser.add_argument('--pr-per-seed', action='store_true',
                        help='With multiple --base-scores: one PR curve per NPZ (not mean overlay)')
    parser.add_argument('--pr-no-daps', action='store_true',
                        help='With --pr-per-seed: baseline only (skip edge-dev curves)')
    parser.add_argument('--edge-dev-alpha', type=float, default=0.1,
                        help='Alpha for edge-dev PR curve when --base-scores is set (default: 0.1)')
    args = parser.parse_args()

    if args.dataset == 'OPTC':
        fig_dir = args.fig_dir
        if fig_dir == DEFAULT_FIG_DIR:
            fig_dir = os.path.join(DEFAULT_FIG_DIR, 'optc_flow')
        run_optc_redteam_degree_plot(
            fig_dir=fig_dir,
            labels_path=args.optc_labels,
            delta=DELTA,
        )
        return

    stats = load_train_stats()
    report_train_stability(stats)

    tr_start, _ = train_window()
    redteam_ids, _ = load_redteam_node_ids(args.redteam_path)
    redteam_events = load_redteam_log_events(args.redteam_path)
    print(f'=== Redteam compromised hosts: {len(redteam_ids)} nodes (fixed global set) ===')
    print(f'  Redteam log events: {len(redteam_events):,}')

    print(f'Loading timeline ({tr_start} → {TIMES["all"]})...')
    full_data = load_partial_lanl(
        start=tr_start, end=TIMES['all'], delta=DELTA, is_test=True,
    )

    benign_ts = collect_redteam_benign_timeseries(full_data, tr_start, redteam_ids)
    benign_std_ts = collect_redteam_benign_std_timeseries(
        full_data, tr_start, redteam_ids, stats,
    )
    log_ts = collect_redteam_log_timeseries(full_data, tr_start, redteam_ids, redteam_events)
    log_std_ts = collect_redteam_log_std_timeseries(
        full_data, tr_start, redteam_ids, redteam_events,
    )
    print('=== Benign auth degrees (attack edges excluded from graph) ===')
    report_redteam_trajectories(benign_ts, tr_start)
    print('=== Benign auth — train-period degree std at active snapshots ===')
    report_redteam_trajectories(benign_std_ts, tr_start)
    print('=== Redteam.txt log degrees only ===')
    report_redteam_trajectories(log_ts, tr_start)
    print('=== Redteam.txt — train-period log degree std at active snapshots ===')
    report_redteam_trajectories(log_std_ts, tr_start)

    test_data, test_start = slice_test_data(full_data, tr_start)
    print_attack_mean_sanity(test_data, test_start, target_day=7.0)
    report_attack_dst_in_degree(stats, test_data, test_start)
    report_zcap_asymmetry(stats, test_data)
    edge_data = collect_test_edge_data(test_data, stats)
    report_test_edges(edge_data)

    if args.pr_curve:
        if args.base_scores:
            pr_test = load_spinup_test_data()
            pr_edge_data = collect_test_edge_data(pr_test, stats)
            paths = args.base_scores
            if args.pr_per_seed or len(paths) > 1:
                labels = [f'seed {i + 1}' for i in range(len(paths))]
                # Prefer seed N from filename if present
                import re
                for i, p in enumerate(paths):
                    m = re.search(r'seed[_-]?(\d+)', os.path.basename(p), re.I)
                    if m:
                        labels[i] = f'seed {m.group(1)}'
                pr_curves = build_per_seed_pr_curves(
                    pr_edge_data, paths,
                    edge_dev_alpha=args.edge_dev_alpha,
                    apply_daps=not args.pr_no_daps,
                    seed_labels=labels,
                )
            else:
                base_scores = load_base_scores_npz(paths[0], pr_edge_data)
                pr_curves = build_pr_curves(
                    pr_edge_data, base_scores=base_scores,
                    edge_dev_alpha=args.edge_dev_alpha,
                )
        else:
            pr_curves = build_pr_curves(
                edge_data, base_scores=None, edge_dev_alpha=args.edge_dev_alpha,
            )
        report_pr_curves(pr_curves)

    if not args.no_figures:
        os.makedirs(args.fig_dir, exist_ok=True)
        p_benign = plot_redteam_benign_trajectories(benign_ts, args.fig_dir, tr_start)
        p_std = plot_redteam_benign_std_trajectories(benign_std_ts, args.fig_dir, tr_start)
        p_log = plot_redteam_log_trajectories(log_ts, args.fig_dir, tr_start)
        p_log_std = plot_redteam_log_std_trajectories(log_std_ts, args.fig_dir, tr_start)
        print(f'Wrote {p_benign}')
        print(f'Wrote {p_std}')
        print(f'Wrote {p_log}')
        print(f'Wrote {p_log_std}')
        if args.legacy_attack_plot:
            redteam_ts = collect_redteam_timeseries(full_data, tr_start, redteam_ids)
            report_attack_above_mean(compute_attack_above_mean_stats(redteam_ts), scope='redteam')
            p0 = plot_redteam_trajectories(redteam_ts, args.fig_dir, tr_origin=tr_start)
            print(f'Wrote {p0}')
        if args.dataset_degree_plot:
            print('Collecting all-node degree snapshots...')
            dataset_ts = collect_dataset_degree_snapshots(full_data, tr_start)
            report_attack_above_mean(
                compute_dataset_attack_above_mean_stats(dataset_ts), scope='dataset',
            )
            report_dataset_degree_means(dataset_ts, tr_start)
            p_ds = plot_dataset_degree_trajectories(
                dataset_ts, args.fig_dir, tr_origin=tr_start,
                benign_subsample=args.benign_subsample,
            )
            print(f'Wrote {p_ds}')
        if not args.no_edge_figures:
            p1 = plot_fig1_out_degree_zscore(edge_data, args.fig_dir)
            p2 = plot_fig2_penalty_histogram(edge_data, args.fig_dir)
            print(f'Wrote {p1}')
            print(f'Wrote {p2}')
        if args.pr_curve and not (args.base_scores and (args.pr_per_seed or len(args.base_scores) > 1)):
            p3 = plot_fig3_pr_curve(pr_curves, args.fig_dir)
            print(f'Wrote {p3}')

    # Multi-seed PR always writes (works with --no-figures to skip other plots)
    if args.pr_curve and args.base_scores and (args.pr_per_seed or len(args.base_scores) > 1):
        os.makedirs(args.fig_dir, exist_ok=True)
        p3 = plot_fig3_pr_per_seed(pr_curves, args.fig_dir)
        print(f'Wrote {p3}')
    elif args.pr_curve and args.no_figures:
        os.makedirs(args.fig_dir, exist_ok=True)
        p3 = plot_fig3_pr_curve(pr_curves, args.fig_dir)
        print(f'Wrote {p3}')

    if args.node_mean_degree_plots or (args.dataset_degree_plot and not args.no_figures):
        p_train_nm, p_test_nm = write_node_mean_degree_plots(full_data, tr_start, args.fig_dir)
        print(f'Wrote {p_train_nm}')
        print(f'Wrote {p_test_nm}')


if __name__ == '__main__':
    main()
