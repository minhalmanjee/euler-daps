"""Edge-level degree deviation features (diagnostic, no GCN retrain)."""

import os
import pickle

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .load_lanl import LANL_FOLDER, load_partial_lanl

EPS = 1e-6
import argparse

# Cap also exported via env so mp.spawn workers match the CLI value.
_ENV_ZSCORE_CAP = 'EULER_ZSCORE_CAP'
_DEFAULT_ZSCORE_CAP = 1.0


def get_zscore_cap() -> float:
    if _ENV_ZSCORE_CAP in os.environ:
        return float(os.environ[_ENV_ZSCORE_CAP])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--zscore-cap', type=float, default=_DEFAULT_ZSCORE_CAP,
        help='Cap for z-score in edge deviation calculation',
    )
    args, _ = parser.parse_known_args()
    return args.zscore_cap


def set_zscore_cap(cap: float) -> None:
    """Set active z-score cap (master + env for spawn workers)."""
    global ZSCORE_CAP
    ZSCORE_CAP = float(cap)
    os.environ[_ENV_ZSCORE_CAP] = str(ZSCORE_CAP)


ZSCORE_CAP = get_zscore_cap()
STATS_FILE = 'train_edge_dev_stats.pkl'
DIAGNOSTIC_ALPHAS = (0.1, 0.5, 1.0)

# Active dataset folder + partial loader (LANL by default; run.py configures OpTC).
# Folder also exported via env so mp.spawn workers resolve the same stats path.
_ENV_FOLDER = 'EULER_EDGE_DEV_FOLDER'
_DATA_FOLDER = os.environ.get(_ENV_FOLDER, LANL_FOLDER)
_LOAD_PARTIAL = load_partial_lanl


def configure(folder: str, load_partial) -> None:
    """Point edge-dev stats / train load at a dataset folder (must end with /)."""
    global _DATA_FOLDER, _LOAD_PARTIAL
    if folder and not folder.endswith('/'):
        folder = folder + '/'
    _DATA_FOLDER = folder
    _LOAD_PARTIAL = load_partial
    os.environ[_ENV_FOLDER] = folder


def stats_path() -> str:
    """Return the full path to train_edge_dev_stats.pkl in the active data folder."""
    folder = os.environ.get(_ENV_FOLDER, _DATA_FOLDER)
    if folder and not folder.endswith('/'):
        folder = folder + '/'
    return folder + STATS_FILE


def _degrees(ei: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    in_d = torch.zeros(num_nodes, dtype=torch.float32)
    out_d = torch.zeros(num_nodes, dtype=torch.float32)
    if ei.numel() == 0: # no edges in the snapshot or elements in the edge index tensor
        return in_d, out_d
    ones = torch.ones(ei.size(1), dtype=torch.float32)
    out_d.scatter_add_(0, ei[0].long(), ones)
    in_d.scatter_add_(0, ei[1].long(), ones)
    return in_d, out_d


def build_train_stats(tr_start: int, tr_end: int, delta: int) -> dict:
    data = _LOAD_PARTIAL(start=tr_start, end=tr_end, delta=delta, is_test=False)
    num_nodes = data.num_nodes
    in_stack, out_stack = [], []
    for ei in data.eis:
        in_d, out_d = _degrees(ei.cpu(), num_nodes)
        in_stack.append(in_d)
        out_stack.append(out_d)
    in_mat = torch.stack(in_stack)
    out_mat = torch.stack(out_stack)
    std_in = in_mat.std(0).clamp_min(EPS) #clamp_min is used to prevent division by zero
    std_out = out_mat.std(0).clamp_min(EPS) #clamp_min is used to prevent division by zero
    stats = {
        'num_nodes': num_nodes,
        'mean_in': in_mat.mean(0),
        'std_in': std_in,
        'mean_out': out_mat.mean(0),
        'std_out': std_out,
        'global_std_in': std_in.max().clamp_min(EPS),
        'global_std_out': std_out.max().clamp_min(EPS),
    }
    with open(stats_path(), 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote edge dev stats -> {stats_path()}')
    return stats


def load_train_stats() -> dict:
    with open(stats_path(), 'rb') as f:
        return pickle.load(f)


def snapshot_node_dev(
    ei: torch.Tensor, stats: dict, num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_d, out_d = _degrees(ei.cpu(), num_nodes)
    std_in = stats['std_in'].clamp_min(stats['global_std_in'])
    std_out = stats['std_out'].clamp_min(stats['global_std_out'])
    cap = get_zscore_cap()
    in_dev = ((in_d - stats['mean_in']) / std_in).clamp(-cap, cap)
    out_dev = ((out_d - stats['mean_out']) / std_out).clamp(-cap, cap)
    return out_dev, in_dev


def edge_penalty(
    ei: torch.Tensor, out_dev: torch.Tensor, in_dev: torch.Tensor,
) -> torch.Tensor:
    """Per-edge lateral-movement signal: out_dev[src] + in_dev[dst]."""
    return out_dev[ei[0].long()] + in_dev[ei[1].long()]


def adjust_logits(
    logits: torch.Tensor, penalty: torch.Tensor, alpha: float,
) -> torch.Tensor:
    if alpha <= 0:
        return logits
    return logits - alpha * penalty


def adjust_edge_scores(
    base_scores: torch.Tensor, penalty: torch.Tensor, alpha: float,
) -> torch.Tensor:
    logits = torch.logit(base_scores.clamp(EPS, 1 - EPS))
    return torch.sigmoid(adjust_logits(logits, penalty, alpha))


# def pick_alpha_val(
#     pscore: torch.Tensor,
#     nscore: torch.Tensor,
#     p_pen: torch.Tensor,
#     n_pen: torch.Tensor,
#     alphas: tuple[float, ...] = (0.0,) + DIAGNOSTIC_ALPHAS,
# ) -> float:
#     """Pick alpha maximizing val existence-edge AUC (includes 0 = baseline)."""
#     from utils import get_score

#     best_alpha, best_auc = 0.0, -1.0
#     for alpha in alphas:
#         if alpha == 0.0:
#             p_adj, n_adj = pscore, nscore
#         else:
#             p_adj = adjust_edge_scores(pscore, p_pen, alpha)
#             n_adj = adjust_edge_scores(nscore, n_pen, alpha)
#         auc, _ = get_score(n_adj, p_adj)
#         if auc > best_auc:
#             best_auc = auc
#             best_alpha = alpha
#     return best_alpha


# def attack_ap(raw_scores: torch.Tensor, labels: torch.Tensor) -> float:
    # """AP for attack labels; low raw score = anomalous."""
    # y = labels.numpy().astype(np.int64)
    # return float(average_precision_score(y, (1 - raw_scores).numpy()))
# 
# 
# def precision_at_tpr(
    # raw_scores: torch.Tensor, labels: torch.Tensor, target_tpr: float = 0.99,
# ) -> float:
    # scores = raw_scores.numpy()
    # y = labels.numpy().astype(bool)
    # n_pos = int(y.sum())
    # if n_pos == 0:
        # return 0.0
    # order = np.argsort(scores)
    # y_sorted = y[order]
    # need = int(np.ceil(target_tpr * n_pos))
    # cum = np.cumsum(y_sorted)
    # cut_idx = int(np.searchsorted(cum, need, side='left'))
    # flagged = np.zeros(len(scores), dtype=bool)
    # flagged[order[: cut_idx + 1]] = True
    # tp = int((flagged & y).sum())
    # fp = int((flagged & ~y).sum())
    # return tp / (tp + fp) if (tp + fp) > 0 else 0.0
# 
# 
# def print_edge_dev_diagnostic(
    # base_scores: torch.Tensor,
    # labels: torch.Tensor,
    # penalty: torch.Tensor,
    # alphas: tuple[float, ...] = DIAGNOSTIC_ALPHAS,
# ):
    # labels = labels.clamp(max=1)
    # base_ap = attack_ap(base_scores, labels)
    # base_p99 = precision_at_tpr(base_scores, labels)
    # print('\nEdge dev diagnostic (attack labels, no retrain)')
    # print(f'  baseline  AP={base_ap:.4f}  precision@99%TPR={base_p99:.4f}')
    # for alpha in alphas:
        # adj = adjust_edge_scores(base_scores, penalty, alpha)
        # ap = attack_ap(adj, labels)
        # p99 = precision_at_tpr(adj, labels)
        # print(f'  alpha={alpha:<3}  AP={ap:.4f}  precision@99%TPR={p99:.4f}')
    # print()
# 