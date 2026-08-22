"""Ranking and classification metrics for link-prediction eval."""

from __future__ import annotations

import torch


def suspicion_from_link_logits(link_logits: torch.Tensor) -> torch.Tensor:
    """Higher = more suspicious (attack). suspicion=0 ⟺ link_logit=0 (train acc threshold)."""
    return -link_logits


def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    if y_true.sum() == 0:
        return float("nan")
    order = torch.argsort(y_score, descending=True)
    y = y_true[order].float()
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1.0 - y, dim=0)
    prec = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / y_true.sum().float()
    recall_prev = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    return float((recall - recall_prev).mul(prec).sum().item())


def auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    if y_true.sum() == 0 or y_true.sum() == y_true.numel():
        return float("nan")
    order = torch.argsort(y_score, descending=True)
    y = y_true[order].float()
    tps = y_true.sum().float()
    fps = (1.0 - y_true).sum().float()
    tp_c = torch.cumsum(y, dim=0)
    fp_c = torch.cumsum(1.0 - y, dim=0)
    tpr = torch.cat([torch.zeros(1), tp_c / tps])
    fpr = torch.cat([torch.zeros(1), fp_c / fps])
    return float(torch.trapz(tpr, fpr).item())


def euler_optimal_threshold(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    lam: float = 0.6,
) -> tuple[float, float, float]:
    """
    Eq. 6: argmin |(1-lam)*TPR - lam*FPR| over thresholds.
    Predict positive if y_score >= tau. Returns (tau, tpr, fpr).
    """
    if y_true.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    tps = y_true.sum().float()
    fps = (1.0 - y_true).sum().float()
    if tps == 0 or fps == 0:
        return float("nan"), float("nan"), float("nan")

    uniq = torch.unique(y_score)
    best_cost, best_tpr, best_fpr, best_tau = float("inf"), 0.0, 0.0, uniq[0].item()

    for tau in uniq:
        pred = y_score >= tau
        tp = (pred & (y_true == 1)).sum().float()
        fp = (pred & (y_true == 0)).sum().float()
        tpr = (tp / tps).item()
        fpr_v = (fp / fps).item()
        cost = abs((1.0 - lam) * tpr - lam * fpr_v)
        if cost < best_cost:
            best_cost, best_tpr, best_fpr, best_tau = cost, tpr, fpr_v, tau.item()

    return best_tau, best_tpr, best_fpr


def tpr_fpr_precision_at_tau(
    y_true: torch.Tensor, y_score: torch.Tensor, tau: float,
) -> tuple[float, float, float]:
    """Returns (TPR, FPR, precision) at threshold tau."""
    if y_true.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    tps = y_true.sum().float()
    fps = (1.0 - y_true).sum().float()
    pred = y_score >= tau
    tp = (pred & (y_true == 1)).sum().float()
    fp = (pred & (y_true == 0)).sum().float()
    tpr = (tp / tps).item()
    fpr = (fp / fps).item()
    prec = (tp / (tp + fp).clamp_min(1e-12)).item() if (tp + fp) > 0 else 0.0
    return tpr, fpr, prec


def tpr_fpr_at_tau(y_true: torch.Tensor, y_score: torch.Tensor, tau: float) -> tuple[float, float]:
    tpr, fpr, _ = tpr_fpr_precision_at_tau(y_true, y_score, tau)
    return tpr, fpr


def rank_of_positive(y_score: torch.Tensor) -> int:
    atk = y_score[0]
    return int((y_score >= atk).sum().item())


def mrr_from_ranks(ranks: list[int]) -> float:
    if not ranks:
        return float("nan")
    return sum(1.0 / r for r in ranks) / len(ranks)


def hits_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return float("nan")
    return sum(1 for r in ranks if r <= k) / len(ranks)


def full_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    ranks: list[int] | None = None,
    tau: float | None = None,
    classification: bool = True,
) -> dict:
    """AP, AUC, MRR, Hits@20/50. tau: fixed threshold for TPR/FPR (haystack only)."""
    out = {
        "AP": average_precision(y_true, y_score),
        "AUC": auc_roc(y_true, y_score),
    }
    if classification:
        if tau is None:
            tau, tpr, fpr = euler_optimal_threshold(y_true, y_score)
            _, _, prec = tpr_fpr_precision_at_tau(y_true, y_score, tau)
        else:
            tpr, fpr, prec = tpr_fpr_precision_at_tau(y_true, y_score, tau)
        out["tau"] = tau
        out["TPR"] = tpr
        out["FPR"] = fpr
        out["P"] = prec
    if ranks is not None:
        out["MRR"] = mrr_from_ranks(ranks)
        out["Hits@20"] = hits_at_k(ranks, 20)
        out["Hits@50"] = hits_at_k(ranks, 50)
    else:
        pos_idx = (y_true == 1).nonzero(as_tuple=True)[0]
        global_ranks = [int((y_score >= y_score[pi]).sum().item()) for pi in pos_idx]
        out["MRR"] = mrr_from_ranks(global_ranks)
        out["Hits@20"] = hits_at_k(global_ranks, 20)
        out["Hits@50"] = hits_at_k(global_ranks, 50)
    return out
