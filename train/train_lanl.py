#!/usr/bin/env python3
"""
Train + evaluate LANL link detection (Euler-aligned static graph).

Models: GCN / GAT (3 heads) / GraphSAGE — 1-d edge_weight (GCN only), Euler regularization.

Eval:
  1. haystack — all attack events + benign test (Euler Table VI comparable)
  2. decoy   — per attack: 1 vs K hard benigns (tiers A–D); MRR, Hits@K

TPR/FPR/P on haystack: default τ=0 on suspicion (= link_logit<0 for dot, logit>0 for mlp).
Optional: --tau-mode val|cal|static for Euler Eq. 6 or fixed τ.

Link decoder: --link-decoder dot (h_u·h_v) or mlp (joint LinkMLP on h_u,h_v,edge_attr).
"""

from __future__ import annotations

from itertools import chain
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import argparse

from euler_models import LinkMLP, build_link_mlp, build_model
from metrics import (
    euler_optimal_threshold,
    full_metrics,
    rank_of_positive,
    suspicion_from_link_logits,
)


class _Tee:
    """Write stdout to console and log file."""

    def __init__(self, log_path: Path):
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = _Tee(log_path)
    print(f"Log file: {log_path}")
    return log_path


def load_graph(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def link_logits(h: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    return (h[ei[0]] * h[ei[1]]).sum(dim=-1)


def train_edge_index(data: dict) -> torch.Tensor:
    m = data["train_mask"]
    return data["edge_index"][:, m]


def graph_edge_index(data: dict) -> torch.Tensor:
    """All pre-attack edges (train ∪ val); forbid when sampling link-prediction negatives."""
    return data["edge_index"]


def train_edge_weight(data: dict) -> torch.Tensor:
    return data["edge_weight"][data["train_mask"]]


def train_edge_attr(data: dict) -> torch.Tensor:
    return data["edge_attr"][data["train_mask"]]


def masked_edge_attr(ea: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ea[mask]


def attack_cal_edge_attr(data: dict) -> torch.Tensor:
    return masked_edge_attr(data["attack_edge_attr"], data["attack_cal_mask"])


def attack_eval_edge_attr(data: dict) -> torch.Tensor:
    return masked_edge_attr(data["attack_edge_attr"], data["attack_eval_mask"])


def benign_cal_edge_attr(data: dict) -> torch.Tensor:
    return masked_edge_attr(data["benign_test_edge_attr"], data["benign_cal_mask"])


def benign_eval_edge_attr(data: dict) -> torch.Tensor:
    return masked_edge_attr(data["benign_test_edge_attr"], data["benign_eval_mask"])


def val_edge_index(data: dict) -> torch.Tensor:
    return data["edge_index"][:, data["val_mask"]]


def val_edge_attr(data: dict) -> torch.Tensor:
    return data["edge_attr"][data["val_mask"]]


def link_decoder_scores(
    h: torch.Tensor,
    ei: torch.Tensor,
    edge_attr: torch.Tensor | None,
    *,
    link_decoder: str,
    link_mlp: LinkMLP | None,
    edge_feat_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-edge link logits (BCE target: 1=edge exists)."""
    if link_decoder == "dot":
        return link_logits(h, ei)
    if link_mlp is None:
        raise ValueError("link_decoder=mlp requires link_mlp")
    ea = edge_attr
    if ea is None:
        ea = torch.zeros(ei.size(1), edge_feat_dim, device=device)
    else:
        ea = ea.to(device)
    return link_mlp(h[ei[0]], h[ei[1]], ea)


def suspicion_from_scores(
    scores: torch.Tensor,
    *,
    link_decoder: str,
) -> torch.Tensor:
    """Higher = more suspicious (attack). Dot uses -link_logit; MLP uses raw logit."""
    if link_decoder == "dot":
        return suspicion_from_link_logits(scores)
    return scores


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


def ensure_cal_splits(data: dict, cal_frac: float = 0.2, cal_seed: int = 43) -> None:
    """Add τ-cal masks to graphs saved before cal split was added."""
    if "attack_cal_mask" in data:
        return
    n_atk = data["attack_edge_index"].size(1)
    data["attack_cal_mask"], data["attack_eval_mask"] = cal_eval_mask(n_atk, cal_frac, cal_seed)
    n_ben = data["benign_test_edge_index"].size(1)
    data["benign_cal_mask"], data["benign_eval_mask"] = cal_eval_mask(
        n_ben, cal_frac, cal_seed + 1,
    )


def masked_edges(ei: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ei[:, mask]


def attack_cal_edges(data: dict) -> torch.Tensor:
    return masked_edges(data["attack_edge_index"], data["attack_cal_mask"])


def attack_eval_edges(data: dict) -> torch.Tensor:
    return masked_edges(data["attack_edge_index"], data["attack_eval_mask"])


def benign_cal_edges(data: dict) -> torch.Tensor:
    return masked_edges(data["benign_test_edge_index"], data["benign_cal_mask"])


def benign_eval_edges(data: dict) -> torch.Tensor:
    return masked_edges(data["benign_test_edge_index"], data["benign_eval_mask"])


def lam_for_model(model_name: str) -> float:
    return 0.5 if model_name == "sage" else 0.6


def haystack_neg_ratio(data: dict) -> float:
    n_atk = int(data["attack_eval_mask"].sum().item())
    n_ben = int(data["benign_eval_mask"].sum().item())
    if n_atk == 0:
        return 200.0
    return n_ben / n_atk


def sample_random_non_edges(num_nodes, pos_ei, n, rng, forbid=None):
    pos_set = set(zip(pos_ei[0].tolist(), pos_ei[1].tolist()))
    if forbid:
        pos_set |= forbid
    negs = []
    tries = 0
    while len(negs) < n and tries < n * 100:
        tries += 1
        u, v = rng.randrange(num_nodes), rng.randrange(num_nodes)
        if u != v and (u, v) not in pos_set:
            negs.append((u, v))
            pos_set.add((u, v))
    if not negs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(negs, dtype=torch.long).t().contiguous()


def uses_edge_weight(model_name: str) -> bool:
    return model_name == "gcn"


def uses_edge_attr(model_name: str, gat_variant: str) -> bool:
    return model_name == "gat" and gat_variant == "v2"


def run_encode(
    model,
    model_name: str,
    x,
    edge_index,
    edge_weight=None,
    edge_attr=None,
    gat_variant: str = "v1",
):
    if uses_edge_weight(model_name):
        return model.encode(x, edge_index, edge_weight)
    if uses_edge_attr(model_name, gat_variant):
        return model.encode(x, edge_index, edge_attr)
    return model.encode(x, edge_index)


@torch.no_grad()
def encode_model(model, model_name: str, data: dict, device, gat_variant: str = "v1"):
    model.eval()
    x = data["x"].to(device)
    ei = train_edge_index(data).to(device)
    ew = train_edge_weight(data).to(device) if uses_edge_weight(model_name) else None
    ea = train_edge_attr(data).to(device) if uses_edge_attr(model_name, gat_variant) else None
    return run_encode(model, model_name, x, ei, ew, ea, gat_variant=gat_variant)


@torch.no_grad()
def score_edge_index(
    h,
    ei,
    device,
    chunk=50000,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_attr: torch.Tensor | None = None,
    edge_feat_dim: int = 7,
):
    ei = ei.to(device)
    if ei.size(1) == 0:
        return torch.empty(0, device=device)
    parts = []
    for s in range(0, ei.size(1), chunk):
        e = ei[:, s : s + chunk]
        ea = None if edge_attr is None else edge_attr[s : s + chunk]
        parts.append(
            link_decoder_scores(
                h, e, ea, link_decoder=link_decoder, link_mlp=link_mlp,
                edge_feat_dim=edge_feat_dim, device=device,
            )
        )
    return torch.cat(parts)


def build_tier_pools(
    tier: str,
    attack_ei: torch.Tensor,
    benign_ei: torch.Tensor,
    train_ei: torch.Tensor,
    num_nodes: int,
    pool_size: int,
    rng: random.Random,
    attack_ea: torch.Tensor | None = None,
    benign_ea: torch.Tensor | None = None,
    attack_ew: torch.Tensor | None = None,
    benign_ew: torch.Tensor | None = None,
    tier_d_candidates: int = 100_000,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    n_atk = attack_ei.size(1)
    pools: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    if tier == "A":
        for _ in range(n_atk):
            pools.append((sample_random_non_edges(num_nodes, train_ei, pool_size, rng), None))

    elif tier == "B":
        n_ben = benign_ei.size(1)
        if n_ben == 0:
            return [(torch.empty((2, 0), dtype=torch.long), None)] * n_atk
        pool_idx = list(range(n_ben))
        for _ in range(n_atk):
            idx = torch.tensor(
                rng.sample(pool_idx, min(pool_size, n_ben)), dtype=torch.long,
            )
            ea = benign_ea[idx] if benign_ea is not None else None
            pools.append((benign_ei[:, idx], ea))

    elif tier == "C":
        src_to_dsts: dict[int, list[tuple[int, int]]] = {}
        if benign_ea is not None:
            for i, (u, v) in enumerate(zip(benign_ei[0].tolist(), benign_ei[1].tolist())):
                src_to_dsts.setdefault(u, []).append((v, i))
        else:
            for u, v in zip(benign_ei[0].tolist(), benign_ei[1].tolist()):
                src_to_dsts.setdefault(u, []).append((v, -1))
        for u, v in zip(attack_ei[0].tolist(), attack_ei[1].tolist()):
            cands = [(d, j) for d, j in src_to_dsts.get(u, []) if d != v]
            rng.shuffle(cands)
            picked = cands[:pool_size]
            if not picked:
                pools.append((torch.empty((2, 0), dtype=torch.long), None))
                continue
            dsts, idxs = zip(*picked)
            ei = torch.tensor([[u] * len(dsts), list(dsts)], dtype=torch.long)
            ea = benign_ea[list(idxs)] if benign_ea is not None else None
            pools.append((ei, ea))

    elif tier == "D":
        # Feature-nearest benign; subsample candidates when benign_eval is huge (window graphs).
        feat = attack_ew if attack_ew is not None else attack_ea
        ben_feat = benign_ew if benign_ew is not None else benign_ea
        if feat is None or ben_feat is None or ben_feat.numel() == 0:
            return [(torch.empty((2, 0), dtype=torch.long), None)] * n_atk
        n_ben = benign_ei.size(1)
        n_cand = min(n_ben, tier_d_candidates)
        if n_ben > n_cand:
            cand_idx = torch.tensor(rng.sample(range(n_ben), n_cand), dtype=torch.long)
            ben_ei_c = benign_ei[:, cand_idx]
            ben_feat_c = ben_feat[cand_idx]
            ben_ea_c = benign_ea[cand_idx] if benign_ea is not None else None
        else:
            ben_ei_c = benign_ei
            ben_feat_c = ben_feat
            ben_ea_c = benign_ea
        for i in range(n_atk):
            a = feat[i].float().flatten()
            if ben_feat_c.dim() == 1:
                dist = (ben_feat_c.float() - a[0]).pow(2)
            else:
                dist = (ben_feat_c.float() - a).pow(2).sum(dim=1)
            k = min(pool_size, dist.size(0))
            _, idx = torch.topk(dist, k=k, largest=False)
            pool_ea = ben_ea_c[idx] if ben_ea_c is not None else None
            pools.append((ben_ei_c[:, idx], pool_ea))
    else:
        raise ValueError(f"Unknown tier {tier}")
    return pools


def select_hardest_decoys(
    h,
    pool_ei,
    k: int,
    device,
    tier: str,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    pool_ea: torch.Tensor | None = None,
    edge_feat_dim: int = 7,
) -> torch.Tensor:
    """
    Tier A (non-edges): hardest = highest link score (false-positive non-edges).
    Tier C (same-source benign): hardest = lowest link score (most attack-like benign).
    """
    if pool_ei.size(1) == 0:
        return pool_ei
    scores = link_decoder_scores(
        h, pool_ei.to(device), pool_ea, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_feat_dim=edge_feat_dim, device=device,
    )
    k = min(k, pool_ei.size(1))
    hardest_by_highest = tier == "A"
    _, idx = torch.topk(scores, k=k, largest=hardest_by_highest)
    return pool_ei[:, idx]


def select_tier_b_decoys(
    pool_ei: torch.Tensor,
    k: int,
    rng: random.Random,
    pool_ea: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Tier B: random K held-out benign edges (baseline decoys, not hardest-mined)."""
    n = min(k, pool_ei.size(1))
    if n == 0:
        return pool_ei, pool_ea
    idx = torch.tensor(rng.sample(range(pool_ei.size(1)), n), dtype=torch.long)
    ea = pool_ea[idx] if pool_ea is not None else None
    return pool_ei[:, idx], ea


@torch.no_grad()
def suspicion_scores(
    h: torch.Tensor,
    ei: torch.Tensor,
    device,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_attr: torch.Tensor | None = None,
    edge_feat_dim: int = 7,
) -> torch.Tensor:
    scores = score_edge_index(
        h, ei, device, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_attr=edge_attr, edge_feat_dim=edge_feat_dim,
    )
    return suspicion_from_scores(scores, link_decoder=link_decoder)


@torch.no_grad()
def tune_tau_on_val(
    model,
    model_name: str,
    data: dict,
    device,
    lam: float = 0.6,
    neg_seed: int = 42,
    gat_variant: str = "v1",
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
) -> tuple[float, float, float]:
    """
    Euler §IV-D / §VI: Eq. 6 on held-out train val edges + equal random non-edges.
    No attack labels. Returns (tau, val_tpr, val_fpr).
    """
    h = encode_model(model, model_name, data, device, gat_variant=gat_variant)
    val_ei = val_edge_index(data)
    n_val = val_ei.size(1)
    if n_val == 0:
        return float("nan"), float("nan"), float("nan")

    neg_ei = sample_random_non_edges(
        data["num_nodes"], graph_edge_index(data).cpu(), n_val, random.Random(neg_seed),
    )
    pos_log = link_decoder_scores(
        h, val_ei.to(device), val_edge_attr(data),
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    neg_log = link_decoder_scores(
        h, neg_ei.to(device), None,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    logits = torch.cat([pos_log, neg_log])
    y_score = suspicion_from_scores(logits, link_decoder=link_decoder)
    y_true = torch.cat([torch.ones(n_val), torch.zeros(neg_log.size(0))])
    tau, val_tpr, val_fpr = euler_optimal_threshold(y_true, y_score, lam=lam)
    return tau, val_tpr, val_fpr


@torch.no_grad()
def tune_tau_on_cal(
    model,
    model_name: str,
    data: dict,
    device,
    lam: float,
    neg_ratio: float | None = None,
    neg_seed: int = 42,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    gat_variant: str = "v1",
    edge_feat_dim: int = 7,
) -> tuple[float, float, float, int, int]:
    """
    Optional: Eq. 6 on attack cal + benign cal. Default n_neg=n_pos (balanced).
    Set neg_ratio>0 to use haystack-like imbalance (can yield extreme TPR/FPR).
    """
    h = encode_model(model, model_name, data, device, gat_variant=gat_variant)
    atk_cal = attack_cal_edges(data)
    ben_cal = benign_cal_edges(data)
    n_pos = atk_cal.size(1)
    if n_pos == 0 or ben_cal.size(1) == 0:
        return float("nan"), float("nan"), float("nan"), n_pos, 0

    if neg_ratio is None:
        n_neg = min(ben_cal.size(1), n_pos)
    else:
        n_neg = min(ben_cal.size(1), max(n_pos, int(n_pos * neg_ratio)))
    rng = random.Random(neg_seed)
    idx = torch.tensor(rng.sample(range(ben_cal.size(1)), n_neg), dtype=torch.long)
    ben_sample = ben_cal[:, idx]
    ben_ea = benign_cal_edge_attr(data)[idx]

    atk_ea = attack_cal_edge_attr(data)
    pos_s = suspicion_scores(
        h, atk_cal.to(device), device, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_attr=atk_ea, edge_feat_dim=edge_feat_dim,
    )
    neg_s = suspicion_scores(
        h, ben_sample.to(device), device, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_attr=ben_ea, edge_feat_dim=edge_feat_dim,
    )
    y_score = torch.cat([pos_s, neg_s])
    y_true = torch.cat([torch.ones(n_pos), torch.zeros(neg_s.size(0))])
    tau, cal_tpr, cal_fpr = euler_optimal_threshold(y_true, y_score, lam=lam)
    return tau, cal_tpr, cal_fpr, n_pos, n_neg


@torch.no_grad()
def eval_decoy_protocol(
    model, model_name: str, data: dict, device,
    tier: str, k_decoys: int = 500, pool_mult: int = 5,
    gat_variant: str = "v1",
    tier_d_candidates: int = 100_000,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
) -> dict:
    h = encode_model(model, model_name, data, device, gat_variant=gat_variant)
    attack_ei = attack_eval_edges(data)
    benign_ei = benign_eval_edges(data)
    attack_mask = data["attack_eval_mask"]
    benign_mask = data["benign_eval_mask"]
    attack_ea = data.get("attack_edge_attr")
    benign_ea = data.get("benign_test_edge_attr")
    attack_ew = data.get("attack_edge_weight")
    benign_ew = data.get("benign_test_edge_weight")
    if attack_ea is not None:
        attack_ea = attack_ea[attack_mask]
    if benign_ea is not None:
        benign_ea = benign_ea[benign_mask]
    if attack_ew is not None:
        attack_ew = attack_ew[attack_mask]
    if benign_ew is not None:
        benign_ew = benign_ew[benign_mask]
    graph_ei = graph_edge_index(data)

    rng = random.Random(42)
    pool_size = max(k_decoys, k_decoys * pool_mult)
    pools = build_tier_pools(
        tier, attack_ei, benign_ei, graph_ei, data["num_nodes"],
        pool_size, rng, attack_ea, benign_ea, attack_ew, benign_ew,
        tier_d_candidates=tier_d_candidates,
    )

    all_true, all_score, ranks = [], [], []
    query_rng = random.Random(42)
    for i in range(attack_ei.size(1)):
        atk = attack_ei[:, i : i + 1]
        pool_ei, pool_ea = pools[i]
        if tier == "B":
            decoys, dec_ea = select_tier_b_decoys(pool_ei, k_decoys, query_rng, pool_ea)
        elif tier == "D":
            n = min(k_decoys, pool_ei.size(1))
            decoys = pool_ei[:, :n]
            dec_ea = pool_ea[:n] if pool_ea is not None else None
        elif pool_ei.size(1) > k_decoys:
            decoys = select_hardest_decoys(
                h, pool_ei, k_decoys, device, tier,
                link_decoder=link_decoder, link_mlp=link_mlp, pool_ea=pool_ea,
                edge_feat_dim=edge_feat_dim,
            )
            dec_ea = None
        else:
            decoys = select_hardest_decoys(
                h, pool_ei, min(k_decoys, pool_ei.size(1)), device, tier,
                link_decoder=link_decoder, link_mlp=link_mlp, pool_ea=pool_ea,
                edge_feat_dim=edge_feat_dim,
            )
            dec_ea = None

        if decoys.size(1) == 0:
            continue

        atk_ea_i = attack_ea[i : i + 1] if attack_ea is not None else None
        atk_s = suspicion_scores(
            h, atk, device, link_decoder=link_decoder, link_mlp=link_mlp,
            edge_attr=atk_ea_i, edge_feat_dim=edge_feat_dim,
        )
        dec_s = suspicion_scores(
            h, decoys, device, link_decoder=link_decoder, link_mlp=link_mlp,
            edge_attr=dec_ea, edge_feat_dim=edge_feat_dim,
        )
        y_s = torch.cat([atk_s, dec_s])
        all_true.append(torch.cat([torch.ones(1), torch.zeros(decoys.size(1))]))
        all_score.append(y_s)
        ranks.append(rank_of_positive(y_s))

    if not all_true:
        nan = float("nan")
        keys = ("AP", "AUC", "MRR", "Hits@20", "Hits@50")
        return {k: nan for k in keys}

    y_true = torch.cat(all_true)
    y_score = torch.cat(all_score)
    m = full_metrics(y_true, y_score, ranks=ranks, classification=False)
    m["n_queries"] = len(ranks)
    # actual K used (varies for Tier C due to sparse same-source pools)
    m["mean_k"] = round((y_true.numel() - len(ranks)) / max(len(ranks), 1))
    return m


@torch.no_grad()
def eval_haystack_protocol(
    model, model_name: str, data: dict, device,
    max_benign: int | None = None, tau: float | None = None,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    gat_variant: str = "v1",
    edge_feat_dim: int = 7,
) -> dict:
    h = encode_model(model, model_name, data, device, gat_variant=gat_variant)
    attack_ei = attack_eval_edges(data)
    benign_ei = benign_eval_edges(data)
    attack_ea = attack_eval_edge_attr(data)
    benign_ea = benign_eval_edge_attr(data)
    edge_dim = edge_feat_dim

    if max_benign is not None and benign_ei.size(1) > max_benign:
        rng = random.Random(42)
        idx = torch.tensor(rng.sample(range(benign_ei.size(1)), max_benign), dtype=torch.long)
        benign_ei = benign_ei[:, idx]
        benign_ea = benign_ea[idx]

    atk_s = suspicion_scores(
        h, attack_ei.to(device), device, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_attr=attack_ea, edge_feat_dim=edge_dim,
    )
    ben_s = suspicion_scores(
        h, benign_ei.to(device), device, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_attr=benign_ea, edge_feat_dim=edge_dim,
    )
    y_score = torch.cat([atk_s, ben_s])
    n_atk = attack_ei.size(1)
    y_true = torch.cat([torch.ones(n_atk), torch.zeros(benign_ei.size(1))])
    m = full_metrics(y_true, y_score, ranks=None, tau=tau)
    m["n_scored"] = n_atk + benign_ei.size(1)
    m["n_attack"] = n_atk
    return m


def link_bce_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Mean of positive-edge and negative-edge BCE (reportable scale ~0.69 random)."""
    return (
        F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    ) / 2


def link_accuracy(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    """Balanced accuracy at logit threshold 0: (acc_pos + acc_neg) / 2."""
    pos_acc = (pos_score > 0).float().mean()
    neg_acc = (neg_score <= 0).float().mean()
    return (pos_acc + neg_acc) / 2


@torch.no_grad()
def eval_split_loss_acc(
    model,
    model_name: str,
    data: dict,
    device,
    pos_ei: torch.Tensor,
    neg_ei: torch.Tensor,
    gat_variant: str = "v1",
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    pos_ea: torch.Tensor | None = None,
    edge_feat_dim: int = 7,
) -> tuple[float, float]:
    """BCE + balanced accuracy on provided positive and negative edge sets."""
    model.eval()
    if link_mlp is not None:
        link_mlp.eval()
    x = data["x"].to(device)
    train_ei = train_edge_index(data)
    ew = train_edge_weight(data).to(device) if uses_edge_weight(model_name) else None
    ea_enc = train_edge_attr(data).to(device) if uses_edge_attr(model_name, gat_variant) else None
    h = run_encode(model, model_name, x, train_ei.to(device), ew, ea_enc, gat_variant=gat_variant)

    pos_score = link_decoder_scores(
        h, pos_ei.to(device), pos_ea,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    neg_score = link_decoder_scores(
        h, neg_ei.to(device), None,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    return (
        link_bce_loss(pos_score, neg_score).item(),
        link_accuracy(pos_score, neg_score).item(),
    )


@torch.no_grad()
def eval_train_metrics(
    model, model_name: str, data: dict, device, nratio: float = 1.0, neg_seed: int = 42,
    gat_variant: str = "v1",
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
) -> tuple[float, float]:
    pos_ei = train_edge_index(data)
    n_neg = max(1, int(pos_ei.size(1) * nratio))
    neg_ei = sample_random_non_edges(
        data["num_nodes"], graph_edge_index(data).cpu(), n_neg, random.Random(neg_seed),
    )
    return eval_split_loss_acc(
        model, model_name, data, device, pos_ei, neg_ei, gat_variant=gat_variant,
        link_decoder=link_decoder, link_mlp=link_mlp,
        pos_ea=train_edge_attr(data), edge_feat_dim=edge_feat_dim,
    )


@torch.no_grad()
def eval_val_metrics(
    model, model_name: str, data: dict, device, nratio: float = 1.0, neg_seed: int = 42,
    gat_variant: str = "v1",
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
) -> tuple[float, float]:
    pos_ei = val_edge_index(data)
    if pos_ei.size(1) == 0:
        return float("inf"), float("nan")
    n_neg = max(1, int(pos_ei.size(1) * nratio))
    neg_ei = sample_random_non_edges(
        data["num_nodes"], graph_edge_index(data).cpu(), n_neg, random.Random(neg_seed),
    )
    return eval_split_loss_acc(
        model, model_name, data, device, pos_ei, neg_ei, gat_variant=gat_variant,
        link_decoder=link_decoder, link_mlp=link_mlp,
        pos_ea=val_edge_attr(data), edge_feat_dim=edge_feat_dim,
    )


@torch.no_grad()
def eval_test_metrics(
    model, model_name: str, data: dict, device, tau: float | None = None,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    gat_variant: str = "v1",
    edge_feat_dim: int = 7,
) -> tuple[float, float]:
    """Eval haystack = attack eval + benign eval; acc at suspicion τ (0 = link_logit threshold)."""
    pos_ei = attack_eval_edges(data)
    neg_ei = benign_eval_edges(data)
    if pos_ei.size(1) == 0 or neg_ei.size(1) == 0:
        return float("nan"), float("nan")

    model.eval()
    if link_mlp is not None:
        link_mlp.eval()
    h = encode_model(model, model_name, data, device, gat_variant=gat_variant)
    pos_ea = attack_eval_edge_attr(data)
    neg_ea = benign_eval_edge_attr(data)
    pos_log = link_decoder_scores(
        h, pos_ei.to(device), pos_ea,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    neg_log = link_decoder_scores(
        h, neg_ei.to(device), neg_ea,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim, device=device,
    )
    loss = link_bce_loss(pos_log, neg_log).item()
    pos_score = suspicion_from_scores(pos_log, link_decoder=link_decoder)
    neg_score = suspicion_from_scores(neg_log, link_decoder=link_decoder)

    if tau is None:
        pred = torch.cat([pos_score, neg_score]) > 0
        n_atk = pos_score.size(0)
        acc = (pred[:n_atk].float().mean().item() + (~pred[n_atk:]).float().mean().item()) / 2
    else:
        y_score = torch.cat([pos_score, neg_score])
        pred = y_score >= tau
        n_atk = pos_score.size(0)
        pos_acc = pred[:n_atk].float().mean().item()
        neg_acc = (~pred[n_atk:]).float().mean().item()
        acc = (pos_acc + neg_acc) / 2
    return loss, acc


def train_epoch(
    model, model_name: str, data: dict, optimizer, device, nratio=1.0,
    gat_variant: str = "v1",
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
):
    model.train()
    if link_mlp is not None:
        link_mlp.train()
    optimizer.zero_grad()
    x = data["x"].to(device)
    ei = train_edge_index(data).to(device)
    ew = train_edge_weight(data).to(device) if uses_edge_weight(model_name) else None
    ea_enc = train_edge_attr(data).to(device) if uses_edge_attr(model_name, gat_variant) else None
    h = run_encode(model, model_name, x, ei, ew, ea_enc, gat_variant=gat_variant)

    n_neg = max(1, int(ei.size(1) * nratio))
    neg_ei = sample_random_non_edges(
        data["num_nodes"], graph_edge_index(data).cpu(), n_neg, random.Random(),
    ).to(device)
    pos_ea = train_edge_attr(data).to(device)
    pos_score = link_decoder_scores(
        h, ei, pos_ea, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_feat_dim=edge_feat_dim, device=device,
    )
    neg_score = link_decoder_scores(
        h, neg_ei, None, link_decoder=link_decoder, link_mlp=link_mlp,
        edge_feat_dim=edge_feat_dim, device=device,
    )
    loss = link_bce_loss(pos_score, neg_score)
    loss.backward()
    optimizer.step()
    return loss.item(), link_accuracy(pos_score, neg_score).item()


def _val_improved(
    val_acc: float, val_loss: float, best_acc: float, best_loss: float, metric: str,
) -> bool:
    """Primary: val_acc (matches logit>0 train objective); tie-break on val_loss."""
    if metric == "loss":
        if val_loss < best_loss - 1e-8:
            return True
        return val_loss <= best_loss + 1e-8 and val_acc > best_acc + 1e-8
    if val_acc > best_acc + 1e-8:
        return True
    return val_acc >= best_acc - 1e-8 and val_loss < best_loss - 1e-8


def train_with_early_stopping(
    model,
    model_name: str,
    data: dict,
    optimizer,
    device,
    max_epochs: int = 1500,
    patience: int = 25,
    log_every: int = 10,
    stop_metric: str = "acc",
    gat_variant: str = "v1",
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    edge_feat_dim: int = 7,
) -> int:
    """
    Euler §V/VI: up to 1500 epochs, stop after `patience` epochs without val improvement.
    Default `stop_metric=acc` (balanced acc at logit 0); val_loss can rise while acc improves.
    Restores best checkpoint before returning.
    """
    best_val_acc = -1.0
    best_val_loss = float("inf")
    min_val_loss = float("inf")
    min_val_loss_ep = 0
    best_ep = 0
    best_state: dict | None = None
    stale = 0
    last_ep = 0

    print(f"  early-stop: {stop_metric} (link_decoder={link_decoder})")

    for ep in range(1, max_epochs + 1):
        last_ep = ep
        train_loss, train_acc = train_epoch(
            model, model_name, data, optimizer, device,
            gat_variant=gat_variant, link_decoder=link_decoder, link_mlp=link_mlp,
            edge_feat_dim=edge_feat_dim,
        )
        val_loss, val_acc = eval_val_metrics(
            model, model_name, data, device, gat_variant=gat_variant,
            link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
        )

        if val_loss < min_val_loss - 1e-8:
            min_val_loss = val_loss
            min_val_loss_ep = ep

        if _val_improved(val_acc, val_loss, best_val_acc, best_val_loss, stop_metric):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_ep = ep
            stale = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            if link_mlp is not None:
                best_state["link_mlp"] = {
                    k: v.detach().cpu().clone() for k, v in link_mlp.state_dict().items()
                }
            loss_note = ""
            if stop_metric == "acc" and val_loss > min_val_loss + 1e-4:
                loss_note = (
                    f"  [loss vs {min_val_loss:.4f}@ep{min_val_loss_ep}; "
                    f"keeping higher acc]"
                )
            print(
                f"  epoch {ep:4d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  *best*{loss_note}"
            )
        else:
            stale += 1
            if ep % log_every == 0 or stale >= patience:
                print(
                    f"  epoch {ep:4d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                    f"patience {stale}/{patience}  (checkpoint=ep{best_ep})"
                )

        if stale >= patience:
            print(
                f"  early stop at epoch {ep} "
                f"(no val {stop_metric} improvement for {patience} epochs)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        if link_mlp is not None and "link_mlp" in best_state:
            link_mlp.load_state_dict(best_state["link_mlp"])
    train_loss, train_acc = eval_train_metrics(
        model, model_name, data, device, gat_variant=gat_variant,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
    )
    val_loss, val_acc = eval_val_metrics(
        model, model_name, data, device, gat_variant=gat_variant,
        link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
    )
    print(
        f"  trained {last_ep} epochs; checkpoint=epoch {best_ep}  "
        f"val_acc={best_val_acc:.4f}  val_loss={best_val_loss:.4f}\n"
        f"  (min val_loss={min_val_loss:.4f} at epoch {min_val_loss_ep}; "
        f"not used when early-stop metric is acc)\n"
        f"  final train  loss={train_loss:.4f}  acc={train_acc:.4f}\n"
        f"  final val    loss={val_loss:.4f}  acc={val_acc:.4f}"
    )
    return last_ep


def print_metrics_block(m: dict, title: str | None = None):
    if title:
        mean_k = m.get("mean_k")
        k_note = f", mean_K={mean_k}" if mean_k is not None else ""
        print(f"  [{title}{k_note}]")
    for k in ("AP", "AUC", "MRR", "Hits@20", "Hits@50"):
        if k in m:
            v = m[k]
            print(f"    {k:10s}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
    if "tau" in m:
        print(f"    {'tau':10s}: {m['tau']:.6f}")
    if "TPR" in m:
        print(f"    {'TPR':10s}: {m['TPR'] * 100:.2f}%")
    if "FPR" in m:
        print(f"    {'FPR':10s}: {m['FPR'] * 100:.4f}%")
    if "P" in m:
        print(f"    {'P':10s}: {m['P']:.6f}")


def run_all_eval(
    model, model_name: str, data: dict, device, k_decoys, haystack_max, tau,
    *,
    link_decoder: str = "dot",
    link_mlp: LinkMLP | None = None,
    gat_variant: str = "v1",
    tier_d_candidates: int = 100_000,
    edge_feat_dim: int = 7,
):
    out = {"decoy": {}, "haystack": {}}
    for t in ["A", "B", "C", "D"]:
        print(f"  eval decoy tier {t}...", flush=True)
        t0 = time.perf_counter()
        out["decoy"][t] = eval_decoy_protocol(
            model, model_name, data, device, t, k_decoys,
            gat_variant=gat_variant, tier_d_candidates=tier_d_candidates,
            link_decoder=link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
        )
        print(f"  eval decoy tier {t} done ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("  eval haystack...", flush=True)
    t0 = time.perf_counter()
    out["haystack_full"] = eval_haystack_protocol(
        model, model_name, data, device, max_benign=None, tau=tau,
        link_decoder=link_decoder, link_mlp=link_mlp, gat_variant=gat_variant,
        edge_feat_dim=edge_feat_dim,
    )
    print(f"  eval haystack done ({time.perf_counter() - t0:.1f}s)", flush=True)
    if haystack_max is not None:
        out["haystack_capped"] = eval_haystack_protocol(
            model, model_name, data, device, max_benign=haystack_max, tau=tau,
            link_decoder=link_decoder, link_mlp=link_mlp, gat_variant=gat_variant,
            edge_feat_dim=edge_feat_dim,
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("processed/lanl/graph.pt"))
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "all", "both"], default="all",
                        help="'all'/'both' = run gcn+gat+sage")
    parser.add_argument("--max-epochs", type=int, default=1500,
                        help="Max epochs (Euler §V: 1500)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stop patience on val metric (Euler §VI: 10)")
    parser.add_argument(
        "--early-stop-metric", choices=["acc", "loss"], default="acc",
        help="acc = balanced val acc at logit 0 (default); loss = min val BCE",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--k-decoys", type=int, default=500)
    parser.add_argument("--haystack-max", type=int, default=None,
                        help="Cap benign edges in haystack eval (recommended for window graphs)")
    parser.add_argument(
        "--tier-d-candidates", type=int, default=100_000,
        help="Max benign candidates for Tier D feature-nearest pool (avoids O(67M) topk)",
    )
    parser.add_argument(
        "--tau-mode", choices=["zero", "val", "cal", "static"], default="zero",
        help="zero = suspicion>=0 (link_logit<=0, matches train acc); val/cal = Eq.6; static = --tau",
    )
    parser.add_argument(
        "--tau", type=float, default=0.0,
        help="Suspicion threshold when --tau-mode static (attack if suspicion >= tau)",
    )
    parser.add_argument("--lam", type=float, default=None,
                        help="Eq.6 λ (default: 0.6; SAGE uses 0.5)")
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--cal-seed", type=int, default=43)
    parser.add_argument(
        "--cal-neg-ratio", type=float, default=None,
        help="τ-cal only: benign:attack ratio (default balanced n_neg=n_pos)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("/home/mmanjee/CARS/output_lanl"),
        help="Directory for timestamped run logs",
    )
    parser.add_argument(
        "--link-decoder", choices=["dot", "mlp"], default="dot",
        help="dot=h_u·h_v; mlp=LinkMLP(h_u,h_v,edge_attr) joint-trained with GNN (window graphs)",
    )
    parser.add_argument(
        "--gat-variant", choices=["v1", "v2"], default="v1",
        help="v1=Euler GAT (no edge_attr); v2=GATv2 with edge_attr in message passing",
    )
    parser.add_argument(
        "--link-mlp-no-dot", action="store_true",
        help="MLP decoder input excludes dot-product link_logit feature",
    )
    args = parser.parse_args()

    setup_logging(args.log_dir)
    data = load_graph(args.graph)
    ensure_cal_splits(data, cal_frac=args.cal_frac, cal_seed=args.cal_seed)
    device = torch.device(args.device)
    models = ["gcn", "gat", "sage"] if args.model in ("all", "both") else [args.model]
    edge_feat_dim = int(data.get("edge_feat_dim", 7))

    print(
        f"Graph: {data['num_nodes']} nodes, granularity={data.get('granularity', '?')}, "
        f"edge_feat_dim={edge_feat_dim}, link_decoder={args.link_decoder}, "
        f"train={data['train_mask'].sum().item()} (val={data['val_mask'].sum().item()}), "
        f"attack_cal={data['attack_cal_mask'].sum().item()}, "
        f"attack_eval={data['attack_eval_mask'].sum().item()}, "
        f"benign_cal={data['benign_cal_mask'].sum().item()}, "
        f"benign_eval={data['benign_eval_mask'].sum().item()}"
    )
    n_ben_eval = int(data["benign_eval_mask"].sum().item())
    if n_ben_eval > 1_000_000 and args.haystack_max is None:
        print(
            f"  WARNING: benign_eval={n_ben_eval:,} — full haystack is slow on CPU. "
            f"Consider --haystack-max 500000",
            flush=True,
        )

    for name in models:
        print(f"\n{'='*60}\n=== {name.upper()} ===")
        gat_variant = args.gat_variant if name == "gat" else "v1"
        if name == "gat" and gat_variant == "v2":
            print(f"  GATv2 with edge_attr dim={edge_feat_dim}")
        model = build_model(
            name, data["node_feat_dim"],
            gat_variant=gat_variant, edge_feat_dim=edge_feat_dim,
        ).to(device)

        link_mlp = None
        if args.link_decoder == "mlp":
            link_mlp = build_link_mlp(
                edge_feat_dim=edge_feat_dim,
                include_dot=not args.link_mlp_no_dot,
            ).to(device)
            print(f"  LinkMLP decoder (include_dot={not args.link_mlp_no_dot})")

        params = (
            chain(model.parameters(), link_mlp.parameters())
            if link_mlp is not None
            else model.parameters()
        )
        opt = torch.optim.Adam(params, lr=args.lr)

        train_with_early_stopping(
            model, name, data, opt, device,
            max_epochs=args.max_epochs, patience=args.patience,
            stop_metric=args.early_stop_metric,
            gat_variant=gat_variant,
            link_decoder=args.link_decoder,
            link_mlp=link_mlp,
            edge_feat_dim=edge_feat_dim,
        )

        if args.tau_mode == "zero":
            tau = 0.0
            print("  tau = 0.0  (suspicion>=0)")
        elif args.tau_mode == "val":
            lam = args.lam if args.lam is not None else lam_for_model(name)
            tau, val_tpr, val_fpr = tune_tau_on_val(
                model, name, data, device, lam=lam, gat_variant=gat_variant,
                link_decoder=args.link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
            )
            print(
                f"  tau_val = {tau:.6f}  (Eq.6 λ={lam}, "
                f"val n={data['val_mask'].sum().item()}, "
                f"val TPR={val_tpr * 100:.2f}% FPR={val_fpr * 100:.4f}%)"
            )
        elif args.tau_mode == "cal":
            lam = args.lam if args.lam is not None else lam_for_model(name)
            tau, cal_tpr, cal_fpr, n_pos, n_neg = tune_tau_on_cal(
                model, name, data, device, lam=lam, neg_ratio=args.cal_neg_ratio,
                gat_variant=gat_variant,
                link_decoder=args.link_decoder, link_mlp=link_mlp, edge_feat_dim=edge_feat_dim,
            )
            print(
                f"  tau_cal = {tau:.6f}  (Eq.6 λ={lam}, cal n_pos={n_pos} n_neg={n_neg}, "
                f"cal TPR={cal_tpr * 100:.2f}% FPR={cal_fpr * 100:.4f}%)"
            )
        else:
            tau = args.tau
            print(f"  static tau = {tau:.6f}")

        test_loss, test_acc = eval_test_metrics(
            model, name, data, device, tau=tau,
            link_decoder=args.link_decoder, link_mlp=link_mlp,
            gat_variant=gat_variant, edge_feat_dim=edge_feat_dim,
        )
        print(f"  final test   loss={test_loss:.4f}  acc={test_acc:.4f}")

        res = run_all_eval(
            model, name, data, device, args.k_decoys, args.haystack_max, tau,
            link_decoder=args.link_decoder, link_mlp=link_mlp,
            gat_variant=gat_variant, tier_d_candidates=args.tier_d_candidates,
            edge_feat_dim=edge_feat_dim,
        )

        print(f"\n  Protocol 1: decoy (link_decoder={args.link_decoder})")
        for t, m in res["decoy"].items():
            print_metrics_block(m, title=f"Tier {t} (K={args.k_decoys})")

        print()
        print_metrics_block(res["haystack_full"], title=f"haystack (link_decoder={args.link_decoder})")
        if "haystack_capped" in res:
            print_metrics_block(
                res["haystack_capped"],
                title=f"haystack capped (max_benign={args.haystack_max})",
            )


if __name__ == "__main__":
    main()
