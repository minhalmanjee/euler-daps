"""Quick src out-degree z-score sanity check on CyberGFM OpTC CSV."""

import argparse
from collections import defaultdict

import numpy as np

DEFAULT_SRC = '/home/mmanjee/CARS/datasets/optc/cybergfm/full_graph.csv'
BIN = 1800


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--bin', type=int, default=BIN)
    args = ap.parse_args()

    # Load edges: rebased ts, src, dst, label
    t_min = None
    edges = []
    with open(args.src, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            ts = float(parts[0])
            if t_min is None:
                t_min = ts
            edges.append((int(ts - t_min), int(parts[1]), int(parts[2]), int(parts[6])))

    first_mal = min((e[0] for e in edges if e[3] == 1), default=None)
    if first_mal is None:
        raise SystemExit('No malicious edges')

    # Train bins: before first mal
    train_out = defaultdict(list)  # node -> list of out-deg per train bin
    bin_edges = defaultdict(list)  # bin -> list of (src,dst,label)

    for ts, src, dst, lab in edges:
        b = ts // args.bin
        bin_edges[b].append((src, dst, lab))

    train_bins = [b for b in bin_edges if b * args.bin < first_mal]
    for b in train_bins:
        out_d = defaultdict(int)
        for src, dst, _ in bin_edges[b]:
            out_d[src] += 1
        nodes = set(out_d)
        for n in nodes:
            train_out[n].append(out_d[n])

    mean_out = {}
    std_out = {}
    for n, vals in train_out.items():
        mean_out[n] = float(np.mean(vals))
        std_out[n] = float(np.std(vals)) if len(vals) > 1 else 1.0
        if std_out[n] < 1e-6:
            std_out[n] = 1.0

    global_std = float(np.median(list(std_out.values()))) if std_out else 1.0

    mal_z, ben_z = [], []
    hub_z = []  # (z, src, bin, n_out, is_mal_snap)

    for b, elist in bin_edges.items():
        if b * args.bin < first_mal:
            continue
        out_d = defaultdict(int)
        has_mal = False
        for src, dst, lab in elist:
            out_d[src] += 1
            if lab == 1:
                has_mal = True
        for src, deg in out_d.items():
            mu = mean_out.get(src, 0.0)
            sd = std_out.get(src, global_std)
            z = (deg - mu) / max(sd, global_std)
            if has_mal and any(lab == 1 and s == src for s, _, lab in elist):
                mal_z.append(z)
            else:
                ben_z.append(z)
            hub_z.append((z, src, b, deg, has_mal))

    hub_z.sort(reverse=True)
    mal_z = np.array(mal_z) if mal_z else np.array([0.0])
    ben_z = np.array(ben_z) if ben_z else np.array([0.0])

    def rate(zs, thr=3.0):
        return float((zs > thr).mean()) if len(zs) else 0.0

    print(f'src={args.src}')
    print(f'|E|={len(edges)} first_mal_rebased={first_mal} bin={args.bin}s')
    print(f'train_bins={len(train_bins)} attack_bins={sum(1 for b in bin_edges if b*args.bin>=first_mal)}')
    print(f'mal src out-z: mean={mal_z.mean():.2f} p50={np.median(mal_z):.2f} p95={np.percentile(mal_z,95):.2f} rate>3={rate(mal_z):.3f}')
    print(f'ben src out-z: mean={ben_z.mean():.2f} p50={np.median(ben_z):.2f} p95={np.percentile(ben_z,95):.2f} rate>3={rate(ben_z):.3f}')
    print('top hubs (z, src, bin, outdeg, attack_snap):')
    for row in hub_z[:10]:
        print(f'  z={row[0]:.1f} src={row[1]} bin={row[2]} out={row[3]} atk={row[4]}')
    use = rate(mal_z) > 2 * max(rate(ben_z), 1e-6) and mal_z.mean() > ben_z.mean()
    print(f'use_daps={"yes" if use else "no"} (informational)')


if __name__ == '__main__':
    main()
