"""Slice CyberGFM OpTC full_graph.csv into Euler DELTA=10000 shard files."""

import argparse
import json
import os
import pickle

from tqdm import tqdm

DEFAULT_SRC = '/home/mmanjee/CARS/datasets/optc/cybergfm/full_graph.csv'
DEFAULT_DST = '/home/mmanjee/CARS/processed/optc_flow/'
DELTA = 10000


def split(src: str, dst: str) -> dict:
    os.makedirs(dst, exist_ok=True)
    if not dst.endswith(os.sep):
        dst = dst + os.sep

    # Pass 1: find t_min (first edge ts)
    t_min = None
    with open(src, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            t_min = float(line.split(',', 1)[0])
            break
    if t_min is None:
        raise SystemExit(f'Empty CSV: {src}')

    nmap = {}  # original id -> contiguous id
    nid = [0]

    def get_or_add(n: int) -> int:
        if n not in nmap:
            nmap[n] = nid[0]
            nid[0] += 1
        return nmap[n]

    cur_time = 0
    f_out = open(dst + '0.txt', 'w')
    n_edges = 0
    n_mal = 0
    first_mal = None
    t_max = 0

    with open(src, 'r') as f_in:
        for line in tqdm(f_in, desc='Slicing OpTC FLOW'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # ts,src,dst,port,user,image,label
            ts_raw = float(parts[0])
            src_id = int(parts[1])
            dst_id = int(parts[2])
            label = int(parts[6])
            ts = int(ts_raw - t_min)
            if ts < 0:
                ts = 0

            while ts >= cur_time + DELTA:
                cur_time += DELTA
                f_out.close()
                f_out = open(dst + str(cur_time) + '.txt', 'w')

            sid = get_or_add(src_id)
            did = get_or_add(dst_id)
            f_out.write(f'{ts},{sid},{did},{label}\n')

            n_edges += 1
            if label == 1:
                n_mal += 1
                if first_mal is None:
                    first_mal = ts
            if ts > t_max:
                t_max = ts

    f_out.close()

    nmap_rev = [None] * (max(nmap.values()) + 1)
    for k, v in nmap.items():
        nmap_rev[v] = k

    with open(dst + 'nmap.pkl', 'wb') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

    if first_mal is None:
        raise SystemExit('No malicious edges found in CSV')

    with open(dst + 'date_of_evil.txt', 'w') as f:
        f.write(str(first_mal) + '\n')

    meta = {
        'src': src,
        'n_edges': n_edges,
        'n_mal': n_mal,
        'n_nodes': len(nmap_rev),
        't_min_unix': t_min,
        't_max_rebased': t_max,
        'date_of_evil': first_mal,
        'delta': DELTA,
        'span': t_max,
    }
    with open(dst + 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(
        f'Wrote {n_edges} edges ({n_mal} mal), |V|={len(nmap_rev)}, '
        f'DATE_OF_EVIL={first_mal}, t_max={t_max} -> {dst}'
    )
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--dst', default=DEFAULT_DST)
    args = ap.parse_args()
    split(args.src, args.dst)


if __name__ == '__main__':
    main()
