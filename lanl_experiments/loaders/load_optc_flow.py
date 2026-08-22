"""Euler loader for CyberGFM OpTC FLOW slices (LANL-shaped API)."""

from copy import deepcopy
import json
import os
import pickle
from joblib import Parallel, delayed

import torch
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w

FILE_DELTA = 10000
OPTC_FLOW_FOLDER = '/home/mmanjee/CARS/processed/optc_flow/'


def _load_meta():
    meta_path = OPTC_FLOW_FOLDER + 'meta.json'
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)


def _load_date_of_evil() -> int:
    path = OPTC_FLOW_FOLDER + 'date_of_evil.txt'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return int(float(f.read().strip()))
    meta = _load_meta()
    if meta and 'date_of_evil' in meta:
        return int(meta['date_of_evil'])
    raise FileNotFoundError(
        f'Need {path} or meta.json — run loaders/split_optc_flow.py first'
    )


DATE_OF_EVIL = None
TIMES = {'all': None}

_meta = None
try:
    if os.path.isdir(OPTC_FLOW_FOLDER):
        DATE_OF_EVIL = _load_date_of_evil()
        _meta = _load_meta()
        if _meta and 't_max_rebased' in _meta:
            TIMES['all'] = int(_meta['t_max_rebased'])
        else:
            TIMES['all'] = DATE_OF_EVIL
except FileNotFoundError:
    pass

torch.set_num_threads(1)


def empty_optc_flow():
    return make_data_obj([], None, None)


def load_optc_flow_dist(
    workers, start=0, end=None, delta=8640, is_test=False, ew_fn=std_edge_w,
):
    if end is None:
        end = TIMES['all']
    if start is None or end is None:
        return empty_optc_flow()

    num_slices = (end - start) // delta
    remainder = (end - start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = min(num_slices, workers)

    if workers <= 1:
        return load_partial_optc_flow(start, end, delta, is_test, ew_fn)

    per_worker = [num_slices // workers] * workers
    rem = num_slices % workers
    if rem:
        for i in range(workers, workers - rem, -1):
            per_worker[i - 1] += 1

    kwargs = []
    prev = start
    for i in range(workers):
        end_t = prev + delta * per_worker[i]
        kwargs.append({
            'start': prev,
            'end': min(end_t - 1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn,
        })
        prev = end_t

    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_optc_flow_job)(i, kwargs[i]) for i in range(workers)
    )

    data_reduce = lambda x: sum([getattr(datas[i], x) for i in range(workers)], [])

    print('Joining Data objects')
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    node_map = datas[0].node_map

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    print('Done')
    return TData(eis, x, ys, masks, ews=ews, node_map=node_map, cnt=cnt)


def load_partial_optc_flow_job(pid, args):
    return load_partial_optc_flow(**args)


def make_data_obj(eis, ys, ew_fn, ews=None, snapshot_starts=None, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(OPTC_FLOW_FOLDER + 'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt + 1)

    eis_t = []
    masks = []
    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    data = TData(eis_t, x, ys, masks, ews=ews, cnt=cnt, node_map=nm)
    if snapshot_starts is not None:
        data.snapshot_starts = list(snapshot_starts)
    return data


def load_partial_optc_flow(start, end, delta, is_test=False, ew_fn=std_edge_w):
    cur_slice = int(start - (start % FILE_DELTA))
    edges = []
    ews = []
    edges_t = {}
    ys = []
    snapshot_starts = []

    node_map = pickle.load(open(OPTC_FLOW_FOLDER + 'nmap.pkl', 'rb'))

    fmt_line = lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3][:-1]))

    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1] + 1)
        else:
            edges_t[et] = (is_anom, 1)

    def flush_bucket():
        ei = list(zip(*edges_t.keys()))
        edges.append(ei)
        y, ew = list(zip(*edges_t.values()))
        ews.append(torch.tensor(ew))
        if is_test:
            ys.append(torch.tensor(y))
        snapshot_starts.append(next_split - delta)
        edges_t.clear()

    def open_slice(slice_t):
        """Open first non-empty shard at or after slice_t. CyberGFM has quiet gaps."""
        while True:
            path = OPTC_FLOW_FOLDER + str(slice_t) + '.txt'
            if not os.path.exists(path):
                return None, '', slice_t
            f = open(path, 'r')
            first = f.readline()
            if first:
                return f, first, slice_t
            f.close()
            slice_t += FILE_DELTA

    scan_prog = tqdm(desc='Finding start', total=max(start - cur_slice - 1, 0))
    prog = tqdm(desc='Seconds read', total=max(end - start - 1, 0))

    keep_reading = True
    next_split = start + delta

    in_f, line, cur_slice = open_slice(cur_slice)
    if in_f is None:
        scan_prog.close()
        prog.close()
        return make_data_obj(
            [], [] if is_test else None, ew_fn, ews=[],
            snapshot_starts=[], node_map=node_map,
        )

    old_ts = fmt_line(line.split(','))[0]
    while keep_reading:
        while line:
            l = line.split(',')
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(max(ts - old_ts, 0))
                old_ts = ts
                continue

            ts, src, dst, label = fmt_line(l)
            et = (src, dst)

            prog.update(max(ts - old_ts, 0))
            old_ts = ts

            if ts >= next_split:
                if edges_t:
                    flush_bucket()

                while next_split <= ts:
                    next_split += delta

                if ts >= end:
                    keep_reading = False
                    break

            if et[0] == et[1]:
                line = in_f.readline()
                continue

            if ts >= end:
                keep_reading = False
                break

            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA
        in_f, line, cur_slice = open_slice(cur_slice)
        if in_f is None:
            keep_reading = False
            break

    if edges_t:
        flush_bucket()

    ys = ys if is_test else None
    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, ys, ew_fn, ews=ews, snapshot_starts=snapshot_starts, node_map=node_map,
    )


if __name__ == '__main__':
    print('DATE_OF_EVIL', DATE_OF_EVIL, 'TIMES', TIMES, 'FOLDER', OPTC_FLOW_FOLDER)
    if DATE_OF_EVIL is not None:
        data = load_optc_flow_dist(2, start=0, end=min(20000, DATE_OF_EVIL), delta=10000)
        print(data)
