"""GPU layout for Euler RPC training.

Default (workers_per_gpu=1): worker i -> cuda:i, master -> cuda:num_workers.
With workers_per_gpu=2:      workers 0,1 -> cuda:0; 2,3 -> cuda:1; ... master -> cuda:ceil(n/wpg).
"""

import math
import torch

_NUM_WORKERS = 1
_WORKERS_PER_GPU = 1
_FORCE_CPU = False


def configure(num_workers: int, workers_per_gpu: int = 1, force_cpu: bool = False):
    global _NUM_WORKERS, _WORKERS_PER_GPU, _FORCE_CPU
    _NUM_WORKERS = max(1, int(num_workers))
    _WORKERS_PER_GPU = max(1, int(workers_per_gpu))
    _FORCE_CPU = force_cpu


def use_cuda() -> bool:
    return torch.cuda.is_available() and not _FORCE_CPU


def num_workers() -> int:
    return _NUM_WORKERS


def workers_per_gpu() -> int:
    return _WORKERS_PER_GPU


def num_worker_gpus() -> int:
    """Number of GPUs used by workers (ceil(num_workers / workers_per_gpu))."""
    return math.ceil(_NUM_WORKERS / _WORKERS_PER_GPU)


def worker_device(rank: int) -> torch.device:
    if use_cuda():
        return torch.device(f'cuda:{rank // _WORKERS_PER_GPU}')
    return torch.device('cpu')


def master_device() -> torch.device:
    if use_cuda():
        return torch.device(f'cuda:{num_worker_gpus()}')
    return torch.device('cpu')


def assert_gpu_layout(workers: int):
    if not use_cuda():
        return
    n_worker_gpus = num_worker_gpus()
    need = n_worker_gpus + 1
    have = torch.cuda.device_count()
    if have < need:
        raise RuntimeError(
            f'Need {need} GPUs ({workers} workers on {n_worker_gpus} GPUs + 1 master) '
            f'but found {have}. Reduce -w or increase --workers-per-gpu.'
        )


def ddp_device_ids(rank: int) -> dict:
    if not use_cuda():
        return {}
    gpu = rank // _WORKERS_PER_GPU
    return {'device_ids': [gpu], 'output_device': gpu}


def move_tdata_to(data, device: torch.device):
    if data is None or not hasattr(data, 'eis'):
        return data
    data.eis = [e.to(device) for e in data.eis]
    if data.dynamic_feats:
        data.xs = [x.to(device) for x in data.xs]
    else:
        data.xs = data.xs.to(device)
    ews = getattr(data, 'ews', None)
    if ews is not None and not isinstance(ews, None.__class__):
        data.ews = [w.to(device) for w in ews]
    ys = getattr(data, 'ys', None)
    if ys is not None and not isinstance(ys, None.__class__):
        data.ys = [y.to(device) for y in ys]
    data.masks = [
        m.to(device) if isinstance(m, torch.Tensor) else m for m in data.masks
    ]
    cnt = getattr(data, 'cnt', None)
    if cnt is not None and not isinstance(cnt, None.__class__):
        data.cnt = [c.to(device) for c in cnt]
    return data


def move_state_to(state, device: torch.device):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, (tuple, list)):
        return type(state)(move_state_to(x, device) for x in state)
    return state


def log_device_map():
    if not use_cuda():
        print('Devices: CPU (no CUDA or --cpu)')
        return
    w = _NUM_WORKERS
    wpg = _WORKERS_PER_GPU
    ng = num_worker_gpus()
    master_gpu = ng
    if wpg == 1:
        print(f'Devices: worker0..worker{w-1} -> cuda:0..cuda:{ng-1}, master -> cuda:{master_gpu}')
    else:
        print(f'Devices: {w} workers on {ng} GPUs ({wpg}/GPU), master -> cuda:{master_gpu}')
