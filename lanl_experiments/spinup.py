import math
import os
import pickle
import random
import time
import warnings

# Suppress known third-party deprecation/future warnings that are not actionable
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', message='leaked sem', category=UserWarning)
warnings.filterwarnings('ignore', message='leaked folder', category=UserWarning)

import numpy as np
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score
import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam, Adadelta

from loaders.tdata import TData
from loaders.load_lanl import load_lanl_dist
from models.euler_detector import DetectorEncoder, DetectorRecurrent 
from models.euler_predictor import PredictorEncoder, PredictorRecurrent
from models.utils import _remote_method_async, _remote_method
from devices import (
    configure, use_cuda, master_device, worker_device, num_worker_gpus,
    assert_gpu_layout, log_device_map, move_state_to,
)
from utils import get_score, get_optimal_cutoff

DDP_PORT = '22032'
RPC_PORT = '22204'


def set_seed(seed, rank=0):
    """Set RNG seeds for reproducibility; offset by rank for worker processes."""
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_TR = {
    'lr': 0.01,
    'epochs': 3,
    'min': 1,
    'patience': 5,
    'nratio': 1,
    'val_nratio': 1,
}

# Defaults
WORKER_ARGS = [32,32]
RNN_ARGS = [32,32,16,1]

WORKERS=3 # number of workers per GPU
W_THREADS=1 # number of threads per worker
M_THREADS=2 # number of threads per master for parallel operations not related to the RNN

TMP_FILE = 'tmp.dat' # temporary file to store results
SCORE_FILE = 'scores.txt' # file to store scores

# Callable that returns TData object
# method signature must match
# workers: int, start=int, end=int, delta=int, is_test=bool 
LOAD_FN = None

torch.set_num_threads(1)




'''
Constructs params for data loaders
'''
def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = math.ceil((end-start) / delta)

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    remainder = slices_needed % num_workers 
    if remainder:
        # Put remaining tasks on last workers since it's likely the 
        # final timeslice is stopped hallambda_paramay (ie it's less than a delta
        # so giving it extra timesteps is more likely okay)
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1 

    # Only uncomment when running late at night
    #load_threads = W_THREADS*2 if isTe else W_THREADS
    load_threads = W_THREADS

    # Make sure workers are collectively using at least 8 threads
    # since loading the data takes forever otherwise
    min_threads = min(8, load_threads*num_workers)
    t_per_worker = 1

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start
    
    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            kwargs.append({
                'start': prev,
                'end': end_t,
                'delta': delta, 
                'is_test': isTe,
                'jobs': t_per_worker
            })
            prev = end_t

    return kwargs
    

def init_workers(num_workers, start, end, delta, isTe, worker_constructor, worker_args):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                worker_constructor,
                args=(LOAD_FN, kwargs[i], *worker_args),
                kwargs={'head': i==0}
            )
        )

    return rrefs


#workers that are not used for training but are used for testing
def init_empty_workers(num_workers, worker_constructor, worker_args):
    empty = {'jobs': 0, 'start': None, 'end': None}
    
    rrefs = [
        rpc.remote(
            'worker'+str(i),
            worker_constructor,
            args=(LOAD_FN, empty, *worker_args),
            kwargs={'head': i==0}
        )
        for i in range(num_workers)
    ]

    return rrefs

def _setup_rpc_device_maps(options, rank, world_size, num_workers):
    """Map worker cuda:gpu <-> master cuda:num_worker_gpus for TensorPipe RPC."""
    if not use_cuda():
        return
    master_idx = num_worker_gpus()
    if rank == world_size - 1:
        for i in range(num_workers):
            w_gpu = worker_device(i).index
            options.set_device_map(f'worker{i}', {w_gpu: master_idx, master_idx: w_gpu})
    else:
        w_gpu = worker_device(rank).index
        options.set_device_map('master', {master_idx: w_gpu, w_gpu: master_idx})

def _build_model(rnn_constructor, rnn_args, rrefs, impl):
    mdev = master_device()
    if use_cuda():
        torch.cuda.set_device(mdev)
    rnn = rnn_constructor(*rnn_args).to(mdev)
    if impl == 'DETECT':
        return DetectorRecurrent(rnn, rrefs, device=mdev)
    return PredictorRecurrent(rnn, rrefs, device=mdev)

def init_procs(rank, world_size, rnn_constructor, rnn_args, worker_constructor, worker_args, 
                times, just_test, lambda_param, impl, load_fn, tr_args, seed,
                num_workers, force_cpu, workers_per_gpu=1,
                edge_dev=False, edge_dev_alpha=None, edge_dev_diagnostic=False,
                save_test_scores=None):
    configure(num_workers, workers_per_gpu=workers_per_gpu, force_cpu=force_cpu)
    if seed is not None:
        set_seed(seed, rank)

    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = DDP_PORT

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:' + RPC_PORT
    _setup_rpc_device_maps(rpc_backend_options, rank, world_size, num_workers)

    # This is a lot easier than actually changing it in all the methods
    # at this point
    global LOAD_FN
    LOAD_FN = load_fn

    # setting up master for RNN operations
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS) #parallelism on master for standard operations in RNN 
        if use_cuda():
            torch.cuda.set_device(master_device())
        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )


        # Evaluating a pre-trained model, so no need to train 
        if just_test:
            rrefs = init_empty_workers(
                world_size-1, 
                worker_constructor, 
                worker_args
            )

            model = _build_model(rnn_constructor, rnn_args, rrefs, impl)

            states = pickle.load(open('model_save.pkl', 'rb'))
            model.load_states(*states['states'])
            h0 = move_state_to(states['h0'], master_device())
            tpe = 0
            tr_time = 0


        # Building and training a fresh model
        else:
            rrefs = init_workers(
                world_size-1, 
                times['tr_start'], times['tr_end'], times['delta'], False,
                worker_constructor, worker_args
            )

            tmp = time.time()
            model, h0, tpe = train(rrefs, tr_args, rnn_constructor, rnn_args, impl)
            tr_time = time.time() - tmp
        
        h0, zs = get_cutoff(
            model, h0, times, tr_args, lambda_param,
            edge_dev=edge_dev, edge_dev_alpha=edge_dev_alpha,
        )
        stats = []

        for te_start,te_end in times['te_times']:
            test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']
            }
            st = test(
                model, h0, test_times, rrefs, edge_dev_diagnostic,
                save_scores_path=save_test_scores,
            )
            for s in st:
                s['TPE'] = tpe
                s['tr_time'] = tr_time

            stats += st

    # Slaves
    else:
        torch.set_num_threads(W_THREADS)
        # NCCL requires each rank to have a unique GPU; fall back to gloo when
        # multiple workers share a GPU (workers_per_gpu > 1).
        from devices import workers_per_gpu as _wpg
        backend = 'nccl' if (use_cuda() and _wpg() == 1) else 'gloo'
        if use_cuda():
            torch.cuda.set_device(worker_device(rank))
        
        # Slaves are their own process group. This allows
        # DDP to work between these processes
        dist.init_process_group(
            backend, rank=rank, 
            world_size=world_size-1
        )

        #initialize RPC for the worker so that the master can call methods on the worker
        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    # Block until all procs complete
    print(f"[rank {rank}] pre-shutdown rpc", flush=True)
    rpc.shutdown()

    # Workers need to explicitly destroy their NCCL process group to avoid
    # "ProcessGroupNCCL not destroyed" warnings on exit
    if rank < world_size - 1 and dist.is_initialized():
        dist.destroy_process_group()

    # Write output to a tmp file to get it back to the parent process
    if rank == world_size-1:
        pickle.dump(stats, open(TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)


def train(rrefs, kwargs, rnn_constructor, rnn_args, impl):
    model = _build_model(rnn_constructor, rnn_args, rrefs, impl)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        with dist_autograd.context() as context_id:
            print("forward")
            st = time.time()
            zs = model.forward(TData.TRAIN)
            loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, loss)
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            l = torch.stack(loss).sum()
            print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            
            ap, auc = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            # Either incriment or update early stopping criteria
            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1 

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break 

    model.load_states(*best[0])

    # Get the best possible h0 to eval with
    zs, h0 = model(TData.TEST, include_h=True)

    states = {'states': best[0], 'h0': h0}
    f = open('model_save.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    
    return model, h0, tpe


def _set_edge_dev_alpha(model, alpha):
    for rref in model.gcns:
        _remote_method(DetectorEncoder.set_edge_dev_alpha, rref, alpha)


'''
Given a trained model, generate the optimal cutoff point - threshold for using to predict anomalies
using the validation data
'''
def get_cutoff(model, h0, times, kwargs, lambda_param,
               edge_dev=False, edge_dev_alpha=None):
    # Weirdly, calling the parent class' method doesn't work
    # whatever. This is a hacky solution, but it works
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent) \
        else PredictorEncoder

    # First load validation data onto one of the GCNs
    _remote_method(
        Encoder.load_new_data,
        model.gcns[0],
        LOAD_FN,
        {
            'start': times['val_start'],
            'end': times['val_end'],
            'delta': times['delta'],
            'jobs': 1,
            'is_test': False
        }
    )

    # Then generate GCN embeds
    model.eval()
    zs = _remote_method(
        Encoder.forward,
        model.gcns[0], 
        TData.ALL,
        True
    )

    # Finally, generate actual embeds
    with torch.no_grad():
        if hasattr(model.rnn, 'reset_stream_cache'):
            h0 = model.rnn.reset_stream_cache(h0)
        h0 = move_state_to(h0, model.device)
        zs = zs.to(model.device)
        zs, h0 = model.rnn(zs, h0, include_h=True)

    # For predictor, prepend a dummy z_{-1} so the head worker's is_head=1
    # offset aligns z[i] with E[i+1] (prediction-style), matching test scoring.
    # Detector needs no adjustment.
    zs_for_score = zs
    if isinstance(model, PredictorRecurrent):
        dummy = torch.zeros(1, *zs.shape[1:], device=zs.device)
        zs_for_score = torch.cat([dummy, zs], dim=0)

    # Apply edge-dev penalty before val cutoff (requires --edge-dev-alpha)
    if edge_dev and isinstance(model, DetectorRecurrent):
        assert edge_dev_alpha is not None, \
            '--edge-dev requires --edge-dev-alpha (val alpha tuning disabled)'
        print('Edge dev alpha = %0.4f' % edge_dev_alpha)
        _set_edge_dev_alpha(model, edge_dev_alpha)
        # # Pass 1 (redundant when alpha is fixed): separate base scores + penalties for pick_alpha_val
        # from loaders.edge_dev import pick_alpha_val
        # p, n, p_pen, n_pen = _remote_method(
        #     DetectorEncoder.score_edges_with_penalties,
        #     model.gcns[0],
        #     zs_for_score, TData.ALL,
        #     kwargs['val_nratio'],
        # )
        # if edge_dev_alpha is None:
        #     edge_dev_alpha = pick_alpha_val(p, n, p_pen, n_pen)
        # print('Edge dev alpha (val-tuned) = %0.4f' % edge_dev_alpha)
        # _set_edge_dev_alpha(model, edge_dev_alpha)

    p,n = _remote_method(
        Encoder.score_edges, 
        model.gcns[0],
        zs_for_score, TData.ALL,
        kwargs['val_nratio']
    )


    # Finally, figure out the optimal cutoff score
    model.cutoff = get_optimal_cutoff(p,n,fw=lambda_param)

    print()
    # Clone last timestep so the full val embedding tensor can be freed before test.
    return h0, zs[-1].detach().clone()

def test(model, h0, times, rrefs, edge_dev_diagnostic=False, save_scores_path=None):
    # For whatever reason, it doesn't know what to do if you call
    # the parent object's methods. Kind of defeats the purpose of 
    # using OOP at all IMO, but whatever
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent) \
        else PredictorEncoder

    # Load train data into workers
    ld_args = get_work_units(
        len(rrefs), 
        times['te_start'], 
        times['te_end'],
        times['delta'], 
        True
    )

    print("Loading test data")
    
    # Make sure there's enough data for each worker to do something
    dont_use = 0
    for ld in ld_args:    
        if ld['start'] == ld['end']:
            dont_use += 1
        else:
            break

    # If we have more workers than work. Tell master not to use them
    futs = [
        _remote_method_async(
            Encoder.load_new_data,
            rrefs[i], 
            LOAD_FN, 
            ld_args[i+dont_use]
        ) for i in range(len(rrefs)-dont_use)
    ]
    model.num_workers = len(futs)

    # Wait until all workers have finished
    [f.wait() for f in futs]
    stats = []

    print("Embedding Test Data...")
    with torch.no_grad():
        model.eval()
        if hasattr(model.rnn, 'reset_stream_cache'):
            h0 = model.rnn.reset_stream_cache(h0)
        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    scores, labels, weights = model.score_all(zs)
    if save_scores_path:
        import numpy as np
        base = torch.cat(scores, dim=0).detach().cpu().numpy()
        y = torch.cat(labels, dim=0).clamp(max=1).detach().cpu().numpy().astype(bool)
        np.savez_compressed(save_scores_path, base_scores=base, y=y)
        print(f'Saved test scores -> {save_scores_path}')
    # if edge_dev_diagnostic and isinstance(model, DetectorRecurrent):
    #     from loaders.edge_dev import print_edge_dev_diagnostic
    #     penalties = model.collect_penalties()
    #     print_edge_dev_diagnostic(
    #         torch.cat(scores, dim=0).detach().cpu(),
    #         torch.cat(labels, dim=0).detach().cpu(),
    #         torch.cat(penalties, dim=0).detach().cpu(),
    #     )
    stats.append(
            score_stats(
            model.__class__.__name__, 
            scores, labels, weights, model.cutoff, ctime
        )       
    )
    
    # Then reset model to having all workers for future tests
    model.num_workers = len(rrefs)
    return stats
    

def score_stats(title, scores, labels, weights, cutoff, ctime):
    # Cat scores from timesteps together bc separation 
    # is no longer necessary 
    scores = torch.cat(scores, dim=0).detach().cpu()
    labels = torch.cat(labels, dim=0).clamp(max=1).cpu()
    weights = torch.cat(weights, dim=0).cpu()

    # Classify using cutoff from earlier (low score => predicted anomaly)
    classified = torch.zeros(labels.size())
    classified[scores <= cutoff] = 1

    pos = labels == 1
    neg = labels == 0
    tp = classified[pos].sum()
    fn = (1 - classified[pos]).sum()
    fp = classified[neg].sum()
    tn = (1 - classified[neg]).sum()

    n_pos = pos.sum().float()
    n_neg = neg.sum().float()
    tpr = tp.true_divide(n_pos) if n_pos else torch.tensor(0.)
    fnr = fn.true_divide(n_pos) if n_pos else torch.tensor(0.)
    fpr = fp.true_divide(n_neg) if n_neg else torch.tensor(0.)
    tnr = tn.true_divide(n_neg) if n_neg else torch.tensor(0.)
    denom_p = tp + fp
    precision = tp.true_divide(denom_p) if denom_p else torch.tensor(0.)

    # Because a low score correlates to a 1 lable, sub from 1 to get
    # accurate AUC/AP scores
    scores = 1-scores

    # Get metrics
    auc = auc_score(labels, scores)
    ap = ap_score(labels, scores)
    f1 = f1_score(labels, classified)

    print(title)
    print("Learned Cutoff %0.4f" % cutoff)
    print("TP: %d  FP: %d  TN: %d  FN: %d" % (tp, fp, tn, fn))
    print("TPR: %0.4f  FPR: %0.4f  TNR: %0.4f  FNR: %0.4f" % (
        tpr, fpr, tnr, fnr))
    print("Precision: %0.4f  F1: %0.8f" % (precision, f1))
    print("AUC: %0.4f  AP: %0.4f\n" % (auc, ap))

    return {
        'Model': title,
        'TP': tp.item(),
        'FP': fp.item(),
        'TN': tn.item(),
        'FN': fn.item(),
        'TPR': tpr.item(),
        'FPR': fpr.item(),
        'TNR': tnr.item(),
        'FNR': fnr.item(),
        'Precision': precision.item(),
        'F1': f1,
        'AUC': auc,
        'AP': ap,
        'FwdTime': ctime
    }

def run_all(workers, rnn_constructor, rnn_args, worker_constructor, 
            worker_args, delta, just_test, lambda_param, impl, load_fn, 
            tr_start, tr_end, val_times, te_times, tr_args, seed=None,
            force_cpu=False, workers_per_gpu=1,
            edge_dev=False, edge_dev_alpha=None, edge_dev_diagnostic=False,
            save_test_scores=None):
    '''
    Starts up proceses, trains validates and tests the model given 
    the inputs 

        workers : int 
            how many worker processes to use
        rnn_constructor : callable -> RNN 
            constructor for RNN model
        rnn_args : list 
            arguments for detector rnn model
        worker_constructor : callable -> Euler_Encoder_Unit 
            constructs an Euler_Encoder wrapped RRef to worker
        worker_args : list 
            non-file loading related worker arguments
        delta : int 
            size of time window to partition graphs
        just_test : boolean 
            Loads pre-trained model from disk and evaluates it
        lambda_param : float
            How much weight to give low FPR when deciding a cutoff;
            defaults to 0.6
        impl : str in ['DETECT', 'PREDICT']
            Class implimenting Euler_Interface
        load_fn : callable -> TGraph
            Function to load a set of snapshots into workers
        tr_start : int
            Timestep the training set starts at
        tr_end : int 
            Timestep the training set ends at
        te_end : int 
            Timestep the test set ends at. By default, loads the full LANL dataset
        tr_args : dict
            Hyperparameters for training. E.g. epochs, patience, etc. 
        '''
    
    # Need at least 2 deltas; default to 5% of tr data if that's enough
    if val_times is None:
        val = max((tr_end - tr_start) // 20, delta*2)
        val_start = tr_end-val
        val_end = tr_end
        tr_end = val_start
    else:
        val_start = val_times[0]
        val_end = val_times[1]

    # Make sure each worker has some data on it
    max_workers = int((tr_end-tr_start) // delta)
    workers = max(min(max_workers, workers), 1)

    configure(workers, workers_per_gpu=workers_per_gpu, force_cpu=force_cpu)
    if use_cuda():
        max_workers = (torch.cuda.device_count() - 1) * workers_per_gpu
        workers = min(workers, max_workers)
        configure(workers, workers_per_gpu=workers_per_gpu, force_cpu=False)
    assert_gpu_layout(workers)
    log_device_map()

    if edge_dev or edge_dev_diagnostic:
        from loaders.edge_dev import build_train_stats
        build_train_stats(tr_start, tr_end, delta)

    times = {
        'tr_start': tr_start,
        'tr_end': tr_end,
        'val_start': val_start,
        'val_end': val_end,
        'te_times': te_times,
        'delta': delta
    }

    print(times)

    # Start workers
    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(
            world_size, 
            rnn_constructor, 
            rnn_args, 
            worker_constructor, 
            worker_args,
            times,
            just_test,
            lambda_param,
            impl,
            load_fn,
            tr_args,
            seed,
            workers,
            force_cpu,
            workers_per_gpu,
            edge_dev,
            edge_dev_alpha,
            edge_dev_diagnostic,
            save_test_scores,
        ),
        nprocs=world_size,
        join=True
    )

    # Retrieve stats, and cleanup temp file
    stats = pickle.load(open(TMP_FILE, 'rb'))
    #os.remove(TMP_FILE)

    print(stats)
    return stats

if __name__ == '__main__':
    print("Please run this file using run.py")