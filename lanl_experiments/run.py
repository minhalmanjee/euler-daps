from argparse import ArgumentParser
import os 

import pandas as pd

import loaders.load_lanl as lanl
import loaders.load_optc_flow as optc
import loaders.edge_dev as edge_dev
from models.recurrent import GRU, LSTM, EmptyModel, CausalTransformer, GRUTransformer, TransformerGRU
from models.embedders import \
    detector_gcn_rref, detector_gat_rref, detector_sage_rref, \
    predictor_gcn_rref, predictor_gat_rref, predictor_sage_rref 

from spinup import run_all

DEFAULT_TR = {
    'anom_lr': 0.05,
    'epochs': 1500,
    'min': 1,
    'nratio': 1,
    'val_nratio': 1
}

OUTPATH = '' # Output folder for results.txt (ending in delimeter)

def get_args():
    global DEFAULT_TR

    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=0.5
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=3,
        help='GCN worker processes (default 3)'
    )

    ap.add_argument(
        '-g', '--workers-per-gpu',
        type=int, default=1, dest='workers_per_gpu',
        help='Workers sharing each GPU (default 1). E.g. -w 6 -g 2 => 6 workers on 3 GPUs, master on cuda:3'
    )

    ap.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU even if CUDA is available'
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=1
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE', 'CTRANS', 'HYBRID', 'TGRU'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int, 
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )

    ap.add_argument(
        '--impl', '-i',
        type=str.upper,
        choices=['DETECT', 'PREDICT', 'D', 'P', 'PRED'],
        default="DETECT"
    )

    # For future new data sets
    ap.add_argument(
        '--dataset',
        default='LANL', 
        type=str.upper
    )

    ap.add_argument(
        '--lr',
        default=0.005,
        type=float
    )
    ap.add_argument(
        '--patience',
        default=10, 
        type=int
    )

    ap.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed; with -t N, uses seed, seed+1, ..., seed+N-1'
    )

    ap.add_argument(
        '--edge-dev-diagnostic',
        action='store_true',
        help='[disabled] After test, print AP/precision@99%%TPR for alpha=0.1,0.5,1.0 vs baseline'
    )

    ap.add_argument(
        '--edge-dev',
        action='store_true',
        help='Apply degree-dev edge scoring at val cutoff + test (requires --edge-dev-alpha)'
    )

    ap.add_argument(
        '--edge-dev-alpha',
        type=float, default=None,
        help='Edge-dev penalty weight (required with --edge-dev)'
    )

    ap.add_argument(
        '--zscore-cap',
        type=float, default=1.0,
        help='Clamp |z| for degree-dev penalty (default 1.0); used with --edge-dev'
    )

    ap.add_argument(
        '--save-test-scores',
        default=None,
        help='Save test edge existence scores to npz (for degree_check --pr-curve)'
    )

    args = ap.parse_args()
    args.te_end = None
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'
    assert args.zscore_cap > 0, '--zscore-cap must be positive'

    # Propagate to edge_dev (and spawn workers via env) before dataset configure.
    edge_dev.set_zscore_cap(args.zscore_cap)

    readable = str(args)
    print(readable)
    print(f'edge_dev ZSCORE_CAP={edge_dev.get_zscore_cap()}')

    model_str = '%s -> %s (%s)' % (args.encoder , args.rnn, args.impl)
    print(model_str)
    
    # Parse dataset info 
    if args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False
        edge_dev.configure(lanl.LANL_FOLDER, lanl.load_partial_lanl)

    elif args.dataset.startswith('O'):  # OPTC / OPTC_FLOW
        if optc.DATE_OF_EVIL is None or optc.TIMES.get('all') is None:
            raise FileNotFoundError(
                'OpTC FLOW slices missing — run: python loaders/split_optc_flow.py'
            )
        args.loader = optc.load_optc_flow_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL
        args.val_times = None
        args.te_times = [(args.tr_end, optc.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False
        edge_dev.configure(optc.OPTC_FLOW_FOLDER, optc.load_partial_optc_flow)

    else:
        raise NotImplementedError('Supported datasets: LANL, OPTC / OPTC_FLOW')

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = detector_gcn_rref if args.impl[0] == 'D' \
            else predictor_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = detector_gat_rref if args.impl[0] == 'D' \
            else predictor_gat_rref
    else:
        args.encoder = detector_sage_rref if args.impl[0] == 'D' \
            else predictor_sage_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    elif args.rnn == 'CTRANS':
        args.rnn = CausalTransformer
    elif args.rnn == 'HYBRID':
        args.rnn = GRUTransformer
    elif args.rnn == 'TGRU':
        args.rnn = TransformerGRU
    else:
        args.rnn = EmptyModel

    return args, readable, model_str

if __name__ == '__main__':
    args, argstr, modelstr = get_args() 
    DEFAULT_TR['lr'] = args.lr
    DEFAULT_TR['patience'] = args.patience

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        # Need to tell workers to output in embed dim
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]

    stats = []
    for i in range(args.tests):
        seed_i = None if args.seed is None else args.seed + i
        save_path = args.save_test_scores
        # With -t N>1, write distinct NPZs per trial (seed suffix)
        if save_path and seed_i is not None and args.tests > 1:
            root, ext = os.path.splitext(save_path)
            if f'seed{seed_i}' not in os.path.basename(root):
                save_path = f'{root}_seed{seed_i}{ext or ".npz"}'
        stats.append(
            run_all(
                args.workers,
                args.rnn,
                rnn_args,
                args.encoder,
                worker_args,
                args.delta,
                args.load,
                args.fpweight,
                args.impl,
                args.loader,
                args.tr_start,
                args.tr_end,
                args.val_times,
                args.te_times,
                DEFAULT_TR,
                seed=seed_i,
                force_cpu=args.cpu,
                workers_per_gpu=args.workers_per_gpu,
                edge_dev=args.edge_dev,
                edge_dev_alpha=args.edge_dev_alpha,
                edge_dev_diagnostic=args.edge_dev_diagnostic,
                save_test_scores=save_path,
            )
        )

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    f = open(OUTPATH+'results.txt', 'a')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(modelstr + '\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    dfs = pd.concat(dfs, axis=0)

    for m in dfs['Model'].unique():
        df = dfs[dfs['Model'] == m]

        compressed = pd.DataFrame(
            [df.mean(numeric_only=True), df.sem(numeric_only=True)],
            index=['mean', 'stderr']
        ).to_csv().replace(',', '\t') # For easier copying into Excel

        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')

        f.write(m + '\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n')

    f.close()