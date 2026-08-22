import warnings
import torch
from torch import nn

warnings.filterwarnings('ignore', message='enable_nested_tensor', category=UserWarning)


class CompressedContextCache:
    '''
    Exact recent tokens + mean-pooled summary tokens for unbounded streams.

    Recent window (max_exact): full-resolution transformer states.
    Distant history (n_summary): stride mean-pool compression into baseline
    summary tokens prepended as attention prefix (KV-cache analogue).
    '''

    def __init__(self, max_exact=80, n_summary=10, pool_stride=20):
        self.max_exact = max_exact
        self.n_summary = n_summary
        self.pool_stride = pool_stride

    @staticmethod
    def compress_stride(tokens, pool_stride):
        '''Mean-pool along time: [T, N, D] -> [T', N, D].'''
        T = tokens.size(0)
        if T == 0:
            return tokens
        n = (T // pool_stride) * pool_stride
        if n == 0:
            return tokens.mean(dim=0, keepdim=True)
        grouped = tokens[:n].view(
            -1, pool_stride, tokens.size(1), tokens.size(2)
        )
        return grouped.mean(dim=1)

    @staticmethod
    def cap_summary(summary, n_summary):
        if summary is None or summary.size(0) <= n_summary:
            return summary
        T = summary.size(0)
        chunk = max(1, (T + n_summary - 1) // n_summary)
        parts = [
            summary[i : i + chunk].mean(dim=0, keepdim=True)
            for i in range(0, T, chunk)
        ]
        return torch.cat(parts[:n_summary], dim=0)

    def prefix(self, summary, exact):
        parts = [p for p in (summary, exact) if p is not None and p.size(0) > 0]
        return torch.cat(parts, dim=0) if parts else None

    def update(self, summary, exact, new_tokens, detach=True):
        '''Append new_tokens to exact window; overflow -> summary pool.'''
        combined = (
            torch.cat([exact, new_tokens], dim=0)
            if exact is not None else new_tokens
        )
        if combined.size(0) <= self.max_exact:
            out = combined.detach() if detach else combined
            s = summary.detach() if detach and summary is not None else summary
            return s, out

        overflow = combined[: -self.max_exact]
        exact_new = combined[-self.max_exact :]
        pooled = self.compress_stride(overflow, self.pool_stride)
        summary = torch.cat([summary, pooled], dim=0) if summary is not None else pooled
        summary = self.cap_summary(summary, self.n_summary)

        if detach:
            summary = summary.detach()
            exact_new = exact_new.detach()
        return summary, exact_new


class GRU(nn.Module):
    '''
    GRU Class; very simple and lightweight
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        '''
        Constructor for GRU model 

        x_dim : int
            The input dimension
        h_dim : int 
            The hidden dimension
        z_dim : int 
            The output dimension
        hidden_units : int 
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        '''
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)
        
        self.z_dim = z_dim 

    def forward(self, xs, h0, include_h=False):
        '''
        Forward method for GRU 

        xs : torch.Tensor 
            The T x N x X_dim input of node embeddings 
        h0 : torch.Tensor 
            A hidden state for the GRU
        include_h : bool 
            If true, return hidden state as well as output
        '''
        xs = self.drop(xs)
        
        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)
        
        if not include_h:
            return self.lin(xs)
        
        return self.lin(xs), h


class LSTM(GRU):
    '''
    Slightly more complex RNN, but about equal at most tasks, though 
    some papers show that LSTM is better in some instances than GRU

    Best practice to use LSTM first, and if GRU performs as well to switch to that
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        '''
        Constructor for LSTM model 

        x_dim : int
            The input dimension
        h_dim : int 
            The hidden dimension
        z_dim : int 
            The output dimension
        hidden_units : int 
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        '''
        super(LSTM, self).__init__(x_dim, h_dim, z_dim, hidden_units=hidden_units)

        # Just swapping out one component with another
        self.rnn = nn.LSTM(
            x_dim, h_dim, num_layers=hidden_units
        )


class Lin(nn.Module):
    '''
    Doesn't take time into account at all, just projects input
    into the output dimension via MLP
    '''
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(Lin, self).__init__()

        self.layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.layers(xs)
        
        return self.layers(xs), None


class CausalTransformer(nn.Module):
    '''
    Causal Transformer Encoder; drop-in replacement for GRU/LSTM.

    Processes GCN snapshot embeddings with masked self-attention so each
    timestep can only attend to itself and earlier timesteps (causality).
    Maintains a context buffer (h0) across streaming worker chunks so that
    later chunks can attend back to earlier ones — solving the key failure
    mode of naive per-chunk processing.

    Hyperparameter choices vs. previous poor-performing transformer:
      nhead=4          h_dim=32 → 8-dim heads; enough capacity, avoids
                       head-dim collapse seen with nhead=8 on h_dim=32
      ffn = h_dim*2    compact FFN (64) prevents overfitting on LANL
      dropout=0.1      transformers are far more sensitive to dropout than
                       GRUs; 0.25 (GRU default) suppresses attention signal
      num_layers≥2     1-layer transformer cannot learn both local and
                       global temporal patterns; clamped to 2 minimum
      max_ctx=80       equals T_train (80 snapshots) so training runs as a
                       single chunk (full-seq equivalent, no BPTT truncation);
                       test sub-chunks at 80-step intervals keeping attn
                       matrix at 160×160 (~6.4 GB) vs 2700×2700 (1.7 TB)
      sinusoidal PE    global position offset passed across worker chunks
                       preserves chronological order in the full sequence
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1, nhead=4,
                 dropout=0.1, max_ctx=80):
        super(CausalTransformer, self).__init__()

        num_layers = max(hidden_units, 2)

        # ensure h_dim is divisible by nhead
        while h_dim % nhead != 0:
            nhead = nhead // 2

        self.input_proj = nn.Linear(x_dim, h_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=nhead,
            dim_feedforward=h_dim * 2,
            dropout=dropout,
            norm_first=True,    # pre-LN: stable from step 0, no warmup needed
            batch_first=False   # expects [seq_len, batch, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(h_dim, z_dim)

        self.h_dim = h_dim
        self.max_ctx = max_ctx
        self.z_dim = z_dim
        # Signals euler_interface to collect all worker embeddings before
        # calling forward, giving the transformer the full sequence at once
        # for proper gradient flow and cross-sequence attention
        self.collect_before_rnn = True

    def forward(self, xs, h0, include_h=False):
        '''
        xs        : [T, N, x_dim]  full sequence of GCN embeddings
        h0        : unused; accepted for interface compatibility
        include_h : if True return (out, None), else return out

        Always uses a sliding-window causal attention of max_ctx steps so
        that training (T=80) and test (T=2700) run identical code paths.
        This eliminates the train/test mismatch that occurs when a different
        forward mode is used at inference time.

        Memory per transformer call: attn matrix ≤ (2*max_ctx)² × N × nhead,
        e.g. max_ctx=40 → 80×80 × 15611 × 4 × 4 bytes ≈ 1.5 GB.
        '''
        T = xs.size(0)
        ctx = None
        outs = []

        n_chunks = (T + self.max_ctx - 1) // self.max_ctx
        for i, start in enumerate(range(0, T, self.max_ctx)):
            if n_chunks > 4:   # only print for long (test-time) sequences
                print(f'\r  [CTRANS] {i+1}/{n_chunks} chunks', end='', flush=True)

            chunk = xs[start : start + self.max_ctx]           # [T_c, N, x_dim]
            T_c = chunk.size(0)
            cur = self.input_proj(chunk)                       # [T_c, N, h_dim]

            seq = torch.cat([ctx, cur], dim=0) if ctx is not None else cur
            T_seq = seq.size(0)

            causal_mask = torch.triu(
                torch.ones(T_seq, T_seq, device=xs.device), diagonal=1
            ).bool()

            out_full = self.transformer(seq, mask=causal_mask) # [T_seq, N, h_dim]
            outs.append(out_full[-T_c:])

            # Rolling context buffer; detach to truncate BPTT across chunks
            ctx = out_full[-self.max_ctx:].detach()

        if n_chunks > 4:
            print()

        out = self.lin(self.drop(torch.cat(outs, dim=0)))      # [T, N, z_dim]
        return (out, None) if include_h else out


class GRUTransformer(nn.Module):
    '''
    GRU → Transformer hybrid; drop-in replacement for GRU/LSTM.

    GRU streams over worker chunks and compresses unbounded history into a
    fixed hidden state (same as vanilla EULER). A causal Transformer then
    refines GRU outputs for local snapshot interactions within each chunk.

    Long-range memory  : GRU hidden state h0 (O(1), no vanishing chunk resets)
    Local structure    : causal self-attention over recent GRU outputs
    Test scalability   : GRU runs over full worker chunk; Transformer
                         sub-chunks only when T > max_ctx (memory bound)

    Does NOT set collect_before_rnn — uses EULER's native streaming path.
    h0 is a tuple (gru_h, trans_ctx) threaded across worker calls.
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1, nhead=4,
                 dropout=0.1, max_ctx=80):
        super(GRUTransformer, self).__init__()

        num_layers = max(hidden_units, 2)
        while h_dim % nhead != 0:
            nhead = nhead // 2

        self.gru = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.gru_drop = nn.Dropout(0.25)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=nhead,
            dim_feedforward=h_dim * 2,
            dropout=dropout,
            norm_first=True,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trans_drop = nn.Dropout(dropout)
        self.lin = nn.Linear(h_dim, z_dim)

        self.h_dim = h_dim
        self.max_ctx = max_ctx
        self.z_dim = z_dim

    @staticmethod
    def _unpack_h0(h0):
        if h0 is None:
            return None, None
        if isinstance(h0, (tuple, list)) and len(h0) == 2:
            return h0[0], h0[1]
        return h0, None

    def _transformer_pass(self, seq_in, trans_ctx, verbose=False):
        '''Causal transformer with optional prefix context; sub-chunks if long.'''
        T = seq_in.size(0)
        ctx = trans_ctx
        outs = []
        n_chunks = max(1, (T + self.max_ctx - 1) // self.max_ctx)

        for i, start in enumerate(range(0, T, self.max_ctx)):
            if verbose:
                print(f'\r  [HYBRID] transformer {i+1}/{n_chunks}', end='', flush=True)

            cur = seq_in[start : start + self.max_ctx]
            T_c = cur.size(0)
            seq = torch.cat([ctx, cur], dim=0) if ctx is not None else cur
            T_seq = seq.size(0)

            mask = torch.triu(
                torch.ones(T_seq, T_seq, device=seq_in.device), diagonal=1
            ).bool()
            out_full = self.transformer(seq, mask=mask)
            outs.append(out_full[-T_c:])
            ctx = out_full[-self.max_ctx:].detach()

        if verbose:
            print()
        return torch.cat(outs, dim=0), ctx

    def forward(self, xs, h0, include_h=False):
        '''
        xs  : [T, N, x_dim]
        h0  : None | (gru_h, trans_ctx)
              gru_h     — GRU hidden state (long-range memory)
              trans_ctx — last max_ctx transformer outputs (local prefix)
        '''
        gru_h, trans_ctx = self._unpack_h0(h0)
        xs = self.gru_drop(xs)

        if gru_h is None:
            gru_out, gru_h = self.gru(xs)
        else:
            gru_out, gru_h = self.gru(xs, gru_h)

        verbose = gru_out.size(0) > self.max_ctx * 2
        trans_out, new_trans_ctx = self._transformer_pass(gru_out, trans_ctx, verbose)
        out = self.lin(self.trans_drop(trans_out))

        if not include_h:
            return out
        return out, (gru_h, new_trans_ctx)


class TransformerGRU(nn.Module):
    '''
    Transformer → GRU hybrid with compressed summary context cache.

    Pipeline per worker chunk:
      1. Causal Transformer on GCN embeddings (local parallel attention)
         with prefix = [summary tokens | exact recent tokens]
      2. GRU compresses transformer outputs into O(1) streaming state

    Context cache (h0[1]):
      exact  — last max_ctx full transformer states (local window)
      summary — mean-pooled baseline tokens from older history (bounded)

    Uses EULER's native streaming path; h0 = (gru_h, (summary, exact)).
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1, nhead=4,
                 dropout=0.1, max_ctx=80, n_summary=10, pool_stride=20):
        super(TransformerGRU, self).__init__()

        num_layers = max(hidden_units, 2)
        while h_dim % nhead != 0:
            nhead = nhead // 2

        self.input_proj = nn.Linear(x_dim, h_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=nhead,
            dim_feedforward=h_dim * 2,
            dropout=dropout,
            norm_first=True,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trans_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(h_dim, h_dim, num_layers=hidden_units)
        self.gru_drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.h_dim = h_dim
        self.max_ctx = max_ctx
        self.z_dim = z_dim
        self.cache = CompressedContextCache(max_ctx, n_summary, pool_stride)

    @staticmethod
    def _unpack_h0(h0):
        if h0 is None:
            return None, None, None
        if isinstance(h0, (tuple, list)) and len(h0) == 2:
            gru_h, cache = h0
            if isinstance(cache, (tuple, list)) and len(cache) == 2:
                return gru_h, cache[0], cache[1]
            return gru_h, None, cache
        return h0, None, None

    def _transformer_pass(self, xs, summary, exact, verbose=False):
        seq_in = self.input_proj(xs)
        T = seq_in.size(0)
        outs = []
        n_chunks = max(1, (T + self.max_ctx - 1) // self.max_ctx)

        for i, start in enumerate(range(0, T, self.max_ctx)):
            if verbose:
                print(f'\r  [TGRU] transformer {i+1}/{n_chunks}', end='', flush=True)

            cur = seq_in[start : start + self.max_ctx]
            T_c = cur.size(0)
            prefix = self.cache.prefix(summary, exact)
            prefix_len = 0 if prefix is None else prefix.size(0)

            seq = torch.cat([prefix, cur], dim=0) if prefix_len else cur
            T_seq = seq.size(0)
            mask = torch.triu(
                torch.ones(T_seq, T_seq, device=xs.device), diagonal=1
            ).bool()

            out_full = self.transformer(seq, mask=mask)
            new_out = out_full[prefix_len:]
            outs.append(new_out)
            # Cache projected inputs (same space as cur), not transformer outputs
            summary, exact = self.cache.update(summary, exact, cur)

        if verbose:
            print()
        return torch.cat(outs, dim=0), summary, exact

    @staticmethod
    def reset_stream_cache(h0):
        '''Clear transformer prefix cache between val/test; keep GRU state.'''
        if h0 is None:
            return None
        if isinstance(h0, (tuple, list)) and len(h0) == 2:
            return (h0[0], (None, None))
        return h0

    def forward(self, xs, h0, include_h=False):
        gru_h, summary, exact = self._unpack_h0(h0)

        verbose = xs.size(0) > self.max_ctx * 2
        trans_out, summary, exact = self._transformer_pass(
            xs, summary, exact, verbose
        )
        trans_out = self.trans_drop(trans_out)

        # Chunk the GRU to avoid cuDNN allocating a workspace proportional
        # to the full sequence length (28+ GiB at test time with ~900 steps).
        chunk_outs = []
        for start in range(0, trans_out.size(0), self.max_ctx):
            chunk = trans_out[start : start + self.max_ctx]
            if gru_h is None:
                chunk_out, gru_h = self.gru(chunk)
            else:
                chunk_out, gru_h = self.gru(chunk, gru_h)
            chunk_outs.append(chunk_out)
        gru_out = torch.cat(chunk_outs, dim=0)

        out = self.lin(self.gru_drop(gru_out))
        new_cache = (summary, exact)

        if not include_h:
            return out
        return out, (gru_h, new_cache)


class EmptyModel(nn.Module):
    '''
    Just returns the input, assumes dims are correctly
    sized
    '''
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(EmptyModel, self).__init__()
        self.id = nn.Identity()

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.id(xs)
        
        return self.id(xs), None