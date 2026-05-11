"""SoT+M (Stream of Thoughts + Matrix Memory) — JAX/Flax Linen.

Single shared memory M, written once per token.
N FFN layers, each with its own memory read.

Per token:
    M_new = mem_write(prev_s, s, M)           # one write
    cur_s = s
    for i in range(N):
        mem_out = mem_read[i](cur_s, M_new)   # per-layer read
        ffn_out = FFN[i](cur_s)
        pg = gate[i](cur_s)
        cur_s = pg * cur_s + ffn_out + mem_out
    predicted = cur_s
    pred_ce = CE(decode(predicted), token)
    fg = gate(predicted)
    grounded = fg * predicted + in_proj(emb)

State: (s, prev_s, M) — single M shared across all layers.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from src.jax.models.sot_m.config import SoTMConfig


# ── Pure-function building blocks ──────────────────────────────────────────

def _rmsnorm(x, weight, eps=1e-6):
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype) * weight


def _swigluffn(x, norm_w, w1_k, w2_k):
    """SwiGLU FFN: norm -> silu(gate) * val -> project."""
    x = _rmsnorm(x, norm_w)
    gate_val = x @ w1_k
    gate, val = jnp.split(gate_val, 2, axis=-1)
    return (jax.nn.silu(gate) * val) @ w2_k


def _gate(x, norm_w, gate_k, gate_b):
    """Compute gate value: sigmoid(linear(norm(x)))."""
    return jax.nn.sigmoid(_rmsnorm(x, norm_w) @ gate_k + gate_b)


def _in_proj(x, norm_w, proj_k):
    """Pre-norm input projection: norm(x) @ W."""
    return _rmsnorm(x, norm_w) @ proj_k


def _l2_normalize(x, axis=-1):
    return x / jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=True) + 1e-12)


def _mem_write(prev_prev_s, prev_s, M, write_params):
    """Write to M via delta rule. Returns M_new.

    K from prev_prev_s, V/beta/decay from prev_s.
    """
    n_heads = write_params['n_heads']
    d_key = write_params['d_key']
    d_val = write_params['d_val']

    kh = _rmsnorm(prev_prev_s, write_params['norm_w'])
    qh = _rmsnorm(prev_s, write_params['norm_w'])

    k = _l2_normalize(jax.nn.relu(
        (kh @ write_params['W_k_k'] + write_params['W_k_b']).reshape(*kh.shape[:-1], n_heads, d_key)))
    v = (qh @ write_params['W_v_k']).reshape(*qh.shape[:-1], n_heads, d_val)
    beta = jax.nn.sigmoid(
        qh @ write_params['W_beta_k']
    ).reshape(*qh.shape[:-1], n_heads, 1, 1)

    g = -jnp.exp(write_params['A_log'].astype(jnp.float32)) * jax.nn.softplus(
        (qh @ write_params['W_a_k']).astype(jnp.float32) + write_params['dt_bias']
    )
    decay = jnp.exp(g).reshape(*qh.shape[:-1], n_heads, 1, 1)

    pred_v = jnp.einsum('...hkv,...hk->...hv', M, k)
    M_new = decay * M + beta * jnp.einsum('...hk,...hv->...hkv', k, v - pred_v)
    return M_new


def _mem_read(cur_s, M, read_params):
    """Read from M with cur_s as query. Returns mem_out."""
    n_heads = read_params['n_heads']
    d_key = read_params['d_key']
    d_val = read_params['d_val']

    qh = _rmsnorm(cur_s, read_params['norm_w'])

    q = _l2_normalize(jax.nn.relu(
        (qh @ read_params['W_q_k'] + read_params['W_q_b']).reshape(*qh.shape[:-1], n_heads, d_key)))
    mem_out = jnp.einsum('...hkv,...hk->...hv', M, q)

    gate = jax.nn.silu(qh @ read_params['W_gate_k'])
    flat_mem_out = mem_out.reshape(*qh.shape[:-1], n_heads * d_val)
    output = (gate * _rmsnorm(flat_mem_out, read_params['out_norm_w'])) @ read_params['W_o_k']
    return output


# ── Initializers ──────────────────────────────────────────────────────────

def _a_log_init(key, shape, dtype=jnp.float32):
    A = jax.random.uniform(key, shape, dtype=dtype, minval=0.0, maxval=16.0)
    return jnp.log(jnp.maximum(A, 1e-8))


def _dt_bias_init(key, shape, dtype=jnp.float32):
    dt = jnp.exp(
        jax.random.uniform(key, shape, dtype=dtype)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = jnp.clip(dt, min=1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


# ── Flax module ───────────────────────────────────────────────────────────

class SoTM(nn.Module):
    """SoT + matrix memory: single shared M, N FFN layers with per-layer reads."""
    cfg: SoTMConfig

    def setup(self):
        cfg = self.cfg
        lni = nn.initializers.lecun_normal()
        N = cfg.n_layers
        H, dk, dv = cfg.n_heads, cfg.d_key, cfg.d_val
        d = cfg.d_state

        # Embedding (shared)
        self.embedding = self.param(
            'embedding', nn.initializers.normal(stddev=0.02),
            (cfg.vocab_size, cfg.d_emb))
        # Decoder (shared)
        self.decode_norm_w = self.param(
            'decode_norm_w', nn.initializers.ones, (d,))
        self.decode_proj_k = self.param(
            'decode_proj_k', lni, (d, cfg.d_emb))
        # Optional decoder bias (d_emb,) — adds a learnable direction to the
        # decode projection before the embedding matmul. Motivated by the DC
        # finding: the state has a single emergent direction that projects to
        # a frequency-like prior. This bias explicitly represents that
        # direction, freeing the state from maintaining the DC attractor.
        if cfg.use_decoder_bias:
            self.decode_bias = self.param(
                'decode_bias', nn.initializers.zeros, (cfg.d_emb,))
        # Fuse (shared)
        self.fuse_norm_w = self.param(
            'fuse_norm_w', nn.initializers.ones, (cfg.d_emb,))
        self.fuse_proj_k = self.param(
            'fuse_proj_k', lni, (cfg.d_emb, d))
        self.fg_norm_w = self.param(
            'fg_norm_w', nn.initializers.ones, (d,))
        self.fg_gate_k = self.param(
            'fg_gate_k', nn.initializers.zeros, (d, d))
        self.fg_gate_b = self.param(
            'fg_gate_b', nn.initializers.zeros, (d,))

        # Memory write params (shared, used once per token)
        self.write_norm_w = self.param('write_norm_w', nn.initializers.ones, (d,))
        self.write_W_k_k = self.param('write_W_k_k', lni, (d, H * dk))
        self.write_W_k_b = self.param('write_W_k_b', nn.initializers.zeros, (H * dk,))
        self.write_W_v_k = self.param('write_W_v_k', lni, (d, H * dv))
        self.write_W_beta_k = self.param('write_W_beta_k', lni, (d, H))
        self.write_A_log = self.param('write_A_log', _a_log_init, (H,))
        self.write_dt_bias = self.param('write_dt_bias', _dt_bias_init, (H,))
        self.write_W_a_k = self.param('write_W_a_k', lni, (d, H))

        # Per-layer params: FFN + predict gate + memory read
        _layers = []
        for i in range(N):
            p = f'L{i}_'
            layer = {
                # FFN
                'ffn_norm_w': self.param(p+'ffn_norm_w', nn.initializers.ones, (d,)),
                'ffn_w1_k': self.param(p+'ffn_w1_k', lni, (d, 2 * cfg.d_ffn)),
                'ffn_w2_k': self.param(p+'ffn_w2_k', nn.initializers.zeros, (cfg.d_ffn, d)),
                # Predict gate
                'pg_norm_w': self.param(p+'pg_norm_w', nn.initializers.ones, (d,)),
                'pg_gate_k': self.param(p+'pg_gate_k', nn.initializers.zeros, (d, d)),
                'pg_gate_b': self.param(p+'pg_gate_b', nn.initializers.zeros, (d,)),
                # Memory read
                'read_norm_w': self.param(p+'read_norm_w', nn.initializers.ones, (d,)),
                'read_W_q_k': self.param(p+'read_W_q_k', lni, (d, H * dk)),
                'read_W_q_b': self.param(p+'read_W_q_b', nn.initializers.zeros, (H * dk,)),
                'read_W_gate_k': self.param(p+'read_W_gate_k', lni, (d, H * dv)),
                'read_out_norm_w': self.param(p+'read_out_norm_w', nn.initializers.ones, (H * dv,)),
                'read_W_o_k': self.param(p+'read_W_o_k', nn.initializers.normal(stddev=0.02), (H * dv, d)),
            }
            _layers.append(layer)
        self.layers = _layers

    def _write_params(self):
        cfg = self.cfg
        return {
            'n_heads': cfg.n_heads, 'd_key': cfg.d_key, 'd_val': cfg.d_val,
            'norm_w': self.write_norm_w,
            'W_k_k': self.write_W_k_k, 'W_k_b': self.write_W_k_b,
            'W_v_k': self.write_W_v_k, 'W_beta_k': self.write_W_beta_k,
            'A_log': self.write_A_log, 'dt_bias': self.write_dt_bias,
            'W_a_k': self.write_W_a_k,
        }

    @staticmethod
    def _read_params(layer, cfg):
        return {
            'n_heads': cfg.n_heads, 'd_key': cfg.d_key, 'd_val': cfg.d_val,
            'norm_w': layer['read_norm_w'],
            'W_q_k': layer['read_W_q_k'], 'W_q_b': layer['read_W_q_b'],
            'W_gate_k': layer['read_W_gate_k'],
            'out_norm_w': layer['read_out_norm_w'],
            'W_o_k': layer['read_W_o_k'],
        }

    def init_state(self, batch_size: int):
        """Returns (s, prev_s, M) — single shared M."""
        cfg = self.cfg
        s = jnp.zeros((batch_size, cfg.d_state))
        prev_s = jnp.zeros((batch_size, cfg.d_state))
        M = jnp.zeros((batch_size, cfg.n_heads, cfg.d_key, cfg.d_val))
        return s, prev_s, M

    def step(self, prev_s, prev_prev_s, M, token_t):
        """One step. Returns (grounded, prev_s_out, M_new, predicted)."""
        cfg = self.cfg
        # Single memory write per token
        M_new = _mem_write(prev_prev_s, prev_s, M, self._write_params())
        # N FFN layers, each with own memory read
        cur_s = prev_s
        for i in range(cfg.n_layers):
            L = self.layers[i]
            mem_out = _mem_read(cur_s, M_new, self._read_params(L, cfg))
            ffn_out = _swigluffn(cur_s, L['ffn_norm_w'], L['ffn_w1_k'], L['ffn_w2_k'])
            pg = _gate(cur_s, L['pg_norm_w'], L['pg_gate_k'], L['pg_gate_b'])
            cur_s = pg * cur_s + ffn_out + mem_out
        predicted = cur_s
        # Fuse with token
        emb = self.embedding[token_t]
        grounding = _in_proj(emb, self.fuse_norm_w, self.fuse_proj_k)
        fg = _gate(predicted, self.fg_norm_w, self.fg_gate_k, self.fg_gate_b)
        grounded = fg * predicted + grounding
        return grounded, prev_s, M_new, predicted

    def __call__(self, tokens, init_state=None,
                 checkpoint_every: int = 0, bf16: bool = False,
                 return_logits: bool = False):
        """Process a window. tokens: (B, T). Returns (loss, final_state, metrics).

        If bf16=True, cast projection weights (matmul matrices) to bf16 for
        compute. State (s, prev_s) and memory matrix M stay in fp32 for
        numerical stability. Norm weights and biases also stay in fp32.

        return_logits=True: additionally returns per-position logits (B, T, V).
            Only supported with checkpoint_every=0.
        """
        B, T = tokens.shape
        cfg = self.cfg
        if init_state is None:
            init_state = self.init_state(B)

        s, prev_s, M = init_state

        def _maybe_cast(x):
            """Cast matmul weights to bf16, keep norms/biases/1D in fp32."""
            if bf16 and x.ndim >= 2:
                return x.astype(jnp.bfloat16)
            return x

        # Capture params as locals for scan closure
        emb_w = _maybe_cast(self.embedding)
        decode = {'norm_w': self.decode_norm_w, 'proj_k': _maybe_cast(self.decode_proj_k)}
        decode_bias = self.decode_bias if cfg.use_decoder_bias else None
        fuse = {
            'norm_w': self.fuse_norm_w, 'proj_k': _maybe_cast(self.fuse_proj_k),
            'gate_norm_w': self.fg_norm_w, 'gate_k': _maybe_cast(self.fg_gate_k), 'gate_b': self.fg_gate_b,
        }
        wp = {k: (_maybe_cast(v) if hasattr(v, 'ndim') else v)
              for k, v in self._write_params().items()}
        layers = [{
            'ffn_norm_w': L['ffn_norm_w'],
            'ffn_w1_k': _maybe_cast(L['ffn_w1_k']),
            'ffn_w2_k': _maybe_cast(L['ffn_w2_k']),
            'pg_norm_w': L['pg_norm_w'],
            'pg_gate_k': _maybe_cast(L['pg_gate_k']),
            'pg_gate_b': L['pg_gate_b'],
            'rp': {k: (_maybe_cast(v) if hasattr(v, 'ndim') else v)
                   for k, v in self._read_params(L, cfg).items()},
        } for L in self.layers]

        _zero = jnp.float32(0.0)
        init_carry = (s, prev_s, M, {'ce': _zero, 'mem_effect': _zero, 'mem_cos': _zero})

        def step_fn(carry, token):
            s, prev_s, M, m = carry
            # Memory write: K from prev_s (grounded from 2 steps back), V from s
            M_new = _mem_write(prev_s, s, M, wp)
            # N FFN layers, each with its own memory read (parallel with FFN)
            cur_s = s
            step_mem_effect = _zero
            step_mem_cos = _zero
            for i in range(cfg.n_layers):
                lp = layers[i]
                mem_out = _mem_read(cur_s, M_new, lp['rp'])
                ffn_out = _swigluffn(cur_s, lp['ffn_norm_w'], lp['ffn_w1_k'], lp['ffn_w2_k'])
                pg = _gate(cur_s, lp['pg_norm_w'], lp['pg_gate_k'], lp['pg_gate_b'])
                state_no_mem = pg * cur_s + ffn_out
                cur_s = state_no_mem + mem_out
                # Memory effect metrics (averaged over batch)
                no_mem_norm = jnp.sqrt(jnp.sum(state_no_mem ** 2, axis=-1) + 1e-12)
                mem_norm = jnp.sqrt(jnp.sum(mem_out ** 2, axis=-1) + 1e-12)
                step_mem_effect = step_mem_effect + jnp.mean(mem_norm / no_mem_norm)
                dot = jnp.sum(state_no_mem * cur_s, axis=-1)
                cur_norm = jnp.sqrt(jnp.sum(cur_s ** 2, axis=-1) + 1e-12)
                step_mem_cos = step_mem_cos + jnp.mean(dot / (no_mem_norm * cur_norm + 1e-12))
            step_mem_effect = step_mem_effect / cfg.n_layers
            step_mem_cos = step_mem_cos / cfg.n_layers
            predicted = cur_s
            # CE loss
            decoded = _in_proj(predicted, decode['norm_w'], decode['proj_k'])
            if decode_bias is not None:
                decoded = decoded + decode_bias
            logits = decoded @ emb_w.T
            pred_ce = optax.softmax_cross_entropy_with_integer_labels(logits, token).sum()
            # Fuse
            emb = emb_w[token]
            grounding = _in_proj(emb, fuse['norm_w'], fuse['proj_k'])
            fg = _gate(predicted, fuse['gate_norm_w'], fuse['gate_k'], fuse['gate_b'])
            grounded = fg * predicted + grounding
            return (grounded, s, M_new, {
                'ce': m['ce'] + pred_ce,
                'mem_effect': m['mem_effect'] + step_mem_effect,
                'mem_cos': m['mem_cos'] + step_mem_cos,
            }), (logits if return_logits else None)

        tokens_t = tokens.T  # (T, B)

        if checkpoint_every > 0 and checkpoint_every < T:
            n_segs = T // checkpoint_every
            tokens_segs = tokens_t.reshape(n_segs, checkpoint_every, B)

            def run_segment(carry, seg_tokens):
                return jax.lax.scan(step_fn, carry, seg_tokens)

            checkpointed_segment = jax.checkpoint(run_segment)

            def outer_scan(carry, seg_tokens):
                carry, _ = checkpointed_segment(carry, seg_tokens)
                return carry, None

            (final_s, final_prev_s, final_M, final_m), _ = jax.lax.scan(
                outer_scan, init_carry, tokens_segs)
            logits_stack = None
        else:
            (final_s, final_prev_s, final_M, final_m), logits_stack = jax.lax.scan(
                step_fn, init_carry, tokens_t)

        loss = final_m['ce'] / (B * T)

        metrics = {
            "pred_ce": loss,
            "state_norm": jnp.sqrt(jnp.mean(final_s ** 2)),
            "mem_norm": jnp.sqrt(jnp.mean(final_M ** 2)),
            "mem_effect": final_m['mem_effect'] / T,
            "mem_cos": final_m['mem_cos'] / T,
        }
        if return_logits:
            logits = jnp.transpose(logits_stack, (1, 0, 2))
            return loss, (final_s, final_prev_s, final_M), metrics, logits
        return loss, (final_s, final_prev_s, final_M), metrics

    def describe(self) -> list[str]:
        c = self.cfg
        return [
            f"SoT+M: d_state={c.d_state}, d_emb={c.d_emb}, d_ffn={c.d_ffn}, n_layers={c.n_layers}",
            f"  memory: {c.n_heads}h, d_key={c.d_key}, d_val={c.d_val} (1 write, {c.n_layers} reads)",
        ]
