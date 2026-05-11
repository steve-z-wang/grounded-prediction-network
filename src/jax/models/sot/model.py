"""SoT (Stream of Thoughts) base model — JAX/Flax Linen.

Per step:
    predicted = sigmoid(gate(prev_s)) * prev_s + FFN(norm(prev_s))
    pred_ce   = CE(decode(predicted), token_t)
    grounded  = sigmoid(fuse_gate(predicted)) * predicted + fuse_proj(norm(emb))

Loss = mean pred_ce over the window.

Uses jax.lax.scan for efficient sequential processing.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from src.jax.models.sot.config import SoTConfig


# ── Pure-function building blocks (no Flax module state) ──────────────────

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


# ── Flax module (parameter management + forward) ─────────────────────────

class SoT(nn.Module):
    """Base SoT: vector state, FFN with gated residual, fuse with gated residual."""
    cfg: SoTConfig

    def setup(self):
        cfg = self.cfg
        # Embedding
        self.embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (cfg.vocab_size, cfg.d_emb),
        )
        # Decoder
        self.decode_norm_w = self.param(
            'decode_norm_w', nn.initializers.ones, (cfg.d_state,))
        self.decode_proj_k = self.param(
            'decode_proj_k', nn.initializers.lecun_normal(), (cfg.d_state, cfg.d_emb))
        # Optional decoder bias (d_emb,) — adds a learnable direction to the
        # decode projection before the embedding matmul. Motivated by the DC
        # finding: state's persistent direction encodes a frequency prior;
        # making it explicit as a bias separates it from the content state.
        if cfg.use_decoder_bias:
            self.decode_bias = self.param(
                'decode_bias', nn.initializers.zeros, (cfg.d_emb,))
        # FFN
        self.ffn_norm_w = self.param(
            'ffn_norm_w', nn.initializers.ones, (cfg.d_state,))
        self.ffn_w1_k = self.param(
            'ffn_w1_k', nn.initializers.lecun_normal(), (cfg.d_state, 2 * cfg.d_ffn))
        self.ffn_w2_k = self.param(
            'ffn_w2_k', nn.initializers.zeros, (cfg.d_ffn, cfg.d_state))
        # Predict gate
        self.pg_norm_w = self.param(
            'pg_norm_w', nn.initializers.ones, (cfg.d_state,))
        self.pg_gate_k = self.param(
            'pg_gate_k', nn.initializers.zeros, (cfg.d_state, cfg.d_state))
        self.pg_gate_b = self.param(
            'pg_gate_b', nn.initializers.zeros, (cfg.d_state,))
        # Fuse input norm + projection
        self.fuse_norm_w = self.param(
            'fuse_norm_w', nn.initializers.ones, (cfg.d_emb,))
        self.fuse_proj_k = self.param(
            'fuse_proj_k', nn.initializers.lecun_normal(), (cfg.d_emb, cfg.d_state))
        # Fuse gate
        self.fg_norm_w = self.param(
            'fg_norm_w', nn.initializers.ones, (cfg.d_state,))
        self.fg_gate_k = self.param(
            'fg_gate_k', nn.initializers.zeros, (cfg.d_state, cfg.d_state))
        self.fg_gate_b = self.param(
            'fg_gate_b', nn.initializers.zeros, (cfg.d_state,))

    def init_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.cfg.d_state))

    def step(self, prev_s: jnp.ndarray, token_t: jnp.ndarray):
        """One step. Returns (grounded, predicted)."""
        ffn_out = _swigluffn(prev_s, self.ffn_norm_w, self.ffn_w1_k, self.ffn_w2_k)
        pg = _gate(prev_s, self.pg_norm_w, self.pg_gate_k, self.pg_gate_b)
        predicted = pg * prev_s + ffn_out
        # Fuse with token
        emb = self.embedding[token_t]
        grounding = _in_proj(emb, self.fuse_norm_w, self.fuse_proj_k)
        fg = _gate(predicted, self.fg_norm_w, self.fg_gate_k, self.fg_gate_b)
        grounded = fg * predicted + grounding
        return grounded, predicted

    def __call__(self, tokens: jnp.ndarray, init_state=None,
                 checkpoint_every: int = 0, bf16: bool = False,
                 return_logits: bool = False):
        """Process a window. tokens: (B, T). Returns (loss, final_state, metrics).

        bf16=True: cast matmul weights to bf16; state stays fp32.
        return_logits=True: additionally returns per-position logits (B, T, V).
            Logits at step t predict token t given state from tokens[:t].
            Only supported with checkpoint_every=0 (eval-only path).
        """
        B, T = tokens.shape
        if init_state is None:
            init_state = self.init_state(B)

        def _cast(x):
            return x.astype(jnp.bfloat16) if bf16 and x.ndim >= 2 else x

        # Capture all parameters as local variables for the scan closure
        emb_w = _cast(self.embedding)
        ffn_nw = self.ffn_norm_w
        ffn_w1 = _cast(self.ffn_w1_k)
        ffn_w2 = _cast(self.ffn_w2_k)
        pg_nw = self.pg_norm_w
        pg_gk = _cast(self.pg_gate_k)
        pg_gb = self.pg_gate_b
        fn_nw = self.fuse_norm_w
        fn_pk = _cast(self.fuse_proj_k)
        fg_nw = self.fg_norm_w
        fg_gk = _cast(self.fg_gate_k)
        fg_gb = self.fg_gate_b

        dn_w = self.decode_norm_w
        dp_k = _cast(self.decode_proj_k)
        decode_bias = self.decode_bias if self.cfg.use_decoder_bias else None
        V = emb_w.shape[0]

        def step_fn(carry, token):
            prev_s, total_ce = carry
            ffn_out = _swigluffn(prev_s, ffn_nw, ffn_w1, ffn_w2)
            pg = _gate(prev_s, pg_nw, pg_gk, pg_gb)
            predicted = pg * prev_s + ffn_out
            # CE loss
            decoded = _in_proj(predicted, dn_w, dp_k)
            if decode_bias is not None:
                decoded = decoded + decode_bias
            logits = decoded @ emb_w.T
            pred_ce = optax.softmax_cross_entropy_with_integer_labels(logits, token).sum()
            # Fuse
            emb = emb_w[token]
            grounding = _rmsnorm(emb, fn_nw) @ fn_pk
            fg = _gate(predicted, fg_nw, fg_gk, fg_gb)
            grounded = fg * predicted + grounding
            return (grounded, total_ce + pred_ce), (logits if return_logits else None)

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

            (final_s, total), _ = jax.lax.scan(
                outer_scan, (init_state, jnp.float32(0.0)), tokens_segs
            )
            logits_stack = None
        else:
            (final_s, total), logits_stack = jax.lax.scan(
                step_fn, (init_state, jnp.float32(0.0)), tokens_t
            )

        loss = total / (B * T)

        metrics = {
            "pred_ce": loss,
            "state_norm": jnp.sqrt(jnp.mean(final_s ** 2)),
        }
        if return_logits:
            # logits_stack: (T, B, V) → (B, T, V)
            logits = jnp.transpose(logits_stack, (1, 0, 2))
            return loss, final_s, metrics, logits
        return loss, final_s, metrics

    def describe(self) -> list[str]:
        c = self.cfg
        return [
            f"SoT: d_state={c.d_state}, d_emb={c.d_emb}, d_ffn={c.d_ffn}",
        ]
