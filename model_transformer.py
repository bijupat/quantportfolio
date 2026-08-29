"""
model_transformer.py - Transformer encoder for multi-stock return prediction

Architecture:
  Input (seq_len, n_features)
  → Linear projection (embedding)
  → Positional encoding
  → N × TransformerEncoder blocks (MultiHeadAttention + FFN)
  → Attention pooling
  → Dense head
  → Scalar output (predicted 30-day return)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from util import log, MODELS_DIR, METRICS_DIR, save_json


# ─────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
        self.pe = tf.constant(pe[None, :, :], dtype=tf.float32)  # (1, T, d)

    def call(self, x):
        return x + self.pe

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


# ─────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads,
                                                dropout=dropout)
        self.ffn   = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention (pre-norm)
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, return_attention_scores=True,
                                            training=training)
        x = x + self.drop1(attn_out, training=training)

        # FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        x = x + self.drop2(ffn_out, training=training)
        return x, attn_weights

    def get_config(self):
        cfg = super().get_config()
        return cfg


# ─────────────────────────────────────────────
# Attention Pooling
# ─────────────────────────────────────────────
class AttentionPooling(layers.Layer):
    """Learnable attention pooling over sequence dimension."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.query_dense = layers.Dense(1)

    def call(self, x):
        # x: (B, T, d_model)
        scores = self.query_dense(x)           # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)  # (B, T, 1)
        pooled = tf.reduce_sum(x * weights, axis=1)  # (B, d_model)
        return pooled, tf.squeeze(weights, axis=-1)


# ─────────────────────────────────────────────
# Full Transformer Model
# ─────────────────────────────────────────────
class QuantTransformer(keras.Model):
    """
    End-to-end Transformer model for stock return regression.
    """

    def __init__(
        self,
        seq_len:    int   = 60,
        n_features: int   = 50,
        d_model:    int   = 128,
        n_heads:    int   = 8,
        ff_dim:     int   = 256,
        n_layers:   int   = 3,
        dropout:    float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len    = seq_len
        self.n_features = n_features
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.ff_dim     = ff_dim
        self.n_layers   = n_layers

        # Input projection
        self.input_proj = layers.Dense(d_model, use_bias=False)
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)

        # Positional encoding
        self.pos_enc = PositionalEncoding(seq_len, d_model)
        self.pos_drop = layers.Dropout(dropout)

        # Encoder blocks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout, name=f"encoder_{i}")
            for i in range(n_layers)
        ]

        # Attention pooling
        self.pool = AttentionPooling(d_model)

        # Regression head
        self.head = keras.Sequential([
            layers.Dense(128, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(64,  activation="gelu"),
            layers.Dropout(dropout * 0.5),
            layers.Dense(1,   activation="linear"),
        ])

        # Store attention weights for visualization
        self._last_attn_weights = None

    def call(self, x, training=False):
        # x: (B, T, n_features)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.pos_drop(x, training=training)

        all_attn = []
        for block in self.encoder_blocks:
            x, attn_w = block(x, training=training)
            all_attn.append(attn_w)

        self._last_attn_weights = all_attn  # store for visualization

        pooled, pool_weights = self.pool(x)
        out = self.head(pooled, training=training)
        return tf.squeeze(out, axis=-1)
# ─────────────────────────────────────────────
# Triple-Branch Hybrid Model
# Relocated from hybrid_main.py — fuses time-series, financial, and
# sentiment streams into a single wide-tensor-compatible model so it
# stays drop-in loadable by build_keras_model() / predictor.py.
# ─────────────────────────────────────────────
class TripleBranchHybridModel(keras.Model):
    """
    Triple-branch deep learning model for stock return prediction.

    Accepts a SINGLE wide tensor (batch, seq_len, n_total_features) where
    n_total_features = n_ts_features + n_financial_features + n_sentiment_features.
    Internally slices into three branches, processes each with a specialised
    sub-network, fuses the embeddings, and regresses to a scalar return.
    """

    def __init__(
        self,
        seq_len:              int   = 120,
        n_ts_features:        int   = 50,
        n_financial_features: int   = 8,
        n_sentiment_features: int   = 4,
        d_model:              int   = 64,
        n_heads:              int   = 4,
        n_layers:              int  = 2,
        lstm_units:           int   = 64,
        dropout:              float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seq_len              = seq_len
        self.n_ts_features        = n_ts_features
        self.n_financial_features = n_financial_features
        self.n_sentiment_features = n_sentiment_features
        self.n_features           = n_ts_features + n_financial_features + n_sentiment_features
        self.d_model              = d_model
        self.n_heads              = n_heads
        self.n_layers             = n_layers
        self.lstm_units           = lstm_units

        # ── Branch 1: Time-Series (LSTM) ──────────────────────
        self.ts_lstm   = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout)
        self.ts_norm   = layers.LayerNormalization(epsilon=1e-6)
        self.ts_dense  = layers.Dense(d_model, activation="gelu")
        self.ts_drop   = layers.Dropout(dropout)

        # ── Branch 2: Financial (Dense MLP) ───────────────────
        self.fin_dense1 = layers.Dense(32, activation="gelu")
        self.fin_norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.fin_drop1  = layers.Dropout(dropout)
        self.fin_dense2 = layers.Dense(d_model, activation="gelu")
        self.fin_drop2  = layers.Dropout(dropout)

        # ── Branch 3: Sentiment (Transformer Encoder) ─────────
        self.sent_proj   = layers.Dense(d_model, use_bias=False)
        self.sent_norm_i = layers.LayerNormalization(epsilon=1e-6)
        self.sent_attn   = layers.MultiHeadAttention(
            num_heads = max(1, n_heads // 2),
            key_dim   = max(1, d_model // max(1, n_heads // 2)),
        )
        self.sent_ffn    = keras.Sequential([
            layers.Dense(d_model * 2, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.sent_norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.sent_norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.sent_drop   = layers.Dropout(dropout)
        self.sent_pool_q = layers.Dense(1)
        self.sent_drop2  = layers.Dropout(dropout)

        # ── Fusion layer ───────────────────────────────────────
        self.fusion_dense1 = layers.Dense(d_model * 2, activation="gelu")
        self.fusion_norm   = layers.LayerNormalization(epsilon=1e-6)
        self.fusion_drop   = layers.Dropout(dropout)
        self.fusion_dense2 = layers.Dense(d_model, activation="gelu")

        # ── Regression head ────────────────────────────────────
        self.head = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dropout(dropout * 0.5),
            layers.Dense(1, activation="linear"),
        ])

        self._last_attn_weights = None

    def call(self, x, training=False):
        """Forward pass: slices x into [ts | financial | sentiment] branches."""
        n_ts   = self.n_ts_features
        n_fin  = self.n_financial_features
        n_sent = self.n_sentiment_features

        x_ts   = x[:, :, :n_ts]
        x_fin  = x[:, 0, n_ts:n_ts + n_fin]
        x_sent = x[:, :, n_ts + n_fin:]

        ts_out = self.ts_lstm(x_ts, training=training)
        ts_out = self.ts_norm(ts_out)
        ts_emb = self.ts_drop(self.ts_dense(ts_out), training=training)

        fin_out = self.fin_dense1(x_fin)
        fin_out = self.fin_norm1(fin_out)
        fin_out = self.fin_drop1(fin_out, training=training)
        fin_emb = self.fin_drop2(self.fin_dense2(fin_out), training=training)

        s = self.sent_proj(x_sent)
        s = self.sent_norm_i(s)
        s_norm  = self.sent_norm1(s)
        s_attn, attn_w = self.sent_attn(
            s_norm, s_norm, return_attention_scores=True, training=training
        )
        s = s + self.sent_drop(s_attn, training=training)
        s_norm = self.sent_norm2(s)
        s = s + self.sent_ffn(s_norm, training=training)
        self._last_attn_weights = [attn_w]

        scores  = self.sent_pool_q(s)
        weights = tf.nn.softmax(scores, axis=1)
        sent_emb = tf.reduce_sum(s * weights, axis=1)
        sent_emb = self.sent_drop2(sent_emb, training=training)

        fused = tf.concat([ts_emb, fin_emb, sent_emb], axis=-1)
        fused = self.fusion_norm(self.fusion_dense1(fused))
        fused = self.fusion_drop(fused, training=training)
        fused = self.fusion_dense2(fused)

        out = self.head(fused, training=training)
        return tf.squeeze(out, axis=-1)


def build_hybrid_model(
    seq_len:              int   = 120,
    n_ts_features:        int   = 50,
    n_financial_features: int   = 8,
    n_sentiment_features: int   = 4,
    d_model:              int   = 64,
    n_heads:              int   = 4,
    n_layers:             int   = 2,
    lstm_units:           int   = 64,
    dropout:              float = 0.15,
    lr:                   float = 1e-4,
) -> TripleBranchHybridModel:
    """Build, compile, and warm-up the triple-branch model."""
    model = TripleBranchHybridModel(
        seq_len=seq_len, n_ts_features=n_ts_features,
        n_financial_features=n_financial_features,
        n_sentiment_features=n_sentiment_features,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        lstm_units=lstm_units, dropout=dropout,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="huber", metrics=["mae"],
    )
    n_total = n_ts_features + n_financial_features + n_sentiment_features
    dummy = tf.zeros((1, seq_len, n_total))
    model(dummy, training=False)
    log.info(
        f"TripleBranchHybridModel built: ts={n_ts_features}  fin={n_financial_features}  "
        f"sent={n_sentiment_features}  total_features={n_total}  params={model.count_params():,}"
    )
    return model


def train_hybrid_model(
    model:       TripleBranchHybridModel,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    model_name:  str,
    epochs:      int = 60,
    batch_size:  int = 32,
    patience:    int = 12,
) -> keras.callbacks.History:
    """Train the hybrid model with early stopping and LR reduction."""
    tmp_path = MODELS_DIR / f"{model_name}_tmp_best.weights.h5"

    class _LogCB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                logs = logs or {}
                log.info(
                    f"  Epoch {epoch+1:3d}  loss={logs.get('loss', 0):.5f}  "
                    f"mae={logs.get('mae', 0):.5f}  val_loss={logs.get('val_loss', 0):.5f}  "
                    f"val_mae={logs.get('val_mae', 0):.5f}"
                )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=patience,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                           min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(tmp_path), monitor="val_mae",
                                         save_best_only=True, save_weights_only=True, verbose=0),
        keras.callbacks.TerminateOnNaN(),
        _LogCB(),
    ]

    log.info(
        f"Training '{model_name}'  Train={X_train.shape[0]:,}  Val={X_val.shape[0]:,}  "
        f"Features={X_train.shape[2]}  Epochs={epochs}  BS={batch_size}"
    )

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0,
    )

    if tmp_path.exists():
        tmp_path.unlink()

    best_val = min(history.history.get("val_mae", [999]))
    log.info(f"Training complete.  Best val_mae = {best_val:.5f}")
    return history

# ─────────────────────────────────────────────
# Build & compile
# ─────────────────────────────────────────────
def build_transformer_model(
    seq_len:    int   = 60,
    n_features: int   = 50,
    d_model:    int   = 128,
    n_heads:    int   = 8,
    ff_dim:     int   = 256,
    n_layers:   int   = 3,
    dropout:    float = 0.15,
    lr:         float = 1e-4,
) -> QuantTransformer:
    model = QuantTransformer(
        seq_len=seq_len,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        ff_dim=ff_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="huber",               # robust to return outliers
        metrics=["mae"],
    )
    # Build weights by calling with dummy input
    dummy = tf.zeros((1, seq_len, n_features))
    model(dummy, training=False)
    log.info(f"Model built: d_model={d_model}, heads={n_heads}, "
             f"layers={n_layers}, params={model.count_params():,}")
    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(
    model:          QuantTransformer,
    X_train:        np.ndarray,
    y_train:        np.ndarray,
    X_val:          np.ndarray,
    y_val:          np.ndarray,
    epochs:         int   = 60,
    batch_size:     int   = 64,
    model_name:     str   = "transformer_model",
    patience:       int   = 12,
) -> keras.callbacks.History:
    """Train model with early stopping, LR reduction, and checkpoint."""

    model_path = MODELS_DIR / f"{model_name}.weights.h5"
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_mae", save_best_only=True,
            save_weights_only=True, verbose=0
        ),
        keras.callbacks.TerminateOnNaN(),
        _LoggingCallback(log),
    ]

    log.info(f"Training {model_name}  |  "
             f"Train={X_train.shape[0]:,}  Val={X_val.shape[0]:,}  "
             f"Epochs={epochs}  BS={batch_size}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # Persist metrics
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(hist_dict, metrics_path)
    log.info(f"Training complete.  Best val_mae = "
             f"{min(history.history.get('val_mae', [999])):.5f}")
    return history


class _LoggingCallback(keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            logs = logs or {}
            msg = (f"Epoch {epoch+1:3d}  "
                   f"loss={logs.get('loss',0):.5f}  "
                   f"mae={logs.get('mae',0):.5f}  "
                   f"val_loss={logs.get('val_loss',0):.5f}  "
                   f"val_mae={logs.get('val_mae',0):.5f}")
            self.logger.info(msg)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate_model(
    model:  QuantTransformer,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute regression metrics on test split."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_true = y_test.flatten()

    # Remove NaN
    mask  = np.isfinite(y_true) & np.isfinite(y_pred)
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0

    # Hit ratio: correct direction
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # IC (Information Coefficient)
    from scipy.stats import spearmanr
    try:
        ic, _ = spearmanr(y_true, y_pred)
    except Exception:
        ic = 0.0

    metrics = {
        "MSE":           mse,
        "MAE":           mae,
        "RMSE":          float(np.sqrt(mse)),
        "Pearson_corr":  corr,
        "HitRatio":      hit,
        "IC_Spearman":   float(ic),
    }
    return metrics


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────
def save_model(model: QuantTransformer, name: str = "transformer_model") -> Path:
    """
    Save weights as a flat ordered h5py file.
    Embeds architecture attrs inside the file so load_model() is
    self-contained — no CLI args or external config needed.

    Also writes _arch.json to metrics/ as a readable companion.
    """
    import h5py

    path = MODELS_DIR / f"{name}.weights.h5"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for i, w in enumerate(model.weights):
            f.create_dataset(f"weight_{i:04d}", data=np.array(w))
        # Embed arch attrs for self-describing load
        f.attrs["seq_len"]    = model.seq_len
        f.attrs["n_features"] = model.n_features
        f.attrs["d_model"]    = model.d_model
        f.attrs["n_heads"]    = model.n_heads
        f.attrs["ff_dim"]     = model.ff_dim
        f.attrs["n_layers"]   = model.n_layers

    arch = {
        "seq_len":    model.seq_len,
        "n_features": model.n_features,
        "d_model":    model.d_model,
        "n_heads":    model.n_heads,
        "ff_dim":     model.ff_dim,
        "n_layers":   model.n_layers,
    }
    save_json(arch, METRICS_DIR / f"{name}_arch.json")
    log.info(f"Model saved → {path.name}  ({model.count_params():,} params)")
    return path


def _load_weights_shape_matched(model, path: Path) -> None:
    """
    Load weights from h5 file using shape-matched positional assignment.

    Bypasses Keras name-based matching which fails when:
      - Optimizer states are co-located with model weights
      - Generic layer names ('dense') cause collisions
      - A different model class is used at load vs save time

    Algorithm:
      1. Collect ALL arrays from h5 with their paths
      2. Build expected shape list from model.weights
      3. For each expected shape, take first unused array that matches
      4. Skips optimizer states (momentum/variance) naturally
    """
    import h5py

    with h5py.File(path, "r") as f:
        # New flat format: weight_0000, weight_0001, ...
        flat_keys = sorted(k for k in f.keys() if k.startswith("weight_"))
        if flat_keys:
            saved = [np.array(f[k]) for k in flat_keys]
            log.info(f"  Flat format: {len(saved)} arrays")
        else:
            # Legacy Keras nested format — collect recursively
            all_arrays = []
            def _collect(obj, prefix=""):
                for key in obj.keys():
                    full = f"{prefix}/{key}"
                    item = obj[key]
                    if isinstance(item, h5py.Dataset):
                        all_arrays.append((full, np.array(item)))
                    elif isinstance(item, h5py.Group):
                        _collect(item, full)
            _collect(f)
            log.info(f"  Legacy format: {len(all_arrays)} total arrays")
            saved = None

    if saved is None:
        # Shape-match against legacy arrays (skips optimizer states)
        expected = [tuple(w.shape) for w in model.weights]
        matched, used = [], set()
        for idx, shape in enumerate(expected):
            for fpath, arr in all_arrays:
                if fpath not in used and arr.shape == shape:
                    matched.append(arr)
                    used.add(fpath)
                    break
            else:
                remaining = [(p, a.shape) for p, a in all_arrays if p not in used]
                log.error(
                    f"  No array with shape {shape} for weight[{idx}] "
                    f"({model.weights[idx].name}). "
                    f"Unused: {[s for _,s in remaining[:5]]}"
                )
                raise ValueError(
                    f"Shape mismatch at weight[{idx}]: expected {shape}. "
                    f"Check n_features matches training."
                )
        skipped = len(all_arrays) - len(matched)
        log.info(f"  Shape-matched {len(matched)} arrays, skipped {skipped} optimizer tensors")
        saved = matched

    if len(saved) != len(model.weights):
        raise ValueError(
            f"Weight count mismatch: file has {len(saved)} arrays, "
            f"model expects {len(model.weights)}. "
            f"Ensure the correct model class is built before loading."
        )

    model.set_weights(saved)
    log.info(f"  set_weights() complete ({len(saved)} arrays)")


def load_model(name: str = "transformer_model"):
    """
    Fully self-contained model loader.

    Reads architecture from the h5 file attrs or _arch.json,
    determines whether the saved model is a standard QuantTransformer
    or a TripleBranchHybridModel (saved by composite_main.py),
    builds the correct class, then loads weights using shape-matched
    h5py assignment — bypasses Keras name-based matching entirely.

    Args:
        name: model name without extension (e.g. 'transformer_banking')

    Returns:
        Loaded model with .seq_len, .n_features, .d_model, .n_heads,
        .n_layers attributes set — compatible with predict.py,
        ensemble_predict.py, and composite_predict.py.
    """
    import h5py

    weights_path = MODELS_DIR / f"{name}.weights.h5"
    arch_path    = METRICS_DIR / f"{name}_arch.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No weights file for '{name}' at {weights_path}.\n"
            f"Train first:  python main.py --model-name {name}"
        )

    # ── Read architecture ─────────────────────────────────────
    arch = {}
    with h5py.File(weights_path, "r") as f:
        attr_keys = ("seq_len", "n_features", "d_model", "n_heads", "n_layers")
        if all(k in f.attrs for k in attr_keys):
            arch = {k: int(f.attrs[k]) for k in attr_keys}
            # Also read hybrid-specific keys if present
            for extra in ("n_ts_features", "n_financial_features",
                          "n_sentiment_features", "model_type"):
                if extra in f.attrs:
                    v = f.attrs[extra]
                    arch[extra] = int(v) if extra != "model_type" else str(v)
            if "ff_dim" in f.attrs:                    
                    arch["ff_dim"] = int(f.attrs["ff_dim"])
            log.info(f"  Architecture from h5 attrs")

    if not arch:
        if not arch_path.exists():
            raise FileNotFoundError(
                f"No _arch.json for '{name}'. "
                f"Re-train or run: python -c \""
                f"from model_transformer import migrate_model; "
                f"migrate_model('{name}')\""
            )
        arch = json.load(open(arch_path))
        log.info(f"  Architecture from {arch_path.name}")

    model_type = arch.get("model_type", "standard")
    ff_dim = arch.get("ff_dim", 256)
    log.info(
        f"  Loading '{name}'  type={model_type}  "
        f"seq={arch['seq_len']}  feat={arch['n_features']}  "
        f"d_model={arch['d_model']}  heads={arch['n_heads']}  "
        f"layers={arch['n_layers']}"
    )

    # ── Build correct model class ─────────────────────────────
    if model_type == "triple_branch_hybrid":
        try:
            model = build_hybrid_model(
                seq_len              = arch["seq_len"],
                n_ts_features        = arch.get("n_ts_features",
                                                arch["n_features"] - 12),
                n_financial_features = arch.get("n_financial_features", 8),
                n_sentiment_features = arch.get("n_sentiment_features", 4),
                d_model              = arch["d_model"],
                n_heads              = arch["n_heads"],
                n_layers             = arch["n_layers"],
            )
            log.info(
                f"  TripleBranchHybridModel built  "
                f"({model.count_params():,} params)"
            )
        except ImportError:
            log.warning(
                "TripleBranchHybridModel not built !!."
                "Prediction may be inaccurate if model was trained as hybrid."
            )
            model = build_transformer_model(
                seq_len    = arch["seq_len"],
                n_features = arch["n_features"],
                d_model    = arch["d_model"],
                n_heads    = arch["n_heads"],
                n_layers   = arch["n_layers"],
            )
    else:
        # Standard QuantTransformer
        model = build_transformer_model(
            seq_len    = arch["seq_len"],
            n_features = arch["n_features"],
            d_model    = arch["d_model"],
            n_heads    = arch["n_heads"],
            ff_dim     = ff_dim,
            n_layers   = arch["n_layers"],
        )

    # ── Load weights (shape-matched — no Keras name matching) ─
    _load_weights_shape_matched(model, weights_path)

    # Stamp arch attributes for downstream access (model.seq_len etc.)
    for k, v in arch.items():
        setattr(model, k, v)
    model.ff_dim = ff_dim
    log.info(
        f"  '{name}' loaded  ({model.count_params():,} params)"
    )
    return model


def migrate_model(name: str) -> None:
    """
    One-time migration: loads an old .weights.h5 model using shape-matched
    h5py reading (handles Keras optimizer state co-location), then re-saves
    in the new flat format.

    Run once for models saved before this fix:
        python -c "from model_transformer import migrate_model; migrate_model('your_model')"
    """
    weights_path = MODELS_DIR / f"{name}.weights.h5"
    arch_path    = METRICS_DIR / f"{name}_arch.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"No weights at {weights_path}")

    if arch_path.exists():
        arch = json.load(open(arch_path))
    else:
        log.warning("No _arch.json — enter architecture manually:")
        arch = {
            "seq_len":    int(input("seq_len    [60]:  ") or 60),
            "n_features": int(input("n_features [50]:  ") or 50),
            "d_model":    int(input("d_model    [64]:  ") or 64),
            "n_heads":    int(input("n_heads    [4]:   ") or 4),
            "n_layers":   int(input("n_layers   [2]:   ") or 2),
        }
    ff_dim = arch.get("ff_dim", 256)
    model = build_transformer_model(**{
        k: arch[k] for k in
        ("seq_len", "n_features", "d_model", "n_heads", "n_layers")
    }, ff_dim=ff_dim)

    _load_weights_shape_matched(model, weights_path)

    for k, v in arch.items():
        setattr(model, k, v)
    model.ff_dim = ff_dim  
    save_model(model, name)
    log.info(f"Migration complete: {name}")


# ─────────────────────────────────────────────
# Attention visualization helper
# ─────────────────────────────────────────────
def get_attention_weights(model: QuantTransformer, X: np.ndarray) -> np.ndarray:
    """
    Forward pass on single sample and return mean attention weights
    across all layers and heads.  Shape: (seq_len, seq_len)
    """
    x = tf.constant(X[np.newaxis], dtype=tf.float32)
    model(x, training=False)
    if model._last_attn_weights:
        # Average across layers and heads
        all_w = [w.numpy()[0].mean(axis=0) for w in model._last_attn_weights]
        return np.mean(all_w, axis=0)
    return np.array([])
