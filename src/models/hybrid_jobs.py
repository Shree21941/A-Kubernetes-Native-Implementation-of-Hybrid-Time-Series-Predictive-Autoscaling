"""The Phase 2 job table, in its own module so it can be imported anywhere.

`hybrid.py` has to parse argv at import time (thread-pool environment variables must be
set before TensorFlow loads), which makes importing it from another script impossible.
Keeping the job definitions here lets the trainer, the artifact verifier and the
comparison script agree on one description of each model rather than three copies.

  source="residual" -> the hybrid stage: the network is trained on Prophet's residuals.
  source="raw"      -> a single-model baseline: the network is trained on raw y.
"""

from __future__ import annotations

JOBS = {
    "hybrid_lstm":   dict(cell="lstm",   source="residual", horizon=1,
                          label="Prophet+LSTM (hybrid)"),
    "hybrid_bilstm": dict(cell="bilstm", source="residual", horizon=1,
                          label="Prophet+BiLSTM (hybrid)"),
    "hybrid_gru":    dict(cell="gru",    source="residual", horizon=1,
                          label="Prophet+GRU (hybrid)"),
    "bilstm_1step":  dict(cell="bilstm", source="raw",      horizon=1,
                          label="BiLSTM 1-step"),
    "bilstm_5step":  dict(cell="bilstm", source="raw",      horizon=5,
                          label="BiLSTM 5-step"),
}
