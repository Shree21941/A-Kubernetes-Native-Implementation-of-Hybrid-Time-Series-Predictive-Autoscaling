# Phase 5 - the predictive autoscaler.
#
# Unlike the target app (app/Dockerfile), which is stdlib-only and deliberately tiny,
# this image needs TensorFlow and Prophet. It is large; that is acceptable because it
# runs once per experiment and is loaded into KIND from the local daemon rather than
# pulled over a network.
FROM python:3.11-slim

# FOUR packages, not seven. The first build installed prophet, pandas and scikit-learn
# as well; the running container imports none of them:
#
#   prophet, pandas  imported only inside Forecaster._yhat_live / _init_prophet, i.e.
#                    the "live" and "both" Prophet modes. The controller runs
#                    prophet_mode="table", which reads the precomputed lookup in
#                    prophet_table.npz. Phase 5 verified table == live to 1e-9 on real
#                    timestamps, so predictions are bit-identical either way - only the
#                    ability to RECOMPUTE Prophet in-cluster is given up, and nothing
#                    in Phases 6-8 does that.
#   scikit-learn     not imported anywhere in src/controller. The residual scaler is two
#                    floats carried in bundle.json, not a pickled MinMaxScaler.
#   matplotlib       only in simulate.py / sweep_compression.py, which are offline
#                    analysis tools and never run in this container.
#
# This matters here for a mundane reason: the host disk is at 97% and the first build
# failed after consuming 4.5 GB. It is also just correct - a container should not carry
# a Stan toolchain it never invokes.
#
# CONSEQUENCE, stated so it is not discovered later: `--prophet-mode live` and `both`
# will raise ImportError inside this image. That is intentional. Run them on the host,
# where the full environment exists, if a re-verification of the table is ever needed.
#
# --retries/--timeout because the first attempt died on a ReadTimeoutError from
# files.pythonhosted.org partway through a ~230 MB wheel.
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
      "tensorflow-cpu==2.16.*" \
      "kubernetes==29.*" \
      "requests==2.*" \
      "numpy<2"

WORKDIR /app
COPY src/controller/ /app/controller/
# The bundle, not models/ - 10 MB rather than 252 MB. See export_bundle.py.
COPY models/controller_bundle/ /app/models/controller_bundle/

# Non-root, matching the target app's convention. The controller holds a token that can
# scale a Deployment; there is no reason for it to also be root inside its container.
RUN useradd --uid 10002 --create-home controller \
    && mkdir -p /var/log/controller \
    && chown -R controller /var/log/controller /app
USER 10002

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/controller \
    TF_CPP_MIN_LOG_LEVEL=3

ENTRYPOINT ["python", "/app/controller/main.py"]
