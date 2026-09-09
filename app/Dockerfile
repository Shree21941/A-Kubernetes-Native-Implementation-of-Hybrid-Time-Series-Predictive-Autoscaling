# Target application image.
#
# python:3.11-slim, not the full python image and not alpine:
#   - full  (~1 GB) would dominate the cold-start measurement with image pull time
#   - alpine uses musl, which changes CPython's performance characteristics enough that
#     our CPU-per-request calibration would not transfer to any normal deployment
#
# There is no `pip install` layer at all - main.py is standard library only - so the
# image is one base layer plus a 9 KB file. That keeps container start dominated by
# CPython's own boot, which is the honest thing to measure.
FROM python:3.11-slim

# Unbuffered stdout so `kubectl logs` shows the startup line immediately rather than
# when the buffer happens to flush. During cold-start debugging that line is the
# evidence for when the app was actually up.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app
COPY main.py /app/main.py

# Run as a non-root user. Not security theatre: a container that only works as root
# will fail the moment a cluster applies a restricted PodSecurity standard, and
# discovering that in Phase 5 under load is a bad time to discover it.
RUN useradd --create-home --uid 10001 appuser
USER 10001

EXPOSE 8080
CMD ["python", "-u", "/app/main.py"]
