"""
Export CPU usage metrics from Prometheus to CSV.

The generated CSV files can be used later for:
- Prophet forecasting
- LSTM training
- AIOps autoscaling experiments
- CPU usage analysis

Prerequisites:
    1. Keep Prometheus port-forward running:
       kubectl port-forward -n monitoring \
       svc/prometheus-kube-prometheus-prometheus 9090:9090

    2. Install dependencies:
       pip install requests pandas

    3. Run:
       python3 export_metrics.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090"

# How much historical data to export
DATA_DURATION_MINUTES = 20

# Prometheus sampling interval
STEP = "15s"


# ============================================================
# PROMQL QUERY
# ============================================================

# CPU usage of all application pods in the default namespace.
#
# container!="" and container!="POD"
# remove empty/infrastructure container series.
#
# rate(...[1m])
# calculates CPU usage per second over the previous 1 minute.
#
# sum by (pod)
# gives one CPU usage value per pod.

QUERY = """
sum by (pod) (
    rate(
        container_cpu_usage_seconds_total{
            namespace="default",
            container!="",
            container!="POD"
        }[1m]
    )
)
"""


# ============================================================
# TIME RANGE
# ============================================================

END_TIME = datetime.now(timezone.utc)

START_TIME = END_TIME - timedelta(
    minutes=DATA_DURATION_MINUTES
)


# ============================================================
# CHECK PROMETHEUS CONNECTION
# ============================================================

def check_prometheus():
    """Check whether Prometheus is reachable."""

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/-/ready",
            timeout=5
        )

        response.raise_for_status()

        print("Prometheus connection: OK")

    except requests.exceptions.RequestException as e:
        print("ERROR: Cannot connect to Prometheus.")
        print(f"URL: {PROMETHEUS_URL}")
        print(f"Reason: {e}")

        print("\nMake sure this is running in another terminal:")
        print(
            "kubectl port-forward -n monitoring "
            "svc/prometheus-kube-prometheus-prometheus 9090:9090"
        )

        return False

    return True


# ============================================================
# FETCH METRICS
# ============================================================

def fetch_metrics():

    params = {
        "query": QUERY,
        "start": START_TIME.timestamp(),
        "end": END_TIME.timestamp(),
        "step": STEP,
    }

    try:

        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:

        print("ERROR while querying Prometheus:")
        print(e)

        return None

    # Prometheus API itself can return status=error
    if data.get("status") != "success":

        print("ERROR: Prometheus query failed.")

        if "error" in data:
            print(data["error"])

        return None

    return data


# ============================================================
# CONVERT PROMETHEUS RESPONSE TO DATAFRAME
# ============================================================

def to_dataframe(data):

    rows = []

    if not data:
        return pd.DataFrame()

    results = data.get("data", {}).get("result", [])

    for series in results:

        pod_name = series["metric"].get(
            "pod",
            "unknown"
        )

        for timestamp, value in series.get("values", []):

            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        float(timestamp),
                        timezone.utc
                    ),
                    "pod": pod_name,
                    "cpu_usage": float(value),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    # --------------------------------------------------------
    # Per-pod CPU metrics
    # --------------------------------------------------------

    output_file = "cpu_metrics2.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved {len(df)} rows to "
        f"{output_file}"
    )


    # --------------------------------------------------------
    # Total CPU across all application pods
    # --------------------------------------------------------

    agg_df = (
        df.groupby("timestamp")["cpu_usage"]
        .sum()
        .reset_index()
    )

    agg_df.columns = [
        "timestamp",
        "total_cpu_usage"
    ]

    agg_df.to_csv(
        "total_cpu_timeseries2.csv",
        index=False
    )

    print(
        "Saved aggregated time series to "
        "total_cpu_timeseries.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Prometheus CPU Metrics Exporter")
    print("=" * 60)

    print(
        f"Prometheus URL : {PROMETHEUS_URL}"
    )

    print(
        f"Start time     : {START_TIME}"
    )

    print(
        f"End time       : {END_TIME}"
    )

    print(
        f"Step           : {STEP}"
    )

    print(
        f"Duration       : "
        f"{DATA_DURATION_MINUTES} minutes"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Check Prometheus
    # --------------------------------------------------------

    if not check_prometheus():
        return


    # --------------------------------------------------------
    # Fetch metrics
    # --------------------------------------------------------

    print("\nFetching metrics from Prometheus...")

    data = fetch_metrics()

    if data is None:
        return


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = to_dataframe(data)


    # --------------------------------------------------------
    # Check for empty result
    # --------------------------------------------------------

    if df.empty:

        print("\nWARNING: Prometheus returned no data.")

        print(
            "\nPossible reasons:"
        )

        print(
            "1. No CPU metrics exist in the selected time range."
        )

        print(
            "2. The load test has not generated traffic yet."
        )

        print(
            "3. The Prometheus query returned no series."
        )

        print(
            "\nTry running the following query directly "
            "in Prometheus:"
        )

        print(
            'sum(rate(container_cpu_usage_seconds_total'
            '{namespace="default"}[1m])) by (pod)'
        )

        return


    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------

    df = df.sort_values(
        by=["timestamp", "pod"]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    save_data(df)


    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


    # --------------------------------------------------------
    # Display basic statistics
    # --------------------------------------------------------

    print("\nPods recorded:")

    print(
        df["pod"]
        .drop_duplicates()
        .to_string(index=False)
    )

    print("\nCPU statistics:")

    print(
        df["cpu_usage"].describe()
    )


    print("\nDone!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
