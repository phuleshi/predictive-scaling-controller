import os
import time
import math
import json
import pickle
import logging
from collections import deque
from datetime import datetime, timezone
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

import requests
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://127.0.0.1:9090"
).rstrip("/")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "rps_prediction_model.pkl"
)

FEATURE_CONFIG_PATH = os.getenv(
    "FEATURE_CONFIG_PATH",
    "feature_config.json"
)

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))
PROMETHEUS_TIMEOUT = int(os.getenv("PROMETHEUS_TIMEOUT", "10"))

FORECAST_HORIZON_SECONDS = int(
    os.getenv("FORECAST_HORIZON_SECONDS", "10")
)

# Scaling policy
POD_CAPACITY = 120
SAFETY_FACTOR = 1.2
MIN_REPLICAS = 2
MAX_REPLICAS = 6

# Kubernetes scaling
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "exam-platform")
K8S_DEPLOYMENT = os.getenv("K8S_DEPLOYMENT", "backend")

SCALE_DOWN_CONFIRMATIONS = 3
SCALE_DOWN_COOLDOWN = 60

apps_api = None

EXPECTED_FEATURES = [
    "rps",
    "rps_lag_1",
    "rps_lag_2",
    "rps_lag_3",
    "rps_rolling_3",
    "rps_rolling_6",
    "cpu",
    "memory",
    "pod_count",
]

# Keep these queries identical to the metric collector used for training.
PROMQL = {
    "rps": """
        sum(
            rate(
                http_server_requests_seconds_count{
                    job="backend",
                    uri="/auth/test-heavy"
                }[1m]
            )
        )
    """,
    "cpu": """
        sum(
            rate(
                container_cpu_usage_seconds_total{
                    namespace="exam-platform",
                    container="backend"
                }[1m]
            )
        )
    """,
    "memory": """
        sum(
            container_memory_working_set_bytes{
                namespace="exam-platform",
                container="backend"
            }
        )
    """,
    "pod_count": """
        count(
            kube_pod_status_phase{
                namespace="exam-platform",
                phase="Running",
                pod=~"backend-.*"
            }
        )
    """,
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("predictive-controller")


# ============================================================
# LOAD MODEL
# ============================================================

# ============================================================
# LOAD MODEL
# ============================================================

def load_artifacts():

    # Kiểm tra model
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}"
        )

    # Kiểm tra feature config
    if not os.path.isfile(FEATURE_CONFIG_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy feature config: {FEATURE_CONFIG_PATH}"
        )

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load feature config
    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Lấy danh sách features và forecast horizon
    if not isinstance(config_data, dict):
        raise ValueError(
            "feature_config.json phải có định dạng JSON object."
        )

    features = config_data.get("features")

    forecast_horizon = config_data.get(
        "forecast_horizon_seconds"
    )

    # Kiểm tra features
    if not isinstance(features, list):
        raise ValueError(
            "feature_config.json phải chứa danh sách features."
        )

    if features != EXPECTED_FEATURES:
        raise ValueError(
            "Feature config không khớp model.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Loaded: {features}"
        )

    # Kiểm tra forecast horizon
    if forecast_horizon is None:
        raise ValueError(
            "feature_config.json thiếu forecast_horizon_seconds."
        )

    try:
        forecast_horizon = int(forecast_horizon)
    except (TypeError, ValueError):
        raise ValueError(
            "forecast_horizon_seconds phải là số nguyên."
        )

    if forecast_horizon <= 0:
        raise ValueError(
            f"Forecast horizon không hợp lệ: {forecast_horizon}"
        )

    if forecast_horizon != FORECAST_HORIZON_SECONDS:
        raise ValueError(
            "Forecast horizon không khớp cấu hình controller.\n"
            f"Expected: {FORECAST_HORIZON_SECONDS}s\n"
            f"Loaded: {forecast_horizon}s"
        )

    # Log thông tin
    logger.info("Loaded model: %s", MODEL_PATH)
    logger.info(
        "Loaded %d model features.",
        len(features)
    )
    logger.info(
        "Forecast horizon: %ss",
        forecast_horizon
    )

    return model, features


# ============================================================
# PROMETHEUS
# ============================================================

def query_prometheus(query, retries=3, retry_delay=2):
    url = f"{PROMETHEUS_URL}/api/v1/query"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params={"query": query},
                timeout=PROMETHEUS_TIMEOUT,
            )
            response.raise_for_status()

            payload = response.json()
            if payload.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed: {payload}")

            results = payload.get("data", {}).get("result", [])
            if not results:
                return None

            value = results[0]["value"][1]
            if value in ("NaN", "Inf", "+Inf", "-Inf"):
                return None

            number = float(value)
            if not math.isfinite(number):
                return None

            return number

        except Exception as exc:
            logger.warning(
                "Prometheus request failed (%d/%d): %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(retry_delay)

    return None


def collect_live_metrics():
    metrics = {}

    for name, query in PROMQL.items():
        metrics[name] = query_prometheus(query)

    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        raise RuntimeError(
            "Thiếu metrics bắt buộc từ Prometheus: " + ", ".join(missing)
        )

    metrics["pod_count"] = int(metrics["pod_count"])

    if metrics["pod_count"] < 1:
        raise RuntimeError("pod_count không hợp lệ.")

    return metrics


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_row(metrics, rps_history, feature_names):
    """
    rps_history chứa tối đa 5 mẫu RPS trước mẫu hiện tại.
    Thứ tự: cũ -> mới.
    """

    previous = list(rps_history)

    if len(previous) < 5:
        raise ValueError(
            f"Cần tối thiểu 5 mẫu RPS trước đó; hiện có {len(previous)}."
        )

    current_rps = float(metrics["rps"])

    lag_1 = float(previous[-1])
    lag_2 = float(previous[-2])
    lag_3 = float(previous[-3])

    last_3 = previous[-2:] + [current_rps]
    last_6 = previous[-5:] + [current_rps]

    values = {
        "rps": current_rps,
        "rps_lag_1": lag_1,
        "rps_lag_2": lag_2,
        "rps_lag_3": lag_3,
        "rps_rolling_3": float(np.mean(last_3)),
        "rps_rolling_6": float(np.mean(last_6)),
        "cpu": float(metrics["cpu"]),
        "memory": float(metrics["memory"]),
        "pod_count": int(metrics["pod_count"]),
    }

    return pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names,
    )

# ============================================================
# REPLICA CALCULATION
# ============================================================

def calculate_scaling_decision(predicted_rps):
    predicted_rps = float(predicted_rps)

    if not math.isfinite(predicted_rps) or predicted_rps < 0:
        raise ValueError(
            f"Predicted RPS không hợp lệ: {predicted_rps}"
        )

    # Bù sai số dự đoán bằng safety factor
    safe_rps = predicted_rps * SAFETY_FACTOR

    # Tính số replica cần thiết
    required_replicas = math.ceil(
        safe_rps / POD_CAPACITY
    )

    # Giới hạn min/max replicas
    target_replicas = max(
        MIN_REPLICAS,
        min(MAX_REPLICAS, required_replicas),
    )

    return {
        "predicted_rps": predicted_rps,
        "safe_rps": safe_rps,
        "required_replicas": required_replicas,
        "target_replicas": target_replicas,
        "capacity_at_target": target_replicas * POD_CAPACITY,
        "capacity_limited": required_replicas > MAX_REPLICAS,
    }

# ============================================================
# KUBERNETES API
# ============================================================

def initialize_kubernetes():
    global apps_api

    # Controller chạy bên trong Kubernetes.
    config.load_incluster_config()

    apps_api = client.AppsV1Api()

    deployment = apps_api.read_namespaced_deployment(
        name=K8S_DEPLOYMENT,
        namespace=K8S_NAMESPACE,
    )

    logger.info(
        "Connected to Kubernetes: namespace=%s, deployment=%s, replicas=%s",
        K8S_NAMESPACE,
        K8S_DEPLOYMENT,
        deployment.spec.replicas,
    )


def get_current_replicas():
    deployment = apps_api.read_namespaced_deployment(
        name=K8S_DEPLOYMENT,
        namespace=K8S_NAMESPACE,
    )

    current = deployment.spec.replicas or 0
    ready = deployment.status.ready_replicas or 0

    return current, ready


def scale_deployment(target_replicas):
    target_replicas = max(
        MIN_REPLICAS,
        min(MAX_REPLICAS, int(target_replicas)),
    )

    body = {
        "spec": {
            "replicas": target_replicas
        }
    }

    apps_api.patch_namespaced_deployment_scale(
        name=K8S_DEPLOYMENT,
        namespace=K8S_NAMESPACE,
        body=body,
    )

    logger.info(
        "KUBERNETES SCALE REQUEST: %s replicas -> %s replicas",
        K8S_DEPLOYMENT,
        target_replicas,
    )

# ============================================================
# CONTROLLER LOOP
# ============================================================

def run_controller():
    global apps_api

    model, feature_names = load_artifacts()

    # Kết nối Kubernetes
    initialize_kubernetes()

    rps_history = deque(maxlen=5)

    consecutive_scale_down = 0
    last_scale_time = 0.0

    logger.info("==============================================")
    logger.info("       Predictive Auto Scaling Controller")
    logger.info("==============================================")
    logger.info("Prometheus URL : %s", PROMETHEUS_URL)
    logger.info("Poll interval  : %ss", POLL_INTERVAL)
    logger.info("Forecast horizon: %ss", FORECAST_HORIZON_SECONDS)
    logger.info("Namespace      : %s", K8S_NAMESPACE)
    logger.info("Deployment     : %s", K8S_DEPLOYMENT)
    logger.info(
        "Policy         : capacity=%s, safety=%s, replicas=%s-%s",
        POD_CAPACITY,
        SAFETY_FACTOR,
        MIN_REPLICAS,
        MAX_REPLICAS,
    )
    logger.info("Scale-down confirmation: %s cycles", SCALE_DOWN_CONFIRMATIONS)
    logger.info("Scale-down cooldown: %ss", SCALE_DOWN_COOLDOWN)
    logger.info("Mode           : ACTIVE KUBERNETES SCALING")

    while True:
        loop_start = time.monotonic()

        try:
            metrics = collect_live_metrics()
            current_rps = float(metrics["rps"])

            if len(rps_history) < 5:
                rps_history.append(current_rps)

                logger.info(
                    "Warming up: %d/5 samples | RPS=%.2f",
                    len(rps_history),
                    current_rps,
                )

            else:
                feature_row = build_feature_row(
                    metrics,
                    rps_history,
                    feature_names,
                )

                prediction = model.predict(feature_row)
                predicted_rps = float(prediction[0])

                decision = calculate_scaling_decision(predicted_rps)

                # Đọc replica thực tế từ Kubernetes, không dùng pod_count
                # của Prometheus để quyết định replica hiện tại.
                current_replicas, ready_replicas = get_current_replicas()

                target_replicas = decision["target_replicas"]
                action = "NO CHANGE"

                # ==========================================
                # SCALE UP: thực hiện ngay
                # ==========================================

                if target_replicas > current_replicas:
                    consecutive_scale_down = 0

                    scale_deployment(target_replicas)
                    last_scale_time = time.monotonic()

                    action = f"SCALE UP: {current_replicas} -> {target_replicas}"

                # ==========================================
                # SCALE DOWN: cần xác nhận và cooldown
                # ==========================================

                elif target_replicas < current_replicas:
                    consecutive_scale_down += 1

                    elapsed_since_scale = (
                        time.monotonic() - last_scale_time
                    )

                    if (
                        consecutive_scale_down >= SCALE_DOWN_CONFIRMATIONS
                        and elapsed_since_scale >= SCALE_DOWN_COOLDOWN
                    ):
                        scale_deployment(target_replicas)

                        last_scale_time = time.monotonic()
                        consecutive_scale_down = 0

                        action = (
                            f"SCALE DOWN: {current_replicas} "
                            f"-> {target_replicas}"
                        )
                    else:
                        action = (
                            f"WAIT SCALE DOWN "
                            f"({consecutive_scale_down}/"
                            f"{SCALE_DOWN_CONFIRMATIONS}), "
                            f"cooldown={max(0, SCALE_DOWN_COOLDOWN - elapsed_since_scale):.0f}s"
                        )

                else:
                    consecutive_scale_down = 0

                logger.info(
                    "RPS=%.2f | Predicted=%.2f | Safe RPS=%.2f | "
                    "Required=%d | Target=%d | Capacity=%d | "
                    "Current=%d | Ready=%d | %s",
                    current_rps,
                    decision["predicted_rps"],
                    decision["safe_rps"],
                    decision["required_replicas"],
                    decision["target_replicas"],
                    decision["capacity_at_target"],
                    current_replicas,
                    ready_replicas,
                    action,
                )

                if decision["capacity_limited"]:
                    logger.warning(
                        "MAX REPLICAS REACHED: Safe RPS=%.2f exceeds max capacity=%d RPS",
                        decision["safe_rps"],
                        MAX_REPLICAS * POD_CAPACITY,
                    )

                rps_history.append(current_rps)

        except KeyboardInterrupt:
            raise

        except ApiException as exc:
            logger.exception(
                "Kubernetes API error: status=%s, reason=%s",
                exc.status,
                exc.reason,
            )

        except Exception as exc:
            logger.exception("Controller iteration failed: %s", exc)

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0, POLL_INTERVAL - elapsed))


def main():
    try:
        run_controller()
    except KeyboardInterrupt:
        logger.info("Controller stopped by user.")


if __name__ == "__main__":
    main()
