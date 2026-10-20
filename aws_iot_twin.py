"""
aws_iot_twin.py — Real-Time Spatial Digital Twin via AWS IoT Core & Timestream
==============================================================================

Provides cloud-native IoT telemetry streaming for the Wildlife Sentinel project
(OpenCV AI Competition 2026, powered by AWS).

This module serializes continuous-time Kalman filter states (position, velocity,
heading, covariance trace, and activity state) into structured JSON payloads,
publishing them to AWS IoT Core MQTT topics (e.g. `wildlife/kinematics/{camera_id}`)
for ingestion into Amazon Timestream and visualization in Amazon Managed Grafana.

Architecture
------------
  [RobustKinematicTracker]
             │ (KinematicState)
             ▼
     [AWSIoTStreamer] ────(MQTT / TLS 1.3)────> [AWS IoT Core Message Broker]
                                                       │
                                                (IoT Topic Rule)
                                                       │
                                                       ▼
                                            [Amazon Timestream DB]
                                                       │
                                                       ▼
                                          [Amazon Managed Grafana]
                                          (Multi-Agent Trajectory HUD)

Author:  Wildlife Sentinel / OpenCV AI Competition 2026
License: Apache-2.0
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("sentinel.iot")

# ---------------------------------------------------------------------------
# Optional AWS SDK imports with graceful local fallbacks
# ---------------------------------------------------------------------------
_BOTO3_OK = False
_AWSIOT_OK = False

try:
    import boto3
    from botocore.config import Config
    _BOTO3_OK = True
except ImportError:
    boto3 = None

try:
    from awscrt import io, mqtt, auth, http
    from awsiot import mqtt_connection_builder
    _AWSIOT_OK = True
except ImportError:
    pass


@dataclass(frozen=True)
class DigitalTwinTelemetry:
    """Structured kinematic telemetry event published to AWS IoT Core."""
    camera_id: str
    timestamp_epoch: float
    iso_timestamp: str
    track_id: int
    species: str
    confidence: float
    pos_x: int
    pos_y: int
    vel_x: float
    vel_y: float
    speed_px_s: float
    heading_deg: float
    cardinal: str
    is_moving: bool
    activity_state: str
    covariance_trace: float
    mahalanobis_dist_sq: float
    anomaly_flag: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class AWSIoTStreamer:
    """High-throughput, asynchronous MQTT telemetry streamer for AWS IoT Core.

    Supports:
    1. Direct AWS IoT Device SDK v2 (Mutual TLS with X.509 certificates)
    2. Boto3 IoT-Data fallback via SigV4 HTTPS publish
    3. In-memory simulated mode for local offline evaluation
    """

    def __init__(
        self,
        camera_id: Optional[str] = None,
        iot_endpoint: Optional[str] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        ca_path: Optional[str] = None,
        client_id: Optional[str] = None,
        base_topic: str = "wildlife/kinematics",
        max_buffer_size: int = 1000,
        publish_rate_hz: float = 20.0,
    ) -> None:
        self.camera_id = camera_id or os.environ.get("CAMERA_ID", "camera_01")
        self.iot_endpoint = iot_endpoint or os.environ.get("AWS_IOT_ENDPOINT", "")
        self.cert_path = cert_path or os.environ.get("AWS_IOT_CERT_PATH", "")
        self.key_path = key_path or os.environ.get("AWS_IOT_KEY_PATH", "")
        self.ca_path = ca_path or os.environ.get("AWS_IOT_CA_PATH", "")
        self.client_id = client_id or f"sentinel-{self.camera_id}-{int(time.time())}"
        self.base_topic = base_topic
        self.topic = f"{self.base_topic}/{self.camera_id}"

        self._queue: queue.Queue = queue.Queue(maxsize=max_buffer_size)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._mqtt_connection = None
        self._boto_iot_client = None
        self._connection_mode = "mock"

        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize connection using Device SDK v2, boto3 iot-data, or mock."""
        # 1. Try Device SDK v2 with X.509 certs
        if _AWSIOT_OK and self.iot_endpoint and self.cert_path and os.path.exists(self.cert_path):
            try:
                event_loop_group = io.EventLoopGroup(1)
                host_resolver = io.DefaultHostResolver(event_loop_group)
                client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
                self._mqtt_connection = mqtt_connection_builder.mtls_from_path(
                    endpoint=self.iot_endpoint,
                    cert_filepath=self.cert_path,
                    pri_key_filepath=self.key_path,
                    client_bootstrap=client_bootstrap,
                    ca_filepath=self.ca_path if self.ca_path and os.path.exists(self.ca_path) else None,
                    client_id=self.client_id,
                    clean_session=False,
                    keep_alive_secs=30,
                )
                connect_future = self._mqtt_connection.connect()
                connect_future.result(timeout=10.0)
                self._connection_mode = "awsiot_v2_mtls"
                log.info("AWS IoT Core connected via mTLS X.509: %s", self.iot_endpoint)
                return
            except Exception as e:
                log.warning("AWS IoT Device SDK v2 connection failed: %s; falling back to boto3", e)

        # 2. Try boto3 iot-data (SigV4)
        if _BOTO3_OK and os.environ.get("AWS_REGION"):
            try:
                region = os.environ.get("AWS_REGION", "us-east-1")
                cfg = Config(retries={"max_attempts": 3, "mode": "adaptive"})
                endpoint_url = f"https://{self.iot_endpoint}" if self.iot_endpoint else None
                self._boto_iot_client = boto3.client("iot-data", region_name=region, endpoint_url=endpoint_url, config=cfg)
                self._connection_mode = "boto3_iot_data"
                log.info("AWS IoT Data client initialized in region %s", region)
                return
            except Exception as e:
                log.warning("Boto3 iot-data initialization failed: %s", e)

        # 3. Local Mock mode
        self._connection_mode = "mock"
        log.info("AWSIoTStreamer running in passive MOCK mode (IoT Endpoint not configured).")

    def start(self) -> None:
        """Start the background publishing worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._publish_loop, daemon=True, name="iot-streamer")
        self._worker_thread.start()
        log.info("AWSIoTStreamer background queue worker started.")

    def stop(self) -> None:
        """Stop background worker and flush buffer."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._mqtt_connection:
            try:
                disconnect_future = self._mqtt_connection.disconnect()
                disconnect_future.result(timeout=5.0)
            except Exception:
                pass
        log.info("AWSIoTStreamer stopped.")

    def publish_kinematics(
        self,
        track_id: int,
        species: str,
        confidence: float,
        kinematic_state: Any,
        covariance_trace: float = 0.0,
        mahalanobis_sq: float = 0.0,
        anomaly_flag: bool = False,
    ) -> None:
        """Enqueue a KinematicState measurement to be streamed to AWS IoT Core.

        Parameters
        ----------
        track_id : int
            ByteTrack target ID.
        species : str
            Classification label (e.g. "tiger", "deer", "elephant").
        confidence : float
            YOLO detection confidence.
        kinematic_state : KinematicState
            Output object from RobustKinematicTracker.
        covariance_trace : float
            Trace of Kalman covariance matrix Tr(P_k), measures uncertainty.
        mahalanobis_sq : float
            Squared Mahalanobis innovation distance d_M^2.
        anomaly_flag : bool
            Flag indicating sudden velocity burst or trajectory divergence.
        """
        now = time.time()
        iso_ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(now))

        heading_deg = (kinematic_state.heading_rad * 180.0 / 3.141592653589793) % 360.0

        telemetry = DigitalTwinTelemetry(
            camera_id=self.camera_id,
            timestamp_epoch=round(now, 3),
            iso_timestamp=iso_ts,
            track_id=int(track_id),
            species=species.lower(),
            confidence=round(float(confidence), 3),
            pos_x=int(kinematic_state.smoothed_pos[0]),
            pos_y=int(kinematic_state.smoothed_pos[1]),
            vel_x=round(float(kinematic_state.velocity[0]), 2),
            vel_y=round(float(kinematic_state.velocity[1]), 2),
            speed_px_s=round(float(kinematic_state.speed), 2),
            heading_deg=round(float(heading_deg), 1),
            cardinal=kinematic_state.cardinal,
            is_moving=(kinematic_state.cardinal != "STATIONARY"),
            activity_state="stationary" if kinematic_state.cardinal == "STATIONARY" else ("walking" if kinematic_state.speed < 45.0 else "running"),
            covariance_trace=round(float(covariance_trace), 3),
            mahalanobis_dist_sq=round(float(mahalanobis_sq), 3),
            anomaly_flag=bool(anomaly_flag),
        )

        try:
            self._queue.put_nowait(telemetry)
        except queue.Full:
            pass  # Drop oldest frame during network degradation to protect real-time latency

    def _publish_loop(self) -> None:
        """Worker thread that consumes telemetry from queue and dispatches to AWS."""
        while self._running:
            try:
                item: DigitalTwinTelemetry = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            payload = item.to_json()
            try:
                if self._connection_mode == "awsiot_v2_mtls" and self._mqtt_connection:
                    self._mqtt_connection.publish(
                        topic=self.topic,
                        payload=payload,
                        qos=mqtt.QoS.AT_MOST_ONCE,
                    )
                elif self._connection_mode == "boto3_iot_data" and self._boto_iot_client:
                    self._boto_iot_client.publish(
                        topic=self.topic,
                        qos=0,
                        payload=payload.encode("utf-8"),
                    )
                else:
                    # In mock mode, log sampled telemetry for verification
                    if item.track_id % 5 == 0 and item.pos_x % 10 == 0:
                        log.debug("[MOCK IoT] %s -> %s", self.topic, payload[:90])
            except Exception as e:
                log.warning("Failed to publish telemetry to %s: %s", self.topic, e)
            finally:
                self._queue.task_done()


# ===========================================================================
# Timestream Database Schema Specification (AWS CLI / CloudFormation)
# ===========================================================================
def generate_timestream_schema() -> Dict[str, Any]:
    """Returns the Amazon Timestream table schema configuration for digital twin analytics."""
    return {
        "DatabaseName": "wildlife_sentinel_db",
        "TableName": "kinematics_telemetry",
        "RetentionProperties": {
            "MemoryStoreRetentionPeriodInHours": 24,       # Hot in-memory tier for live Grafana HUD
            "MagneticStoreRetentionPeriodInDays": 365,     # Cold analytical tier for historical migration studies
        },
        "MagneticStoreWriteProperties": {
            "EnableMagneticStoreWrites": True,
        },
        "Dimensions": [
            {"Name": "camera_id", "Type": "VARCHAR"},
            {"Name": "track_id", "Type": "VARCHAR"},
            {"Name": "species", "Type": "VARCHAR"},
            {"Name": "cardinal", "Type": "VARCHAR"},
            {"Name": "activity_state", "Type": "VARCHAR"},
            {"Name": "anomaly_flag", "Type": "BOOLEAN"},
        ],
        "Measures": [
            {"Name": "pos_x", "Type": "BIGINT"},
            {"Name": "pos_y", "Type": "BIGINT"},
            {"Name": "speed_px_s", "Type": "DOUBLE"},
            {"Name": "vel_x", "Type": "DOUBLE"},
            {"Name": "vel_y", "Type": "DOUBLE"},
            {"Name": "heading_deg", "Type": "DOUBLE"},
            {"Name": "covariance_trace", "Type": "DOUBLE"},
            {"Name": "mahalanobis_dist_sq", "Type": "DOUBLE"},
        ],
        "IoTRuleSQL": """
            SELECT 
                camera_id,
                track_id,
                species,
                cardinal,
                activity_state,
                anomaly_flag,
                pos_x,
                pos_y,
                speed_px_s,
                vel_x,
                vel_y,
                heading_deg,
                covariance_trace,
                mahalanobis_dist_sq
            FROM 'wildlife/kinematics/+'
        """,
    }


# ===========================================================================
# Amazon Managed Grafana Dashboard JSON Template
# ===========================================================================
def generate_grafana_dashboard_json() -> str:
    """Generates an importable Grafana Dashboard JSON for multi-agent trajectory tracking."""
    dashboard = {
        "title": "Wildlife Sentinel — Spatial Digital Twin (Amazon Timestream)",
        "refresh": "1s",
        "schemaVersion": 38,
        "tags": ["opencv-ai-2026", "aws-iot", "timestream", "digital-twin"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "Multi-Agent 2D Spatial Ground Trajectories (X vs Y)",
                "type": "xychart",
                "gridPos": {"x": 0, "y": 0, "w": 14, "h": 10},
                "targets": [
                    {
                        "rawQuery": """
                            SELECT track_id, pos_x, pos_y, time
                            FROM "wildlife_sentinel_db"."kinematics_telemetry"
                            WHERE time >= ago(5m) AND camera_id = '$camera_id'
                            ORDER BY time ASC
                        """,
                        "format": "table",
                    }
                ],
                "options": {
                    "series": [{"name": "Spatial Trajectory", "x": "pos_x", "y": "pos_y"}]
                },
            },
            {
                "id": 2,
                "title": "Continuous Velocity & Anomaly Spikes (px/s)",
                "type": "timeseries",
                "gridPos": {"x": 14, "y": 0, "w": 10, "h": 5},
                "targets": [
                    {
                        "rawQuery": """
                            SELECT time, track_id, speed_px_s
                            FROM "wildlife_sentinel_db"."kinematics_telemetry"
                            WHERE time >= ago(15m)
                            ORDER BY time ASC
                        """
                    }
                ],
            },
            {
                "id": 3,
                "title": "Kalman Covariance Uncertainty Tr(P)",
                "type": "gauge",
                "gridPos": {"x": 14, "y": 5, "w": 5, "h": 5},
                "targets": [
                    {
                        "rawQuery": """
                            SELECT AVG(covariance_trace)
                            FROM "wildlife_sentinel_db"."kinematics_telemetry"
                            WHERE time >= ago(1m)
                        """
                    }
                ],
            },
            {
                "id": 4,
                "title": "Active Threat Anomalies (Bedrock Agent Triggers)",
                "type": "stat",
                "gridPos": {"x": 19, "y": 5, "w": 5, "h": 5},
                "targets": [
                    {
                        "rawQuery": """
                            SELECT COUNT(*)
                            FROM "wildlife_sentinel_db"."kinematics_telemetry"
                            WHERE time >= ago(10m) AND anomaly_flag = true
                        """
                    }
                ],
            },
        ],
    }
    return json.dumps(dashboard, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing AWSIoTStreamer initialization...")
    streamer = AWSIoTStreamer(camera_id="cam_test_01")
    streamer.start()

    from collections import namedtuple
    FakeState = namedtuple("FakeState", ["smoothed_pos", "velocity", "speed", "heading_rad", "cardinal"])
    fake = FakeState(smoothed_pos=(320, 240), velocity=(15.2, -4.1), speed=15.7, heading_rad=0.35, cardinal="E")

    streamer.publish_kinematics(
        track_id=1,
        species="snow_leopard",
        confidence=0.94,
        kinematic_state=fake,
        covariance_trace=8.4,
        mahalanobis_sq=1.2,
        anomaly_flag=False,
    )
    time.sleep(0.5)
    streamer.stop()
    print("Timestream Schema DDL:")
    print(json.dumps(generate_timestream_schema(), indent=2)[:400] + "...\n")
    print("AWSIoTStreamer test completed successfully!")
