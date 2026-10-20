"""
aws_integration.py — AWS Integration Module for Wildlife Sentinel
=================================================================

Provides cloud-native AWS service integrations for the Wildlife Sentinel
project (OpenCV AI Competition 2026).  This module is fully standalone and
imports nothing from the main application code-base so it can be deployed
independently (e.g. inside an AWS Lambda package).

Services covered
----------------
* **Amazon S3**          – Upload analysed video clips, detection snapshots,
                           and JSON analysis reports.
* **AWS Rekognition**    – Secondary species verification / classification to
                           supplement local YOLO detections.
* **Amazon DynamoDB**    – Structured metadata logging of detection events
                           (timestamp, species, count, confidence, GPS, …).
* **Amazon SNS**         – Cloud-native push notifications as an alternative
                           to Telegram / email alerting.
* **AWS Lambda**         – A ``lambda_handler`` entry-point stub for
                           triggering analysis jobs from Lambda invocations.

Configuration
-------------
All settings are read from environment variables.  A master toggle
``USE_AWS`` (default ``False``) gates every outbound AWS call so the rest of
the application can run without AWS credentials during local development.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
    USE_AWS              Master on/off toggle (``"true"`` to enable).
    AWS_REGION           AWS region name          (default ``"us-east-1"``).
    AWS_S3_BUCKET        S3 bucket for uploads    (required when enabled).
    AWS_DYNAMODB_TABLE   DynamoDB table name      (required when enabled).
    AWS_SNS_TOPIC_ARN    SNS topic ARN for alerts (required when enabled).
    AWS_REKOGNITION_MIN_CONFIDENCE
                         Minimum Rekognition confidence % (default ``70``).
    AWS_S3_PREFIX        Key prefix for all S3 uploads   (default ``""``).
    AWS_RETRY_MAX        Maximum boto3 retry attempts     (default ``3``).

Usage
-----
>>> from aws_integration import init_aws, upload_to_s3, log_detection_event
>>> init_aws()                           # validates config, creates clients
>>> url = upload_to_s3("clip.mp4", "videos/clip.mp4")
>>> log_detection_event({"species": "Amur Leopard", "confidence": 0.97, ...})
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("wildlife_sentinel.aws")
logger.addHandler(logging.NullHandler())  # library-friendly default

# ---------------------------------------------------------------------------
# Configuration dataclass (populated by init_aws)
# ---------------------------------------------------------------------------

class _AWSConfig:
    """Internal container for validated AWS configuration values."""

    def __init__(self) -> None:
        self.use_aws: bool = False
        self.region: str = "us-east-1"
        self.s3_bucket: str = ""
        self.s3_prefix: str = ""
        self.dynamodb_table: str = ""
        self.sns_topic_arn: str = ""
        self.rekognition_min_confidence: float = 70.0
        self.retry_max: int = 3

        # boto3 service clients (created lazily by init_aws)
        self.s3_client: Any = None
        self.dynamodb_resource: Any = None
        self.dynamodb_table_resource: Any = None
        self.sns_client: Any = None
        self.rekognition_client: Any = None

        self._initialised: bool = False

    # Convenience -----------------------------------------------------------
    @property
    def initialised(self) -> bool:
        return self._initialised

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<_AWSConfig use_aws={self.use_aws} region={self.region!r} "
            f"bucket={self.s3_bucket!r} table={self.dynamodb_table!r} "
            f"topic={self.sns_topic_arn!r} initialised={self._initialised}>"
        )


# Module-level singleton
_cfg = _AWSConfig()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable, stripping surrounding whitespace."""
    val = os.environ.get(name)
    if val is not None:
        return val.strip()
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean flag.

    Truthy values: ``"1"``, ``"true"``, ``"yes"`` (case-insensitive).
    """
    val = _env(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_float(name: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, falling back to *default*."""
    val = _env(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float for %s=%r — using default %s", name, val, default)
        return default


def _env_int(name: str, default: int = 0) -> int:
    """Read an environment variable as an int, falling back to *default*."""
    val = _env(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for %s=%r — using default %d", name, val, default)
        return default


def _utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _guard() -> None:
    """Raise if AWS integration has not been initialised or is disabled."""
    if not _cfg.use_aws:
        raise RuntimeError(
            "AWS integration is disabled (USE_AWS is not set to true)."
        )
    if not _cfg.initialised:
        raise RuntimeError(
            "AWS integration has not been initialised — call init_aws() first."
        )

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_aws() -> bool:
    """Validate configuration and create boto3 service clients.

    Reads all ``AWS_*`` environment variables, validates that required values
    are present when ``USE_AWS`` is enabled, and creates boto3 clients with
    retry configuration.

    Returns
    -------
    bool
        ``True`` if AWS integration is now active, ``False`` if it was
        disabled or if initialisation failed gracefully.

    Raises
    ------
    ImportError
        If ``boto3`` is not installed and ``USE_AWS`` is ``True``.
    ValueError
        If a required environment variable is missing and ``USE_AWS`` is
        ``True``.
    """
    _cfg.use_aws = _env_bool("USE_AWS", default=False)

    if not _cfg.use_aws:
        logger.info("AWS integration is disabled (USE_AWS != true).")
        return False

    # ------------------------------------------------------------------
    # Import boto3 only when actually needed
    # ------------------------------------------------------------------
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS integration.  "
            "Install it with:  pip install boto3"
        ) from exc

    # ------------------------------------------------------------------
    # Read and validate environment variables
    # ------------------------------------------------------------------
    _cfg.region = _env("AWS_REGION", "us-east-1") or "us-east-1"
    _cfg.s3_bucket = _env("AWS_S3_BUCKET", "") or ""
    _cfg.s3_prefix = _env("AWS_S3_PREFIX", "") or ""
    _cfg.dynamodb_table = _env("AWS_DYNAMODB_TABLE", "") or ""
    _cfg.sns_topic_arn = _env("AWS_SNS_TOPIC_ARN", "") or ""
    _cfg.rekognition_min_confidence = _env_float(
        "AWS_REKOGNITION_MIN_CONFIDENCE", 70.0
    )
    _cfg.retry_max = _env_int("AWS_RETRY_MAX", 3)

    missing: List[str] = []
    if not _cfg.s3_bucket:
        missing.append("AWS_S3_BUCKET")
    if not _cfg.dynamodb_table:
        missing.append("AWS_DYNAMODB_TABLE")
    if not _cfg.sns_topic_arn:
        missing.append("AWS_SNS_TOPIC_ARN")
    if missing:
        raise ValueError(
            "The following required AWS environment variables are missing: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------
    # Build boto3 clients with retry config
    # ------------------------------------------------------------------
    retry_config = BotoConfig(
        region_name=_cfg.region,
        retries={"max_attempts": _cfg.retry_max, "mode": "adaptive"},
    )

    try:
        _cfg.s3_client = boto3.client("s3", config=retry_config)
        _cfg.rekognition_client = boto3.client("rekognition", config=retry_config)
        _cfg.sns_client = boto3.client("sns", config=retry_config)
        _cfg.dynamodb_resource = boto3.resource("dynamodb", config=retry_config)
        _cfg.dynamodb_table_resource = _cfg.dynamodb_resource.Table(
            _cfg.dynamodb_table
        )
    except Exception:
        logger.exception("Failed to create boto3 clients.")
        raise

    _cfg._initialised = True
    logger.info("AWS integration initialised successfully.  %r", _cfg)
    return True


# ---------------------------------------------------------------------------
# S3 — Upload artefacts
# ---------------------------------------------------------------------------

def upload_to_s3(
    local_path: str,
    s3_key: str,
    *,
    content_type: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Upload a local file to Amazon S3.

    Parameters
    ----------
    local_path : str
        Absolute or relative path to the file on the local filesystem.
    s3_key : str
        Object key (path) inside the S3 bucket.  If ``AWS_S3_PREFIX`` is
        set, it is automatically prepended.
    content_type : str, optional
        MIME type override.  When omitted boto3 will guess from the
        extension.
    extra_args : dict, optional
        Additional keyword arguments forwarded to
        ``S3.Client.upload_file``.

    Returns
    -------
    str
        The public or virtual-hosted S3 URL of the uploaded object,
        e.g. ``https://<bucket>.s3.amazonaws.com/<key>``.

    Raises
    ------
    FileNotFoundError
        If *local_path* does not exist.
    RuntimeError
        If AWS integration is disabled or uninitialised.
    botocore.exceptions.ClientError
        On S3 upload failure after retries.
    """
    _guard()

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Prepend the configured prefix (if any) to the key
    full_key = f"{_cfg.s3_prefix}/{s3_key}".lstrip("/") if _cfg.s3_prefix else s3_key

    upload_extra: Dict[str, Any] = extra_args.copy() if extra_args else {}
    if content_type:
        upload_extra["ContentType"] = content_type

    logger.info("Uploading %s → s3://%s/%s", local_path, _cfg.s3_bucket, full_key)

    try:
        _cfg.s3_client.upload_file(
            Filename=str(path),
            Bucket=_cfg.s3_bucket,
            Key=full_key,
            ExtraArgs=upload_extra or None,
        )
    except Exception:
        logger.exception(
            "S3 upload failed for %s → s3://%s/%s",
            local_path,
            _cfg.s3_bucket,
            full_key,
        )
        raise

    s3_url = (
        f"https://{_cfg.s3_bucket}.s3.{_cfg.region}.amazonaws.com/{full_key}"
    )
    logger.info("Upload complete: %s", s3_url)
    return s3_url


# ---------------------------------------------------------------------------
# Rekognition — Secondary species verification
# ---------------------------------------------------------------------------

def rekognition_verify(
    image_bytes: bytes,
    local_labels: Optional[List[Dict[str, Any]]] = None,
    *,
    max_labels: int = 20,
) -> List[Dict[str, Any]]:
    """Run AWS Rekognition label detection and merge with local YOLO labels.

    This function calls ``DetectLabels`` on the provided image and returns a
    unified list of enriched label dictionaries.  Each entry contains:

    * ``Name``       – species or object name
    * ``Confidence`` – Rekognition confidence (0–100)
    * ``Source``     – ``"rekognition"``, ``"local"``, or ``"both"``
    * ``Parents``    – Rekognition taxonomy parents (if available)
    * ``LocalConfidence`` – original local model confidence (if matched)

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (JPEG or PNG) to send to Rekognition.
    local_labels : list of dict, optional
        Labels produced by the local YOLO detector.  Each dict should have
        at least a ``"name"`` (str) and ``"confidence"`` (float 0–1) key.
    max_labels : int
        Maximum number of labels to request from Rekognition (default 20).

    Returns
    -------
    list of dict
        Enriched label list as described above.

    Raises
    ------
    RuntimeError
        If AWS integration is disabled or uninitialised.
    botocore.exceptions.ClientError
        On Rekognition API failure.
    """
    _guard()

    if local_labels is None:
        local_labels = []

    # Build a lookup of local labels keyed by lower-cased name for merging
    local_lookup: Dict[str, Dict[str, Any]] = {
        lbl.get("name", "").lower(): lbl for lbl in local_labels
    }

    logger.info(
        "Calling Rekognition DetectLabels (min_confidence=%.1f, max_labels=%d)",
        _cfg.rekognition_min_confidence,
        max_labels,
    )

    try:
        response = _cfg.rekognition_client.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=max_labels,
            MinConfidence=_cfg.rekognition_min_confidence,
        )
    except Exception:
        logger.exception("Rekognition DetectLabels call failed.")
        raise

    # ---- Merge Rekognition results with local labels ----------------------
    enriched: List[Dict[str, Any]] = []
    seen_names: set = set()

    for rek_label in response.get("Labels", []):
        name = rek_label.get("Name", "Unknown")
        name_lower = name.lower()
        seen_names.add(name_lower)

        entry: Dict[str, Any] = {
            "Name": name,
            "Confidence": rek_label.get("Confidence", 0.0),
            "Parents": [p.get("Name", "") for p in rek_label.get("Parents", [])],
            "Instances": rek_label.get("Instances", []),
            "Source": "rekognition",
        }

        # If the local model also detected this label, mark as "both"
        if name_lower in local_lookup:
            local_conf = local_lookup[name_lower].get("confidence", 0.0)
            entry["Source"] = "both"
            entry["LocalConfidence"] = local_conf

        enriched.append(entry)

    # Append any local-only labels that Rekognition did not return
    for name_lower, lbl in local_lookup.items():
        if name_lower not in seen_names:
            enriched.append(
                {
                    "Name": lbl.get("name", name_lower),
                    "Confidence": None,
                    "Parents": [],
                    "Instances": [],
                    "Source": "local",
                    "LocalConfidence": lbl.get("confidence", 0.0),
                }
            )

    logger.info(
        "Rekognition returned %d labels; merged total: %d",
        len(response.get("Labels", [])),
        len(enriched),
    )
    return enriched


# ---------------------------------------------------------------------------
# DynamoDB — Detection event logging
# ---------------------------------------------------------------------------

def log_detection_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Store a detection event in Amazon DynamoDB.

    The function automatically adds:

    * ``event_id``   – a UUID-v4 primary key (if not already present).
    * ``logged_at``  – ISO-8601 UTC timestamp of when the record was written.

    Parameters
    ----------
    event_data : dict
        Arbitrary detection metadata.  Recommended keys include:

        * ``timestamp``  (str)   – ISO-8601 capture time.
        * ``species``    (str)   – detected species name.
        * ``count``      (int)   – number of individuals.
        * ``confidence`` (float) – model confidence score (0–1).
        * ``location``   (dict)  – ``{"lat": …, "lon": …}`` GPS coordinates.
        * ``source``     (str)   – ``"yolo"``, ``"rekognition"``, etc.
        * ``video_key``  (str)   – S3 key of the source video clip.
        * ``snapshot_key`` (str) – S3 key of the detection snapshot image.

    Returns
    -------
    dict
        The item as written to DynamoDB (including generated fields).

    Raises
    ------
    RuntimeError
        If AWS integration is disabled or uninitialised.
    botocore.exceptions.ClientError
        On DynamoDB put failure.
    """
    _guard()

    item = dict(event_data)  # shallow copy to avoid mutating caller's dict

    # Ensure a unique primary key
    if "event_id" not in item:
        item["event_id"] = str(uuid.uuid4())

    # Server-side write timestamp
    item["logged_at"] = _utc_iso()

    # DynamoDB does not support float directly in all SDKs — convert to
    # Decimal via JSON round-trip is one approach, but boto3's resource
    # interface handles Python floats transparently when using
    # ``TypeDeserializer``/``TypeSerializer``.  We convert floats to strings
    # wrapped in Decimal for safety.
    from decimal import Decimal

    def _sanitise(obj: Any) -> Any:
        """Recursively convert floats to Decimal for DynamoDB compat."""
        if isinstance(obj, float):
            return Decimal(str(obj))
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitise(v) for v in obj]
        return obj

    safe_item = _sanitise(item)

    logger.info(
        "Logging detection event %s to DynamoDB table %s",
        item["event_id"],
        _cfg.dynamodb_table,
    )

    try:
        _cfg.dynamodb_table_resource.put_item(Item=safe_item)
    except Exception:
        logger.exception(
            "DynamoDB put_item failed for event %s", item["event_id"]
        )
        raise

    logger.info("Detection event %s logged successfully.", item["event_id"])
    return item


# ---------------------------------------------------------------------------
# SNS — Cloud-native alerting
# ---------------------------------------------------------------------------

def send_sns_alert(subject: str, message: str) -> str:
    """Publish an alert message to the configured Amazon SNS topic.

    Parameters
    ----------
    subject : str
        Short subject line (≤100 characters for SMS compatibility).
    message : str
        Full notification body.

    Returns
    -------
    str
        The SNS ``MessageId`` of the published message.

    Raises
    ------
    RuntimeError
        If AWS integration is disabled or uninitialised.
    botocore.exceptions.ClientError
        On SNS publish failure.
    """
    _guard()

    # SNS subject must be ≤ 100 characters
    safe_subject = subject[:100] if len(subject) > 100 else subject

    logger.info(
        "Publishing SNS alert to %s — subject: %s",
        _cfg.sns_topic_arn,
        safe_subject,
    )

    try:
        response = _cfg.sns_client.publish(
            TopicArn=_cfg.sns_topic_arn,
            Subject=safe_subject,
            Message=message,
        )
    except Exception:
        logger.exception("SNS publish failed for topic %s", _cfg.sns_topic_arn)
        raise

    message_id: str = response.get("MessageId", "")
    logger.info("SNS alert published.  MessageId=%s", message_id)
    return message_id


# ---------------------------------------------------------------------------
# Lambda — Entry-point handler
# ---------------------------------------------------------------------------

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry-point for triggering Wildlife Sentinel analysis jobs.

    This handler is designed to be referenced in your Lambda function
    configuration as ``aws_integration.lambda_handler``.

    Expected *event* payload
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. code-block:: json

        {
            "action": "analyse",
            "s3_bucket": "my-bucket",
            "s3_key": "incoming/video_001.mp4",
            "location": {"lat": -2.345, "lon": 34.876},
            "notify": true
        }

    Supported actions:

    * ``"analyse"`` — Download the video from S3, run detection (stub),
      log results to DynamoDB, and optionally send an SNS alert.
    * ``"health"``  — Return a simple health-check response.

    Parameters
    ----------
    event : dict
        Lambda invocation event payload.
    context : object
        Lambda runtime context (provides request ID, remaining time, etc.).

    Returns
    -------
    dict
        A response dict with ``statusCode`` and ``body`` keys suitable for
        API Gateway proxy integration.
    """
    request_id = getattr(context, "aws_request_id", "unknown") if context else "local"
    logger.info("Lambda invocation — request_id=%s", request_id)

    # Ensure AWS clients are ready (idempotent after first call)
    try:
        init_aws()
    except Exception as exc:
        logger.exception("init_aws() failed inside Lambda handler.")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "AWS initialisation failed", "detail": str(exc)}
            ),
        }

    action = event.get("action", "").lower()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    if action == "health":
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "ok",
                    "service": "wildlife-sentinel",
                    "request_id": request_id,
                    "timestamp": _utc_iso(),
                }
            ),
        }

    # ------------------------------------------------------------------
    # Analyse action
    # ------------------------------------------------------------------
    if action == "analyse":
        source_bucket = event.get("s3_bucket", _cfg.s3_bucket)
        source_key = event.get("s3_key", "")
        location = event.get("location", {})
        notify = event.get("notify", False)

        if not source_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required field 's3_key'."}),
            }

        # ------ Step 1: Download source video from S3 (stub) ----------
        # In a real deployment this would download the file to /tmp and
        # invoke the local YOLO inference pipeline.
        logger.info(
            "Analyse job: s3://%s/%s  location=%s", source_bucket, source_key, location
        )

        # Placeholder detection results (replace with actual inference)
        detections = [
            {
                "species": "placeholder_species",
                "confidence": 0.0,
                "count": 0,
                "bbox": [],
            }
        ]

        # ------ Step 2: Log detection event to DynamoDB ---------------
        event_record: Dict[str, Any] = {
            "timestamp": _utc_iso(),
            "source_bucket": source_bucket,
            "source_key": source_key,
            "location": location,
            "detections": detections,
            "lambda_request_id": request_id,
        }

        try:
            logged = log_detection_event(event_record)
        except Exception as exc:
            logger.exception("Failed to log detection event.")
            logged = {"error": str(exc)}

        # ------ Step 3: (Optional) Send SNS notification --------------
        sns_message_id = None
        if notify:
            try:
                summary_lines = [
                    f"[Wildlife Sentinel] Detection from {source_key}",
                    f"Location: {location}",
                    f"Detections: {len(detections)}",
                ]
                sns_message_id = send_sns_alert(
                    subject="Wildlife Sentinel Detection Alert",
                    message="\n".join(summary_lines),
                )
            except Exception:
                logger.exception("SNS notification failed — continuing.")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Analysis complete.",
                    "event_id": logged.get("event_id"),
                    "sns_message_id": sns_message_id,
                    "request_id": request_id,
                }
            ),
        }

    # ------------------------------------------------------------------
    # Unknown action
    # ------------------------------------------------------------------
    return {
        "statusCode": 400,
        "body": json.dumps(
            {
                "error": f"Unknown action '{action}'.",
                "supported_actions": ["analyse", "health"],
            }
        ),
    }


# ---------------------------------------------------------------------------
# Convenience — Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    print("Wildlife Sentinel — AWS Integration Module")
    print("=" * 50)

    try:
        active = init_aws()
        print(f"AWS active: {active}")
    except Exception as err:
        print(f"Initialisation failed (expected in local dev): {err}")

    print("\nConfiguration snapshot:")
    print(f"  USE_AWS            = {_cfg.use_aws}")
    print(f"  AWS_REGION         = {_cfg.region}")
    print(f"  AWS_S3_BUCKET      = {_cfg.s3_bucket}")
    print(f"  AWS_DYNAMODB_TABLE = {_cfg.dynamodb_table}")
    print(f"  AWS_SNS_TOPIC_ARN  = {_cfg.sns_topic_arn}")
    print(f"  Initialised        = {_cfg.initialised}")
