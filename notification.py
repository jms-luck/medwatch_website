"""Minimal AWS SNS helpers for SMS and topic publish.

Requires AWS credentials (env vars or config file) and AWS_REGION (default: us-east-1).
"""

import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError


def _client():
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("sns", region_name=region)


def send_sms(phone_number: str, message: str) -> None:
    """Send an SMS via SNS. phone_number must be in E.164 format (e.g., +15551234567)."""
    try:
        _client().publish(PhoneNumber=phone_number, Message=message)
    except (NoCredentialsError, BotoCoreError, ClientError) as err:
        raise RuntimeError(f"SNS SMS send failed: {err}") from err


def send_topic(topic_arn: str, message: str, subject: Optional[str] = None) -> None:
    """Publish a message to an SNS topic."""
    publish_kwargs = {"TopicArn": topic_arn, "Message": message}
    if subject:
        publish_kwargs["Subject"] = subject
    try:
        _client().publish(**publish_kwargs)
    except (NoCredentialsError, BotoCoreError, ClientError) as err:
        raise RuntimeError(f"SNS topic publish failed: {err}") from err


__all__ = ["send_sms", "send_topic"]
