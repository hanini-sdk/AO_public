"""Shared helper utilities (Utility layer)."""

import json
import re


def slugify(text: str) -> str:
    """Turn an arbitrary title into a URL-friendly slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def format_response(payload: object) -> str:
    """Serialize a payload (dict / dataclass list) to a JSON string."""
    def encode(value: object):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value

    return json.dumps(encode(payload), indent=2)
