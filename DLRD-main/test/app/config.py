"""Configuration: load/save the LLMAAS connection settings and app tunables.

The secret ``api_key`` is NEVER written to the repo, logged, or placed in the
generated knowledge graph. It is persisted only to a gitignored local file
(``config.local.json``) or read from the environment / ``.env``.

Precedence when loading: the gitignored ``config.local.json`` (written by the
in-browser config screen) wins; any field it does not set falls back to an
environment variable (optionally provided via a gitignored ``.env``). This lets
the work machine be configured either through the browser or by pre-seeding
environment variables — never by committing anything secret.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

try:  # optional: load a gitignored .env if present. Never required.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ---------------------------------------------------------------------------
# Filesystem layout (all runtime data stays inside the project, gitignored).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.local.json"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = ROOT / "uploads"
WEB_DIR = ROOT / "web"
DASHBOARD_DIR = WEB_DIR / "dashboard"

# Allowed output languages for the LLM-generated content (summaries, tours).
# Exactly English and French — English is the default. The dashboard's own UI
# chrome is always English regardless of this setting.
SUPPORTED_LANGUAGES = ("en", "fr")
LANGUAGE_NAMES = {"en": "English", "fr": "French"}

# Hard ceiling on enrichment concurrency (prompt requirement: max 5).
MAX_CONCURRENCY = 5

_TRUE = {"1", "true", "yes", "on"}


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else None


@dataclass
class Settings:
    # --- LLMAAS connection (entered on the config screen) ---
    api_base: str = ""        # e.g. https://llmaas.internal/v1  (keep /v1 as given)
    api_key: str = ""         # SECRET — never logged or committed
    model: str = ""
    language: str = "en"      # output language for generated content; UI stays English
    supports_system_message: bool = True  # set False for gemma & co.
    # Path to a CA-certificate bundle (PEM) used to verify the LLMAAS TLS
    # certificate when it is signed by an internal/private CA. Empty = use the
    # default public CA store. This only selects which CA validates the cert;
    # it never disables verification and never changes which host is allowed.
    ca_cert_path: str = ""

    # --- LLM request tuning ---
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: float = 90.0
    concurrency: int = 5      # clamped to MAX_CONCURRENCY
    max_retries: int = 3

    # --- Analysis limits ---
    max_file_bytes: int = 300_000   # skip files larger than this when scanning
    llm_char_limit: int = 24_000    # cap on file content / distilled payload sent to the LLM
    max_files: int = 4000           # safety cap on number of files analyzed

    # --- Server ---
    port: int = 8765
    debug: bool = False             # off by default; never logs file contents

    # Non-secret fields that are safe to round-trip through the browser/UI.
    _PUBLIC_FIELDS = (
        "api_base", "model", "language", "supports_system_message", "ca_cert_path",
        "temperature", "max_tokens", "request_timeout", "concurrency",
        "max_file_bytes", "llm_char_limit", "max_files", "port", "debug",
    )

    def __post_init__(self) -> None:
        self.concurrency = max(1, min(int(self.concurrency), MAX_CONCURRENCY))
        if self.language not in SUPPORTED_LANGUAGES:
            self.language = "en"

    # -- safety: never leak the key via repr/str (e.g. in tracebacks/logs) --
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Settings(api_base={self.api_base!r}, model={self.model!r}, "
            f"language={self.language!r}, api_key={'***set***' if self.api_key else '(unset)'})"
        )

    __str__ = __repr__

    # ----------------------------------------------------------------- load
    @classmethod
    def load(cls) -> "Settings":
        data: dict = {}

        # 1) gitignored local file written by the config screen (primary store)
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}

        # 2) environment / .env fallback for any unset field
        env_map = {
            "api_base": _env("UA_API_BASE"),
            "api_key": _env("UA_API_KEY"),
            "model": _env("UA_MODEL"),
            "language": _env("UA_LANGUAGE"),
            "ca_cert_path": _env("UA_CA_CERT"),
        }
        for key, val in env_map.items():
            if val is not None and not data.get(key):
                data[key] = val

        ssm = _env("UA_SUPPORTS_SYSTEM_MESSAGE")
        if ssm is not None and "supports_system_message" not in data:
            data["supports_system_message"] = ssm.lower() in _TRUE
        if (port := _env("UA_PORT")) and "port" not in data:
            data["port"] = int(port)
        if (dbg := _env("UA_DEBUG")) is not None and "debug" not in data:
            data["debug"] = dbg.lower() in _TRUE

        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    # ----------------------------------------------------------------- save
    def save(self) -> None:
        """Persist to the gitignored local config file (includes the key)."""
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def update_public(self, payload: dict) -> None:
        """Apply only known public fields coming from the browser UI."""
        for key in self._PUBLIC_FIELDS:
            if key in payload and payload[key] is not None:
                setattr(self, key, payload[key])
        self.__post_init__()

    def set_api_key(self, key: str | None) -> None:
        # Empty string from the UI means "keep the existing key" (UI never
        # echoes the key back), so only overwrite on a non-empty value.
        if key:
            self.api_key = key

    # -------------------------------------------------------------- helpers
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)

    def to_public_dict(self) -> dict:
        """Config for the browser: no secret, just whether a key is set."""
        out = {k: getattr(self, k) for k in self._PUBLIC_FIELDS}
        out["api_key_set"] = bool(self.api_key)
        out["is_configured"] = self.is_configured()
        return out
