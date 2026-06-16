"""LLMAAS client (OpenAI-compatible).

All network access goes through the guarded httpx client from ``http_guard`` —
this module is the ONLY place an OpenAI/network client is constructed. The
client is built with ``http_client=<guarded>`` so the SDK cannot reach any host
other than the configured ``apiBase``.

Handles the documented pitfalls:
  * ``api_base`` may end with ``/v1`` — we pass it through verbatim, never guess.
  * Some models (notably gemma) reject ``system`` messages — when
    ``supports_system_message`` is False we fold the system text into the first
    user message.
  * Endpoints can be slow/flaky — bounded retries with exponential backoff.
  * Responses are not always valid JSON — defensive extraction + one structured
    retry before giving up.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from .config import Settings
from .http_guard import EgressBlockedError, build_guarded_client

log = logging.getLogger("data_lineage_retro_documentation.llm")

Message = dict[str, str]


def fold_system_into_user(messages: list[Message]) -> list[Message]:
    """Merge any system messages into the first user message (gemma & co)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not system_parts:
        return rest
    preamble = "\n\n".join(system_parts).strip()
    for i, m in enumerate(rest):
        if m.get("role") == "user":
            merged = dict(m)
            merged["content"] = f"{preamble}\n\n{m['content']}"
            return rest[:i] + [merged] + rest[i + 1 :]
    # No user message at all — synthesize one.
    return [{"role": "user", "content": preamble}, *rest]


def _egress_blocked_in_chain(exc: BaseException | None) -> EgressBlockedError | None:
    """Walk the cause/context chain looking for an egress block.

    The SDK wraps transport errors, so an egress block surfaces as an
    APIConnectionError whose __cause__ is our EgressBlockedError. We must never
    retry a security block — we surface it immediately.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, EgressBlockedError):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def extract_json_object(text: str) -> Any:
    """Best-effort parse of a JSON object from a model response.

    Strips ```json fences and, failing a direct parse, grabs the outermost
    {...} span. Raises ValueError if nothing parseable is found.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        # remove the opening fence line (``` or ```json) and any closing fence
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass
    raise ValueError("No valid JSON object found in model response")


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._http = build_guarded_client(
            settings.api_base,
            timeout=settings.request_timeout,
            ca_cert_path=settings.ca_cert_path,
        )
        # max_retries=0: we own the retry/backoff loop so we can also retry on
        # malformed JSON and never retry an egress block.
        self._client = OpenAI(
            base_url=settings.api_base,
            api_key=settings.api_key,
            http_client=self._http,
            max_retries=0,
        )

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ chat
    def _create(self, messages: list[Message], *, max_tokens: int, temperature: float) -> str:
        if not self.settings.supports_system_message:
            messages = fold_system_into_user(messages)
        resp = self._client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0] if resp.choices else None
        return (choice.message.content if choice and choice.message else "") or ""

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Single completion with bounded exponential-backoff retries."""
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = self.settings.temperature if temperature is None else temperature
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                return self._create(messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # noqa: BLE001 - we re-raise after inspecting
                blocked = _egress_blocked_in_chain(exc)
                if blocked is not None:
                    raise blocked  # security block: never retry, surface clearly
                last_exc = exc
                if attempt < self.settings.max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
        assert last_exc is not None
        raise last_exc

    def chat_json(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
    ) -> Any:
        """Completion that must return a JSON object; defensive parse + 1 retry."""
        text = self.chat(messages, max_tokens=max_tokens)
        try:
            return extract_json_object(text)
        except ValueError:
            # one structured retry: restate the requirement firmly
            retry_messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. Reply again with "
                        "ONLY a single valid JSON object and nothing else."
                    ),
                },
            ]
            text2 = self.chat(retry_messages, max_tokens=max_tokens)
            return extract_json_object(text2)

    # ------------------------------------------------------------ connection
    def test_connection(self) -> tuple[bool, str]:
        """Validate apiBase + apiKey + model + the egress guard, all at once."""
        try:
            reply = self._create(
                [
                    {"role": "system", "content": "You are a connection test."},
                    {"role": "user", "content": "Reply with the single word: OK"},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            snippet = (reply or "").strip()[:80] or "(empty reply)"
            return True, f"Connection OK. Model '{self.settings.model}' replied: {snippet}"
        except Exception as exc:  # noqa: BLE001
            blocked = _egress_blocked_in_chain(exc)
            if blocked is not None:
                return False, f"Egress blocked by guard: {blocked}"
            return False, f"Connection failed: {type(exc).__name__}: {exc}"
