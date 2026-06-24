"""The single, centralized network egress point of the whole backend.

SECURITY (absolute priority): the application must NEVER open an outbound
connection to anything other than the configured LLMAAS endpoint (``apiBase``).

Every outbound HTTP request made by this backend goes through the guarded
httpx client built here. The guard inspects the destination of *every* request
(including any redirect the client would follow) and refuses any host, port or
scheme that does not exactly match ``apiBase``. The OpenAI SDK is constructed
with this guarded client (``http_client=...``), so even the SDK's own internals
cannot reach any other host.

To audit network egress, you only need to read this file and confirm that:
  1. ``llm.py`` is the only module that builds/uses a network client, and it
     uses ``build_guarded_client`` / ``build_guarded_async_client`` exclusively;
  2. nothing else in ``app/`` imports ``httpx``, ``requests``, ``urllib`` for
     outbound calls, or constructs an OpenAI client without the guarded client.
"""

from __future__ import annotations

import os
import ssl
from urllib.parse import urlparse

import httpx

# Default ports for the schemes we allow, used so that an apiBase written
# without an explicit port still matches requests the SDK sends to that scheme.
_DEFAULT_PORTS = {"http": 80, "https": 443}


class EgressBlockedError(PermissionError):
    """Raised when an outbound request targets a host other than apiBase."""


class EgressGuard:
    """Allowlist of exactly one destination: the host of ``api_base``.

    The guard compares scheme, hostname and effective port. Hostname matching
    is case-insensitive (DNS is case-insensitive) but otherwise strict: no
    suffix/subdomain wildcards, no IP-range logic — exactly one destination.
    """

    def __init__(self, api_base: str) -> None:
        parsed = urlparse(api_base)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                f"apiBase must be an absolute URL with scheme and host, got: {api_base!r}"
            )
        self.allowed_scheme = parsed.scheme.lower()
        self.allowed_host = parsed.hostname.lower()
        self.allowed_port = parsed.port or _DEFAULT_PORTS.get(self.allowed_scheme)

    def _effective_port(self, scheme: str, port: int | None) -> int | None:
        return port or _DEFAULT_PORTS.get(scheme.lower())

    def check(self, url: str) -> None:
        """Raise EgressBlockedError unless ``url`` targets exactly apiBase.

        This is the concept from the prompt, hardened to also pin scheme and
        port so an http downgrade or an alternate port is rejected too.
        """
        parsed = urlparse(str(url))
        host = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "").lower()
        port = self._effective_port(scheme, parsed.port)

        if host != self.allowed_host:
            raise EgressBlockedError(
                f"Network egress blocked to host {host!r}. "
                f"Only {self.allowed_host!r} (the apiBase host) is allowed."
            )
        if scheme != self.allowed_scheme:
            raise EgressBlockedError(
                f"Network egress blocked: scheme {scheme!r} for host {host!r}. "
                f"Only {self.allowed_scheme!r} is allowed."
            )
        if port != self.allowed_port:
            raise EgressBlockedError(
                f"Network egress blocked: port {port!r} on host {host!r}. "
                f"Only port {self.allowed_port!r} is allowed."
            )

    def describe(self) -> str:
        return f"{self.allowed_scheme}://{self.allowed_host}:{self.allowed_port}"


class GuardedTransport(httpx.BaseTransport):
    """Sync httpx transport that checks every request URL before sending."""

    def __init__(self, guard: EgressGuard, inner: httpx.BaseTransport | None = None) -> None:
        self._guard = guard
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._guard.check(request.url)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class GuardedAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that checks every request URL before sending."""

    def __init__(self, guard: EgressGuard, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._guard = guard
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._guard.check(request.url)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def resolve_verify(ca_cert_path: str | None):
    """Map an optional CA-bundle path to an httpx ``verify`` value.

    TLS verification is ALWAYS on:
      * a non-empty path -> an SSLContext that trusts exactly that CA bundle
        (used for an internal LLMAAS whose cert is signed by a private CA);
      * empty/unset      -> ``True`` (the default public CA store), so the
        OpenAI connection test and any public endpoint keep working.

    This function can never return ``False`` — there is no code path that
    disables certificate verification. A configured-but-missing bundle raises
    a clear error instead of silently downgrading.
    """
    path = (ca_cert_path or "").strip()
    if not path:
        return True
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CA certificate bundle not found: {path}")
    # check_hostname + CERT_REQUIRED are on by default in create_default_context.
    return ssl.create_default_context(cafile=path)


def build_guarded_client(
    api_base: str, timeout: float = 60.0, ca_cert_path: str | None = None
) -> httpx.Client:
    """Build a sync httpx.Client locked to the apiBase host.

    ``follow_redirects`` is disabled: the LLMAAS API does not need redirects,
    and disabling them removes an entire class of redirect-to-elsewhere risk.
    Even if it were enabled, the transport guard would still inspect the
    redirected request and block a foreign host.

    ``ca_cert_path`` only changes which CA bundle validates the server
    certificate; it never affects which host is allowed (that stays pinned by
    the EgressGuard) and never disables verification.
    """
    guard = EgressGuard(api_base)
    inner = httpx.HTTPTransport(verify=resolve_verify(ca_cert_path))
    return httpx.Client(
        transport=GuardedTransport(guard, inner),
        timeout=timeout,
        follow_redirects=False,
        # Defense-in-depth (audit H1): never trust the environment for proxies
        # (HTTP(S)_PROXY/NO_PROXY) or .netrc, so the single-egress guarantee is
        # independent of httpx internals and a future upgrade can't reintroduce
        # env-proxy routing around the guard. TLS is unaffected (verify= is
        # explicit on the transport).
        trust_env=False,
    )


def build_guarded_async_client(
    api_base: str, timeout: float = 60.0, ca_cert_path: str | None = None
) -> httpx.AsyncClient:
    guard = EgressGuard(api_base)
    inner = httpx.AsyncHTTPTransport(verify=resolve_verify(ca_cert_path))
    return httpx.AsyncClient(
        transport=GuardedAsyncTransport(guard, inner),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,  # see build_guarded_client (audit H1): env proxies/.netrc never trusted
    )
