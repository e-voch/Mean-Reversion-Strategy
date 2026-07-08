"""meanrev: mean-reversion strategy research, validation, and (eventually) live trading."""

import os
from pathlib import Path

import truststore

truststore.inject_into_ssl()

_LOCAL_CA_BUNDLE = Path(__file__).resolve().parents[2] / ".local" / "ca-bundle.pem"
if _LOCAL_CA_BUNDLE.exists():
    # Some machines run antivirus HTTPS scanning (e.g. Norton) whose proxy cert fails strict
    # OpenSSL validation. yfinance's HTTP client (curl_cffi) doesn't use truststore, so it needs
    # an explicit CA bundle that includes the AV's root cert. See README for setup.
    bundle = str(_LOCAL_CA_BUNDLE)
    os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    os.environ.setdefault("SSL_CERT_FILE", bundle)
