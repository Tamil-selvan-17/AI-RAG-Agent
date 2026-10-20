"""
Optional: trust the operating system's certificate store for outgoing HTTPS.

Python normally verifies HTTPS certificates against its own bundled CA list
(certifi). On corporate networks that inspect HTTPS traffic, requests to
Gemini/Qdrant Cloud are re-signed by the company's own CA, which Windows /
macOS already trust but certifi doesn't -- so every call fails with
CERTIFICATE_VERIFY_FAILED. With USE_SYSTEM_CERTS=true, verification uses the
OS store instead (via `truststore`). Verification is never disabled.
"""

from app.core.config import get_settings
from app.core.logging import logger

_configured = False


def configure_tls() -> None:
    global _configured
    if _configured or not get_settings().use_system_certs:
        return
    import truststore

    truststore.inject_into_ssl()
    _configured = True
    logger.info("HTTPS certificate verification uses the operating system's certificate store")
