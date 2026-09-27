"""
delta_client.py — Authenticated REST client for Delta Exchange India.

Handles:
- Rate limiting (candle endpoint weight = 3)
- HMAC-SHA256 authentication for private endpoints
- Retry logic with exponential backoff
- Public market data endpoints
"""

import time
import hmac
import hashlib
import requests
import logging
import urllib.parse

from typing import Optional, Dict, Any

from config.settings import (
    DELTA_BASE_URL,
    DELTA_API_KEY,
    DELTA_API_SECRET,
    API_REQUESTS_PER_MINUTE,
    API_RETRY_ATTEMPTS,
    API_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket style rate limiter for API calls."""

    def __init__(self, max_per_minute: int = 20):
        self.max_per_minute = max_per_minute
        self.interval = 60.0 / max_per_minute
        self._last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_call

        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)

        self._last_call = time.time()


class DeltaClient:
    """
    REST client for Delta Exchange India API.

    Base URL:
        https://api.india.delta.exchange
    """

    def __init__(self):
        self.base_url = DELTA_BASE_URL.rstrip("/")
        self.api_key = DELTA_API_KEY
        self.api_secret = DELTA_API_SECRET

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        self._rate_limiter = RateLimiter(
            API_REQUESTS_PER_MINUTE
        )

    # ─────────────────────────────────────────
    # Signature generation
    # ─────────────────────────────────────────

    def _generate_signature(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = "",
        timestamp: Optional[int] = None,
    ) -> Dict[str, str]:

        if timestamp is None:
            timestamp = int(time.time())

        msg = (
            method.upper()
            + str(timestamp)
            + path
            + query_string
            + body
        )

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "api-key": self.api_key,
            "timestamp": str(timestamp),
            "signature": signature,
        }

    # ─────────────────────────────────────────
    # Generic request with retry
    # ─────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        auth: bool = False,
    ) -> Optional[Any]:

        url = f"{self.base_url}{path}"

        backoff = API_RETRY_BACKOFF

        for attempt in range(
            1,
            API_RETRY_ATTEMPTS + 1,
        ):

            self._rate_limiter.wait()

            response = None

            try:

                headers = {}

                if (
                    auth
                    and self.api_key
                    and self.api_secret
                ):

                    qs = urllib.parse.urlencode(
                        params or {}
                    )

                    sig_headers = (
                        self._generate_signature(
                            method.upper(),
                            path,
                            qs,
                        )
                    )

                    headers.update(sig_headers)

                response = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=10,
                )

                response.raise_for_status()

                return response.json()

            except requests.exceptions.HTTPError as e:

                status_code = (
                    response.status_code
                    if response is not None
                    else None
                )

                logger.warning(
                    f"HTTP error on attempt "
                    f"{attempt}/{API_RETRY_ATTEMPTS}: {e}"
                )

                # 404 = endpoint/path issue.
                # 400/401/403/404/422 are normally
                # non-retryable for this client.
                if status_code in (
                    400,
                    401,
                    403,
                    404,
                    422,
                ):

                    logger.error(
                        f"Non-retryable error "
                        f"{status_code}: "
                        f"{response.text if response else ''}"
                    )

                    return None

            except requests.exceptions.RequestException as e:

                logger.warning(
                    f"Request error on attempt "
                    f"{attempt}/{API_RETRY_ATTEMPTS}: {e}"
                )

            except ValueError as e:

                logger.error(
                    f"Invalid JSON response from "
                    f"{path}: {e}"
                )

                return None

            if attempt < API_RETRY_ATTEMPTS:

                logger.info(
                    f"Retrying in {backoff:.1f}s …"
                )

                time.sleep(backoff)

                backoff *= 2

        logger.error(
            f"All {API_RETRY_ATTEMPTS} attempts "
            f"failed for {path}"
        )

        return None

    # ─────────────────────────────────────────
    # Public endpoints
    # ─────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        resolution: str,
        start: int,
        end: int,
    ) -> Optional[list]:

        """
        Fetch OHLCV candles.

        Args:
            symbol:
                e.g. BTCUSD

            resolution:
                4h, 1h, 15m

            start:
                Unix timestamp seconds

            end:
                Unix timestamp seconds

        Returns:
            List of candle objects.
        """

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": start,
            "end": end,
        }

        data = self._request(
            "GET",
            "/v2/history/candles",
            params=params,
        )

        if isinstance(data, dict):

            result = data.get("result")

            if isinstance(result, list):
                return result

        if isinstance(data, list):
            return data

        logger.warning(
            f"Unexpected candle response for "
            f"{symbol} {resolution}: "
            f"{type(data).__name__}"
        )

        return []

    # ─────────────────────────────────────────
    # Instruments
    # ─────────────────────────────────────────

    def get_instruments(self) -> Optional[list]:
        """Fetch all available trading instruments."""

        data = self._request(
            "GET",
            "/v2/products",
        )

        if isinstance(data, dict):

            result = data.get("result")

            if isinstance(result, list):
                return result

        if isinstance(data, list):
            return data

        logger.warning(
            "Unexpected instruments response type: "
            f"{type(data).__name__}"
        )

        return []

    # ─────────────────────────────────────────
    # Ticker
    # ─────────────────────────────────────────

    def get_ticker(
        self,
        symbol: str,
    ) -> Optional[Any]:

        """
        Fetch current ticker for a symbol.

        Delta can return the result as:
        - dict
        - list
        """

        data = self._request(
            "GET",
            "/v2/tickers",
            params={
                "symbol": symbol
            },
        )

        if not data:
            return None

        if isinstance(data, dict):

            if "result" in data:
                return data["result"]

            # Direct ticker object
            if (
                "symbol" in data
                or "close" in data
                or "last_price" in data
            ):
                return data

        if isinstance(data, list):
            return data

        logger.warning(
            f"Unexpected ticker response type "
            f"for {symbol}: "
            f"{type(data).__name__}"
        )

        return None

    # ─────────────────────────────────────────
    # L2 Orderbook
    # ─────────────────────────────────────────

    def get_orderbook(
        self,
        symbol: str,
        depth: int = 5,
    ) -> Optional[Any]:

        """
        Fetch L2 order book for spread calculation.

        IMPORTANT:
        Delta Exchange endpoint:

            /v2/l2orderbook/{symbol}

        Example:

            /v2/l2orderbook/BTCUSD?depth=1
        """

        # IMPORTANT FIX:
        # Old incorrect endpoint:
        #
        # /v2/l2orderbook?symbol=BTCUSD
        #
        # Correct endpoint:
        #
        # /v2/l2orderbook/BTCUSD

        path = f"/v2/l2orderbook/{symbol}"

        data = self._request(
            "GET",
            path,
            params={
                "depth": depth
            },
        )

        if not data:
            return None

        if isinstance(data, dict):

            if "result" in data:
                return data["result"]

            return data

        if isinstance(data, list):
            return data

        logger.warning(
            f"Unexpected orderbook response type "
            f"for {symbol}: "
            f"{type(data).__name__}"
        )

        return None