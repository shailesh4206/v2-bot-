"""
websocket.py — Real-time WebSocket client for Delta Exchange India.

Purpose:
    - Outcome tracking for active signals (TP/SL/expiry monitoring)
    - NOT used for signal generation (signals only on confirmed candle closes)

Uses the public ticker channel — no trading permissions required.
"""

import json
import time
import logging
import threading
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Delta Exchange WebSocket URL
WS_URL_LIVE = "wss://socket.india.delta.exchange"
WS_URL_TEST = "wss://testnet-socket.india.delta.exchange"


class DeltaWebSocket:
    """
    Async-capable WebSocket client for Delta Exchange real-time tickers.

    Runs in a background thread.
    Calls price_callback(symbol, price) when a ticker update arrives.
    """

    def __init__(
        self,
        symbols: list,
        price_callback: Callable[[str, float], None],
        use_testnet: bool = False,
    ):
        self.symbols = symbols
        self.price_callback = price_callback
        self.ws_url = WS_URL_TEST if use_testnet else WS_URL_LIVE
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscribed: Set[str] = set()

    def start(self):
        """Start the WebSocket in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="DeltaWS")
        self._thread.start()
        logger.info(f"WebSocket thread started for symbols: {self.symbols}")

    def stop(self):
        """Stop the WebSocket cleanly."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WebSocket stopped")

    def _run(self):
        """Main WebSocket loop with reconnect."""
        try:
            import websocket as ws_lib
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return

        while self._running:
            try:
                logger.info(f"Connecting to Delta WebSocket: {self.ws_url}")
                self._ws = ws_lib.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                logger.info("WebSocket reconnecting in 5s …")
                time.sleep(5)

    def _on_open(self, ws):
        """Subscribe to ticker channels on connect."""
        logger.info("WebSocket connected")
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "v2/ticker", "symbols": self.symbols}
                ]
            },
        }
        ws.send(json.dumps(payload))
        self._subscribed = set(self.symbols)
        logger.info(f"Subscribed to tickers: {self.symbols}")

    def _on_message(self, ws, message: str):
        """Parse incoming ticker messages and invoke callback."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            # Handle ticker update
            if msg_type in ("v2/ticker", "ticker"):
                symbol = data.get("symbol") or data.get("s")
                price = (
                    data.get("close")
                    or data.get("last_price")
                    or data.get("mark_price")
                    or data.get("c")
                )
                if symbol and price:
                    self.price_callback(symbol, float(price))

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"WebSocket message parse error: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")


class PriceMonitor:
    """
    High-level price monitor that wraps DeltaWebSocket.

    Used by the signal lifecycle manager to check if
    TP/SL levels are breached by real-time price.
    """

    def __init__(self, symbols: list, use_testnet: bool = False):
        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._ws = DeltaWebSocket(
            symbols=symbols,
            price_callback=self._on_price,
            use_testnet=use_testnet,
        )

    def start(self):
        self._ws.start()

    def stop(self):
        self._ws.stop()

    def _on_price(self, symbol: str, price: float):
        with self._lock:
            self._prices[symbol] = price

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)
