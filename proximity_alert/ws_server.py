"""
WebSocket server that pushes ProximityAlert JSON to connected clients.

Any web-based mission-control UI can subscribe by connecting to
ws://localhost:<port> and receiving JSON alert messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

try:
    import websockets
    import websockets.server
    HAS_WS = True
except ImportError:
    HAS_WS = False

from .engine import AlertEngine, ProximityAlert, ThreatLevel

log = logging.getLogger(__name__)


class AlertWSServer:
    """
    Lightweight async WebSocket server.

    Every time the engine produces an alert above NONE, it is broadcast
    as a JSON object to all connected clients.
    """

    def __init__(self, engine: AlertEngine, host: str = "0.0.0.0", port: int = 9800):
        if not HAS_WS:
            raise ImportError(
                "websockets package required: pip install websockets"
            )
        self.engine = engine
        self.host = host
        self.port = port
        self._clients: Set[websockets.server.ServerConnection] = set()
        self._server = None

    async def _handler(self, ws):
        self._clients.add(ws)
        log.info("Client connected (%d total)", len(self._clients))
        try:
            async for _msg in ws:
                pass
        finally:
            self._clients.discard(ws)
            log.info("Client disconnected (%d remaining)", len(self._clients))

    async def broadcast(self, alert: ProximityAlert):
        if not self._clients:
            return
        payload = json.dumps(alert.to_dict())
        stale = set()
        for ws in self._clients:
            try:
                await ws.send(payload)
            except Exception:
                stale.add(ws)
        self._clients -= stale

    async def start(self):
        self._server = await websockets.serve(
            self._handler, self.host, self.port
        )
        log.info("Alert WS server listening on ws://%s:%d", self.host, self.port)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


def run_ws_bridge(engine: AlertEngine, host: str = "0.0.0.0", port: int = 9800):
    """
    Blocking helper: start the WS server and keep it running.
    Call from a dedicated thread or process alongside the detector.
    """
    server = AlertWSServer(engine, host, port)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    log.info("WS bridge running (Ctrl+C to stop)")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()
