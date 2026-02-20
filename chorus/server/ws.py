"""WebSocket connection manager for live notifications."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("chorus.ws")


class ConnectionManager:
    """Manages WebSocket connections for real-time event broadcasting."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}  # client_id -> ws
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections[client_id] = ws
        logger.info(f"WebSocket connected: {client_id} ({len(self._connections)} total)")

    async def disconnect(self, client_id: str):
        async with self._lock:
            self._connections.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def broadcast(self, event: dict):
        """Send event to all connected clients."""
        message = json.dumps(event)
        async with self._lock:
            dead = []
            for cid, ws in self._connections.items():
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(cid)
            for cid in dead:
                self._connections.pop(cid, None)

    @property
    def connected_count(self) -> int:
        return len(self._connections)
