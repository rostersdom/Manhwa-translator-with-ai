import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.job_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        if job_id not in self.job_connections:
            self.job_connections[job_id] = set()
        self.job_connections[job_id].add(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        if job_id in self.job_connections:
            self.job_connections[job_id].discard(ws)
            if not self.job_connections[job_id]:
                del self.job_connections[job_id]

    async def broadcast(self, job_id: str, data: dict):
        if job_id not in self.job_connections:
            return
        dead = set()
        for ws in self.job_connections[job_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.job_connections[job_id].discard(ws)
        if not self.job_connections.get(job_id):
            del self.job_connections[job_id]


manager = ConnectionManager()
