from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    async def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            await self.active_connections[task_id].close()
            del self.active_connections[task_id]

    async def send_status_update(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
            except Exception as e:
                print(f"Error sending status update: {str(e)}")

connection_manager = ConnectionManager()