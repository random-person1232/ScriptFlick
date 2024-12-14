from typing import Dict
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket = None):
        """Connect a client with a specific task_id"""
        if websocket:
            await websocket.accept()
        self.active_connections[task_id] = websocket

    async def disconnect(self, task_id: str):
        """Disconnect a client"""
        if task_id in self.active_connections:
            if self.active_connections[task_id]:
                await self.active_connections[task_id].close()
            del self.active_connections[task_id]

    async def send_message(self, task_id: str, message: dict):
        """Send a message to a specific client"""
        if task_id in self.active_connections and self.active_connections[task_id]:
            await self.active_connections[task_id].send_json(message)

# Create a single instance to be imported by other modules
connection_manager = ConnectionManager()