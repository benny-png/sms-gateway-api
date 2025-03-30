# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import asyncio
import uuid
import json
import logging
import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time SMS Gateway API",
    description="A FastAPI-based service for sending SMS messages through connected Android devices",
    version="1.0.0",
)

# Database setup
DATABASE_PATH = os.getenv("DATABASE_PATH", "sms_gateway.db")

# Create database tables if they don't exist
def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sms_messages (
            id TEXT PRIMARY KEY,
            phone_number TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
        ''')
        conn.commit()
        logger.info("Database initialized")

# Context manager for database connections
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# SMS Message model
class SMSMessage(BaseModel):
    id: str = Field(..., description="Unique UUID for the SMS message", example="123e4567-e89b-12d3-a456-426614174000")
    phone_number: str = Field(..., description="Recipient phone number with country code", example="+1234567890")
    message: str = Field(..., description="Text content of the message", example="Hello from the SMS Gateway!")
    status: str = Field(default="pending", description="Current status of the message (pending, processing, sent, failed)", example="pending")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO timestamp of when the message was created", example="2023-07-01T12:30:45.123456")

# Store for connected devices
class ConnectionManager:
    def __init__(self):
        # Map of device_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Map of device_id -> device info
        self.device_info: Dict[str, dict] = {}
        
    async def connect(self, device_id: str, websocket: WebSocket, info: dict):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        self.device_info[device_id] = info
        logger.info(f"Device connected: {device_id}, info: {info}")
        
    def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]
        if device_id in self.device_info:
            del self.device_info[device_id]
        logger.info(f"Device disconnected: {device_id}")
            
    async def send_message(self, device_id: str, message: dict):
        if device_id in self.active_connections:
            await self.active_connections[device_id].send_json(message)
            return True
        return False
    
    def get_available_devices(self):
        return list(self.active_connections.keys())
    
    def get_device_info(self, device_id: str):
        return self.device_info.get(device_id, {})
    
    def update_device_info(self, device_id: str, info: dict):
        if device_id in self.device_info:
            self.device_info[device_id].update(info)

# Initialize connection manager
manager = ConnectionManager()

# Security
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "your-secret-api-key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header or api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header

# Request and response models
class SMSRequest(BaseModel):
    phone_number: str = Field(..., description="Recipient phone number with country code", example="+1234567890")
    message: str = Field(..., description="Text content of the message", example="Hello from the SMS Gateway!")

class SMSResponse(BaseModel):
    message_id: str = Field(..., description="Unique ID of the created message", example="123e4567-e89b-12d3-a456-426614174000")
    status: str = Field(..., description="Initial status of the message", example="processing")

class DeviceInfo(BaseModel):
    device_name: str = Field(..., description="Name of the Android device", example="Google Pixel 6")
    battery_level: Optional[float] = Field(None, description="Battery level percentage", example=85.5)
    signal_strength: Optional[int] = Field(None, description="Signal strength as dBm or bars", example=4)
    android_version: Optional[str] = Field(None, description="Android OS version", example="12.0")

# Database operations
def save_sms_message(message: SMSMessage):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sms_messages (id, phone_number, message, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (message.id, message.phone_number, message.message, message.status, message.created_at)
        )
        conn.commit()

def update_sms_status(message_id: str, status: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sms_messages SET status = ? WHERE id = ?",
            (status, message_id)
        )
        conn.commit()
        return cursor.rowcount > 0

def get_sms_message(message_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sms_messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def get_all_sms_messages(limit: int = 100, offset: int = 0):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sms_messages ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# WebSocket endpoint for Android devices
@app.websocket("/ws/device/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    try:
        # Wait for initial connection message with device info
        await websocket.accept()
        init_data = await websocket.receive_text()
        device_info = json.loads(init_data)
        
        # Register the device
        await manager.connect(device_id, websocket, device_info)
        
        # Send acknowledgment
        await websocket.send_json({"type": "connection_ack", "device_id": device_id})
        
        # Listen for messages from the device
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "status_update":
                # Update SMS message status
                message_id = data.get("message_id")
                status = data.get("status")
                
                if update_sms_status(message_id, status):
                    logger.info(f"Updated message {message_id} status to {status}")
            
            elif data.get("type") == "device_update":
                # Update device information
                info_update = data.get("info", {})
                manager.update_device_info(device_id, info_update)
                
            # You can handle other message types here
                
    except WebSocketDisconnect:
        manager.disconnect(device_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(device_id)

# REST API endpoint to send SMS
@app.post(
    "/api/send-sms", 
    response_model=SMSResponse, 
    dependencies=[Depends(get_api_key)],
    summary="Send a new SMS message",
    description="""
    Sends an SMS message via connected Android devices. 
    
    The message will be queued if no devices are available.
    
    Example curl command:
    ```
    curl -X POST "http://localhost:8000/api/send-sms" \\
      -H "X-API-Key: your-secret-api-key" \\
      -H "Content-Type: application/json" \\
      -d '{"phone_number": "+1234567890", "message": "Hello from the SMS Gateway!"}'
    ```
    """
)
async def send_sms(sms_request: SMSRequest):
    # Generate unique message ID
    message_id = str(uuid.uuid4())
    
    # Create and store the message
    message = SMSMessage(
        id=message_id,
        phone_number=sms_request.phone_number,
        message=sms_request.message
    )
    
    # Save to database
    save_sms_message(message)
    
    # Get available devices
    available_devices = manager.get_available_devices()
    
    if not available_devices:
        # No devices connected
        return {"message_id": message_id, "status": "queued_no_device"}
    
    # Simple load balancing - just take the first device
    # In production, implement smarter selection based on battery, signal, etc.
    device_id = available_devices[0]
    
    # Send message to the device in real-time
    message_sent = await manager.send_message(
        device_id,
        {
            "type": "new_sms",
            "message_id": message_id,
            "phone_number": sms_request.phone_number,
            "message": sms_request.message
        }
    )
    
    if message_sent:
        update_sms_status(message_id, "processing")
        return {"message_id": message_id, "status": "processing"}
    else:
        return {"message_id": message_id, "status": "queued"}

# Check message status
@app.get(
    "/api/sms/{message_id}", 
    dependencies=[Depends(get_api_key)],
    summary="Get SMS message status",
    description="""
    Retrieves the current status of an SMS message by its ID.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/sms/123e4567-e89b-12d3-a456-426614174000" \\
      -H "X-API-Key: your-secret-api-key"
    ```
    """
)
async def get_sms_status(message_id: str):
    message = get_sms_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {
        "message_id": message["id"],
        "status": message["status"],
        "created_at": message["created_at"]
    }

# List all SMS messages
@app.get(
    "/api/sms", 
    dependencies=[Depends(get_api_key)],
    summary="List all SMS messages",
    description="""
    Lists all SMS messages with pagination support.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/sms?limit=10&offset=0" \\
      -H "X-API-Key: your-secret-api-key"
    ```
    """
)
async def list_sms_messages(
    limit: int = 100,
    offset: int = 0
):
    messages = get_all_sms_messages(limit, offset)
    return {
        "count": len(messages),
        "messages": messages
    }

# List connected devices
@app.get(
    "/api/devices", 
    dependencies=[Depends(get_api_key)],
    summary="List connected devices",
    description="""
    Lists all currently connected Android devices.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/devices" \\
      -H "X-API-Key: your-secret-api-key"
    ```
    """
)
async def list_devices():
    device_list = []
    for device_id in manager.get_available_devices():
        device_list.append({
            "device_id": device_id,
            **manager.get_device_info(device_id)
        })
    
    return {"devices": device_list, "count": len(device_list)}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)