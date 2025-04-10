# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import uuid
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from openai import OpenAI
import requests
from serpapi import GoogleSearch
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE = os.getenv("LOG_FILE", "")
LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Create logger
logger = logging.getLogger("sms_gateway")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Create formatter
formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create file handler if LOG_FILE is set
if LOG_FILE:
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=LOG_MAX_SIZE, 
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Log startup information
logger.info("Starting SMS Gateway API")
logger.info(f"Log level: {LOG_LEVEL}")
if LOG_FILE:
    logger.info(f"Logging to file: {LOG_FILE}")

app = FastAPI(
    title="Real-Time SMS Gateway API",
    description="A FastAPI-based service for sending SMS messages through connected Android devices",
    version="1.0.0",
)

# Generate a unique instance ID for this server instance
INSTANCE_ID = str(uuid.uuid4())[:8]
logger.info(f"Server instance ID: {INSTANCE_ID}")

# OpenAI/OpenRouter client setup
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
)
logger.info("OpenAI client initialized")

# SerpAPI key setup
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
logger.info("SerpAPI configuration loaded")

# Database setup
DATABASE_PATH = os.getenv("DATABASE_PATH", "sms_gateway.db")
logger.info(f"Using database: {DATABASE_PATH}")

# Create database tables if they don't exist
def init_db():
    logger.info("Initializing database")
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
        logger.info("Database schema initialized")

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
    logger.info("Application startup event triggered")
    init_db()
    logger.info("Application startup completed")

# Log shutdown events
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown event triggered")
    logger.info("Application shutdown completed")

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request started: {request_id} - {request.method} {request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request completed: {request_id} - {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s")
    
    return response

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
        logger.info("Connection manager initialized")
        
    async def connect(self, device_id: str, websocket: WebSocket, info: dict):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        self.device_info[device_id] = info
        logger.info(f"Device connected: {device_id}, info: {info}")
        logger.info(f"Total connected devices: {len(self.active_connections)}")
        
    def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]
        if device_id in self.device_info:
            del self.device_info[device_id]
        logger.info(f"Device disconnected: {device_id}")
        logger.info(f"Total connected devices: {len(self.active_connections)}")
            
    async def send_message(self, device_id: str, message: dict):
        if device_id in self.active_connections:
            logger.info(f"Sending message to device {device_id}: {message}")
            await self.active_connections[device_id].send_json(message)
            return True
        logger.warning(f"Failed to send message: device {device_id} not connected")
        return False
    
    def get_available_devices(self):
        devices = list(self.active_connections.keys())
        logger.debug(f"Available devices: {devices}")
        return devices
    
    def get_device_info(self, device_id: str):
        return self.device_info.get(device_id, {})
    
    def update_device_info(self, device_id: str, info: dict):
        if device_id in self.device_info:
            logger.info(f"Updating device info for {device_id}: {info}")
            self.device_info[device_id].update(info)

# Initialize connection manager
manager = ConnectionManager()

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
    logger.debug(f"Saving SMS message to database: {message.id}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sms_messages (id, phone_number, message, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (message.id, message.phone_number, message.message, message.status, message.created_at)
        )
        conn.commit()
    logger.info(f"SMS message saved to database: {message.id}")

def update_sms_status(message_id: str, status: str):
    logger.debug(f"Updating SMS status in database: {message_id} to {status}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sms_messages SET status = ? WHERE id = ?",
            (status, message_id)
        )
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"SMS status updated: {message_id} -> {status}")
        else:
            logger.warning(f"SMS status update failed: {message_id} not found")
        return updated

def get_sms_message(message_id: str):
    logger.debug(f"Retrieving SMS message from database: {message_id}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sms_messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        if row:
            logger.debug(f"SMS message found: {message_id}")
            return dict(row)
        logger.warning(f"SMS message not found: {message_id}")
        return None

def get_all_sms_messages(limit: int = 100, offset: int = 0):
    logger.debug(f"Retrieving all SMS messages with limit={limit}, offset={offset}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sms_messages ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cursor.fetchall()
        logger.debug(f"Retrieved {len(rows)} SMS messages")
        return [dict(row) for row in rows]

# WebSocket endpoint for Android devices
@app.websocket("/ws/device/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    logger.info(f"WebSocket connection attempt from device: {device_id}")
    try:
        # Wait for initial connection message with device info
        await websocket.accept()
        logger.debug(f"WebSocket connection accepted for device: {device_id}")
        
        init_data = await websocket.receive_text()
        logger.debug(f"Received initial data from device {device_id}: {init_data}")
        
        device_info = json.loads(init_data)
        
        # Register the device
        await manager.connect(device_id, websocket, device_info)
        
        # Send acknowledgment
        await websocket.send_json({"type": "connection_ack", "device_id": device_id})
        logger.debug(f"Sent connection acknowledgment to device: {device_id}")
        
        # Listen for messages from the device
        while True:
            data = await websocket.receive_json()
            logger.debug(f"Received data from device {device_id}: {data}")
            
            # Handle different message types
            if data.get("type") == "status_update":
                # Update SMS message status
                message_id = data.get("message_id")
                status = data.get("status")
                
                logger.info(f"Status update from device {device_id}: message {message_id} -> {status}")
                
                if update_sms_status(message_id, status):
                    logger.info(f"Updated message {message_id} status to {status}")
            
            elif data.get("type") == "device_update":
                # Update device information
                info_update = data.get("info", {})
                logger.info(f"Device info update from {device_id}: {info_update}")
                manager.update_device_info(device_id, info_update)
                
            # You can handle other message types here
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {device_id}")
        manager.disconnect(device_id)
    except Exception as e:
        logger.error(f"WebSocket error for device {device_id}: {str(e)}", exc_info=True)
        manager.disconnect(device_id)

# REST API endpoint to send SMS
@app.post(
    "/api/send-sms", 
    response_model=SMSResponse,
    summary="Send a new SMS message",
    description="""
    Sends an SMS message via connected Android devices. 
    
    The message will be queued if no devices are available.
    
    Example curl command:
    ```
    curl -X POST "http://localhost:8000/api/send-sms" \\
      -H "Content-Type: application/json" \\
      -d '{"phone_number": "+1234567890", "message": "Hello from the SMS Gateway!"}'
    ```
    """
)
async def send_sms(sms_request: SMSRequest):
    # Generate unique message ID
    message_id = str(uuid.uuid4())
    logger.info(f"New SMS request: id={message_id}, to={sms_request.phone_number}, length={len(sms_request.message)}")
    
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
        logger.warning(f"No devices available to send SMS {message_id}")
        return {"message_id": message_id, "status": "queued_no_device"}
    
    # Simple load balancing - just take the first device
    # In production, implement smarter selection based on battery, signal, etc.
    device_id = available_devices[0]
    logger.info(f"Selected device {device_id} for sending SMS {message_id}")
    
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
        logger.info(f"SMS {message_id} sent to device {device_id} for processing")
        return {"message_id": message_id, "status": "processing"}
    else:
        logger.warning(f"SMS {message_id} queued - failed to send to device {device_id}")
        return {"message_id": message_id, "status": "queued"}

# Check message status
@app.get(
    "/api/sms/{message_id}",
    summary="Get SMS message status",
    description="""
    Retrieves the current status of an SMS message by its ID.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/sms/123e4567-e89b-12d3-a456-426614174000"
    ```
    """
)
async def get_sms_status(message_id: str):
    logger.info(f"Status check for SMS {message_id}")
    message = get_sms_message(message_id)
    if not message:
        logger.warning(f"Status check failed: SMS {message_id} not found")
        raise HTTPException(status_code=404, detail="Message not found")
    
    logger.info(f"Status check result for SMS {message_id}: {message['status']}")
    return {
        "message_id": message["id"],
        "status": message["status"],
        "created_at": message["created_at"]
    }

# List all SMS messages
@app.get(
    "/api/sms",
    summary="List all SMS messages",
    description="""
    Lists all SMS messages with pagination support.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/sms?limit=10&offset=0"
    ```
    """
)
async def list_sms_messages(
    limit: int = 100,
    offset: int = 0
):
    logger.info(f"Listing SMS messages with limit={limit}, offset={offset}")
    messages = get_all_sms_messages(limit, offset)
    logger.info(f"Retrieved {len(messages)} SMS messages")
    return {
        "count": len(messages),
        "messages": messages
    }

# List connected devices
@app.get(
    "/api/devices",
    summary="List connected devices",
    description="""
    Lists all currently connected Android devices.
    
    Example curl command:
    ```
    curl -X GET "http://localhost:8000/api/devices"
    ```
    """
)
async def list_devices():
    logger.info("Listing connected devices")
    device_list = []
    for device_id in manager.get_available_devices():
        device_list.append({
            "device_id": device_id,
            **manager.get_device_info(device_id)
        })
    
    logger.info(f"Found {len(device_list)} connected devices")
    return {"devices": device_list, "count": len(device_list)}

# SerpAPI tools
def search_google(query: str, num_results: int = 5):
    """
    Search Google using SerpAPI and return the results.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"SerpAPI search request {request_id}: query='{query}', num_results={num_results}")
    start_time = time.time()
    
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results
        }
        
        logger.debug(f"SerpAPI request {request_id} parameters: {json.dumps({k: v for k, v in params.items() if k != 'api_key'})}")
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        duration = time.time() - start_time
        logger.info(f"SerpAPI search {request_id} completed in {duration:.2f}s")
        
        # Extract organic results
        organic_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:num_results]:
                organic_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
            logger.debug(f"SerpAPI search {request_id} found {len(organic_results)} organic results")
        else:
            logger.warning(f"SerpAPI search {request_id} did not return organic results")
        
        # Extract knowledge graph if available
        knowledge_graph = None
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            knowledge_graph = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", "")
            }
            logger.debug(f"SerpAPI search {request_id} found knowledge graph: {kg.get('title', '')}")
        
        search_results = {
            "organic_results": organic_results,
            "knowledge_graph": knowledge_graph
        }
        
        logger.info(f"SerpAPI search {request_id} successful with {len(organic_results)} results")
        return search_results
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"SerpAPI search {request_id} failed after {duration:.2f}s: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search for information on Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Google"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool mapping dictionary
TOOL_MAPPING = {
    "search_google": search_google
}

# AI API Request and response models
class AIRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the AI model", example="What is the meaning of life?")
    model: str = Field(default="google/gemini-1.5-pro:latest", description="The AI model to use", example="google/gemini-1.5-pro:latest")
    enable_tools: bool = Field(default=True, description="Whether to enable tool usage for this request")

class AIResponse(BaseModel):
    response: str = Field(..., description="The AI model's response to the prompt")

# AI endpoint to generate responses via OpenRouter
@app.post(
    "/api/ai/chat",
    response_model=AIResponse,
    summary="Get a response from an AI model",
    description="""
    Sends a prompt to an AI model via OpenRouter and returns the response.
    Supports tool calling using SerpAPI for Google search.
    
    Example curl command:
    ```
    curl -X POST "http://localhost:8000/api/ai/chat" \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "What is the capital of France?", "model": "google/gemini-1.5-pro:latest", "enable_tools": true}'
    ```
    """
)
async def generate_ai_response(ai_request: AIRequest):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"AI request {request_id}: model={ai_request.model}, enable_tools={ai_request.enable_tools}")
    logger.debug(f"AI request {request_id} prompt: '{ai_request.prompt}'")
    
    start_time = time.time()
    
    try:
        site_url = os.getenv("SITE_URL", "https://sms-gateway-api.example.com")
        site_name = os.getenv("SITE_NAME", "SMS Gateway API")
        
        # Initialize messages array with user's prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to search tools."
            },
            {
                "role": "user",
                "content": ai_request.prompt
            }
        ]
        
        # Set up request parameters
        request_params = {
            "extra_headers": {
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            "extra_body": {},
            "model": ai_request.model,
            "messages": messages
        }
        
        # Add tools if enabled
        if ai_request.enable_tools:
            request_params["tools"] = TOOLS
            logger.debug(f"AI request {request_id} using tools: {json.dumps([t['function']['name'] for t in TOOLS])}")
        
        # First API call - might request tool use
        logger.debug(f"AI request {request_id} making first API call")
        first_call_start = time.time()
        
        response = openai_client.chat.completions.create(**request_params)
        
        first_call_duration = time.time() - first_call_start
        logger.debug(f"AI request {request_id} first API call completed in {first_call_duration:.2f}s")
        logger.debug(f"AI request {request_id} raw response: {response}")
        
        # Check if the response is valid
        if not hasattr(response, 'choices') or not response.choices:
            logger.warning(f"AI request {request_id} received empty choices in response")
            return AIResponse(response="I apologize, but I couldn't generate a response at this time. Please try again later.")
            
        # Get the response message
        response_message = response.choices[0].message
        
        # Check if the message is valid
        if not response_message or not hasattr(response_message, 'content'):
            logger.warning(f"AI request {request_id} received invalid message in response")
            return AIResponse(response="I apologize, but I couldn't generate a response at this time. Please try again later.")
        
        # Add the LLM's response to messages
        try:
            messages.append(response_message.model_dump())
        except Exception as e:
            logger.warning(f"AI request {request_id} failed to serialize message: {str(e)}")
            # Fallback - create a simplified message dict
            messages.append({
                "role": "assistant",
                "content": response_message.content
            })
        
        # Check if the model wants to use a tool
        has_tool_calls = hasattr(response_message, 'tool_calls') and response_message.tool_calls
        
        if has_tool_calls:
            logger.info(f"AI request {request_id} model requested tool use")
            
            # Process each tool call
            for i, tool_call in enumerate(response_message.tool_calls):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"AI request {request_id} executing tool call {i+1}: {tool_name}")
                logger.debug(f"AI request {request_id} tool call {i+1} arguments: {json.dumps(tool_args)}")
                
                # Execute the tool
                if tool_name in TOOL_MAPPING:
                    tool_start_time = time.time()
                    
                    tool_response = TOOL_MAPPING[tool_name](**tool_args)
                    
                    tool_duration = time.time() - tool_start_time
                    logger.info(f"AI request {request_id} tool call {i+1} executed in {tool_duration:.2f}s")
                    
                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_response)
                    })
                    logger.debug(f"AI request {request_id} tool call {i+1} response added to messages")
                else:
                    logger.warning(f"AI request {request_id} requested unknown tool: {tool_name}")
            
            # Make a second API call with the tool results
            logger.debug(f"AI request {request_id} making second API call with tool results")
            second_call_start = time.time()
            
            second_response = openai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": site_name,
                },
                extra_body={},
                model=ai_request.model,
                messages=messages,
                tools=TOOLS if ai_request.enable_tools else None
            )
            
            second_call_duration = time.time() - second_call_start
            logger.debug(f"AI request {request_id} second API call completed in {second_call_duration:.2f}s")
            
            # Validate second response
            if not hasattr(second_response, 'choices') or not second_response.choices:
                logger.warning(f"AI request {request_id} received empty choices in second response")
                return AIResponse(response="I apologize, but I couldn't generate a final response after using tools. Please try again later.")
            
            final_response = second_response.choices[0].message.content if hasattr(second_response.choices[0].message, 'content') else "No response content."
            
            total_duration = time.time() - start_time
            logger.info(f"AI request {request_id} completed in {total_duration:.2f}s (with tool use)")
            
            # Return the final response
            return AIResponse(response=final_response)
        else:
            logger.info(f"AI request {request_id} completed without tool use")
            
            total_duration = time.time() - start_time
            logger.info(f"AI request {request_id} completed in {total_duration:.2f}s")
            
            # No tool was used, return the first response
            return AIResponse(response=response_message.content if hasattr(response_message, 'content') else "No response content.")
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"AI request {request_id} failed after {duration:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating AI response: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)