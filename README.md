# Real-Time SMS Gateway API

A FastAPI-based service that provides a WebSocket connection for Android devices to send SMS messages through a centralized API.

## Features

- Real-time SMS sending via WebSocket connections to Android devices
- REST API for sending SMS messages and checking status
- Device management and monitoring (battery level, signal strength)
- Simple load balancing across connected devices
- SQLite database for message storage

## Requirements

- Python 3.7+
- FastAPI
- Pydantic
- Uvicorn
- WebSockets
- SQLite (included with Python)

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Configure your settings in `.env` file (copy from `.env.example`)

## Usage

### Starting the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `POST /api/send-sms` - Send a new SMS message
- `GET /api/sms/{message_id}` - Check status of a specific message
- `GET /api/sms` - List all SMS messages with pagination
- `GET /api/devices` - List all connected devices
- `WebSocket /ws/device/{device_id}` - WebSocket endpoint for device connections

### Example API Calls

#### Sending an SMS

```bash
curl -X POST "http://localhost:8000/api/send-sms" \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+1234567890", "message": "Hello from the SMS Gateway!"}'
```

#### Checking SMS status

```bash
curl -X GET "http://localhost:8000/api/sms/123e4567-e89b-12d3-a456-426614174000"
```

#### Listing SMS messages

```bash
curl -X GET "http://localhost:8000/api/sms?limit=10&offset=0"
```

#### Listing connected devices

```bash
curl -X GET "http://localhost:8000/api/devices"
```

## Device Connection Protocol

Android devices connect via WebSocket and must:

1. Establish a WebSocket connection to `/ws/device/{device_id}`
2. Send initial device information (device name, battery level, etc.)
3. Listen for incoming SMS messages to send
4. Report status updates for messages being processed

## Database

The application uses SQLite for storing SMS messages. The database file is created automatically at startup and is configured via the `DATABASE_PATH` environment variable.

### Message IDs

Message IDs are generated as UUIDs (Universally Unique Identifiers) in the format: `123e4567-e89b-12d3-a456-426614174000`. These provide a reliable way to uniquely identify each message without the risk of collision.

## Swagger UI Documentation

The API comes with built-in Swagger UI documentation, which can be accessed at:

```
http://localhost:8000/docs
```

The documentation includes:
- Interactive endpoints that can be tested directly from the browser
- Request and response schemas
- Example curl commands for each endpoint
- Detailed descriptions of parameters and responses

## Environments

- **Development**: Run with reload flag for quick development
- **Testing**: Use test fixtures instead of real device connections
- **Production**: Consider replacing SQLite with a more robust database for high-load environments

## Security Considerations

For production use:
- Consider implementing authentication for API endpoints
- Consider using a more scalable database like PostgreSQL for high loads
- Encrypt WebSocket communications
- Implement rate limiting

## License

[MIT License](LICENSE)
