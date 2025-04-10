#!/bin/bash

# SMS Gateway API Deployment Script
# This script builds and deploys the SMS Gateway API using Docker

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
IMAGE_NAME="sms-gateway-api"
CONTAINER_NAME="sms-gateway"
API_PORT=5000
DATA_DIR="./data"
LOG_DIR="./logs"

# Print a message with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log "Docker is not installed. Please install Docker first."
    exit 1
fi

# Create data directory for SQLite database if it doesn't exist
mkdir -p $DATA_DIR
log "Data directory: $DATA_DIR"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR
log "Log directory: $LOG_DIR"

# Build Docker image
log "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# Check if container already exists and remove it
if docker ps -a | grep -q $CONTAINER_NAME; then
    log "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# Run the container
log "Starting container: $CONTAINER_NAME"
docker run -d \
    --name $CONTAINER_NAME \
    -p $API_PORT:8000 \
    -v $DATA_DIR:/app/data \
    -v $LOG_DIR:/app/logs \
    -e DATABASE_PATH=/app/data/sms_gateway.db \
    -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-your-openrouter-api-key}" \
    -e SITE_URL="${SITE_URL:-https://sms-gateway-api.example.com}" \
    -e SITE_NAME="${SITE_NAME:-SMS Gateway API}" \
    -e SERPAPI_API_KEY="${SERPAPI_API_KEY:-your-serpapi-api-key}" \
    -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
    -e LOG_FILE="${LOG_FILE:-/app/logs/sms_gateway.log}" \
    -e LOG_MAX_SIZE="${LOG_MAX_SIZE:-10485760}" \
    -e LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-5}" \
    --restart unless-stopped \
    $IMAGE_NAME

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    log "Container $CONTAINER_NAME is now running"
    log "API is accessible at http://localhost:$API_PORT"
    log "API documentation is available at http://localhost:$API_PORT/docs"
    log "Logs are available in $LOG_DIR"
else
    log "Failed to start container. Please check the logs with: docker logs $CONTAINER_NAME"
    exit 1
fi 