#!/bin/bash

# SMS Gateway API Deployment Script
# This script builds and deploys the SMS Gateway API using Docker

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
IMAGE_NAME="sms-gateway-api"
CONTAINER_NAME="sms-gateway"
API_PORT=5000
DATA_DIR="./data"

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
    -e DATABASE_PATH=/app/data/sms_gateway.db \
    -e API_KEY="${API_KEY:-your-secret-api-key}" \
    --restart unless-stopped \
    $IMAGE_NAME

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    log "Container $CONTAINER_NAME is now running"
    log "API is accessible at http://localhost:$API_PORT"
    log "API documentation is available at http://localhost:$API_PORT/docs"
else
    log "Failed to start container. Please check the logs with: docker logs $CONTAINER_NAME"
    exit 1
fi 