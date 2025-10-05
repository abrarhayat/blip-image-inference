#!/usr/bin/env bash
# start.sh — Start the FastAPI app with Uvicorn in development mode
# Usage: ./start.sh [--env-file path/to/.env] [--port 8000]
# Requirements: Python 3.11 environment, dependencies from requirements.txt

# Default values
ENV_FILE=".env"
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do # While there are arguments left to process
    case $1 in # Check what the first argument is
        --env-file) #pattern1)
            ENV_FILE="$2"
            shift 2 #- Removes the first **two** arguments (--env-file and its value) from the argument list
            ;;
        --port) #pattern2)
            PORT="$2"
            shift 2
            ;;
        -h|--help) #pattern3)
            echo "Usage: $0 [--env-file FILE] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --env-file FILE    Path to .env file (default: .env)"
            echo "  --port PORT        Port to run on (default: 8000)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *) # default case
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if the env file exists
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    uvicorn app.main:app --reload --port "$PORT" --env-file "$ENV_FILE"
else
    echo "Warning: $ENV_FILE not found. Starting without explicit env file."
    echo "Environment variables will be loaded from system or .env via python-dotenv"
    uvicorn app.main:app --reload --port "$PORT"
fi