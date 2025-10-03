#!/bin/bash

# This script downloads the vectordb using azcopy and then starts the application

set -e  # Exit on any error

echo "Starting TAIC Smart Assistant..."

# Check if vectordb directory already exists and has content
if [ -d "/app/vectordb/all_document_types.lance" ] && [ "$(ls -A /app/vectordb/all_document_types.lance 2>/dev/null)" ]; then
    echo "Vector database already exists, skipping download..."
else
    echo "Vector database not found, downloading..."
    
    # Check if SAS_TOKEN environment variable is set
    if [ -z "$SAS_TOKEN" ]; then
        echo "Error: SAS_TOKEN environment variable is not set."
        echo "Please set the SAS_TOKEN environment variable with a valid SAS token for the Azure Blob Storage."
        exit 1
    fi
    
    # Download the vector database using azcopy
    echo "Downloading vector database using azcopy..."
    azcopy cp "https://taicdocumentsearcherdata.blob.core.windows.net/vectordb/prod/*?${SAS_TOKEN}" /app/vectordb --recursive
    
    echo "Vector database download completed!"
fi

echo "Starting the application..."

# Start the FastAPI application
exec uvicorn app:app --host 0.0.0.0 --port 7860