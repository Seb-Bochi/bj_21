#!/bin/bash
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed or not in PATH. Please install Docker first."
    exit 1
fi

docker build -f dockerfiles/train_uv.dockerfile . -t train:project
docker run -v $(pwd)/models:/models train:project
