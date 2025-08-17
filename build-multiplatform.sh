#!/bin/bash

# Create a new builder instance for multi-platform builds
docker buildx create --name multiplatform-builder --use --bootstrap

# Build for both AMD64 and ARM64 platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag topiaruss/haiku-rag:latest \
  --push \
  .

echo "The Multi-platform build is nowcomplete for linux/amd64 and linux/arm64"