#!/bin/bash
set -e

# Extract version from pyproject.toml
VERSION=$(awk -F'"' '/^version =/ {print $2}' haiku_rag_slim/pyproject.toml)

if [ -z "$VERSION" ]; then
  echo "Error: Could not extract version from haiku_rag_slim/pyproject.toml"
  exit 1
fi

echo "Building Docker images for version: $VERSION"
echo ""

# Build slim image (haiku.rag-slim)
echo "Building slim image (haiku.rag-slim)..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/ggozad/haiku.rag-slim:latest \
  --tag ghcr.io/ggozad/haiku.rag-slim:$VERSION \
  --file docker/Dockerfile.slim \
  --push \
  .

echo "✓ Slim image built and pushed"
echo ""

# Build full image (haiku.rag)
echo "Building full image (haiku.rag)..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/ggozad/haiku.rag:latest \
  --tag ghcr.io/ggozad/haiku.rag:$VERSION \
  --file docker/Dockerfile \
  --push \
  .

echo "✓ Full image built and pushed"
