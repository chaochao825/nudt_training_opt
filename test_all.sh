#!/bin/bash

# Configuration
IMAGE_NAME="decision-ai-training:v1.0"
PROJECT_DIR=$(pwd)
INPUT_DIR="/data6/user23215430/nudt_training_opt/input"
OUTPUT_DIR="/data6/user23215430/nudt_training_opt/output"

# Create output directory if not exists
mkdir -p "$OUTPUT_DIR"

# Build the image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "------------------------------------------------"
echo "Starting Training Test - VGG16 on CIFAR-10"
echo "------------------------------------------------"

docker run --rm --shm-size=1g \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    -e INPUT_DIR=/input \
    -e OUTPUT_DIR=/output \
    -e model=vgg16 \
    -e dataset=CIFAR-10 \
    -e epochs=1 \
    -e batch_size=8 \
    -e lr=0.001 \
    -e process=train \
    -e DEBUG_MODE=true \
    "$IMAGE_NAME"

echo "------------------------------------------------"
echo "Starting Optimization Test - ResNet50 on CIFAR-10"
echo "------------------------------------------------"

docker run --rm --shm-size=1g \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    -e INPUT_DIR=/input \
    -e OUTPUT_DIR=/output \
    -e model=resnet50 \
    -e dataset=CIFAR-10 \
    -e epochs=1 \
    -e batch_size=8 \
    -e lr=0.0001 \
    -e process=optimize \
    -e DEBUG_MODE=true \
    "$IMAGE_NAME"

echo "------------------------------------------------"
echo "Starting Development Test - Inception V3 on CIFAR-10"
echo "------------------------------------------------"

docker run --rm --shm-size=1g \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    -e INPUT_DIR=/input \
    -e OUTPUT_DIR=/output \
    -e model=inception_v3 \
    -e dataset=CIFAR-10 \
    -e epochs=1 \
    -e batch_size=8 \
    -e lr=0.0001 \
    -e process=develop \
    -e DEBUG_MODE=true \
    "$IMAGE_NAME"

echo "Tests completed. Results are in $OUTPUT_DIR"
