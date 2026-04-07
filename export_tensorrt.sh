#!/bin/bash
# Script to convert exported YOLOv8 ONNX model to TensorRT engine on Jetson Nano
# This assumes you have already exported the PyTorch (.pt) model to ONNX using ultralytics.

MODEL_NAME="best"

echo "Converting ${MODEL_NAME}.onnx to ${MODEL_NAME}.engine utilizing FP16 optimization..."

# Using standard TensorRT executable installed on JetPack
# Set workspace up to 2048 to allow the builder more scratch memory.
/usr/src/tensorrt/bin/trtexec \
  --onnx=${MODEL_NAME}.onnx \
  --saveEngine=${MODEL_NAME}.engine \
  --fp16 \
  --workspace=2048

echo "Conversion Complete."
echo "You can now use ${MODEL_NAME}.engine in your jetson_inference.py script to achieve maximum FPS rates."
