# mAjor jet - Pothole Detection

Real-time pothole detection pipeline utilizing YOLOv8 and optimized for the NVIDIA Jetson platform with TensorRT.

## Project Overview

This repository contains the code and models to perform real-time inference on a live camera stream or video file for pothole detection. The pipeline handles:
- Real-time video processing using OpenCV.
- Object detection leveraging the YOLOv8 architecture.
- Exporting models to ONNX and deploying with TensorRT for accelerated inference on Jetson devices.

## Requirements

Please see `requirements.txt` for Python dependencies.

## Key Files
- `jetson_inference.py`: Python script for running inference using YOLOv8 models.
- `export_tensorrt.sh`: Script to convert models to TensorRT engines.
- `master_guide.md`: Main project documentation.
- Model weights: `best.pt`, `last.pt`, `best.onnx`
