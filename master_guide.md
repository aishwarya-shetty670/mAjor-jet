# 🚀 Pothole Detection Master Guide (YOLOv8 & Jetson Nano)

Follow these end-to-end steps to prepare your downloaded Roboflow dataset, train your YOLOv8 model in Google Colab, export it properly, and run real-time hardware-accelerated inference on your NVIDIA Jetson Nano.

## 1. Google Colab Setup and Training

Since you already have the dataset exported from Roboflow in YOLOv8 format, we will use Google Colab to offload the heavy GPU processing required for training.

**Steps:**
1. Upload the provided `colab_training.ipynb` from this workspace into your Google Drive, and open it via Google Colab.
2. Go to **Runtime > Change runtime type** in the top titlebar and select **T4 GPU** (or A100/V100 if you have Pro).
3. Paste your Roboflow API configuration snippet into the second code block.
4. Run all notebook cells. The dataset will be downloaded directly to Colab's workspace.
5. The **YOLOv8n (nano)** model is chosen because its minimal parameter footprint runs significantly faster on edge devices like the Jetson Nano.
    - **Optimized parameters** used: `epochs=50`, `imgsz=640`, `batch=16`.
6. Once training finishes, the notebook will automatically display the validation/results curves (`results.png`).

---

## 2. Saving the Model Permanently
At the bottom of the `colab_training.ipynb` sequence, the script handles data retention:
1. It requests permission to mount your Google Drive.
2. It automatically transfers the `best.pt` file (the highest validation-performing model state) and `last.pt` into a dedicated folder `Pothole_Detection_Model`.
3. By doing this, you capture the final artifact securely. You simply need to download the `best.pt` file locally inside VS Code for testing and deployment.

---

## 3. Exporting the Model to ONNX
You cannot easily run an unoptimized PyTorch `best.pt` on the Jetson Nano at acceptable frame rates. We must export it to an ONNX interoperability layer, and later to TensorRT.
1. Download the `best.pt` model to your local PC.
2. Using Ultralytics on your PC or inside Colab, execute the Python export code:
   ```python
   from ultralytics import YOLO
   model = YOLO("best.pt")
   
   # Use smaller resolution for edge deployment
   model.export(format="onnx", imgsz=320, dynamic=True) 
   ```
3. A `best.onnx` file will be generated in the same directory. Download this newly generated ONNX file.

---

## 4. Jetson Nano Deployment 

**Transferring the Architecture:**
1. Transfer the `best.onnx` file and `jetson_inference.py` to your Jetson Nano, via a USB memory stick or SSH/SCP:
   ```bash
   scp best.onnx username@jetson_ip:/home/username/pothole_workspace
   ```

**TensorRT Hardware Conversion (`trtexec`):**
1. The Jetson Nano contains integrated Nvidia Maxell/Volta hardware cores that are executed effectively via TensorRT configurations.
2. On your Jetson Nano console, you MUST run the `export_tensorrt.sh` bash script provided:
   ```bash
   chmod +x export_tensorrt.sh
   ./export_tensorrt.sh
   ```
3. This creates an ultra-fast `best.engine` file using standard FP16 mathematical precision (which the Jetson's GPU handles natively instead of rigid FP32 sizes). 

---

## 5. Real-Time Detection Inference
With the optimized `best.engine` output generated successfully, execute the real-time Python script.

Run the logic script on Jetson Nano:
```bash
python3 jetson_inference.py
```
- Your application will leverage OpenCV (`cv2.VideoCapture`) to seamlessly capture visual frames from your attached Web Camera module.
- The Python script overlays standard colored bounding box rectangles and percentages dynamically directly to the interface view.

---

## 6. Performance Optimization Variables

To mitigate and effectively avoid video lagging, stutters, and delay on specifically the Jetson Nano series limitations:

1. **Resolution Bounds Settings**: We have strictly defined `imgsz=320` in the `.predict()` call line inside the interface script. Anything significantly larger than a 416 window frame size will overload the Jetson's local memory blocks and plummet frame yields.
2. **Expected Execution Hardware FPS Rate Targets**: 
    - Directly invoking PyTorch (`best.pt`): ~1 to 3 frames per second. Too slow for driving implementations.
    - Post-TensorRT Engine execution (`best.engine`) dialed to 320px utilizing FP16: Expected Output **~15 to 25 FPS**, perfectly standard for a realtime dashboard stream view.
3. **Jetson Cooling / Throttle Commands**: Guarantee that your Jetson is provisioned with peak operating wattage limits active and non-throttled. Execute these lines right after a Nano reboot:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

---

## 7. VS Code Environment Project Setup

Standardized file hierarchy map inside of your current VS Code workspace (`d:\mAjor jet` folder):
```text
/mAjor jet
  ├── colab_training.ipynb      <-- Used inside your Google Colab browser session
  ├── jetson_inference.py       <-- Main target script that detects streams
  ├── export_tensorrt.sh        <-- Model bridging/conversion bash utility
  ├── requirements.txt          <-- Global dependency listings
  ├── master_guide.md           <-- This exact documentation guide
  ├── best.pt                   <-- Unoptimized raw YOLOv8 model weighting layer
  └── best.engine               <-- Hardware-specific binary (Only works on Nano OS!)
```

**Executing Tests Locally on Windows/Mac (Inside VS Code) initially**:
Run `pip install -r requirements.txt`.
Make absolutely certain that the variable `model_path = "best.pt"` within your python logic script remains targeted to the generic PyTorch suffix for testing runs. Desktop architectures process PyTorch layers cleanly, only shifting logic to the `.engine` variant when porting logic physically on edge.

---

## 8. Common Errors Handling 

### Within your Google Colab Environment Workspace
- **Issue:** *CUDA Core "Out of Memory" Server Faults (OOM)*
    - **Correction Fix:** Adjust and downsize the internal mathematical `batch` limit size scalar within the notebook from 16 to 8 or 4 in the `.train()` arguments parameter.
- **Issue:** *Missing Roboflow Location / "YAML Object Mapping Not Found"*
    - **Correction Fix:** Ensure your exact project name ID and dataset `version` digits match the dashboard correctly. Utilize standard Python pathing `data=dataset.location + '/data.yaml'` explicitly to fetch definitions correctly.

### Within your Jetson Nano Device Console
- **Issue:** *Error Output: `trtexec command not found` processing ONNX to TensorRT.*
    - **Correction Fix:** Explicitly register the NVidia libraries application binary pathway strings to your shell's global references vector mapping list. It commonly resides locally at: `/usr/src/tensorrt/bin/trtexec`.
- **Issue:** *Slow FPS lag outputs even after applying TensorRT engine acceleration configurations.*
    - **Correction Fix:** Verify precisely whether the Jetson Nano processor board represents bottleneck throttling resulting primarily from local voltage load limiters dropping operations (Assure your device connects leveraging standard 5V 4A DC barrel jack inputs over weaker micro-USB protocols), or severe unmonitored thermal constraint throttling behaviors (Safely attach standard rotational heatsink fans onto your board fins!).
- **Issue:** *Blank Black Outputs / Camera Output Index Not Initializing (`Camera Open Source == 0` runtime failures).*
    - **Correction Fix:** Should you physically attach an interface like the Raspberry Pi Video Module v2 (CSI ribbon configurations), baseline OpenCV software distributions absolutely lack correct encoding capabilities natively reading arrays without a bridge layer. Mutate your `cam_source` scalar allocation block string nested deeply sequentially inside the Python script logic to execute a rigid NVIDIA GStreamer standard configuration parameter variable (The string variable is readily included securely as a comment within your file base).
