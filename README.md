# MonoDepth-Assist: Monocular Detection + Depth Fusion for Navigation

**MonoDepth-Assist** is a monocular object detection and depth estimation pipeline designed to enhance navigation for visually impaired users.

By combining [YOLOv8](https://github.com/ultralytics/ultralytics) for real-time object detection with [Depth-Anything](https://huggingface.co/spaces/DepthAnything/Depth-Anything) for high-quality monocular depth estimation, the system:

- Detects and classifies obstacles from a single RGB image  
- Estimates their relative distance and absolute depth  
- Provides intuitive Left / Center / Right spatial labeling  
- Outputs annotated images and structured CSV reports  

All of this is achieved without LiDAR, stereo cameras, or additional sensors.

---

## Key Features

- Real-time object detection using [YOLOv8](https://github.com/ultralytics/ultralytics)  
- Monocular depth estimation with [Depth-Anything](https://huggingface.co/spaces/DepthAnything/Depth-Anything)  
- Detection–depth fusion: median depth per object + spatial direction labeling  
- Annotated image output with bounding boxes, distance, and depth error metrics  
- Optional evaluation against ground-truth depth maps  
- Lightweight – no calibration or additional sensors required  

---

## Installation

Recommended: Python 3.9+ with PyTorch and CUDA (optional, for GPU acceleration)

### 1. Install PyTorch  
(choose the correct CUDA version for your GPU)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install other dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install ultralytics opencv-python matplotlib pandas scikit-image transformers timm tqdm notebook
```

---

## Dataset

This project uses the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) for validation and testing.

- RGB images: `leftImg8bit_trainvaltest/`  
- Optional ground truth depth: `gt_depth/` or `gtFine_trainvaltest/`  

The dataset is large (~10 GB+), so these folders are ignored in Git via `.gitignore`.

---

## Usage

Set the paths in your script or Jupyter Notebook:

```
IMG_PATH = "./leftImg8bit_trainvaltest/val/frankfurt/frankfurt_000000_003025_leftImg8bit.png"
GT_DEPTH_PATH = "./gt_depth/frankfurt_000000_003025_depth.png"  # optional
```

Run the pipeline (Jupyter Notebook recommended):

```
MonoDepth-Assist.ipynb
```

---

## Example Output

| Class   | Confidence | Rel. Depth | Est. Distance (m) | Direction |
|--------|------------|------------|------------------|-----------|
| car    | 0.95       | 0.32       | 35.1             | Center    |
| person | 0.87       | 0.45       | 29.4             | Left      |

Annotated images include:

- Bounding boxes  
- Relative depth and estimated distance labels  
- Left / Center / Right direction markers  
- Absolute and relative depth errors (optional)

---

## Optional Configuration

| Parameter           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| YOLO weights        | [yolov8n.pt](https://github.com/ultralytics/ultralytics/releases), [yolov8s.pt](https://github.com/ultralytics/ultralytics/releases), [yolov8m.pt](https://github.com/ultralytics/ultralytics/releases), [yolov8l.pt](https://github.com/ultralytics/ultralytics/releases) (choose size/accuracy trade-off) |
| Depth model         | `"depth-anything/Depth-Anything-V2-small-hf"` ([Hugging Face link](https://huggingface.co/spaces/DepthAnything/Depth-Anything)) – larger models available for higher quality |
| Output directory    | Change `OUT_DIR` in the script to save results elsewhere                    |

---

## Performance Highlights

- Most objects have an absolute depth error < 1 m  
- Estimated distances are concentrated between 5 – 25 m, consistent with typical urban navigation scenarios  
- Mean IoU per class shows strong performance on large, nearby objects and reveals challenges with small or distant objects

---

## License & References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – Real-time object detection  
- [Depth-Anything](https://huggingface.co/spaces/DepthAnything/Depth-Anything) – Monocular depth estimation  
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) – Urban street scenes for evaluation
