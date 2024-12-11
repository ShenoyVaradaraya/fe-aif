# YOLO Detection on Fisheye Videos

This project demonstrates object detection on fisheye videos using the YOLO model. The fisheye videos are first unwrapped into a panoramic view to ensure more effective object detection by reducing distortions inherent in the fisheye format.

## Features
- Converts fisheye videos to panoramic format using OpenCV.
- Applies YOLO detection to identify objects in the unwrapped panoramic video.

## Requirements
- Python 3.8+
- The following Python libraries:
  - `opencv-python`
  - `torch`
  - `ultralytics`
  - `numpy`
  - `matplotlib`

You can install the dependencies with the following command:

```bash
pip3 install -r requirements.txt
```

## Usage
1. **Unwrapping the Fisheye Video**:
   Prepare the fisheye video for processing by unwrapping it into a panoramic format. Update the script with the path to your fisheye video.

2. **Running the Detection**:
   Use the provided Python script to perform object detection on the panoramic video. Replace `yolo_model_path` with the path to your YOLO weights file.

### Command to Run
```bash
python3 fisheye2pano.py --video <path_to_fisheye_video> 
```

### Arguments
- `--input_video`: Path to the fisheye video file.

### Example
```bash
python fisheye2pano.py --video data/input_fisheye.mp4
```
<span style="color:red;">**Note:** The repository does not contain any videos , only the python scripts.</span>
## Output
The script shows YOLO detections overlaid on the unwrapped panoramic format.

## Notes
- Adjust the YOLO confidence threshold and model path based on your specific use case and requirements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
