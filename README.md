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
 1. **Calibrate the fisheye camera**:
   Calibrate the camera to determine the extrinsics and intrinsics of the camera.
      ### Command to run calibration 
      ```bash
      python3 scripts/calibrator.py --calibrate --image_dir --save_calibration --load_calibration --undistort_image --undistort_video --output_path
      ```
      ### Arguments 
      - ```--calibrate```: Perform calibration
      - ```--image_dir```: Directory containing chessboard images
      - ```--save_calibration``` :"Path to save calibration data
      - ```--load_calibration``` :"Path to load calibration data
      - ```--undistort_image``` :"Path to input image for undistortion
      - ```--undistort_video``` :"Path to input video for undistortion
      - ```--output_path``` :"Path to save output image or video

2. **Unwrapping the Fisheye Video**:
   Prepare the fisheye video for processing by unwrapping it into a panoramic format. Update the script with the path to your fisheye video.

3. **Running the Detection**:
   Use the provided Python script to perform object detection on the panoramic video. Replace `yolo_model_path` with the path to your YOLO weights file.
   ### Command to Run
   ```bash
   python3 fisheye2pano.py --video <path_to_fisheye_video> --start_nv 0 --end_nv 1
   ```

   ### Arguments
   - `--input_video`: Path to the fisheye video file.
   - `--start_nv` : to run the detection and tracking for a specific view
   - `--end_nv`: `start_nv+1` such that only the `start_nv` view is used for detection and tracking

   ### To run multiple views at once 
   ```bash
   ./scripts/run_multiple_view.sh
   ```

   ### Example
   ```bash
   python fisheye2pano.py --video data/input_fisheye.mp4
   ```
4. **Transformation to SPC System**
   Transform pixel co-ordinates to state-plane coordinate systems using ```pyproj``
   ### Command to run an example 
   ```bash 
   python3 GeoPixelTransformer.py 
   ```

   <span style="color:red;">**Note:** The repository does not contain any videos , only the python scripts.</span>
   ## Output
   The script shows YOLO detections and tracks overlaid on the unwrapped panoramic format with the bounding box color signifying how probable the vehicle will stop. Red means higher probability, green means low probability.


## Notes
- Adjust the YOLO confidence threshold and model path based on your specific use case and requirements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
