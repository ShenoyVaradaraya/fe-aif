import glob
import json
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm

class FisheyeCalibrator:
    def __init__(self, chessboard_size=(8, 8), square_size=30):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))

    def calibrate(self, image_dir):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        objp = np.zeros((1, self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

        objpoints = []
        imgpoints = []

        images = glob.glob(f'{image_dir}/*.png')
        calibration_flags = (cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             cv.fisheye.CALIB_CHECK_COND +
                             cv.fisheye.CALIB_FIX_SKEW)

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            detect_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
            ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, detect_flags)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
                imgpoints.append(corners2)

        rms, _, _, _, _ = cv.fisheye.calibrate(
            objpoints, imgpoints, gray.shape[::-1], self.K, self.D, None, None,
            calibration_flags, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

        print("Calibration RMS error:", rms)
        return rms

    def save_calibration(self, filename,sensorW=5.54, sensorH=5.54, focal=1.38, imgrows=2992, imgcols=2992):
        calibration_data = {
            "sensorW": sensorW,
            "sensorH": sensorH,
            "focal": focal,
            "imgrows": imgrows,
            "imgcols": imgcols,
            "K": self.K.tolist(),
            "D": self.D.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
            self.K = np.array(calibration_data['K'])
            self.D = np.array(calibration_data['D'])

class FisheyeUndistorter:
    @staticmethod
    def undistort_image(image, K, D, balance=1):
        h, w = image.shape[:2]
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)
        undistorted = cv.remap(image, map1, map2, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        return undistorted

    @staticmethod
    def process_video(input_path, output_path, K, D):
        cap = cv.VideoCapture(input_path)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
        for _ in tqdm(range(frame_count), desc="Processing Frames"):
            ret, frame = cap.read()
            if not ret:
                break
            undistorted_frame = FisheyeUndistorter.undistort_image(frame, K, D)
            out.write(undistorted_frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fisheye Camera Calibration and Undistortion")
    parser.add_argument('--calibrate', action='store_true', help="Perform calibration")
    parser.add_argument('--image_dir', type=str, help="Directory containing chessboard images")
    parser.add_argument('--save_calibration', type=str, help="Path to save calibration data")
    parser.add_argument('--load_calibration', type=str, help="Path to load calibration data")
    parser.add_argument('--undistort_image', type=str, help="Path to input image for undistortion")
    parser.add_argument('--undistort_video', type=str, help="Path to input video for undistortion")
    parser.add_argument('--output_path', type=str, help="Path to save output image or video")
    args = parser.parse_args()

    calibrator = FisheyeCalibrator()

    if args.calibrate:
        if not args.image_dir:
            print("Error: --image_dir is required for calibration.")
            exit(1)
        calibrator.calibrate(args.image_dir)
        if args.save_calibration:
            calibrator.save_calibration(args.save_calibration)

    if args.load_calibration:
        calibrator.load_calibration(args.load_calibration)

    if args.undistort_image:
        if not args.output_path:
            print("Error: --output_path is required for saving the undistorted image.")
            exit(1)
        image = cv.imread(args.undistort_image)
        undistorted = FisheyeUndistorter.undistort_image(image, calibrator.K, calibrator.D)
        cv.imwrite(args.output_path, undistorted)

    if args.undistort_video:
        if not args.output_path:
            print("Error: --output_path is required for saving the undistorted video.")
            exit(1)
        FisheyeUndistorter.process_video(args.undistort_video, args.output_path, calibrator.K, calibrator.D)
