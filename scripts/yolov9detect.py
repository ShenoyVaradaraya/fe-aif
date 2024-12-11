import cv2
import numpy as np
import torch
from ultralytics import YOLO


class yolov9_detect:
    def __init__(self, conf=0.2, iou=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('yolov9e-seg.pt').to(self.device)
        # Classes : 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign', 12: 'parking meter'
        self.classes = [0, 1, 2, 3, 5, 7, 9, 11, 12] 
        self.target_size = 640
        self.conf = conf
        self.iou = iou

    def resize_frame_for_yolov9(self, frame, target_size=640):
        """
        Resize the frame to 640x640 for YOLOv9 while maintaining the aspect ratio.
        Padding is applied to fill the remaining space.
        
        Args:
            frame (numpy array): Input frame.
            target_size (int): Target size for YOLOv9 (default is 640).
        
        Returns:
            resized_frame (numpy array): Frame resized to the target size.
            scale (float): Scale factor applied to the frame.
            padding (tuple): Padding added (top, bottom, left, right).
        """
        h, w, _ = frame.shape
        scale = target_size / max(h, w)  # Scale factor to fit the largest side
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize frame while keeping aspect ratio
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Calculate padding
        top_pad = (target_size - new_h) // 2
        bottom_pad = target_size - new_h - top_pad
        left_pad = (target_size - new_w) // 2
        right_pad = target_size - new_w - left_pad

        # Add padding to make it 640x640
        padded_frame = cv2.copyMakeBorder(
            resized_frame, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))  # Use gray padding

        return padded_frame, scale, (top_pad, bottom_pad, left_pad, right_pad)

    def detect(self, frame, resize=True):
        if resize:
            # Resize frame to 640x640
            resized_frame, scale, padding = self.resize_frame_for_yolov9(frame, target_size=640)
        else:
            resized_frame = frame
        
        # Convert frame to tensor and move to GPU
        resized_frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        
        # Run YOLOv9 detection
        results = self.model.track(resized_frame_tensor, persist=True, conf=self.conf,
                                   iou=self.iou, show=False, classes=self.classes, tracker="bytetrack.yaml")

        # Visualize results (convert tensor back to numpy for OpenCV)
        annotated_frame = results[0].plot()

        if not resize:
            # Scale back to original size
            h, w, _ = frame.shape
            original_frame = cv2.resize(annotated_frame, (w, h))
        else:
            original_frame = annotated_frame

        return original_frame, results


if __name__ == '__main__':
    yolov9 = yolov9_detect()
    video = cv2.VideoCapture('./videos/panoramic.avi')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        annotated_frame, _ = yolov9.detect(frame)
        cv2.imshow("YOLOv9 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
