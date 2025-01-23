import os
import time
from collections import defaultdict, deque

import aif
import cv2
import GeoPixelTransformer as tf
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


class DetectandTrackingModule:
    def __init__(self, fps, conf=0.2, iou=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('yolo11n-seg.pt').to(self.device)
        self.class_map = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign', 12: 'parking meter'}
        self.classes = [0, 1, 2, 3, 5, 7, 9, 11, 12]
        self.target_size = 640
        self.conf = conf
        self.iou = iou
        self.track_history = defaultdict(lambda: [])
        self.latitudes, self.longitudes = tf.prepare_latlon('data/Portland_66th_GE_Points.xlsx')
        self.trails = defaultdict(lambda: deque(maxlen=10))  # Store trails
        self.speeds = defaultdict(lambda: deque(maxlen=5))  # Store recent speeds
        self.stop_probability = defaultdict(lambda: deque(maxlen=5))  # Store recent stop probabilities
        self.last_timestamps = {}  # Store timestamps for speed calculation
        self.fps = 0.1  # Frame rate - adjust based on your setup

    def transform_to_easting_northing(self, x, y, shape):
        transformer = tf.GeoPixelTransformer(shape, self.latitudes, self.longitudes)
        return transformer.get_relative_spc_from_pixel(x, y)

    def resize_frame(self, frame, target_size=640):
        h, w, _ = frame.shape
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        top_pad = (target_size - new_h) // 2
        bottom_pad = target_size - new_h - top_pad
        left_pad = (target_size - new_w) // 2
        right_pad = target_size - new_w - left_pad
        padded_frame = cv2.copyMakeBorder(
            resized_frame, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded_frame, scale, (top_pad, bottom_pad, left_pad, right_pad)

    def calculate_speed(self, trail,shape,dt):
        if len(trail) < 2:
            return 0.0
        p1 = trail[-2]
        p2 = trail[-1]
        transformer1 = tf.GeoPixelTransformer(shape, self.latitudes, self.longitudes)
        p1_spc_e,p1_spc_n = transformer1.get_relative_spc_from_pixel(p1[1], p1[0])
        p2_spc_e,p2_spc_n = transformer1.get_relative_spc_from_pixel(p2[1], p2[0])
        distance = np.sqrt((p2_spc_e - p1_spc_e) ** 2 + (p2_spc_n - p1_spc_n) ** 2)
        time_diff = 1.0 / self.fps
        return distance / time_diff

    def get_average_speed(self, object_id):
        if not self.speeds[object_id]:
            return 0.0
        return sum(self.speeds[object_id]) / len(self.speeds[object_id])

    def get_average_stop_probability(self, object_id):
        if not self.stop_probability[object_id]:
            return 0.0
        return sum(self.stop_probability[object_id]) / len(self.stop_probability[object_id])

    def detect_and_track(self, frame, nv, dt, resize=True):
        csv_rows = []
        history_rows = []
        resized_frame, scale, padding = self.resize_frame(frame, target_size=self.target_size)
        results = self.model.track(resized_frame, persist=True, conf=self.conf,iou=self.iou,classes=list(self.class_map.keys()), tracker="bytetrack.yaml")
        boxes = results[0].boxes.xywh.cpu()

        # Read existing CSV if it exists to track previously detected objects
        file_path = f'data/tracking_data/tracking_data_{nv}.csv'
        try:
            df_existing = pd.read_csv(file_path)
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=["view_id", "track_id", "object_class", "northing", "easting", "speed", "stop_probability"])

         # Read the history CSV to maintain the history of easting/northing for each track_id
        history_file_path = f'data/tracking_data/tracking_history_{nv}.csv'
        try:
            df_history = pd.read_csv(history_file_path)
        except FileNotFoundError:
            df_history = pd.DataFrame(columns=["track_id", "view_id", "northing", "easting"])
        if results[0].boxes.id is not None and results[0].boxes.cls is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            object_classes = results[0].boxes.cls.int().cpu().tolist()
            annotated_frame = results[0].plot()

            for box, track_id, obj_class in zip(boxes, track_ids, object_classes):
                x, y, w, h = box
                easting, northing = self.transform_to_easting_northing(x, y, annotated_frame.shape)
                trail = self.trails[track_id]
                trail.append((float(x), float(y)))
                if len(trail) > 30:
                    trail.pop(0)

                speed = self.calculate_speed(trail, annotated_frame.shape, dt) * 0.681818  # Convert to miles/h
                stop_prob = aif.process_trajectories_from_file(trail)
                self.speeds[track_id].append(speed)
                self.stop_probability[track_id].append(stop_prob)
                class_name = self.class_map.get(obj_class, "unknown")

                # Update CSV rows, append new detections or update existing ones
                new_row = {
                    "view_id": nv,
                    "track_id": track_id,
                    "object_class": class_name,
                    "northing": northing,
                    "easting": easting,
                    "speed": speed,
                    "stop_probability": stop_prob
                }

                # Check if the track_id already exists in the CSV (i.e., previously detected object)
                existing_row_idx = df_existing[df_existing['track_id'] == track_id].index
                if len(existing_row_idx) > 0:
                    # Update existing row
                    df_existing.loc[existing_row_idx, ['speed', 'stop_probability']] = [speed, stop_prob]
                else:
                    # Add new row if object is not in the CSV
                    csv_rows.append(new_row)

                # Add the easting and northing to the history CSV
                history_row = {
                    "track_id": track_id,
                    "view_id": nv,
                    "northing": northing,
                    "easting": easting
                }
                history_rows.append(history_row)
                points = np.hstack(trail).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=3)

                # Draw bounding box with color based on stop probability
                color = (0, 0, 255) if stop_prob > 0.4 else (0, 255, 0)
                top_left = (int(x - w / 2), int(y - h / 2))
                bottom_right = (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(annotated_frame, top_left, bottom_right, color, 2)
        else:
            annotated_frame = resized_frame

        # Resize back to original frame if necessary
        if not resize:
            h, w, _ = frame.shape
            original_frame = cv2.resize(annotated_frame, (w, h))
        else:
            original_frame = annotated_frame

        # Append any new rows to the DataFrame and save it
        df_existing = df_existing.append(csv_rows, ignore_index=True)
        df_existing.to_csv(file_path, index=False)
        if history_rows:
            df_history = df_history.append(history_rows, ignore_index=True)
            df_history.to_csv(history_file_path, index=False)

        return original_frame, df_existing

