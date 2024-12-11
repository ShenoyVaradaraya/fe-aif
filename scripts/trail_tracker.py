import time
from collections import defaultdict, deque

import cv2
import numpy as np

import aif


class ObjectTracker:
    def __init__(self, max_disappeared=4, max_trail_length=10, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store tracked objects: ID -> bbox
        self.disappeared = defaultdict(int)  # Count frames object has disappeared
        self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))  # Store trails
        self.speeds = defaultdict(lambda: deque(maxlen=5))  # Store recent speeds
        self.stop_probability = defaultdict(lambda: deque(maxlen=5))  # Store recent stop probabilities
        self.last_timestamps = {}  # Store timestamps for speed calculation
        
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.pixels_per_meter = 50  # Calibration factor - adjust based on your setup
        self.fps = 30  # Frame rate - adjust based on your setup
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0
    
    def calculate_center(self, bbox):
        """Calculate center point of bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def calculate_speed(self, trail, timestamp):
        """Calculate speed in meters per second using recent trail points."""
        if len(trail) < 2:
            return 0.0
            
        # Calculate distance between last two points
        p1 = trail[-2]
        p2 = trail[-1]
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convert to meters
        distance = pixel_distance / self.pixels_per_meter
        
        # Calculate time difference
        time_diff = 1.0 / self.fps
            
        # Calculate speed in meters per second
        speed = distance / time_diff
        return speed
    
    def get_average_speed(self, object_id):
        """Get average speed over recent measurements."""
        if not self.speeds[object_id]:
            return 0.0
        return sum(self.speeds[object_id]) / len(self.speeds[object_id])
    
    def update(self, detections):
        """Update object tracking with new detections."""
        current_timestamp = time.time()
        
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_tracking_results()
        
        # Initialize array for current detections
        current_boxes = np.array([det[:4] for det in detections])
        
        # If we have no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self.register(det)
        else:
            # Get IDs of existing objects
            object_ids = list(self.objects.keys())
            previous_boxes = np.array([self.objects[id_] for id_ in object_ids])
            
            # Calculate IoU between each existing object and each new detection
            iou_matrix = np.zeros((len(previous_boxes), len(current_boxes)))
            for i, previous_box in enumerate(previous_boxes):
                for j, current_box in enumerate(current_boxes):
                    iou_matrix[i, j] = self.calculate_iou(previous_box, current_box)
            
            # Find best matches using IoU
            matched_indices = []
            for i in range(len(previous_boxes)):
                if np.max(iou_matrix[i]) > self.iou_threshold:
                    j = np.argmax(iou_matrix[i])
                    matched_indices.append((i, j))
                    iou_matrix[:, j] = 0  # Mark this detection as matched
            
            # Update matched objects
            unmatched_objects = set(range(len(previous_boxes)))
            unmatched_detections = set(range(len(current_boxes)))
            
            for i, j in matched_indices:
                object_id = object_ids[i]
                self.objects[object_id] = current_boxes[j]
                self.disappeared[object_id] = 0
                
                # Update trail and calculate speed
                center = self.calculate_center(current_boxes[j])
                self.trails[object_id].append((int(center[0]), int(center[1])))
                
                speed = self.calculate_speed(self.trails[object_id], current_timestamp+0.1)
                stop_prob = aif.process_trajectories_from_file(self.trails[object_id])
                self.stop_probability[object_id].append(stop_prob)
                self.speeds[object_id].append(speed)
                self.last_timestamps[object_id] = current_timestamp
                
                unmatched_objects.remove(i)
                unmatched_detections.remove(j)
            
            # Handle unmatched objects
            for i in unmatched_objects:
                object_id = object_ids[i]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects for unmatched detections
            for j in unmatched_detections:
                self.register(detections[j])
                
        return self.get_tracking_results()
    
    def get_average_stop_probability(self, object_id):
        """Get average stop probability over recent measurements."""
        if not self.stop_probability[object_id]:
            return 0.0
        return sum(self.stop_probability[object_id]) / len(self.stop_probability[object_id])
    def register(self, detection):
        """Register new object."""
        bbox = detection[:4]
        # print(bbox)
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        center = self.calculate_center(bbox)
        self.trails[self.next_object_id].append((int(center[0]), int(center[1])))
        self.last_timestamps[self.next_object_id] = time.time()
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister disappeared object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trails[object_id]
        del self.speeds[object_id]
        del self.last_timestamps[object_id]
    
    def get_tracking_results(self):
        """Return current tracking results including trails and speeds."""
        results = {
            'objects': self.objects,
            'trails': dict(self.trails),
            'speeds': {id_: self.get_average_speed(id_) for id_ in self.objects},
            'stop_probabilities': {id_: self.get_average_stop_probability(id_) for id_ in self.objects}

        }
        return results
    def draw_trails(self, img):
        # print(self.trails.values())
        """Draw trails on the image"""
        for trail in self.trails.values():
            for pt in trail:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (255, 255, 255), -1)
        # cv2.imshow("Trails", img)

    def show_speed_with_text(self,img, results):
        """Show speed of each object on the image."""
        for object_id, bbox in results['objects'].items():
            trajectory = results['trails'][object_id]
            speed = results['speeds'][object_id]
            speed_text = f"       {speed:.2f} m/s"
            stop_prob = results['stop_probabilities'][object_id]
            p1,p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            text = f"{stop_prob:.3f}"
            # cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            if stop_prob > 0.4:
                cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
            else:
                cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
            # cv2.putText(img, speed_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# from collections import defaultdict, deque

# import cv2
# import numpy as np


# class TrailTracker:
#     def __init__(self, max_trail_length=1000):
#         self.trails = defaultdict(lambda: deque(maxlen=max_trail_length))
#         self.max_trail_length = max_trail_length
#         self.next_id = 1
    
#     def update(self, det):
#         """Update trails with new detections"""
#         if len(det):
#             for *xyxy, conf, cls in det:
#                 # Use center point of bbox as trail point
#                 center_x = (xyxy[0] + xyxy[2]) / 2
#                 center_y = (xyxy[1] + xyxy[3]) / 2
                
#                 # Create unique ID based on class and position (simple approach)
#                 # In practice, you might want to use a proper tracking algorithm
#                 box_id = f"{int(cls)}_{int(center_x)}_{int(center_y)}"
                
#                 # print(box_id, center_x, center_y)
#                 self.trails[box_id].append((int(center_x), int(center_y)))
            
#         return self.trails
    
#     def draw_trails(self, img):

#         # print(self.trails.values())
#         """Draw trails on the image"""
#         for trail in self.trails.values():
#             for pt in trail:
#                 cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
#         # cv2.imshow("Trails", img)

#     def cleanup(self):
#         """Remove old trails"""
#         current_trails = dict(self.trails)
#         for box_id, trail in current_trails.items():
#             if len(trail) > 2:
#                 del self.trails[box_id]

#     def transform_trail_to_fisheye(self):

#         calib_json = "/home/varadaraya-shenoy/RnD/umntc/Thesis/thesis_ws/cts/code/calib/WV-SFV481_calib.json"
#         calib = json.load(open(calib_json))
#         K = np.array(calib['K'])
#         D = np.array(calib['D'])
        
#         current_trails = dict(self.trails)
#         for box_id, trail in current_trails.items():
#             points = trail.reshape(-1, 1, 2)
        
#             # Transform points using fisheye model
#             distorted_points = cv2.fisheye.distortPoints(points, K, D)
            
#             # Reshape back to Nx2
#             distorted_points = distorted_points.reshape(-1, 2)
            
#             self.trails[box_id] = distorted_points
#         return self.trails