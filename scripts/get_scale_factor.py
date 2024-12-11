import cv2
from geopy.distance import geodesic

# Global variables to store points selected by the user
points = []

# Mouse callback function to capture points on the image
def select_points(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the points when the user clicks on the image
        points.append((x, y))

        # Draw a circle on the selected point and display it
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", frame)

        # If two points are selected, calculate the pixel distance
        if len(points) == 2:
            # Draw a line between the points
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Select Points", frame)
            print(f"Point 1: {points[0]}")
            print(f"Point 2: {points[1]}")

# Function to calculate scale factor
def calculate_scale_factor(gps_coords_1, gps_coords_2, pixel_distance):
    # Step 1: Calculate the real-world distance between the GPS points (in meters)
    print(f"Pixel distance between the points: {pixel_distance} pixels")
    real_world_distance = geodesic(gps_coords_1, gps_coords_2).feet

    # Step 2: Calculate the scale factor (pixels per meter)
    scale_factor = pixel_distance / real_world_distance
    return scale_factor

def convert_position_to_pixels(x_in_feet, y_in_feet, pixels_per_foot):
    # Convert the x and y positions from feet to pixels
    x_in_pixels = x_in_feet * pixels_per_foot
    y_in_pixels = y_in_feet * pixels_per_foot
    return x_in_pixels, y_in_pixels

# Example list of (x, y) positions in feet

# Load your image
video = cv2.VideoCapture('./Tue_2017-03-14_073002_reduced_fps.avi')
ret, frame = video.read()

# # Display the image and wait for user input
cv2.imshow("Select Points", frame)
cv2.setMouseCallback("Select Points", select_points)

# # Wait for the user to select two points
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Once points are selected, calculate the scale factor (example using predefined GPS coordinates)
# gps_coords_1 = (44.8837613, -93.26788027)  # Point 1: San Francisco
# gps_coords_2 = (44.88375531, -93.2677683)  # Point 2: Close to San Francisco

# # Measure pixel distance between selected points
# pixel_distance = cv2.norm(points[0], points[1], cv2.NORM_L2)
# scale_factor = calculate_scale_factor(gps_coords_1, gps_coords_2, pixel_distance)

scale_factor = 10.6
print(f"Scale factor (pixels per meter): {scale_factor}")
positions_in_feet = [(72.888,31.224)]

# Convert positions to pixels
positions_in_pixels = [convert_position_to_pixels(x, y, scale_factor) for x, y in positions_in_feet]
for x, y in positions_in_pixels:
    # Draw a circle at each position (in pixels)
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circle with a radius of 5

# Display the frame with overlay
cv2.imshow("Image with Overlay", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()